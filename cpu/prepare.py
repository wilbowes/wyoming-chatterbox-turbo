"""First run of the CPU image: make the three derived model files from
Resemble's published weights, on the user's own machine, into MODELS_DIR.

- t3-q8_0.gguf      T3 for chatterbox.cpp (its own converter, 8-bit: on CPUs
                    with VNNI, 4-bit was slower)
- estimator.onnx    the S3Gen flow estimator, for OpenVINO
- hift_core.onnx    HiFT's convolution core (conv_pre .. conv_post), for OpenVINO

The two ONNX graphs are traced from one sentence in the model's built-in voice
and checked against PyTorch before they are kept. Each file is written under a
temporary name and renamed, so an interrupted run starts again cleanly.
Takes a few minutes; later starts skip it.
"""
import inspect
import logging
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("wyoming-chatterbox-cpu")
CONVERTER = "/opt/chatterbox.cpp/scripts/convert-t3-turbo-to-gguf.py"
LINE = "It's nineteen degrees outside, with a light breeze from the south."
FILES = ("t3-q8_0.gguf", "estimator.onnx", "hift_core.onnx")


def missing(models_dir):
    return [f for f in FILES if not os.path.exists(os.path.join(models_dir, f))]


def _t3(models_dir):
    out = os.path.join(models_dir, "t3-q8_0.gguf")
    tmp = out + ".tmp"
    subprocess.run([sys.executable, CONVERTER, "--quant", "q8_0", "--out", tmp], check=True)
    os.replace(tmp, out)


def _check(name, path, feed, ref):
    import openvino as ov
    got = ov.Core().compile_model(path, "CPU").create_infer_request().infer(feed)[0]
    rel = float(np.abs(got - ref).max() / (np.abs(ref).max() or 1.0))
    log.info("%s: OpenVINO vs PyTorch, max relative difference %.1e", name, rel)
    if rel > 1e-3:
        raise RuntimeError(f"{name}: exported graph does not match PyTorch ({rel:.1e})")


def _estimator(model, models_dir):
    est = model.s3gen.flow.decoder.estimator
    fwd, cap = est.forward, {}
    sig = inspect.signature(fwd)

    def grab(*a, **k):
        if not cap:  # by parameter name, positional or not: the names become the ONNX inputs
            for n, v in sig.bind(*a, **k).arguments.items():
                if torch.is_tensor(v):
                    cap[n] = v.detach().clone()
        return fwd(*a, **k)
    est.forward = grab
    with torch.inference_mode():
        model.generate(LINE)
    est.forward = fwd
    names = list(cap)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.est = est

        def forward(self, *xs):
            return fwd(**dict(zip(names, xs)))
    args = tuple(cap[n].clone() for n in names)
    with torch.no_grad():
        ref = Wrap()(*args).float().numpy()
    T = cap["x"].shape[-1]
    dyn = {n: {2: "T"} for n in names if cap[n].dim() == 3 and cap[n].shape[-1] == T}
    out = os.path.join(models_dir, "estimator.onnx")
    tmp = out + ".tmp.onnx"
    torch.onnx.export(Wrap(), args, tmp, input_names=names, output_names=["dxdt"],
                      dynamic_axes={**dyn, "dxdt": {2: "T"}}, opset_version=17, dynamo=False)
    _check("estimator", tmp, {n: cap[n].float().numpy() for n in names}, ref)
    os.replace(tmp, out)


def _hift(model, models_dir):
    h, cap = model.s3gen.mel2wav, {}
    hi = model.s3gen.hift_inference

    def grab(mel, cache_source=None):
        cap.setdefault("mel", mel.detach().clone())
        return hi(mel, cache_source)
    model.s3gen.hift_inference = grab
    with torch.inference_mode():
        model.generate(LINE)
    model.s3gen.hift_inference = hi

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.h = h

        def forward(self, x, s_stft):
            h = self.h
            x = h.conv_pre(x)
            for i in range(h.num_upsamples):
                x = h.ups[i](F.leaky_relu(x, h.lrelu_slope))
                if i == h.num_upsamples - 1:
                    x = h.reflection_pad(x)
                x = x + h.source_resblocks[i](h.source_downs[i](s_stft))
                xs = None
                for j in range(h.num_kernels):
                    r = h.resblocks[i * h.num_kernels + j](x)
                    xs = r if xs is None else xs + r
                x = xs / h.num_kernels
            return h.conv_post(F.leaky_relu(x))
    mel = cap["mel"].clone()
    with torch.no_grad():
        f0 = h.f0_predictor(mel)
        s = h.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = h.m_source(s)
        re, im = h._stft(s.transpose(1, 2).squeeze(1))
        s_stft = torch.cat([re, im], dim=1)
        ref = Core()(mel, s_stft).numpy()
    out = os.path.join(models_dir, "hift_core.onnx")
    tmp = out + ".tmp.onnx"
    torch.onnx.export(Core(), (mel, s_stft), tmp, input_names=["x", "s_stft"], output_names=["out"],
                      dynamic_axes={"x": {2: "T"}, "s_stft": {2: "TS"}, "out": {2: "TO"}},
                      opset_version=17, dynamo=False)
    _check("hift", tmp, {"x": mel.numpy(), "s_stft": s_stft.numpy()}, ref)
    os.replace(tmp, out)


def prepare(models_dir):
    todo = missing(models_dir)
    if not todo:
        return
    os.makedirs(models_dir, exist_ok=True)
    log.info("first run: making %s in %s (a few minutes, once)", ", ".join(todo), models_dir)
    if "t3-q8_0.gguf" in todo:
        _t3(models_dir)
    if "estimator.onnx" in todo or "hift_core.onnx" in todo:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device="cpu")
        if model.conds is None:
            raise RuntimeError("the checkpoint has no built-in voice (conds.pt) to trace the export with")
        inf = model.s3gen.inference
        model.s3gen.inference = lambda *a, **k: inf(*a, **{**k, "n_cfm_timesteps": 1})
        torch.manual_seed(0)
        if "estimator.onnx" in todo:
            _estimator(model, models_dir)
        if "hift_core.onnx" in todo:
            _hift(model, models_dir)
    log.info("first run done")
