#!/usr/bin/env python3
"""shim.py without a GPU: the same Wyoming server and text handling, with
the model split across the CPU and, when there is one, an Intel iGPU.
- T3 runs in t3d, a resident chatterbox.cpp process (ggml Q8, ~100 tok/s vs
  PyTorch CPU 21 on an i5-12500) holding the voice prefix in its KV cache, fed
  Python's own text tokens so punc_norm/tokenizer stay identical. The image has
  one t3d per x86 feature level; pick_t3d() chooses from /proc/cpuinfo.
- The S3Gen flow estimator runs in OpenVINO: on an Intel GPU (f16) when
  OpenVINO finds one, zero-padded to a 128-frame bucket so no reply meets a
  shape recompile (padding with mask=0 is exact: rel 4e-7), every bucket
  compiled at startup; otherwise on the CPU, unpadded.
- HiFT's conv core (conv_pre..conv_post, 99% of HiFT) runs in OpenVINO on the
  CPU: 0.72 s vs PyTorch 1.0 s on 6.2 s of audio, output identical (rel 2e-6);
  an iGPU is SLOWER for it (UHD 770: f16 1.37 s, f32 2.43 s).
- The rest (flow encoder, f0, source, STFT/ISTFT, watermark) stays in PyTorch.
- One flow step (meanflow), not tts_turbo's 2: no audible difference in a
  blind A/B, flow time -40%.
In a blind A/B the iGPU f16 estimator could not be told from PyTorch.

Within a sentence, audio streams (STREAM=1, StreamHandler): t3d sends each
speech token as it samples it, and the sentence is rendered in growing chunks.
The flow attends both ways, so each chunk re-runs flow over every token so
far and keeps only the new mel frames (finalize=False holds back the last 3
tokens' frames for lookahead); HiFT runs on the new frames plus an 8-frame
mel cache, its source carried over, and the seam is crossfaded with a Hamming
window: CosyVoice2's token2wav streaming, which Chatterbox's S3Gen descends
from. Loudness cannot be measured before the sentence exists, so each voice's
gain is calibrated once at startup and the soft limiter runs per chunk.

Without STREAM, sentences are pipelined (PipeHandler): T3 for the next sentence runs in t3d
while this one is rendered, so a long second sentence no longer leaves a gap
after a short first one (it was ~0.9 s, 11-14).
"""
import argparse, asyncio, json, logging, os, subprocess, tempfile, time

import numpy as np
import openvino as ov
import torch
from chatterbox.tts_turbo import S3GEN_SIL, ChatterboxTurboTTS, punc_norm
from chatterbox.models.s3gen.s3gen import S3Token2Mel
import pyloudnorm as ln
import threading
from concurrent.futures import ThreadPoolExecutor
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

import shim
from cpu.prepare import prepare

log = logging.getLogger("wyoming-chatterbox-cpu")
BUCKET, MAX_BUCKET = 128, 1536


def physical_cores():
    """Hyperthreads and E-cores make it slower, not faster (i5-12500: 6 threads
    beat 12), so the default is one thread per physical core, at most 8."""
    try:
        cores = {l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("core id")}
        return max(1, min(8, len(cores) or os.cpu_count() // 2))
    except OSError:
        return max(1, min(8, (os.cpu_count() or 2) // 2))


def pick_t3d():
    """The t3d built for this CPU: AVX-512 with VNNI, AVX2 with AVX-VNNI, or AVX2."""
    flags = set()
    for l in open("/proc/cpuinfo"):
        if l.startswith("flags"):
            flags = set(l.split(":")[1].split())
            break
    if {"avx512f", "avx512bw", "avx512vl", "avx512_vnni"} <= flags:
        level = "avx512"
    elif {"avx2", "fma", "f16c", "avx_vnni"} <= flags:
        level = "avx2-vnni"
    elif {"avx2", "fma", "f16c"} <= flags:
        level = "avx2"
    else:
        raise SystemExit("this CPU has no AVX2: Chatterbox would be far slower than real time on it")
    return os.environ.get("T3D") or f"/opt/t3d/{level}/t3d", level


THREADS = int(os.environ.get("THREADS") or physical_cores())
EST_DEVICE = os.environ.get("EST_DEVICE", "auto")  # auto | GPU | CPU (OpenVINO device names)
T3_THREADS = int(os.environ.get("T3_THREADS", str(THREADS)))  # t3d runs beside the renderer while streaming
STREAM = os.environ.get("STREAM", "1") == "1"
FIRST_TOK = int(os.environ.get("FIRST_TOK", "25"))  # first chunk; it emits ~(n-3)*40 ms - 160 ms of audio
MIN_TOK = int(os.environ.get("MIN_TOK", "20"))  # later chunks wait for at least this many new tokens
MEL_CACHE, SRC_CACHE = 8, 8 * 480  # CosyVoice2 values: 8 mel frames = 160 ms at 24 kHz
# Every chunk re-runs flow over the reference prompt (10 s = 500 frames). The last 3 s sounded
# the same in an A/B; whole-sentence S3Gen 2.1 -> 1.45 s. 0 = the full clip.
PROMPT_S = int(os.environ.get("PROMPT_S", "3"))
PIPE = os.environ.get("PIPE", "first")  # off | on | first
MODELS = os.environ.get("MODELS_DIR", "/models")
T3_GGUF = os.path.join(MODELS, "t3-q8_0.gguf")  # 8-bit: 4-bit was slower with VNNI
T3D, T3D_LEVEL = pick_t3d()


class Estimator:
    def __init__(self, path, cache, want="auto"):
        core = ov.Core()
        core.set_property({"CACHE_DIR": cache})
        self.device = ("GPU" if "GPU" in core.available_devices else "CPU") if want == "auto" else want
        if self.device == "GPU":
            name = core.get_property("GPU", "FULL_DEVICE_NAME")
            self.rq = core.compile_model(core.read_model(path), "GPU",
                                         {"INFERENCE_PRECISION_HINT": "f16"}).create_infer_request()
        else:
            name = "CPU"
            self.rq = core.compile_model(core.read_model(path), "CPU",
                                         {"INFERENCE_NUM_THREADS": THREADS}).create_infer_request()
        log.info("flow estimator on %s (OpenVINO)", name)

    def prewarm(self, sample):
        if self.device != "GPU":  # the CPU plugin does not recompile per length: nothing to pad or warm
            return
        t = time.monotonic()
        for b in range(BUCKET, MAX_BUCKET + 1, BUCKET):
            short = {k: v[..., :min(b, v.shape[-1])] if v.ndim == 3 else v for k, v in sample.items()}
            for _ in range(2):  # the GPU plugin settles on the second call at a new shape
                self.run(short, b)
        log.info("estimator buckets to %d compiled in %.1fs", MAX_BUCKET, time.monotonic() - t)

    def run(self, feed, bucket=None):
        T = feed["x"].shape[-1]
        if self.device != "GPU":
            return self.rq.infer(feed)[0]
        b = bucket or -(-T // BUCKET) * BUCKET
        padded = {k: np.pad(v, ((0, 0), (0, 0), (0, b - v.shape[-1]))) if v.ndim == 3 else v
                  for k, v in feed.items()}
        return self.rq.infer(padded)[0][..., :T]


def ov_hift_decode(h, path, cache):
    """HiFTGenerator.decode with the conv core in OpenVINO (CPU). The STFT of
    the source and the ISTFT stay in torch: they are ~1% of the time."""
    core = ov.Core()
    core.set_property({"CACHE_DIR": cache})
    rq = core.compile_model(core.read_model(path), "CPU", {"INFERENCE_NUM_THREADS": THREADS}).create_infer_request()
    nf = h.istft_params["n_fft"] // 2 + 1

    def decode(x, s=torch.zeros(1, 1, 0)):
        re, im = h._stft(s.squeeze(1))
        feed = {"x": x.float().numpy(), "s_stft": torch.cat([re, im], dim=1).float().numpy()}
        out = torch.from_numpy(rq.infer(feed)[0].copy())
        wav = h._istft(torch.exp(out[:, :nf, :]), torch.sin(out[:, nf:, :]))
        return torch.clamp(wav, -h.audio_limit, h.audio_limit)
    return decode


CALIBRATION = "It's nineteen degrees outside, with a light breeze from the south. Should be a nice arvo."
WINDOW = torch.from_numpy(np.hamming(2 * SRC_CACHE).astype("float32"))


def limit(wav):
    """The live level()'s soft knee, alone: per chunk, since the peak of a
    sentence is not known until it has all been made."""
    knee, ceil = 10 ** (shim.KNEE_DBFS / 20), 10 ** (shim.PEAK_DBFS / 20)
    over = np.abs(wav) > knee
    wav[over] = np.sign(wav[over]) * (knee + (ceil - knee) * np.tanh((np.abs(wav[over]) - knee) / (ceil - knee)))
    return wav


class CpuTTS(shim.TTS):
    def tokens(self, name, text):
        """Stage 1 (t3d, its own process): text -> speech tokens, as generate() does."""
        conds, temperature = self.voices[name]
        ids = self.model.tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True).input_ids
        t = time.monotonic()
        tok = self._t3(conds.t3, ids, temperature=temperature)
        tok = tok[tok < 6561]
        return torch.cat([tok, torch.tensor([S3GEN_SIL] * 3, dtype=torch.long)]), time.monotonic() - t

    def render(self, name, tokens):
        """Stage 2 (iGPU estimator + CPU): speech tokens -> PCM bytes, levelled
        per piece with the live shim's level() and its inter-sentence gap."""
        conds, _ = self.voices[name]
        t = time.monotonic()
        with torch.inference_mode():
            wav, _ = self.model.s3gen.inference(speech_tokens=tokens, ref_dict=conds.gen)
            wav = wav.squeeze(0).numpy()
            wav = self.model.watermarker.apply_watermark(wav, sample_rate=self.model.sr)
        wav = np.concatenate([shim.level(wav, self.model.sr), np.zeros(int(self.model.sr * shim.GAP_S))])
        return (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes(), time.monotonic() - t

    def say(self, name, text):
        return self.render(name, self.tokens(name, text)[0])[0]

    def calibrate(self):
        """Per-voice gain the live level() would pick, from one whole sentence."""
        self.gain = {}
        # The default voice last: t3d keeps the last voice's prefix in its KV cache,
        # so the first reply does not pay ~1 s to re-evaluate it.
        names = sorted(self.voices, key=lambda n: n == self.default_voice)
        for name in names:
            wav = self.model.s3gen.inference(speech_tokens=self.tokens(name, CALIBRATION)[0],
                                             ref_dict=self.voices[name][0].gen)[0].squeeze(0).numpy()
            lufs = ln.Meter(self.model.sr).integrated_loudness(wav)
            peak = float(np.max(np.abs(wav))) or 1.0
            self.gain[name] = min(shim.TARGET_LUFS - lufs, shim.PEAK_DBFS + shim.MAX_SQUASH_DB - 20 * np.log10(peak))
            log.info("voice %s: %.1f LUFS, gain %+.1f dB", name, lufs, self.gain[name])

    def stream(self, name, text, emit):
        """Speak one sentence, calling emit(pcm_bytes) for each chunk as it is ready."""
        conds, temperature = self.voices[name]
        ids = self.model.tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True).input_ids
        seed = int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
        req = [self.ref_dirs[id(conds.t3)], temperature, 1000, 0.95, 1.2, seed, 1000,
               ",".join(map(str, ids.reshape(-1).tolist()))]
        if self.t3d is None or self.t3d.poll() is not None:
            self.t3d = subprocess.Popen([T3D, T3_GGUF, str(T3_THREADS)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        toks, done, cv, state = [], [False], threading.Condition(), {"emitted_chunks": 0, "retry_late": False}

        def reader():  # t3d's stdout -> toks, while the chunks render
            while True:
                line = self.t3d.stdout.readline()
                with cv:
                    if line.startswith("T "):
                        state.setdefault("tok1", time.monotonic())
                        if not state["retry_late"]:
                            t = int(line[2:])
                            if t < 6561:
                                toks.append(t)
                    elif line.startswith("R"):
                        if state["emitted_chunks"]:  # audio already out: keep what was said
                            state["retry_late"] = True
                        else:
                            toks.clear()
                    else:
                        state["end"] = line
                        done[0] = True
                    cv.notify()
                    if done[0]:
                        return
        self.t3d.stdin.write("\t".join(map(str, req)) + "\n")
        rt = threading.Thread(target=reader, daemon=True); rt.start()
        t0, st, first_at = time.monotonic(), {"emitted": 0, "cache": None}, None
        st["noise"] = torch.randn(1, 80, 2 * 1010)
        rendered, chunks = 0, []
        try:
            self._chunks(name, conds, toks, done, cv, state, st, emit, t0, chunks)
        finally:
            rt.join()  # t3d must finish this request before the next one is written
        rendered, first_at = st["rendered"], st["first_at"]
        end = state.get("end", "")
        if not end.startswith("OK "):
            raise RuntimeError(f"t3d: {end.strip() or 'exited'}")
        stats = end[3:].split("\t")[0].split()
        attempts = stats[3]
        log.info("  t3: prefix %s ms, decode %s ms for %s tokens, first token at %.2fs",
                 stats[0], stats[1], stats[2], state.get("tok1", t0) - t0)
        log.info("streamed %d tokens in chunks %s, first chunk %.2fs, total %.2fs%s", rendered, chunks,
                 first_at or 0, time.monotonic() - t0,
                 f", t3 attempts {attempts}" + (" (retry after audio: kept the first take)" if state["retry_late"] else "")
                 if attempts != "1" else "")

    def _chunks(self, name, conds, toks, done, cv, state, st, emit, t0, chunks):
        rendered, first_at = 0, None
        while True:
            with cv:
                need = FIRST_TOK if rendered == 0 else rendered + MIN_TOK
                cv.wait_for(lambda: done[0] or state["retry_late"] or len(toks) >= need)
                final = done[0] or state["retry_late"]
                cur = list(toks)
            if final or len(cur) > rendered:
                tc = time.monotonic()
                pcm = self._chunk(name, conds, cur, final, st)
                log.info("  chunk %d tok (%d new) at %.2fs: flow %.2f hift %.2f total %.2f", len(cur), len(cur) - rendered,
                         tc - t0, st.get("t_flow", 0), st.get("t_hift", 0), time.monotonic() - tc)
                with cv:
                    state["emitted_chunks"] += 1
                if pcm:
                    emit(pcm)
                    first_at = first_at or time.monotonic() - t0
                chunks.append(len(cur) - rendered)
                rendered = len(cur)
            if final:
                break
        st["rendered"], st["first_at"] = rendered, first_at

    def _chunk(self, name, conds, toks, final, st):
        s3 = self.model.s3gen
        tt = torch.tensor(toks + ([S3GEN_SIL] * 3 if final else []), dtype=torch.long).unsqueeze(0)
        if tt.shape[1] <= 3 and not final:
            return b""
        n_frames = tt.shape[1] * 2 - (0 if final else 2 * 3)
        with torch.inference_mode():
            tf = time.monotonic()
            mels = S3Token2Mel.forward(s3, tt, ref_wav=None, ref_sr=None, ref_dict=conds.gen, n_cfm_timesteps=1,
                                       finalize=final, noised_mels=st["noise"][..., :n_frames])
            st["t_flow"] = time.monotonic() - tf
            new = mels[..., st["emitted"]:]
            st["emitted"] = mels.shape[-1]
            cache = st["cache"]
            if new.shape[-1] == 0 and cache is None:
                return b""
            mel_in = torch.cat([cache["mel"], new], dim=2) if cache else new
            th = time.monotonic()
            wav, source = s3.hift_inference(mel_in, cache["source"] if cache else None)
            st["t_hift"] = time.monotonic() - th
            if cache:
                ov = SRC_CACHE
                wav[:, :ov] = wav[:, :ov] * WINDOW[:ov] + cache["speech"] * WINDOW[ov:]
            else:
                wav[:, :len(s3.trim_fade)] *= s3.trim_fade
            if not final:
                st["cache"] = {"mel": mel_in[..., -MEL_CACHE:], "source": source[..., -SRC_CACHE:],
                               "speech": wav[:, -SRC_CACHE:].clone()}
                wav = wav[:, :-SRC_CACHE]
            wav = wav.squeeze(0).numpy()
            wav = self.model.watermarker.apply_watermark(wav, sample_rate=self.model.sr)
        wav = limit(wav * 10 ** (self.gain.get(name, 0.0) / 20))
        if final:
            wav = np.concatenate([wav, np.zeros(int(self.model.sr * shim.GAP_S))])
        return (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()

    def __init__(self, voice_dir, voices, est_path, cache):
        torch.set_num_threads(THREADS)
        log.info("t3d %s, %d threads (T3 %d)", T3D_LEVEL, THREADS, T3_THREADS)
        self.model = ChatterboxTurboTTS.from_pretrained(device="cpu")
        s3_inference = self.model.s3gen.inference
        self.model.s3gen.inference = lambda *a, **k: s3_inference(*a, **{**k, "n_cfm_timesteps": 1})
        self.tmp = tempfile.mkdtemp(prefix="cbcpu-")
        self.voices, self.ref_dirs = {}, {}
        for name, v in voices.items():
            self.model.prepare_conditionals(os.path.join(voice_dir, v["ref"]))
            conds = self.model.conds
            d = os.path.join(self.tmp, name)
            os.makedirs(d)
            # tts-cli reads these as flat float32 / int32 and does not check the dtype.
            np.save(f"{d}/speaker_emb.npy", conds.t3.speaker_emb.float().numpy().reshape(-1).astype("<f4"))
            np.save(f"{d}/cond_prompt_speech_tokens.npy",
                    conds.t3.cond_prompt_speech_tokens.numpy().reshape(-1).astype("<i4"))
            if PROMPT_S:  # S3Gen's reference: keep the last PROMPT_S seconds (tokens 25/s, mel 50/s)
                g, nt = conds.gen, PROMPT_S * 25
                if g["prompt_token"].shape[1] > nt:
                    g["prompt_token"], g["prompt_token_len"] = g["prompt_token"][:, -nt:], torch.tensor([nt])
                    g["prompt_feat"] = g["prompt_feat"][:, -2 * nt:, :]
                    if g.get("prompt_feat_len") is not None:
                        g["prompt_feat_len"] = torch.tensor([2 * nt])
            self.voices[name] = (conds, v.get("temperature", 0.8))
            self.ref_dirs[id(conds.t3)] = d
            log.info("voice %s from %s", name, v["ref"])
        self.t3d = None
        self.model.t3.inference_turbo = self._t3
        self.est = Estimator(est_path, cache, EST_DEVICE)
        h = self.model.s3gen.mel2wav
        h.decode = ov_hift_decode(h, os.path.join(os.path.dirname(est_path), "hift_core.onnx"), cache)
        dec = self.model.s3gen.flow.decoder.estimator
        self._torch_est, self._seen = dec.forward, None
        dec.forward = self._estimate
        self.say(next(iter(self.voices)), "Warm up.")  # captures a real feed for the prewarm
        self.est.prewarm(self._seen)
        if STREAM:
            self.calibrate()

    def _t3(self, t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95,
            repetition_penalty=1.2, max_gen_len=1000):
        """One request to the resident t3d (src-cbcpp/src/t3d.cpp): the voice's
        prefix stays in its KV cache, so only this sentence's text is evaluated."""
        seed = int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
        req = [self.ref_dirs[id(t3_cond)], temperature, top_k, top_p, repetition_penalty, seed, max_gen_len,
               ",".join(map(str, text_tokens.reshape(-1).tolist()))]
        if self.t3d is None or self.t3d.poll() is not None:
            self.t3d = subprocess.Popen([T3D, T3_GGUF, str(T3_THREADS)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        self.t3d.stdin.write("\t".join(map(str, req)) + "\n")
        reply = self.t3d.stdout.readline()
        while reply[:2] in ("T ", "R\n"):  # streamed tokens: the OK line repeats them
            reply = self.t3d.stdout.readline()
        if not reply.startswith("OK "):
            raise RuntimeError(f"t3d: {reply.strip() or 'exited'}")
        stats, toks = reply[3:].rstrip("\n").split("\t")
        prefix_ms, t3_ms, n = stats.split()[:3]
        if int(prefix_ms): log.info("t3 voice prefix %s ms", prefix_ms)
        return torch.tensor([int(x) for x in toks.split(",") if x], dtype=torch.long).unsqueeze(0)

    def _estimate(self, x, mask, mu, t, spks=None, cond=None, r=None, **kw):
        if x.shape[0] != 1 or r is None or kw:  # not the shape exported: stay correct, stay slow
            return self._torch_est(x, mask, mu, t, spks=spks, cond=cond, r=r, **kw)
        # flow.inference with finalize=False trims h by the lookahead but builds the
        # mask from the untrimmed lengths: 6 frames too long. B=1: all ones, so crop.
        mask = mask[..., :x.shape[-1]]
        feed = {"x": x, "mask": mask, "mu": mu, "t": t, "spks": spks, "cond": cond, "r": r}
        feed = {k: v.float().numpy() for k, v in feed.items()}
        self._seen = self._seen or feed
        return torch.from_numpy(self.est.run(feed).copy()).to(x.dtype)


class PipeHandler(shim.Handler):
    """shim.Handler with a two-stage pipeline per reply. _speak only queues the
    sentence, so the next text event is read at once; stage 1 makes tokens
    (t3d), stage 2 renders and sends audio in order. The executors are shared
    by every connection (one t3d, one estimator), so replies queue as in the
    live shim. Same reply shape: one audio-start, one audio-stop."""
    T3 = ThreadPoolExecutor(1, thread_name_prefix="t3")
    S3 = ThreadPoolExecutor(1, thread_name_prefix="s3")

    def _begin(self, voice):
        super()._begin(voice)
        self.texts, self.toks = asyncio.Queue(), asyncio.Queue()
        self.stages = [asyncio.create_task(self._stage1()), asyncio.create_task(self._stage2())]
        self.timing, self.first_done = [], asyncio.Event()

    async def _speak(self, text: str):
        text = shim.tags(shim.speakable(text.strip()))
        if shim.re.search(r"[A-Za-z0-9]", text):
            await self.texts.put(text)

    async def _stage1(self):
        loop = asyncio.get_running_loop()
        n = 0
        while (text := await self.texts.get()) is not None:
            if n == 1 and PIPE == "first":  # sentence 1 renders alone: it is the first-audio path
                await self.first_done.wait()
            tok, t3s = await loop.run_in_executor(self.T3, self.tts.tokens, self.voice, text)
            await self.toks.put((text, tok, t3s, time.monotonic()))
            n += 1
        await self.toks.put(None)

    async def _stage2(self):
        loop, rate = asyncio.get_running_loop(), self.tts.model.sr
        step = int(rate * shim.CHUNK_S) * shim.WIDTH
        while (item := await self.toks.get()) is not None:
            text, tok, t3s, ready = item
            pcm, rs = await loop.run_in_executor(self.S3, self.tts.render, self.voice, tok)
            if not self.started:
                await self.write_event(AudioStart(rate=rate, width=shim.WIDTH, channels=shim.CHANNELS).event())
                self.started, self.first = True, time.monotonic() - self.t0
            for i in range(0, len(pcm), step):
                await self.write_event(AudioChunk(rate=rate, width=shim.WIDTH, channels=shim.CHANNELS,
                                                  audio=pcm[i:i + step]).event())
            self.samples += len(pcm) // shim.WIDTH
            self.spoken.append(text)
            self.first_done.set()
            self.timing.append(f"{len(tok)}tok t3 {t3s:.2f}+render {rs:.2f}s")

    async def _end(self):
        await self.texts.put(None)
        await asyncio.gather(*self.stages)
        log.info("pipeline: %s", "; ".join(self.timing))
        await super()._end()


class StreamHandler(shim.Handler):
    """shim.Handler where each sentence streams in chunks (CpuTTS.stream). _speak
    queues the sentence so text events keep being read; one worker speaks the
    queue in order. One audio-start, one audio-stop per reply."""
    POOL = ThreadPoolExecutor(1, thread_name_prefix="stream")

    def _begin(self, voice):
        super()._begin(voice)
        self.texts = asyncio.Queue()
        self.worker = asyncio.create_task(self._work())

    async def _speak(self, text: str):
        text = shim.tags(shim.speakable(text.strip()))
        if shim.re.search(r"[A-Za-z0-9]", text):
            await self.texts.put(text)

    async def _send(self, pcm):
        rate = self.tts.model.sr
        if not self.started:
            await self.write_event(AudioStart(rate=rate, width=shim.WIDTH, channels=shim.CHANNELS).event())
            self.started, self.first = True, time.monotonic() - self.t0
        step = int(rate * shim.CHUNK_S) * shim.WIDTH
        for i in range(0, len(pcm), step):
            await self.write_event(AudioChunk(rate=rate, width=shim.WIDTH, channels=shim.CHANNELS,
                                              audio=pcm[i:i + step]).event())
        self.samples += len(pcm) // shim.WIDTH

    async def _work(self):
        loop = asyncio.get_running_loop()
        emit = lambda pcm: asyncio.run_coroutine_threadsafe(self._send(pcm), loop).result()
        while (text := await self.texts.get()) is not None:
            try:
                await loop.run_in_executor(self.POOL, self.tts.stream, self.voice, text, emit)
                self.spoken.append(text)
            except Exception:  # end the reply cleanly rather than leave HA waiting
                log.exception("sentence failed: %r", text)

    async def _end(self):
        await self.texts.put(None)
        await self.worker
        await super()._end()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="tcp://0.0.0.0:10304")
    ap.add_argument("--voice-dir", default="/voices")
    ap.add_argument("--default-voice", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare(MODELS)
    args.estimator, args.cache = os.path.join(MODELS, "estimator.onnx"), os.path.join(MODELS, "ov-cache")
    voices = json.load(open(os.path.join(args.voice_dir, "voices.json")))
    args.default_voice = args.default_voice or next(iter(voices))
    CpuTTS.default_voice = args.default_voice
    tts = CpuTTS(args.voice_dir, voices, args.estimator, args.cache)
    attribution = Attribution(name="Resemble AI", url="https://huggingface.co/ResembleAI/chatterbox-turbo")
    info = Info(tts=[TtsProgram(
        name="chatterbox-turbo-cpu", description="Chatterbox-Turbo 350M (MIT), on the CPU",
        attribution=attribution, installed=True, version="0.1.7", supports_synthesize_streaming=True,
        voices=[TtsVoice(name=n, description=f"clone of {v['ref']}", attribution=attribution,
                         installed=True, version="turbo", languages=["en"]) for n, v in voices.items()])])
    lock = asyncio.Lock()
    log.info("listening on %s, voices %s (default %s)", args.uri, list(voices), args.default_voice)
    await AsyncServer.from_uri(args.uri).run(lambda *a, **kw: (StreamHandler if STREAM else shim.Handler if PIPE == "off" else PipeHandler)(tts, info, args, lock, *a, **kw))

if __name__ == "__main__":
    asyncio.run(main())
