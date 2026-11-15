# wyoming-chatterbox-turbo

Text-to-speech for Home Assistant from
[Chatterbox-Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo), over
the [Wyoming protocol](https://github.com/OHF-Voice/wyoming). The model runs
in the same process as the Wyoming server, and each voice is a short
reference clip.

It came out of [EchoMuse](https://github.com/wilbowes/EchoMuse), where it has
been the voice on one household's Echo Dots since late September 2026.

## What it does

- **Streams.** Home Assistant sends the reply as the LLM writes it, and each
  sentence is spoken when its full stop arrives, so the first sentence plays
  while the rest are generated.
- **Speaks in sentence groups** of up to 200 characters. Chatterbox drifts
  into nonsense syllables on long text.
- **Strips Markdown**: headings, bullets, emphasis, links, tables and emoji,
  and reads `°C` as "degrees".
- **Sets the loudness** of every piece to -16 LUFS with a soft limiter. Turbo
  alone speaks at about -24 to -27.
- **Maps vocal events**: `(laugh)`, `(sigh)`, `(cough)` in the text become
  Turbo's own `[chuckle]`, `[sigh]`, `[cough]` tags.

English only.

## Requirements

A GPU with about 2 GB of free memory, or no GPU and a CPU with AVX2
([Without a GPU](#without-a-gpu)). One image per stack:

| GPU | Image | Compose file | Host needs |
|-----|-------|--------------|------------|
| NVIDIA | `:latest` | `docker-compose.yml` | the NVIDIA driver and Container Toolkit |
| AMD (ROCm 6.4) | `:rocm` | `docker-compose.amd.yml` | the `amdgpu` kernel driver |
| Intel Arc, Core Ultra (XPU) | `:xpu` | `docker-compose.intel.yml` | the `i915` or `xe` kernel driver |
| none | `:cpu` | `docker-compose.cpu.yml` | an x86-64 CPU with AVX2; an Intel iGPU helps |

The NVIDIA image is built on PyTorch 2.9.1 with CUDA 12.8, which RTX 50-series
cards need and older cards also run. The AMD and Intel images use PyTorch's
own 2.9.1 wheels, which carry their GPU runtime, so the host needs only the
kernel driver.

**The AMD and Intel images are built and checked in CI but have not yet been
run on AMD or Intel hardware.** If you run one, an issue saying which card and
what `docker logs` printed on its first line would help.

The first log line names the GPU in use. Without one the GPU images run on
the CPU, several times slower than real time; use the `:cpu` image instead. `--device cuda|xpu|cpu` (or `DEVICE=`)
chooses instead of detecting; AMD under ROCm is `cuda`.

## Without a GPU

The `:cpu` image splits the model so it streams in real time on a desktop
CPU:

- **T3** (text to speech tokens) runs in
  [chatterbox.cpp](https://github.com/gianni-cor/chatterbox.cpp), 8-bit, as
  one resident process that keeps the voice loaded between sentences and
  hands over each token as it is made. The image carries a build for AVX2,
  AVX2 with VNNI and AVX-512, and picks one for the CPU it starts on.
- **The flow estimator** runs in OpenVINO, on an Intel iGPU if one is passed
  in, otherwise on the CPU.
- **Each sentence streams in chunks**, so the first audio does not wait for
  the whole sentence.
- **One flow step** instead of Turbo's two, and **the last 3 seconds** of
  the reference clip for the audio stage. Neither could be heard in a blind
  A/B.

Measured on one machine only, an Intel i5-12500 (6 cores) with UHD 770
graphics, time from the reply's text to its first audio:

| Reply | iGPU | CPU only |
|-------|------|----------|
| "Done." | 0.49 s | 0.48 s |
| one 15-word sentence | 1.31 s | 1.43 s |
| "Done." then a 15-word sentence | 0.45 s, 0.29 s gap | 0.49 s, 0.47 s gap |

In a full Home Assistant voice turn (speech to text, a local LLM, this) it
started speaking 1.8 to 2.0 s after the end of speech, against 1.3 to 2.3 s
for the NVIDIA image on an RTX 5060 Ti.

The first start makes three model files from Resemble's weights in
`./models` (about 850 MB, a minute or two) and checks the two OpenVINO
graphs against PyTorch. To pass an Intel iGPU in, uncomment the `devices` and
`group_add` lines in `docker-compose.cpu.yml`; startup then takes about 40 s
longer to compile its shapes. `THREADS` defaults to one per physical core, at
most 8: hyperthreads and efficiency cores made it slower, not faster.

**Not yet run on an AMD CPU.** If you run it on one, an issue with the CPU
model and the `t3d` line from `docker logs` would help.

## Voices

Put reference clips in `voices/` and list them in `voices/voices.json`
(`voices.json.example` shows the shape):

```json
{
  "my-voice": {"ref": "my-voice.wav"},
  "my-other-voice": {"ref": "my-other-voice.wav", "temperature": 0.5}
}
```

A clip must be longer than 5 seconds, one speaker, continuous. If the clone
loses the accent of a short clip, loop the clip to about 15 seconds and set
`"temperature": 0.5`. No voices are included: use recordings you have the
right to clone.

## Run

```
git clone https://github.com/wilbowes/wyoming-chatterbox-turbo && cd wyoming-chatterbox-turbo
cp voices/voices.json.example voices/voices.json   # then add your clips
docker compose pull                     # AMD: -f docker-compose.amd.yml, Intel: -f docker-compose.intel.yml, no GPU: -f docker-compose.cpu.yml
docker compose up -d
```

The image is `ghcr.io/wilbowes/wyoming-chatterbox-turbo` (linux/amd64,
versions tagged as `:0.1.0`, `:0.1.0-rocm`, `:0.1.0-xpu` and `:0.1.0-cpu`;
the NVIDIA one is about 18 GB, the CPU one 2.7 GB). To build it yourself, use
`docker compose up -d --build`. Each image carries a build-provenance
attestation:
`gh attestation verify oci://ghcr.io/wilbowes/wyoming-chatterbox-turbo:0.1.0 --repo wilbowes/wyoming-chatterbox-turbo`.

The first start downloads the model into `./hf-cache`. The default voice is
the first one in `voices.json`, or add `command: --default-voice my-voice` to
the compose file.

In Home Assistant: Settings → Devices & services → Add integration →
**Wyoming Protocol**, with this machine's address and port `10304`. Then pick
`chatterbox-turbo` and a voice as the text-to-speech of your Assist pipeline.

Each reply is logged with its length, time to first audio and total time:
`docker logs -f wyoming-chatterbox-turbo`.

## Testing the stream

`stream_test.py` replays the order Home Assistant sends a streamed reply in,
word by word, and reports the time to first audio:

```
pip install wyoming==1.8.0
HOST=192.168.1.10 python3 stream_test.py out.wav "Hello there. This is a test." my-voice
```

## Licence

MIT. Chatterbox-Turbo is Resemble AI's, also MIT. The CPU image builds
chatterbox.cpp (MIT) with the patch in `cpu/`.
