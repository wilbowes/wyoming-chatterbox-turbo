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

An NVIDIA GPU with the NVIDIA Container Toolkit. Turbo takes about 1.8 GB of
VRAM. The image is built on PyTorch 2.9.1 with CUDA 12.8, which RTX 50-series
cards need and older cards also run.

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
docker compose up -d --build
```

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

MIT. Chatterbox-Turbo is Resemble AI's, also MIT.
