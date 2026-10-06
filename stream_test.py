#!/usr/bin/env python3
"""Replay HA's streaming TTS order (components/wyoming/tts.py, dev 2026-09):
synthesize-start, synthesize-chunk per LLM token, synthesize (full text),
synthesize-stop; read until synthesize-stopped. Words arrive at ~40 tok/s,
roughly gemma's rate, so 'first audio' is what an Echo would wait for."""
import asyncio, os, sys, time, wave
from wyoming.audio import AudioChunk, AudioStart
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped, SynthesizeVoice

async def main(out, text, voice=None):
    v = SynthesizeVoice(name=voice) if voice else None
    c = AsyncTcpClient(os.environ.get("HOST", "127.0.0.1"), 10304); await c.connect()
    t0 = time.monotonic()
    async def write():
        await c.write_event(SynthesizeStart(voice=v).event())
        for w in text.split(" "):
            await c.write_event(SynthesizeChunk(text=w + " ").event()); await asyncio.sleep(0.025)
        print(f"  text finished at {(time.monotonic()-t0)*1000:.0f}ms")
        await c.write_event(Synthesize(text=text, voice=v).event())
        await c.write_event(SynthesizeStop().event())
    asyncio.create_task(write())
    pcm, first, fmt, starts = b"", None, None, 0
    while (e := await c.read_event()):
        if AudioStart.is_type(e.type): fmt = fmt or AudioStart.from_event(e); starts += 1
        elif AudioChunk.is_type(e.type):
            first = first or time.monotonic() - t0; pcm += AudioChunk.from_event(e).audio
        elif SynthesizeStopped.is_type(e.type): break
    await c.disconnect()
    with wave.open(out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fmt.rate); w.writeframes(pcm)
    print(f"  {out}: {len(pcm)/(fmt.rate*2):.1f}s audio, FIRST AUDIO {first*1000:.0f}ms, "
          f"done {(time.monotonic()-t0)*1000:.0f}ms, audio-starts {starts}")

asyncio.run(main(*sys.argv[1:]))
