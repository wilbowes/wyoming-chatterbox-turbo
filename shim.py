#!/usr/bin/env python3
"""Wyoming text-to-speech from Chatterbox-Turbo, in-process.

Voices are reference clips (over 5 s, one speaker) named in voices.json:
  {"irish-woman": {"ref": "irish-woman.wav"}, "x": {"ref": "x.wav", "temperature": 0.5}}
A short clip loses its accent to the model's American prior. Two things held
it: loop the clip to fill Turbo's 15 s speech-prompt window, and sample at
temperature 0.5 instead of 0.8. Conditionals are computed once per voice at
startup, not per reply.

An LLM prompted for vocal events in parentheses writes "(laugh)"; Turbo's
native tags are "[laugh]". EVENTS maps one to the other.
"""
import argparse, asyncio, json, logging, os, re, time

import numpy as np
import pyloudnorm as ln
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

log = logging.getLogger("wyoming-chatterbox")
WIDTH, CHANNELS, CHUNK_S = 2, 1, 0.1
EVENTS = {"laugh": "chuckle", "laughs": "chuckle",  # [laugh] is a cackle
          "chuckle": "chuckle", "chuckles": "chuckle", "giggle": "chuckle", "giggles": "chuckle",
          "cough": "cough", "coughs": "cough",
          "sigh": "sigh", "sighs": "sigh", "clears throat": "clear throat"}
PAREN = re.compile(r"\(([a-z ]+)\)", re.I)


def tags(text: str) -> str:
    return PAREN.sub(lambda m: f"[{EVENTS[m[1].lower()]}]" if m[1].lower() in EVENTS else m[0], text)


TARGET_LUFS, PEAK_DBFS, KNEE_DBFS, MAX_SQUASH_DB = -16.0, -1.0, -6.0, 6.0
MAX_CHARS, GAP_S = 200, 0.12
# A sentence ends at . ! ? (but not "1." opening a list item) or at a line
# break: gemma writes lists and headings with no full stop, and without the
# line break they run together as one breathless sentence.
SENTENCE = re.compile(r"(?<=[.!?])(?<!\d\.)\s+|\s*\n\s*")

# An LLM writes Markdown for a screen; spoken, "*" is "asterisk".
MARKDOWN = [
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),         # [text](url) -> text
    (re.compile(r"^\s*(?:#+|>+|[-*+•]|\d+[.)])\s+"), ""),     # heading, quote, bullet, "1."
    (re.compile(r"[*`~]+|(?<!\w)_+|_+(?!\w)"), ""),         # emphasis, code; keeps snake_case
    (re.compile(r"\s*\|\s*"), ", "),                          # table cells
    (re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]"), ""),  # emoji
    (re.compile(r"\s*°\s*C\b"), " degrees"), (re.compile(r"\s*°\s*F\b"), " degrees Fahrenheit"),
    (re.compile(r"\s*°"), " degrees"), (re.compile(r"\s+&\s+"), " and "),
    (re.compile(r"\s{2,}"), " "),
]


def speakable(text: str) -> str:
    """One sentence or line, as it should be said. A line with no closing
    punctuation (a list item, a heading) gets a full stop, so it is spoken as
    a statement rather than trailing off."""
    for pat, rep in MARKDOWN:
        text = pat.sub(rep, text)
    text = text.strip(" ,")
    if text and text[-1] not in ".!?:;,)\"'":
        text += "."
    return text


def chunks(text: str) -> list:
    """Whole sentences, packed up to MAX_CHARS; a longer sentence goes alone."""
    out, cur = [], ""
    for s in filter(None, map(speakable, SENTENCE.split(text.strip()))):
        if cur and len(cur) + 1 + len(s) > MAX_CHARS:
            out.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    return out + ([cur] if cur else [])


def level(wav, sr):
    """Turbo speaks at -24 to -27 LUFS (it normalises its reference to -27);
    on an Echo Dot it was too quiet at -18 as well. Raise to
    TARGET_LUFS, and soft-limit the few peaks that would then pass KNEE_DBFS
    into a ceiling of PEAK_DBFS. A hard peak cap bound first on most replies,
    leaving them short of the target; the limiter is allowed MAX_SQUASH_DB of
    peak reduction and no more, beyond which the gain is cut instead."""
    try:
        lufs = ln.Meter(sr).integrated_loudness(wav)
    except ValueError:  # shorter than one 400 ms gating block
        return wav
    peak = float(np.max(np.abs(wav))) or 1.0
    if not np.isfinite(lufs):
        return wav
    gain_db = min(TARGET_LUFS - lufs, PEAK_DBFS + MAX_SQUASH_DB - 20 * np.log10(peak))
    wav = wav * 10 ** (gain_db / 20)
    knee, ceil = 10 ** (KNEE_DBFS / 20), 10 ** (PEAK_DBFS / 20)
    over = np.abs(wav) > knee
    wav[over] = np.sign(wav[over]) * (knee + (ceil - knee) * np.tanh((np.abs(wav[over]) - knee) / (ceil - knee)))
    return wav


class Handler(AsyncEventHandler):
    """Speaks sentence by sentence, sending each one's audio as soon as it is
    made, so the first sentence plays while the rest are generated.

    Streaming (HA sends synthesize-start / -chunk… / -stop as gemma writes):
    each sentence is spoken the moment its full stop arrives, and the reply
    ends with synthesize-stopped. HA also sends a plain synthesize carrying the
    whole text for servers that cannot stream; inside a stream it is ignored,
    or the reply would be spoken twice. Either way there is ONE audio-start and
    ONE audio-stop per reply: the non-streaming reader stops at the first
    audio-stop."""

    def __init__(self, tts, info, args, lock, *a, **kw):
        super().__init__(*a, **kw)
        self.tts, self.info, self.args, self.lock = tts, info, args, lock
        self.streaming, self.buf, self.voice, self.started = False, "", None, False
        self.t0, self.first, self.samples, self.spoken = 0.0, None, 0, []

    def _voice(self, v):
        return v.name if v and v.name in self.tts.voices else self.args.default_voice

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if SynthesizeStart.is_type(event.type):
            self._begin(SynthesizeStart.from_event(event).voice)
            self.streaming = True
            return True
        if SynthesizeChunk.is_type(event.type):
            self.buf += SynthesizeChunk.from_event(event).text
            *done, self.buf = SENTENCE.split(self.buf)
            for sentence in done:
                await self._speak(sentence)
            return True
        if SynthesizeStop.is_type(event.type):
            await self._speak(self.buf)
            await self._end()
            await self.write_event(SynthesizeStopped().event())
            return False
        if Synthesize.is_type(event.type):
            if self.streaming:
                return True
            syn = Synthesize.from_event(event)
            self._begin(syn.voice)
            for chunk in chunks(syn.text):
                await self._speak(chunk)
            await self._end()
            return False
        return True

    def _begin(self, voice):
        self.voice, self.buf, self.started = self._voice(voice), "", False
        self.t0, self.first, self.samples, self.spoken = time.monotonic(), None, 0, []

    async def _speak(self, text: str):
        text = tags(speakable(text.strip()))
        if not re.search(r"[A-Za-z0-9]", text):
            return
        async with self.lock:  # one GPU, one model: requests queue
            pcm = await asyncio.get_running_loop().run_in_executor(None, self.tts.say, self.voice, text)
        rate = self.tts.model.sr
        if not self.started:
            await self.write_event(AudioStart(rate=rate, width=WIDTH, channels=CHANNELS).event())
            self.started, self.first = True, time.monotonic() - self.t0
        step = int(rate * CHUNK_S) * WIDTH
        for i in range(0, len(pcm), step):
            await self.write_event(AudioChunk(rate=rate, width=WIDTH, channels=CHANNELS,
                                              audio=pcm[i:i + step]).event())
        self.samples += len(pcm) // WIDTH
        self.spoken.append(text)

    async def _end(self):
        rate = self.tts.model.sr
        if not self.started:  # nothing speakable: still a well-formed reply
            await self.write_event(AudioStart(rate=rate, width=WIDTH, channels=CHANNELS).event())
        await self.write_event(AudioStop().event())
        ms = (time.monotonic() - self.t0) * 1000
        log.info("%s%s: %.1fs audio, first audio %.0fms, total %.0fms, %d sentence(s): %r",
                 self.voice, " (streamed)" if self.streaming else "", self.samples / rate,
                 (self.first or 0) * 1000, ms, len(self.spoken), " ".join(self.spoken))


class TTS:
    def __init__(self, voice_dir: str, voices: dict):
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        self.voices = {}
        for name, v in voices.items():
            self.model.prepare_conditionals(os.path.join(voice_dir, v["ref"]))
            self.voices[name] = (self.model.conds, v.get("temperature", 0.8))
            log.info("voice %s from %s", name, v["ref"])

    def say(self, name: str, text: str) -> bytes:
        """One sentence group per call: Chatterbox drifts into babble on long
        text (a 19.6 s story came out as nonsense syllables). Loudness is
        set per piece, since the whole reply never exists at once."""
        self.model.conds, temperature = self.voices[name]
        with torch.inference_mode():
            wav = self.model.generate(text, temperature=temperature).squeeze(0).numpy()
        wav = np.concatenate([level(wav, self.model.sr), np.zeros(int(self.model.sr * GAP_S))])
        return (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="tcp://0.0.0.0:10304")
    ap.add_argument("--voice-dir", default="/voices")
    ap.add_argument("--default-voice", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    voices = json.load(open(os.path.join(args.voice_dir, "voices.json")))
    args.default_voice = args.default_voice or next(iter(voices))
    tts = TTS(args.voice_dir, voices)
    attribution = Attribution(name="Resemble AI", url="https://huggingface.co/ResembleAI/chatterbox-turbo")
    info = Info(tts=[TtsProgram(
        name="chatterbox-turbo", description="Chatterbox-Turbo 350M (MIT)", attribution=attribution,
        installed=True, version="0.1.7", supports_synthesize_streaming=True,
        voices=[TtsVoice(name=n, description=f"clone of {v['ref']}", attribution=attribution,
                         installed=True, version="turbo", languages=["en"]) for n, v in voices.items()])])
    lock = asyncio.Lock()
    log.info("listening on %s, voices %s (default %s)", args.uri, list(voices), args.default_voice)
    await AsyncServer.from_uri(args.uri).run(lambda *a, **kw: Handler(tts, info, args, lock, *a, **kw))

if __name__ == "__main__":
    asyncio.run(main())
