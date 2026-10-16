from __future__ import annotations

import json
import os
import threading
import time

_lock = threading.Lock()
_current: TurnTimer | None = None


class TurnTimer:
    def __init__(self, origin: str, t0: float | None = None, vad_wait_ms: int = 0) -> None:
        self.origin = origin
        self.t0 = t0 if t0 is not None else time.perf_counter()
        self.vad_wait_ms = vad_wait_ms
        self.claimed = False
        self.marks: dict[str, int] = {}
        self.llm_calls: list[dict] = []
        self.tts_segments: list[dict] = []
        self.filler_text: str | None = None
        self._lock = threading.Lock()

    def now_ms(self) -> int:
        return round((time.perf_counter() - self.t0) * 1000)

    def mark(self, name: str, last: bool = False) -> None:
        with self._lock:
            if last:
                self.marks[name] = self.now_ms()
            else:
                self.marks.setdefault(name, self.now_ms())

    def record_llm(self, kind: str, started_ms: int, first_chunk_ms: int | None,
                   final_chunk: dict, tool_calls: int) -> None:
        ns_to_ms = lambda key: round(final_chunk.get(key, 0) / 1e6)
        out_tokens = final_chunk.get("eval_count", 0)
        decode_ms = ns_to_ms("eval_duration")
        with self._lock:
            self.llm_calls.append({
                "kind": kind,
                "start": started_ms,
                "first_chunk": first_chunk_ms,
                "end": self.now_ms(),
                "load_ms": ns_to_ms("load_duration"),
                "prefill_tokens": final_chunk.get("prompt_eval_count"),
                "prefill_ms": ns_to_ms("prompt_eval_duration"),
                "output_tokens": out_tokens,
                "decode_tps": round(out_tokens / (decode_ms / 1000), 1) if decode_ms else None,
                "tool_calls": tool_calls,
            })

    def record_tts(self, chars: int, synth_ms: int, audio_ms: int) -> None:
        with self._lock:
            self.tts_segments.append({"chars": chars, "synth_ms": synth_ms, "audio_ms": audio_ms})

    def audio_started(self, text: str) -> None:
        self.mark("first_audio")
        if text != self.filler_text:
            self.mark("first_answer_audio")

    def summary(self) -> dict:
        with self._lock:
            return {
                "origin": self.origin,
                "vad_wait_ms": self.vad_wait_ms,
                "marks": dict(self.marks),
                "llm_calls": list(self.llm_calls),
                "tts_segments": list(self.tts_segments),
                "filler_text": self.filler_text,
            }

    def report(self) -> None:
        s = self.summary()
        print(_format(s))
        log_path = os.environ.get("IFA_LATENCY_LOG")
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), **s}) + "\n")


class _NullTimer(TurnTimer):
    def __init__(self) -> None:
        super().__init__("none")

    def mark(self, name: str, last: bool = False) -> None: ...
    def record_llm(self, *args, **kwargs) -> None: ...
    def record_tts(self, *args, **kwargs) -> None: ...
    def audio_started(self, text: str) -> None: ...


_NULL = _NullTimer()


def begin(origin: str, t0: float | None = None, vad_wait_ms: int = 0) -> TurnTimer:
    global _current
    timer = TurnTimer(origin, t0, vad_wait_ms)
    with _lock:
        _current = timer
    return timer


def claim_or_begin(origin: str) -> TurnTimer:
    with _lock:
        timer = _current
        if timer is not None and not timer.claimed:
            timer.claimed = True
            return timer
    timer = begin(origin)
    timer.claimed = True
    return timer


def current() -> TurnTimer:
    return _current or _NULL


def _format(s: dict) -> str:
    m = s["marks"]
    lines = []

    def row(label: str, at: int | None, detail: str = "") -> None:
        at_text = f"{at / 1000:6.2f}s" if at is not None else "     – "
        lines.append(f"   {at_text}  {label:<20s} {detail}")

    if s["vad_wait_ms"]:
        row("capture ended", m.get("captured"), f"VAD silence wait {s['vad_wait_ms']} ms")
    if "stt_done" in m:
        row("transcribed", m["stt_done"], f"STT {m['stt_done'] - m.get('captured', 0)} ms")
    for i, call in enumerate(s["llm_calls"], 1):
        detail = (
            f"first chunk +{(call['first_chunk'] or call['end']) - call['start']} ms · "
            f"prefill {call['prefill_tokens']} tok/{call['prefill_ms']} ms · "
            f"{call['output_tokens']} tok @ {call['decode_tps']} tok/s"
        )
        if call["load_ms"] > 50:
            detail += f" · model load {call['load_ms']} ms"
        if call["tool_calls"]:
            detail += f" · {call['tool_calls']} tool call(s)"
        row(f"{call['kind']} #{i} done", call["end"], detail)
    if s["filler_text"] is not None:
        row("filler", m.get("filler_queued"), repr(s["filler_text"]))
    row("first sentence", m.get("first_sentence"))
    if s["tts_segments"]:
        seg = s["tts_segments"][0]
        rtf = seg["synth_ms"] / seg["audio_ms"] if seg["audio_ms"] else 0
        row("tts 1st segment", None,
            f"synth {seg['synth_ms']} ms for {seg['audio_ms']} ms audio (RTF {rtf:.2f})")
    row("first audio", m.get("first_audio"))
    row("first ANSWER audio", m.get("first_answer_audio"), "← TTFS")
    row("speech done", m.get("audio_done"))

    ttfs = m.get("first_answer_audio")
    headline = f"{ttfs / 1000:.2f}s" if ttfs is not None else "n/a (nothing spoken)"
    return f"⏱  [{s['origin']}] TTFS {headline}\n" + "\n".join(lines)
