"""Live tool-call bench — measures the real model's tool choices. NOT run in CI.

test_agent_stream.py checks the loop's plumbing with a scripted fake LLM.
This bench checks the other half: given Ifa's real system prompt and real
tool schemas, does the configured model pick the right tools, with valid
arguments, and chain them when a request needs more than one step?

Sandboxed: every tool keeps its real name/description/schema (that's what
the model sees), but handlers are swapped for fakes returning canned
results. Running the bench never sets reminders, writes memory.md, opens
apps, plays music, or sends WhatsApp messages.

Usage (needs Ollama running and IFA_OLLAMA_MODEL set, e.g. via .env):
    PYTHONPATH=. venv/bin/python -m ifa.tests.test_tool_call_bench
    PYTHONPATH=. venv/bin/python -m ifa.tests.test_tool_call_bench --repeat 3
    PYTHONPATH=. venv/bin/python -m ifa.tests.test_tool_call_bench --only multi_step

Each case is run --repeat times; a case's score is its pass rate across
runs, because the same prompt can produce different tool calls run to run.
Exit code is 0 only if every category meets its threshold.
"""
import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

# Same rule as ifa/main.py: .env must load before any ifa.* import, because
# modules read their settings (e.g. IFA_OLLAMA_MODEL) at import time.
load_dotenv()

from jsonschema import ValidationError, validate  # noqa: E402

from ifa.core import agent_stream  # noqa: E402
from ifa.core.context import AgentContext  # noqa: E402
from ifa.core.memory import Memory  # noqa: E402
from ifa.skills.vibe.vibe import VibeManager  # noqa: E402
from ifa.tools import register_all, registry  # noqa: E402
from ifa.tools.registry import Tool  # noqa: E402


@dataclass
class Call:
    name: str
    args: dict


@dataclass
class BenchCase:
    utterance: str
    expected: list[str]  # tool names; [] = should answer directly
    category: str
    ordered: bool = False  # True when step 2 depends on step 1's result
    # Extra check on the recorded calls, e.g. "did step 2 use step 1's output?"
    check: Callable[[list[Call]], bool] | None = field(default=None, repr=False)


# What each fake tool "returns". Multi-step cases rely on these values.
FAKE_RESULTS = {
    "get_time": "Current time is 14:05",
    "analyze_screen": "VS Code is open showing a TypeError in orchestrator.py on line 40.",
}
DEFAULT_FAKE_RESULT = "ok"


def _task_mentions(word: str) -> Callable[[list[Call]], bool]:
    def check(calls: list[Call]) -> bool:
        return any(
            c.name == "set_reminder" and word in str(c.args.get("task", "")).lower()
            for c in calls
        )
    return check


BENCH: list[BenchCase] = [
    # get_time — deterministic, should always be selected
    BenchCase("what time is it?", ["get_time"], "get_time"),
    BenchCase("can you tell me the current time?", ["get_time"], "get_time"),
    BenchCase("hey what's the clock say", ["get_time"], "get_time"),
    BenchCase("do you know what time it is right now", ["get_time"], "get_time"),

    # set_reminder — varied timeframes
    BenchCase("remind me to stretch in 30 seconds", ["set_reminder"], "set_reminder"),
    BenchCase("set a reminder to call mom in 2 minutes", ["set_reminder"], "set_reminder"),
    BenchCase("in 10 seconds, remind me to take my pills", ["set_reminder"], "set_reminder"),
    BenchCase("remind me about the meeting in 5 minutes", ["set_reminder"], "set_reminder"),

    # remember — the tool is for explicit requests; passing mentions are not stored
    BenchCase("remember that my cat is named Luna", ["remember"], "remember"),
    BenchCase("note down that I work remotely on Thursdays", ["remember"], "remember"),
    BenchCase("please remember I prefer tea over coffee", ["remember"], "remember"),
    BenchCase("I'm tired today", [], "direct_response"),
    BenchCase("it's raining outside", [], "direct_response"),

    # call_n8n_workflow
    BenchCase("trigger my home_summary workflow", ["call_n8n_workflow"], "call_n8n_workflow"),
    BenchCase("run the quick_ping workflow please", ["call_n8n_workflow"], "call_n8n_workflow"),
    BenchCase("fire off my morning_routine automation", ["call_n8n_workflow"], "call_n8n_workflow"),

    # open_app
    BenchCase("open vs code", ["open_app"], "open_app"),
    BenchCase("open youtube in the browser", ["open_app"], "open_app"),

    # spotify_control
    BenchCase("play some lofi on spotify", ["spotify_control"], "spotify_control"),
    BenchCase("pause the music", ["spotify_control"], "spotify_control"),
    BenchCase("skip this song", ["spotify_control"], "spotify_control"),

    # switch_mode
    BenchCase("turn on the vibe", ["switch_mode"], "switch_mode"),
    BenchCase("switch back to normal mode", ["switch_mode"], "switch_mode"),

    # analyze_screen
    BenchCase("what error is on my screen?", ["analyze_screen"], "analyze_screen"),
    BenchCase("read this popup for me", ["analyze_screen"], "analyze_screen"),

    # launch_game
    BenchCase("launch valorant", ["launch_game"], "launch_game"),
    BenchCase("start minecraft", ["launch_game"], "launch_game"),

    # whats_app
    BenchCase("send john a whatsapp saying I'll be late", ["whats_app"], "whats_app"),

    # direct response — no tool
    BenchCase("how are you today?", [], "direct_response"),
    BenchCase("thanks!", [], "direct_response"),
    BenchCase("ok got it", [], "direct_response"),

    # multi_step — needs more than one tool in a single turn
    BenchCase(
        "look at my screen and remind me in 10 minutes to fix whatever error is there",
        ["analyze_screen", "set_reminder"], "multi_step", ordered=True,
        check=_task_mentions("typeerror"),
    ),
    BenchCase(
        "check what time it is and remind me in 1 minute to drink water",
        ["get_time", "set_reminder"], "multi_step",
    ),
    BenchCase(
        "turn on gaming mode and launch valorant",
        ["switch_mode", "launch_game"], "multi_step",
    ),
]

THRESHOLDS = {
    "get_time": 1.00,
    "set_reminder": 0.75,
    "remember": 0.66,  # ⅔ — 0.67 would fail a 2-of-3 score (0.666…)
    "call_n8n_workflow": 0.66,  # ⅔ — 0.67 would fail a 2-of-3 score (0.666…)
    "open_app": 0.75,
    "spotify_control": 0.75,
    "switch_mode": 0.75,
    "analyze_screen": 0.75,
    "launch_game": 0.75,
    "whats_app": 0.75,
    "direct_response": 0.66,  # ⅔ — 0.67 would fail a 2-of-3 score (0.666…)
    "multi_step": 0.50,
    "aggregate": 0.75,
    "valid_args": 0.95,
}


def _sandbox_tools() -> None:
    """Register every real tool, then swap each handler for a canned fake."""
    register_all()
    for tool in registry.all_tools():
        result = FAKE_RESULTS.get(tool.name, DEFAULT_FAKE_RESULT)
        registry.register(Tool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            handler=lambda args, ctx, result=result: result,
        ))


def _args_valid(call: Call) -> bool:
    tool = registry.get(call.name)
    if tool is None or not isinstance(call.args, dict):
        return False
    try:
        validate(instance=call.args, schema=tool.parameters)
        return True
    except ValidationError:
        return False


def _run_case(case: BenchCase) -> dict:
    """Run one full agent turn and record every tool call the model made."""
    calls: list[Call] = []
    real_stream_chat = agent_stream.stream_chat

    def recording_stream_chat(*args, **kwargs):
        for chunk in real_stream_chat(*args, **kwargs):
            for tc in (chunk.get("message") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                calls.append(Call(fn.get("name"), fn.get("arguments")))
            yield chunk

    ctx = AgentContext(tts=MagicMock(), db_path=":memory:", n8n_config={})
    started = time.perf_counter()
    with patch.object(agent_stream, "stream_chat", recording_stream_chat):
        reply = agent_stream.agent_turn_stream(case.utterance, ctx, Memory())
    elapsed_ms = (time.perf_counter() - started) * 1000

    names = [c.name for c in calls]
    if case.ordered:
        tools_match = names == case.expected
    else:
        tools_match = Counter(names) == Counter(case.expected)
    check_ok = case.check(calls) if case.check else True

    return {
        "utterance": case.utterance,
        "category": case.category,
        "expected": case.expected,
        "actual": names,
        "args": [c.args for c in calls],
        "correct": tools_match and check_ok,
        "check_failed": tools_match and not check_ok,
        "valid_args": all(_args_valid(c) for c in calls),
        "latency_ms": round(elapsed_ms),
        "reply": reply,
    }


def run_bench(repeat: int = 1, only: str | None = None) -> dict:
    """Run every bench case `repeat` times and return a structured summary."""
    _sandbox_tools()
    VibeManager(MagicMock())  # the system prompt reads the current vibe

    cases = [c for c in BENCH if only is None or c.category == only]
    results = [_run_case(case) for case in cases for _ in range(repeat)]

    by_category: dict[str, list[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)
    per_category = {
        cat: sum(r["correct"] for r in rs) / len(rs) for cat, rs in by_category.items()
    }
    aggregate = sum(r["correct"] for r in results) / len(results)
    valid_args_rate = sum(r["valid_args"] for r in results) / len(results)
    latencies = sorted(r["latency_ms"] for r in results)

    gates_passed = (
        aggregate >= THRESHOLDS["aggregate"]
        and valid_args_rate >= THRESHOLDS["valid_args"]
        and all(acc >= THRESHOLDS.get(cat, 0) for cat, acc in per_category.items())
    )

    return {
        "model": agent_stream.MODEL,
        "max_tool_hops": agent_stream.MAX_TOOL_HOPS,
        "repeat": repeat,
        "total_runs": len(results),
        "aggregate_accuracy": aggregate,
        "valid_args_rate": valid_args_rate,
        "per_category_accuracy": per_category,
        "latency_ms": {
            "p50": latencies[len(latencies) // 2],
            "p90": latencies[int(len(latencies) * 0.9)],
            "max": latencies[-1],
        },
        "gates_passed": gates_passed,
        "failures": [r for r in results if not r["correct"]],
        "results": results,
    }


def _print_report(summary: dict) -> None:
    def fmt(tools: list[str]) -> str:
        return " → ".join(tools) if tools else "(direct)"

    print(f"\n=== Tool-call bench: {summary['model']} "
          f"(max_tool_hops={summary['max_tool_hops']}, repeat={summary['repeat']}) ===\n")
    print(f"{'Utterance':<48s}  {'Expected':<30s}  {'Actual':<30s}  {'ms':>6s}  OK")
    print("-" * 124)
    for r in summary["results"]:
        mark = "✓" if r["correct"] else ("✗ check" if r["check_failed"] else "✗")
        print(f"{r['utterance'][:48]:<48s}  {fmt(r['expected'])[:30]:<30s}  "
              f"{fmt(r['actual'])[:30]:<30s}  {r['latency_ms']:>6d}  {mark}")
    print()
    print(f"Aggregate accuracy: {summary['aggregate_accuracy']:.1%} "
          f"(≥{THRESHOLDS['aggregate']:.0%})")
    print(f"Valid-args rate:    {summary['valid_args_rate']:.1%} "
          f"(≥{THRESHOLDS['valid_args']:.0%})")
    lat = summary["latency_ms"]
    print(f"Latency per turn:   p50 {lat['p50']} ms · p90 {lat['p90']} ms · max {lat['max']} ms")
    for cat, acc in sorted(summary["per_category_accuracy"].items()):
        threshold = THRESHOLDS.get(cat, 0)
        print(f"  {cat:<20s}: {acc:6.1%} (≥{threshold:.0%}) {'✓' if acc >= threshold else '✗'}")
    print(f"\nGATES PASSED: {summary['gates_passed']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=1, help="runs per case")
    parser.add_argument("--only", help="run a single category, e.g. multi_step")
    parser.add_argument("--json", action="store_true", help="also print the JSON summary")
    cli = parser.parse_args()

    if agent_stream.MODEL is None:
        sys.exit("IFA_OLLAMA_MODEL is not set (add it to .env).")

    summary = run_bench(repeat=cli.repeat, only=cli.only)
    _print_report(summary)
    if cli.json:
        print(json.dumps(summary, indent=2, default=str))
    sys.exit(0 if summary["gates_passed"] else 1)
