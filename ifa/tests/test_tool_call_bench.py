import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

load_dotenv()

from jsonschema import ValidationError, validate

from ifa.core import agent_stream
from ifa.core.context import AgentContext
from ifa.core.memory import Memory
from ifa.skills.vibe.vibe import VibeManager
from ifa.tools import register_all, registry
from ifa.tools.registry import Tool


@dataclass
class Call:
    name: str
    args: dict


@dataclass
class BenchCase:
    utterance: str
    expected: list[str]
    category: str
    ordered: bool = False
    check: Callable[[list[Call]], bool] | None = field(default=None, repr=False)


FAKE_RESULTS = {
    "get_time": "Current time is 14:05",
    "analyze_screen": "VS Code is open showing a TypeError in orchestrator.py on line 40.",
    "spotify_control": "Paused Tum Hi Ho by Arijit Singh.",
}
DEFAULT_FAKE_RESULT = "ok"


def _youtube(query_word: str, play: bool = True) -> Callable[[list[Call]], bool]:
    def check(calls: list[Call]) -> bool:
        return any(
            c.name == "open_website" and c.args.get("site") == "youtube"
            and query_word in str(c.args.get("query", "")).lower()
            and bool(c.args.get("play", True)) == play
            for c in calls
        )
    return check


def _task_mentions(word: str) -> Callable[[list[Call]], bool]:
    def check(calls: list[Call]) -> bool:
        return any(
            c.name == "set_reminder" and word in str(c.args.get("task", "")).lower()
            for c in calls
        )
    return check


BENCH: list[BenchCase] = [
    BenchCase("what time is it?", ["get_time"], "get_time"),
    BenchCase("can you tell me the current time?", ["get_time"], "get_time"),
    BenchCase("hey what's the clock say", ["get_time"], "get_time"),
    BenchCase("do you know what time it is right now", ["get_time"], "get_time"),

    BenchCase("remind me to stretch in 30 seconds", ["set_reminder"], "set_reminder"),
    BenchCase("set a reminder to call mom in 2 minutes", ["set_reminder"], "set_reminder"),
    BenchCase("in 10 seconds, remind me to take my pills", ["set_reminder"], "set_reminder"),
    BenchCase("remind me about the meeting in 5 minutes", ["set_reminder"], "set_reminder"),

    BenchCase("remember that my cat is named Luna", ["remember"], "remember"),
    BenchCase("note down that I work remotely on Thursdays", ["remember"], "remember"),
    BenchCase("please remember I prefer tea over coffee", ["remember"], "remember"),
    BenchCase("I'm tired today", [], "direct_response"),
    BenchCase("it's raining outside", [], "direct_response"),

    BenchCase("trigger my home_summary workflow", ["call_n8n_workflow"], "call_n8n_workflow"),
    BenchCase("run the quick_ping workflow please", ["call_n8n_workflow"], "call_n8n_workflow"),
    BenchCase("fire off my morning_routine automation", ["call_n8n_workflow"], "call_n8n_workflow"),

    BenchCase("open vs code", ["open_app"], "open_app"),

    BenchCase("open youtube in the browser", ["open_website"], "open_website"),
    BenchCase("play tum hi ho on youtube", ["open_website"], "open_website", check=_youtube("tum hi ho")),
    BenchCase("search youtube for lofi beats", ["open_website"], "open_website", check=_youtube("lofi", play=False)),
    BenchCase("google the weather in delhi", ["open_website"], "open_website"),

    BenchCase("play some lofi on spotify", ["spotify_control"], "spotify_control"),
    BenchCase("pause the music", ["spotify_control"], "spotify_control"),
    BenchCase("skip this song", ["spotify_control"], "spotify_control"),

    BenchCase("turn on the vibe", ["switch_mode"], "switch_mode"),
    BenchCase("switch back to normal mode", ["switch_mode"], "switch_mode"),

    BenchCase("what error is on my screen?", ["analyze_screen"], "analyze_screen"),
    BenchCase("read this popup for me", ["analyze_screen"], "analyze_screen"),

    BenchCase("launch valorant", ["launch_game"], "launch_game"),
    BenchCase("start minecraft", ["launch_game"], "launch_game"),

    BenchCase("send john a whatsapp saying I'll be late", ["whats_app"], "whats_app"),

    BenchCase("alright that's everything I needed, talk to you later", ["end_conversation"], "end_conversation"),
    BenchCase("you can stop listening now, I'm done", ["end_conversation"], "end_conversation"),
    BenchCase("restart yourself", ["restart_ifa"], "restart_ifa"),
    BenchCase("ifa please reboot, I changed your settings", ["restart_ifa"], "restart_ifa"),

    BenchCase("how are you today?", [], "direct_response"),
    BenchCase("thanks!", [], "direct_response"),
    BenchCase("ok got it", [], "direct_response"),

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
        "stop this song on spotify and play it on youtube",
        ["spotify_control", "open_website"], "multi_step", ordered=True,
        check=_youtube("tum hi ho"),
    ),
    BenchCase(
        "turn on gaming mode and launch valorant",
        ["switch_mode", "launch_game"], "multi_step",
    ),
]

THRESHOLDS = {
    "get_time": 1.00,
    "set_reminder": 0.75,
    "remember": 0.66,
    "call_n8n_workflow": 0.66,
    "open_app": 0.75,
    "open_website": 0.75,
    "end_conversation": 0.75,
    "restart_ifa": 0.66,
    "spotify_control": 0.75,
    "switch_mode": 0.75,
    "analyze_screen": 0.75,
    "launch_game": 0.75,
    "whats_app": 0.75,
    "direct_response": 0.66,
    "multi_step": 0.50,
    "aggregate": 0.75,
    "valid_args": 0.95,
}


def _sandbox_tools() -> None:
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
    _sandbox_tools()
    VibeManager(MagicMock())

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
    parser = argparse.ArgumentParser(description="Measure how reliably the model picks the right tools.")
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
