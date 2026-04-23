"""Manual tool-call acceptance bench — NOT run in CI.

Measures qwen2.5:7b-instruct's tool-selection accuracy against a fixed
20-utterance set. Gates Unit 6b (deletion of old code paths) in the plan:
the bench's per-tool accuracy floors MUST be met before brain.py /
manager.py / detect_intent / extract_fact are deleted.

Usage:
    PYTHONPATH=. venv/bin/python -m ifa.tests.test_tool_call_bench

Requires:
    - Ollama running locally
    - `ollama pull qwen2.5:7b-instruct` (or quantized equivalent)

Outputs a machine-readable JSON summary to stdout plus a human-readable
per-utterance table. The JSON lets downstream scripts gate the deletion
commit on numeric thresholds, not prose.

Thresholds (from docs/plans/2026-04-23-002-feat-stage1-llm-tool-dispatch-plan.md):
    get_time:           ≥100% (deterministic)
    set_reminder:       ≥75%
    remember_fact:      ≥67%  (judgment-based, hardest case)
    call_n8n_workflow:  ≥67%
    direct-response:    ≥67%  (no tool)
    Aggregate:          ≥75% (hard floor); ≥85% aspirational
    Valid-JSON rate:    ≥95%

Each (utterance, expected_tool) pair is hard-coded — no rubric retrofit.
"""
import json
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

from ifa.core import agent
from ifa.core.context import AgentContext
from ifa.core.memory import Memory
from ifa.services.ollama_client import chat
from ifa.tools import register_all, registry


@dataclass
class BenchCase:
    utterance: str
    expected_tool: str | None  # None = expected to be a direct response


BENCH: list[BenchCase] = [
    # get_time (4) — deterministic, LLM should always select this
    BenchCase("what time is it?", "get_time"),
    BenchCase("can you tell me the current time?", "get_time"),
    BenchCase("hey what's the clock say", "get_time"),
    BenchCase("do you know what time it is right now", "get_time"),

    # set_reminder (4) — varied timeframes
    BenchCase("remind me to stretch in 30 seconds", "set_reminder"),
    BenchCase("set a reminder to call mom in 2 minutes", "set_reminder"),
    BenchCase("in 10 seconds, remind me to take my pills", "set_reminder"),
    BenchCase("remind me about the meeting in 5 minutes", "set_reminder"),

    # remember_fact (6) — selectivity: durable facts vs. ephemeral info
    BenchCase("my cat is named Luna", "remember_fact"),
    BenchCase("I work remotely on Thursdays", "remember_fact"),
    BenchCase("my favorite color is green", "remember_fact"),
    BenchCase("I'm tired today", None),  # ephemeral, should NOT remember
    BenchCase("it's raining outside", None),  # world state, not user fact
    BenchCase("I prefer tea over coffee", "remember_fact"),

    # call_n8n_workflow (3) — valid, unknown, ambiguous
    BenchCase("trigger my home_summary workflow", "call_n8n_workflow"),
    BenchCase("run the quick_ping workflow please", "call_n8n_workflow"),
    BenchCase("fire off my morning_routine automation", "call_n8n_workflow"),

    # direct-response (3) — should not trigger any tool
    BenchCase("how are you today?", None),
    BenchCase("thanks!", None),
    BenchCase("ok got it", None),
]

THRESHOLDS = {
    "get_time": 1.00,
    "set_reminder": 0.75,
    "remember_fact": 0.67,
    "call_n8n_workflow": 0.67,
    "direct_response": 0.67,
    "aggregate": 0.75,
    "valid_json": 0.95,
}


def _observe_tool_call(user_text: str) -> tuple[str | None, bool]:
    """Run agent_turn and return (selected_tool_name, valid_json).

    Returns tool_name=None if the LLM emitted a direct response.
    valid_json=False if Ollama returned a tool call with malformed arguments.
    """
    ctx = AgentContext(tts=MagicMock(), db_path=":memory:", n8n_config={})
    memory = Memory()

    # Intercept the first chat() call to observe what the LLM emitted
    original_chat = chat
    observed = {"tool": None, "valid_json": True}

    def observer(model, messages, tools=None, timeout=60.0):
        response = original_chat(model, messages, tools=tools, timeout=timeout)
        msg = response.get("message", {})
        tool_calls = msg.get("tool_calls") or []
        if tool_calls and observed["tool"] is None:
            tc = tool_calls[0]
            fn = tc.get("function") or {}
            observed["tool"] = fn.get("name")
            if not isinstance(fn.get("arguments"), dict):
                observed["valid_json"] = False
        return response

    import ifa.core.agent as agent_module
    original = agent_module.chat
    agent_module.chat = observer
    try:
        agent.agent_turn(user_text, ctx, memory)
    finally:
        agent_module.chat = original

    return observed["tool"], observed["valid_json"]


def run_bench() -> dict:
    """Run every bench case and return a structured summary."""
    register_all()

    results = []
    per_tool_correct: dict[str, int] = {}
    per_tool_total: dict[str, int] = {}
    valid_json_count = 0

    for case in BENCH:
        expected_key = case.expected_tool or "direct_response"
        per_tool_total[expected_key] = per_tool_total.get(expected_key, 0) + 1

        actual_tool, valid_json = _observe_tool_call(case.utterance)

        # Normalize: None = direct-response
        correct = (actual_tool == case.expected_tool)
        if correct:
            per_tool_correct[expected_key] = per_tool_correct.get(expected_key, 0) + 1
        if valid_json:
            valid_json_count += 1

        results.append({
            "utterance": case.utterance,
            "expected": case.expected_tool,
            "actual": actual_tool,
            "correct": correct,
            "valid_json": valid_json,
        })

    per_tool_accuracy = {
        key: per_tool_correct.get(key, 0) / per_tool_total[key]
        for key in per_tool_total
    }
    aggregate_accuracy = sum(r["correct"] for r in results) / len(results)
    json_valid_rate = valid_json_count / len(results)

    gates_passed = (
        aggregate_accuracy >= THRESHOLDS["aggregate"]
        and json_valid_rate >= THRESHOLDS["valid_json"]
        and all(
            per_tool_accuracy.get(key, 0) >= threshold
            for key, threshold in THRESHOLDS.items()
            if key not in ("aggregate", "valid_json") and key in per_tool_accuracy
        )
    )

    return {
        "model": agent.MODEL,
        "total": len(BENCH),
        "aggregate_accuracy": aggregate_accuracy,
        "json_valid_rate": json_valid_rate,
        "per_tool_accuracy": per_tool_accuracy,
        "per_tool_thresholds": {k: v for k, v in THRESHOLDS.items() if k not in ("aggregate", "valid_json")},
        "gates_passed": gates_passed,
        "failures": [r for r in results if not r["correct"]],
        "results": results,
    }


if __name__ == "__main__":
    summary = run_bench()

    # Human-readable table
    print(f"\n=== Tool-call bench results for {summary['model']} ===\n")
    print(f"{'Utterance':<50s}  {'Expected':<20s}  {'Actual':<20s}  OK")
    print("-" * 100)
    for r in summary["results"]:
        expected = r["expected"] or "(direct response)"
        actual = r["actual"] or "(direct response)"
        mark = "✓" if r["correct"] else "✗"
        print(f"{r['utterance'][:50]:<50s}  {expected:<20s}  {actual:<20s}  {mark}")
    print()
    print(f"Aggregate accuracy: {summary['aggregate_accuracy']:.1%} "
          f"(threshold {THRESHOLDS['aggregate']:.0%})")
    print(f"Valid-JSON rate:    {summary['json_valid_rate']:.1%} "
          f"(threshold {THRESHOLDS['valid_json']:.0%})")
    for tool, acc in summary["per_tool_accuracy"].items():
        threshold = THRESHOLDS.get(tool, 0)
        mark = "✓" if acc >= threshold else "✗"
        print(f"  {tool:<20s}: {acc:.1%} (≥{threshold:.0%}) {mark}")
    print()
    print(f"GATES PASSED: {summary['gates_passed']}")
    print()

    # Machine-readable JSON
    print("JSON_SUMMARY_START")
    print(json.dumps(summary, indent=2))
    print("JSON_SUMMARY_END")

    sys.exit(0 if summary["gates_passed"] else 1)
