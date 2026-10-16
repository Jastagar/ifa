from __future__ import annotations

MAX_RESULT_CHARS = 160


class Memory:
    def __init__(self, max_turns: int = 4) -> None:
        self.turns: list[list[dict]] = []
        self.max_turns = max_turns

    def add_turn(self, user_text: str, steps: list[dict], reply: str) -> None:
        self.turns.append([{"role": "user", "content": user_text}, *steps, {"role": "assistant", "content": reply}])
        self.turns = self.turns[-self.max_turns:]

    def recent(self) -> list[dict]:
        return [message for turn in self.turns for message in turn]


def tool_step(assistant_msg: dict, results: list[tuple[str, str]]) -> list[dict]:
    calls = [{"function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
             for tc in assistant_msg["tool_calls"]]
    step = [{"role": "assistant", "content": assistant_msg.get("content") or "", "tool_calls": calls}]
    for name, result in results:
        first_line = (result.strip().splitlines() or [""])[0]
        if len(first_line) > MAX_RESULT_CHARS:
            first_line = first_line[:MAX_RESULT_CHARS] + "..."
        step.append({"role": "tool", "tool_name": name, "content": first_line})
    return step
