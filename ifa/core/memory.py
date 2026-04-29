"""Short-term conversation history — the rolling window the agent loop sends back to the LLM each turn.

This is the "you said... I said..." chat history that gives Ifa
short-term context. It is **not** long-term knowledge — long-term
facts ("my favorite color is blue") are stored separately in SQLite
via the ``remember_fact`` tool (see ``ifa/tools/memory.py``) and read
back into the system prompt by ``ifa/core/agent.py`` at the start of
each turn.

Lifecycle
---------
- A single ``Memory`` is constructed in ``orchestrator.run()`` once
  per session and threaded into every ``agent_turn`` call.
- ``agent_turn`` reads recent entries to build the chat-history
  portion of the messages sent to Ollama.
- After the LLM produces a final reply, ``orchestrator.run()`` (or
  the agent loop, depending on flow) appends both the user's input
  and the assistant's reply via ``add()``.
- The buffer is bounded by ``max_history`` so a long session doesn't
  grow context windows unboundedly.

Why so simple?
--------------
Stage 1 ships with a trivially-small in-memory deque. It works for
single-session use and avoids a database round-trip per turn. If
multi-session continuity becomes a goal, swap this class for one that
persists to SQLite — every caller uses the same ``add`` / ``get_recent``
interface, so the swap is local.
"""


class Memory:
    """Bounded list of {role, content} dicts, latest at the end.

    ``max_history`` caps how many turns we keep. With ``get_recent(n)``
    the caller can read back the tail; the default n=5 ≈ 2-3
    user/assistant exchanges, which is what the agent prompt currently
    needs.
    """

    def __init__(self, max_history: int = 10) -> None:
        self.history: list[dict] = []
        self.max_history = max_history

    def add(self, role: str, content: str) -> None:
        """Append a turn; truncate from the front if we exceed max_history.

        Roles are normalized so callers can pass "Ifa" / "user" / "ASSISTANT"
        interchangeably and the LLM sees the canonical "user" / "assistant"
        labels Ollama expects.
        """
        if role.lower() == "user":
            role = "user"
        elif role.lower() in ["ifa", "assistant"]:
            role = "assistant"

        self.history.append({"role": role, "content": content})

        # Bounded buffer — drop oldest entries when we go over capacity.
        # Keeps Ollama context windows manageable for long sessions.
        self.history = self.history[-self.max_history :]

    def get_recent(self, n: int = 5) -> list[dict]:
        """Return the last ``n`` turns. Caller decides how much context is enough."""
        return self.history[-n:]
