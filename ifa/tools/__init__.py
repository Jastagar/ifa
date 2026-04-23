"""Tool registry + adapters for the agent loop.

Tools are declared via `register()`, dispatched via `dispatch()`, and
exported to Ollama via `as_ollama_schema()`. Tool results are wrapped in
a per-turn nonce delimiter via `delimit_as_data()` so that crafted tool
content cannot close the data block and inject instructions.

Call `register_all()` once at startup to ensure every Stage 1 tool is
loaded — the individual modules register themselves on import, but an
explicit entry point avoids forgetting one.
"""


def register_all() -> None:
    """Explicitly register every Stage 1 tool.

    Each tool module exports a `TOOL` constant and runs `register(TOOL)`
    on first import. But if anything has called `registry.clear()` since
    then (notably tests), re-importing is a no-op because Python caches
    modules. Calling `register(TOOL)` here directly with the cached
    `TOOL` constants guarantees the registry is populated regardless.
    """
    from ifa.tools import memory, n8n, reminder, time
    from ifa.tools.registry import register

    for mod in (time, reminder, memory, n8n):
        register(mod.TOOL)
