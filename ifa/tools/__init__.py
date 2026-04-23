"""Tool registry + adapters for the agent loop.

Tools are declared via `register()`, dispatched via `dispatch()`, and
exported to Ollama via `as_ollama_schema()`. Tool results are wrapped in
a per-turn nonce delimiter via `delimit_as_data()` so that crafted tool
content cannot close the data block and inject instructions.
"""
