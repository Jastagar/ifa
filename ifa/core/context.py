"""AgentContext — dependency carrier threaded through the agent loop to tool handlers.

Tools need access to stateful services (TTS for reminder playback, DB for
persistence, n8n config for outbound calls). Rather than module-level
singletons, the orchestrator constructs this once and passes it to every
`agent_turn`, which forwards it to every tool handler as `handler(args, ctx)`.
"""
from dataclasses import dataclass, field

from ifa.services.tts_service import TTSService


@dataclass
class AgentContext:
    tts: TTSService
    db_path: str
    n8n_config: dict = field(default_factory=dict)
