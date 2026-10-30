from dataclasses import dataclass, field

from typing import TYPE_CHECKING

from ifa.services.tts_service import TTSService

if TYPE_CHECKING:
    from ifa.services.notifications import NotificationCenter


@dataclass
class AgentContext:
    tts: TTSService
    db_path: str
    n8n_config: dict = field(default_factory=dict)
    contacts: dict = field(default_factory=dict)
    notifications: "NotificationCenter | None" = None
    turn: int = 0
    user_text: str = ""
    end_conversation: bool = False
    restart_requested: bool = False
    finish_silently: bool = False
