import os
import sys

from ifa.core.context import AgentContext
from ifa.skills.commands import CommandRunner
from ifa.tools.registry import Tool

SHELL = "cmd.exe" if sys.platform == "win32" else "the macOS shell" if sys.platform == "darwin" else "the shell"


def enabled() -> bool:
    return os.environ.get("IFA_COMMANDS_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def build_tool() -> Tool | None:
    if not enabled():
        return None
    runner = CommandRunner(timeout=float(os.environ.get("IFA_COMMAND_TIMEOUT_SEC", "30")))

    def handler(args: dict, ctx: AgentContext) -> str:
        return runner.handle(args["command"], ctx.turn, ctx.user_text)

    return Tool(
        name="run_command",
        description=(
            f"Run one command in {SHELL} on this computer and read its output. Use it for system tasks "
            "Sir asks for, like checking the IP address, disk space, running processes, pinging a site, "
            "or a command he names. Do not use it to open apps (use open_app) or to control music. "
            "Use the simplest standard command for the job, not long pipelines. "
            "Every command needs Sir's spoken yes: the first call only proposes it and returns NOT RUN YET. "
            "Then ask him in one short sentence that says what the command does in plain words and "
            "names the command if it is short, and wait for his answer. When the result arrives, "
            "tell him what it means in a sentence or two instead of reading raw output."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": f"A single command line for {SHELL}.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        handler=handler,
    )
