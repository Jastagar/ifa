from ifa.core.context import AgentContext
from ifa.skills.system import Application
from ifa.skills.enums import App
from ifa.tools.registry import Tool, register

def _handler(args: dict, ctx: AgentContext) -> str:
    print("APP TRIED TO OPEN")
    appName = args["app"]
    print(args)
    arguments = args.get("args","")
    print(arguments)
    return Application(ctx.tts).open_app(appName,arguments)

TOOL = Tool(
    name="open_app",
    description=(
        "Open an app on this computer. Call it whenever the user says to open, launch or start an app, "
        "without asking for confirmation. "
        "If the request also asks for something inside the app (for example 'open Spotify and play X'), "
        "call open_app first and then the tool for the rest (spotify_control); the other tool never opens the app. "
        "To open a website, use app 'brave' and put www.<websitename> in args. "
        "For adult or NSFW content (sex, violence, drugs) or when the user asks for it, use 'private brave'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "app": {
                "type": "string",
                "enum": [a.value for a in App],
                "description": (
                    "when user askes for specific app to open"
                )
            },
            "args":{
                "type": "string",
                "description": (
                    "space seprated arguments for cmd commands"
                    "eg. website url for opening on browser"
                ),
            }
        },
        "required":["app"],
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)