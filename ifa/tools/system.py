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
        "call this when user wants you to open an app. in this can you must pass 'app' property ask for confirmation first"
        "if asked to open a website specifically or asked to open something on browser, just pass in www.<websitename> in args"
        "if asked to open some adult or NSFW thing, open them in private browser, anything related to sex, violance, drugs or if user specifically asks"
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