from ifa.core.context import AgentContext
from ifa.services import custom_apps
from ifa.skills.enums import App
from ifa.skills.system import Application, launch_custom
from ifa.tools.registry import Tool


def build_tool(apps_file=custom_apps.APPS_FILE) -> Tool:
    custom = {app["name"].lower(): app for app in custom_apps.load(apps_file)}
    builtin = [a.value for a in App if a.value not in custom]
    names = builtin + [app["name"] for app in custom.values()]

    def handler(args: dict, ctx: AgentContext) -> str:
        name = args["app"]
        app = custom.get(name.lower())
        if app is None:
            return Application(ctx.tts).open_app(name, args.get("args", ""))
        print(f"[apps] opening {app['name']}: {app['command']}")
        launch_custom(app["command"])
        return f"Opening {app['name']}"

    return Tool(
        name="open_app",
        description=(
            "Open an app on this computer. Call it whenever the user says to open, launch or start an app, "
            "without asking for confirmation. "
            "If the request also asks for something inside the app (for example 'open Spotify and play X'), "
            "call open_app first and then the tool for the rest (spotify_control); the other tool never opens the app. "
            "To open a website, search, or play something on YouTube, use open_website instead. "
            "Pick the closest name from the list, even if the user says it differently."
        ),
        parameters={
            "type": "object",
            "properties": {
                "app": {
                    "type": "string",
                    "enum": names,
                    "description": "The app to open.",
                },
                "args": {
                    "type": "string",
                    "description": "Only for brave / private brave: a website to open. Prefer open_website.",
                },
            },
            "required": ["app"],
            "additionalProperties": False,
        },
        handler=handler,
    )
