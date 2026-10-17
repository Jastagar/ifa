def register_all() -> None:
    from ifa.tools import (
        browser, commands, gamelauncher, memory, n8n, notifications, reminder, spotify, system, time, vibe, vision,
        whatsApp,
    )
    from ifa.tools.registry import register

    for mod in (time, reminder, memory, n8n, vibe, system, browser, spotify, whatsApp, vision, gamelauncher, notifications, commands):
        tool = mod.build_tool() if hasattr(mod, "build_tool") else mod.TOOL
        if tool is not None:
            register(tool)
