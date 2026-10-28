def register_all() -> None:
    from ifa.tools import (
        browser, calendar, commands, gamelauncher, memory, n8n, notifications, reminder, session, spotify, system, time, vibe,
        vision, websearch, whatsApp,
    )
    from ifa.tools.registry import register

    for mod in (time, reminder, memory, n8n, vibe, system, browser, spotify, whatsApp, vision, gamelauncher, notifications, commands, calendar, websearch):
        tool = mod.build_tool() if hasattr(mod, "build_tool") else mod.TOOL
        if tool is not None:
            register(tool)
    register(memory.RECALL_TOOL)
    register(memory.FORGET_TOOL)
    register(session.END_TOOL)
    register(session.RESTART_TOOL)
