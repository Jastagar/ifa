def register_all() -> None:
    from ifa.tools import memory, n8n, reminder, time, vibe, system, spotify, whatsApp, vision, gamelauncher, notifications
    from ifa.tools.registry import register

    for mod in (time, reminder, memory, n8n, vibe, system, spotify, whatsApp, vision, gamelauncher, notifications):
        register(mod.TOOL)
