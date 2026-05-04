"""Base class for legacy text-routing skills.

**This is mostly historical.** Stage 0 of Ifa used a "skill manager"
that walked a list of ``Skill`` instances, called ``can_handle(text)``
to find the right one, then ``handle(text)`` to execute. Stage 1
replaced that with LLM tool dispatch (see ``ifa/core/agent.py`` and
``ifa/tools/registry.py``), where the LLM picks tools by name and
JSON-Schema-validated arguments.

The only ``Skill`` subclass that survives Stage 1 cleanup is
``ReminderSkill`` (see ``ifa/skills/reminder.py``), and only its
``schedule(task, seconds)`` method is used — by the ``set_reminder``
tool adapter in ``ifa/tools/reminder.py``. The ``can_handle`` /
``handle`` interface defined here is dead but kept for any future
skill that needs the same shape.

If you're adding a new capability today, **don't subclass Skill** —
register a tool. Look at ``ifa/tools/time.py`` for a minimal example:
declare a ``Tool`` with a JSON Schema and a handler, call
``register(TOOL)`` at module level, then add the import to
``register_all()`` in ``ifa/tools/__init__.py``.
"""


class Skill:
    """Pre-Stage-1 base class. New code should not subclass this — see module docstring."""

    def can_handle(self, text: str) -> bool:
        raise NotImplementedError

    def handle(self, text: str) -> str:
        raise NotImplementedError
