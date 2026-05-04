from pywhatkit import whats
from ifa.tools.registry import Tool, register
from ifa.core.context import AgentContext

def _handler(args: dict, ctx: AgentContext) -> str:
    print(args)
    name = args.get("toSend","myself")
    print("name")
    print(name)
    message = args.get("message","Heelo")
    print("message")
    print(message)
    number = ctx.contacts[str.lower(name)]
    print("number")
    print(number)
    return whats.sendwhatmsg_instantly(number,message)

TOOL = Tool(
    name="whats_app",
    description=(
        "call this when user asks to send a message to a specific other user on whatsapp."
        "incase user has not given a payload info, follow up with the user asking the required details."
    ),
    parameters={
        "type": "object",
        "properties": {
            "toSend": {
                "type": "string",
            },
            "message": {
                "type": "string",
            },
        },
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)