class Memory:
    def __init__(self, max_history: int = 10) -> None:
        self.history: list[dict] = []
        self.max_history = max_history

    def add(self, role: str, content: str) -> None:
        if role.lower() == "user":
            role = "user"
        elif role.lower() in ["ifa", "assistant"]:
            role = "assistant"

        self.history.append({"role": role, "content": content})

        self.history = self.history[-self.max_history :]

    def get_recent(self, n: int = 5) -> list[dict]:
        return self.history[-n:]
