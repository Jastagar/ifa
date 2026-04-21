# ifa/core/memory.py

class Memory:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history

    def add(self, role: str, content: str):
        # normalize roles
        if role.lower() == "user":
            role = "user"
        elif role.lower() in ["ifa", "assistant"]:
            role = "assistant"

        self.history.append({
            "role": role,
            "content": content
        })

        # keep only recent
        self.history = self.history[-self.max_history:]

    def get_recent(self, n=5):
        return self.history[-n:]