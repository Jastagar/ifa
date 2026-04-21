class Skill:
    def can_handle(self, text: str) -> bool:
        raise NotImplementedError

    def handle(self, text: str) -> str:
        raise NotImplementedError