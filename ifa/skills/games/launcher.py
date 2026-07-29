from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ifa.skills.base import Skill


class GameLauncher(Skill):

    def __init__(self):
        self.games = self._load_games()

    def _load_games(self):
        path = Path(__file__).parent / "games.json"

        with open(path, "r") as f:
            return json.load(f)

    def launch(self, game_name: str) -> str:

        game_name = game_name.lower()

        if game_name not in self.games:
            return f"I don't know how to launch {game_name}"

        game = self.games[game_name]

        try:
            subprocess.Popen(
                game["path"],
                shell=True
            )

            return f"Launching {game_name}"

        except Exception as e:
            return f"Failed to launch {game_name}: {e}"