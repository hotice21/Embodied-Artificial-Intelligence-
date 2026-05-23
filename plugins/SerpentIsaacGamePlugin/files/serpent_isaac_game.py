import os

from serpent.game import Game
from serpent.input_controller import InputControllers
from serpent.utilities import Singleton

from .api.api import IsaacAPI


def resolve_game_root():
    project_root = os.getcwd()
    candidates = [
        os.environ.get("ISAAC_GAME_ROOT"),
        os.path.join(project_root, "The Binding of Isaac Rebirth Repentance"),
        os.path.join(os.path.dirname(project_root), "The Binding of Isaac Rebirth Repentance"),
    ]

    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "isaac-ng.exe")):
            return os.path.abspath(candidate)

    return os.path.abspath(candidates[1])


class SerpentIsaacGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        game_root = resolve_game_root()

        kwargs["platform"] = "executable"
        kwargs["window_name"] = os.environ.get("ISAAC_WINDOW_NAME", "Binding of Isaac: Repentance")
        kwargs["executable_path"] = os.path.join(game_root, "isaac-ng.exe")
        kwargs["input_controller"] = InputControllers.NATIVE_WIN32

        super().__init__(**kwargs)

        self.game_root = game_root
        self.api_class = IsaacAPI
        self.api_instance = None

    @property
    def screen_regions(self):
        return {
            "REGION_FULL": (0, 0, 960, 540),
            "REGION_GAMEPLAY": (0, 0, 960, 540),
            "REGION_HEALTH": (40, 20, 260, 90),
            "REGION_STATS": (0, 80, 180, 260),
            "REGION_MAP": (760, 20, 950, 150),
            "REGION_ACTIVE_ITEM": (35, 120, 110, 210),
        }

    @property
    def ocr_presets(self):
        return {}
