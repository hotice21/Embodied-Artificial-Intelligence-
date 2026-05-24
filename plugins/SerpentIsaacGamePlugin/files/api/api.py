from serpent.game_api import GameAPI
from serpent.input_controller import InputController
from sneakysnek.keyboard_keys import KeyboardKey


class IsaacAPI(GameAPI):
    ACTION_KEY_MAP = {
        "MOVE_UP": (KeyboardKey.KEY_W,),
        "MOVE_DOWN": (KeyboardKey.KEY_S,),
        "MOVE_LEFT": (KeyboardKey.KEY_A,),
        "MOVE_RIGHT": (KeyboardKey.KEY_D,),
        "SHOOT_UP": (KeyboardKey.KEY_UP,),
        "SHOOT_DOWN": (KeyboardKey.KEY_DOWN,),
        "SHOOT_LEFT": (KeyboardKey.KEY_LEFT,),
        "SHOOT_RIGHT": (KeyboardKey.KEY_RIGHT,),
        "BOMB": (KeyboardKey.KEY_E,),
        "ACTIVE_ITEM": (KeyboardKey.KEY_SPACE,),
        "DROP": (KeyboardKey.KEY_LEFT_CTRL,),
        "MENU_CONFIRM": (KeyboardKey.KEY_RETURN,),
        "WAIT": tuple(),
    }

    def __init__(self, game=None):
        super().__init__(game=game)
        self.input_controller = InputController(game=game, backend=game.input_controller)

    @classmethod
    def available_actions(cls):
        return sorted(cls.ACTION_KEY_MAP.keys())

    @classmethod
    def keys_for_action(cls, action):
        if action not in cls.ACTION_KEY_MAP:
            raise KeyError("Unknown Isaac action: %s" % action)
        return cls.ACTION_KEY_MAP[action]

    def tap_action(self, action, duration=0.05):
        keys = self.keys_for_action(action)
        if not keys:
            return
        self.input_controller.tap_keys(keys, duration=duration, force=True)
