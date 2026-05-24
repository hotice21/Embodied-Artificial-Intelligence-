import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from plugins.SerpentIsaacGamePlugin.files.api.api import IsaacAPI
from plugins.SerpentIsaacGamePlugin.files.serpent_isaac_game import SerpentIsaacGame


DEFAULT_ACTIONS = [
    "MOVE_RIGHT",
    "MOVE_LEFT",
    "SHOOT_RIGHT",
    "SHOOT_LEFT",
    "WAIT",
]


def key_names(keys):
    return [key.name for key in keys]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate or send Isaac action inputs.")
    parser.add_argument(
        "--send-actions",
        action="store_true",
        help="Focus the running game window and send the requested actions."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.05,
        help="Key tap duration in seconds when --send-actions is used."
    )
    parser.add_argument(
        "actions",
        nargs="*",
        default=DEFAULT_ACTIONS,
        help="Action names to validate or send."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for action in args.actions:
        keys = IsaacAPI.keys_for_action(action)
        print("%s=%s" % (action, key_names(keys)))

    if not args.send_actions:
        print("send_actions=False")
        return

    game = SerpentIsaacGame()
    game.launch(dry_run=True)
    api = IsaacAPI(game=game)

    sent_count = 0
    for action in args.actions:
        keys = IsaacAPI.keys_for_action(action)
        if keys:
            api.tap_action(action, duration=args.duration)
            sent_count += 1
        time.sleep(0.1)

    print("send_actions=True")
    print("sent_count=%s" % sent_count)
    print("window_name=%s" % game.window_name)


if __name__ == "__main__":
    main()
