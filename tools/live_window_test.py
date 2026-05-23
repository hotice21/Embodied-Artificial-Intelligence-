import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from plugins.SerpentIsaacGamePlugin.files.serpent_isaac_game import SerpentIsaacGame


def main():
    game = SerpentIsaacGame()
    game.launch(dry_run=True)
    print("window_id=%s" % game.window_id)
    print("window_geometry=%s" % game.window_geometry)
    print("window_name=%s" % game.window_name)


if __name__ == "__main__":
    main()
