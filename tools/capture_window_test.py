import os
import sys
import time

import mss
import mss.tools


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from plugins.SerpentIsaacGamePlugin.files.serpent_isaac_game import SerpentIsaacGame


def main():
    game = SerpentIsaacGame()
    game.launch(dry_run=True)

    geometry = game.window_geometry
    monitor = {
        "left": geometry["x_offset"],
        "top": geometry["y_offset"],
        "width": geometry["width"],
        "height": geometry["height"],
    }

    os.makedirs(os.path.join(PROJECT_ROOT, "artifacts", "frames"), exist_ok=True)
    output_path = os.path.join(PROJECT_ROOT, "artifacts", "frames", "isaac_window_capture.png")

    time.sleep(1)
    with mss.mss() as screen_capture:
        image = screen_capture.grab(monitor)
        mss.tools.to_png(image.rgb, image.size, output=output_path)

    print("window_geometry=%s" % geometry)
    print("capture_path=%s" % output_path)


if __name__ == "__main__":
    main()
