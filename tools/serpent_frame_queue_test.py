import os
import sys
import time

import numpy as np
from redis import StrictRedis

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from serpent.config import config
from plugins.SerpentIsaacGamePlugin.files.serpent_isaac_game import SerpentIsaacGame


def main():
    client = StrictRedis(**config["redis"])
    redis_key = config["frame_grabber"]["redis_key"]

    game = SerpentIsaacGame()
    game.launch(dry_run=True)

    client.delete(redis_key)
    game.start_frame_grabber()

    try:
        deadline = time.time() + 15
        frame_count = 0
        while time.time() < deadline:
            frame_count = client.llen(redis_key)
            if frame_count > 0:
                break
            time.sleep(0.25)

        if frame_count == 0:
            raise RuntimeError("No frames were pushed to Redis by FrameGrabber.")

        frame_data = client.lindex(redis_key, 0)
        timestamp, shape, dtype, frame_bytes = frame_data.split(b"~", maxsplit=3)
        frame_shape = tuple(int(part) for part in shape.decode("utf-8").split(", "))
        frame = np.frombuffer(frame_bytes, dtype=dtype.decode("utf-8")).reshape(frame_shape)

        print("redis_key=%s" % redis_key)
        print("frame_count=%s" % frame_count)
        print("frame_shape=%s" % (frame.shape,))
        print("frame_dtype=%s" % frame.dtype)
        print("frame_mean=%.2f" % frame.mean())
        print("timestamp=%s" % timestamp.decode("utf-8"))
    finally:
        game.stop_frame_grabber()


if __name__ == "__main__":
    main()
