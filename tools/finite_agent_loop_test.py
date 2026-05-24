import os
import sys
import time

import numpy as np
from redis import StrictRedis

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from serpent.config import config
from serpent.game_frame import GameFrame
from serpent.input_controller import InputController

from plugins.SerpentIsaacGamePlugin.files.serpent_isaac_game import SerpentIsaacGame
from plugins.SerpentIsaacSystemTestGameAgentPlugin.files.serpent_isaac_system_test_game_agent import (
    SerpentIsaacSystemTestGameAgent,
)


def decode_frame(frame_data):
    timestamp, shape, dtype, frame_bytes = frame_data.split(b"~", maxsplit=3)
    frame_shape = tuple(int(part) for part in shape.decode("utf-8").split(", "))
    frame = np.frombuffer(frame_bytes, dtype=dtype.decode("utf-8")).reshape(frame_shape)
    return GameFrame(frame, timestamp=timestamp.decode("utf-8"))


def wait_for_frame(client, redis_key, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame_data = client.lpop(redis_key)
        if frame_data:
            return decode_frame(frame_data)
        time.sleep(0.1)

    raise RuntimeError("No frame was available in Redis before timeout.")


def main():
    frame_limit = int(os.environ.get("ISAAC_AGENT_TEST_FRAMES", "5"))
    client = StrictRedis(**config["redis"])
    redis_key = config["frame_grabber"]["redis_key"]

    game = SerpentIsaacGame()
    game.launch(dry_run=True)

    agent = SerpentIsaacSystemTestGameAgent(
        game=game,
        input_controller=InputController(game=game, backend=game.input_controller)
    )

    client.delete(redis_key)
    game.start_frame_grabber()

    try:
        for _ in range(frame_limit):
            game_frame = wait_for_frame(client, redis_key)
            agent.on_game_frame(game_frame, frame_handler="PLAY")

        agent.db.finish_episode(
            agent.episode_id,
            notes="finite_agent_loop_test completed"
        )

        print("redis_key=%s" % redis_key)
        print("episode_id=%s" % agent.episode_id)
        print("frames_processed=%s" % agent.frame_count)
        print("db_path=%s" % agent.db.db_path)
        print("db_counts=%s" % agent.db.counts())
    finally:
        game.stop_frame_grabber()


if __name__ == "__main__":
    main()
