import os
import sys
import time

from serpent.game_agent import GameAgent

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.isaac_experiment_db import IsaacExperimentDB


class SerpentIsaacSystemTestGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_count = 0
        self.db = IsaacExperimentDB()
        self.episode_id = None

    def setup_play(self, **kwargs):
        os.makedirs(os.path.join("artifacts", "logs"), exist_ok=True)
        self.db.initialize()
        self.episode_id = self.db.start_episode(notes="SerpentIsaacSystemTest PLAY run")

    def handle_play(self, game_frame):
        self.frame_count += 1
        log_path = os.path.join("artifacts", "logs", "interface_frames.log")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("%s,frame=%s,shape=%s\n" % (
                time.time(),
                self.frame_count,
                getattr(game_frame.frame, "shape", None)
            ))

        self.db.record_step(
            episode_id=self.episode_id,
            step_id=self.frame_count,
            action="WAIT",
            reward=0.0,
            health="unknown",
            room_id="unknown",
            done=False
        )
