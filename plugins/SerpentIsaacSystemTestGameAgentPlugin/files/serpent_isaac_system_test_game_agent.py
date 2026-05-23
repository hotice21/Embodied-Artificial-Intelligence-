import os
import time

from serpent.game_agent import GameAgent


class SerpentIsaacSystemTestGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_count = 0

    def setup_play(self):
        os.makedirs(os.path.join("artifacts", "logs"), exist_ok=True)

    def handle_play(self, game_frame):
        self.frame_count += 1
        log_path = os.path.join("artifacts", "logs", "interface_frames.log")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("%s,frame=%s,shape=%s\n" % (
                time.time(),
                self.frame_count,
                getattr(game_frame.frame, "shape", None)
            ))
