import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from tools.isaac_experiment_db import IsaacExperimentDB


def main():
    db = IsaacExperimentDB()
    episode_id = db.start_episode(notes="database smoke test")
    db.record_step(
        episode_id=episode_id,
        step_id=1,
        action="WAIT",
        reward=0.0,
        health="unknown",
        room_id="unknown",
        done=False
    )
    db.record_artifact(
        episode_id=episode_id,
        kind="log",
        path=os.path.join("artifacts", "logs", "interface_frames.log"),
        notes="created by smoke test"
    )
    db.finish_episode(episode_id, notes="database smoke test completed")

    print("db_path=%s" % db.db_path)
    print("episode_id=%s" % episode_id)
    print("counts=%s" % db.counts())
    print("latest_episode=%s" % db.latest_episode())


if __name__ == "__main__":
    main()
