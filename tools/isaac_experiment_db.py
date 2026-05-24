import json
import os
import sqlite3
import uuid
from datetime import datetime


DEFAULT_DB_PATH = os.path.join("data", "isaac_experiments.sqlite3")


def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class IsaacExperimentDB(object):
    """Small SQLite store for Isaac AI episodes, steps, and evaluation records."""

    def __init__(self, db_path=None):
        self.db_path = db_path or os.environ.get("ISAAC_EXPERIMENT_DB") or DEFAULT_DB_PATH

    def initialize(self):
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir and not os.path.isdir(db_dir):
            os.makedirs(db_dir)

        with self._connect() as connection:
            connection.executescript("""
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    seed TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_reward REAL NOT NULL DEFAULT 0,
                    rooms_cleared INTEGER NOT NULL DEFAULT 0,
                    boss_defeated INTEGER NOT NULL DEFAULT 0,
                    death_reason TEXT,
                    notes TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    step_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    reward REAL NOT NULL DEFAULT 0,
                    health TEXT,
                    room_id TEXT,
                    frame_path TEXT,
                    next_frame_path TEXT,
                    done INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE (episode_id, step_id),
                    FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
                );

                CREATE TABLE IF NOT EXISTS model_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    algorithm TEXT,
                    hyperparameters TEXT,
                    training_steps INTEGER,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    checkpoint_id TEXT,
                    average_reward REAL,
                    survival_time REAL,
                    rooms_cleared INTEGER,
                    boss_win_rate REAL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (checkpoint_id) REFERENCES model_checkpoints (checkpoint_id)
                );
            """)

    def start_episode(self, episode_id=None, seed=None, notes=None):
        self.initialize()
        episode_id = episode_id or "episode-%s" % uuid.uuid4().hex[:12]
        now = utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO episodes (
                    episode_id, seed, start_time, notes, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (episode_id, seed, now, notes, now)
            )

        return episode_id

    def record_step(
        self,
        episode_id,
        step_id,
        action,
        reward=0.0,
        health=None,
        room_id=None,
        frame_path=None,
        next_frame_path=None,
        done=False
    ):
        self.initialize()
        now = utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO steps (
                    episode_id, step_id, action, reward, health, room_id,
                    frame_path, next_frame_path, done, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    step_id,
                    action,
                    float(reward or 0.0),
                    health,
                    room_id,
                    frame_path,
                    next_frame_path,
                    1 if done else 0,
                    now
                )
            )
            connection.execute(
                """
                UPDATE episodes
                   SET total_reward = total_reward + ?,
                       updated_at = ?
                 WHERE episode_id = ?
                """,
                (float(reward or 0.0), now, episode_id)
            )

    def finish_episode(
        self,
        episode_id,
        total_reward=None,
        rooms_cleared=None,
        boss_defeated=None,
        death_reason=None,
        notes=None
    ):
        self.initialize()
        now = utc_now()

        assignments = ["end_time = ?", "updated_at = ?"]
        values = [now, now]

        optional_values = [
            ("total_reward", total_reward),
            ("rooms_cleared", rooms_cleared),
            ("boss_defeated", boss_defeated),
            ("death_reason", death_reason),
            ("notes", notes),
        ]

        for column, value in optional_values:
            if value is not None:
                assignments.append("%s = ?" % column)
                values.append(value)

        values.append(episode_id)
        query = "UPDATE episodes SET %s WHERE episode_id = ?" % ", ".join(assignments)

        with self._connect() as connection:
            connection.execute(query, values)

    def record_artifact(self, kind, path, episode_id=None, notes=None):
        self.initialize()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    episode_id, kind, path, notes, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (episode_id, kind, path, notes, utc_now())
            )

    def record_model_checkpoint(
        self,
        model_path,
        checkpoint_id=None,
        algorithm=None,
        hyperparameters=None,
        training_steps=None
    ):
        self.initialize()
        checkpoint_id = checkpoint_id or "checkpoint-%s" % uuid.uuid4().hex[:12]

        if hyperparameters is not None and not isinstance(hyperparameters, str):
            hyperparameters = json.dumps(hyperparameters, sort_keys=True)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_checkpoints (
                    checkpoint_id, model_path, algorithm, hyperparameters,
                    training_steps, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    model_path,
                    algorithm,
                    hyperparameters,
                    training_steps,
                    utc_now()
                )
            )

        return checkpoint_id

    def record_evaluation_run(
        self,
        checkpoint_id=None,
        run_id=None,
        average_reward=None,
        survival_time=None,
        rooms_cleared=None,
        boss_win_rate=None,
        notes=None
    ):
        self.initialize()
        run_id = run_id or "eval-%s" % uuid.uuid4().hex[:12]

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, checkpoint_id, average_reward, survival_time,
                    rooms_cleared, boss_win_rate, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    checkpoint_id,
                    average_reward,
                    survival_time,
                    rooms_cleared,
                    boss_win_rate,
                    notes,
                    utc_now()
                )
            )

        return run_id

    def counts(self):
        self.initialize()
        result = {}

        with self._connect() as connection:
            for table_name in [
                "episodes",
                "steps",
                "artifacts",
                "model_checkpoints",
                "evaluation_runs",
            ]:
                cursor = connection.execute("SELECT COUNT(*) AS count FROM %s" % table_name)
                result[table_name] = cursor.fetchone()["count"]

        return result

    def latest_episode(self):
        self.initialize()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT episode_id, start_time, end_time, total_reward,
                       rooms_cleared, boss_defeated, death_reason, notes
                  FROM episodes
              ORDER BY start_time DESC
                 LIMIT 1
                """
            )
            row = cursor.fetchone()

        return dict(row) if row is not None else None

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def main():
    db = IsaacExperimentDB()
    db.initialize()
    print("db_path=%s" % db.db_path)
    print("counts=%s" % db.counts())


if __name__ == "__main__":
    main()
