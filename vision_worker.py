from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

try:
    import cv2
except Exception:
    cv2 = None

from vision_detector import VisionCombatDetector, VisionConfig


class VisionWorker:
    """Run VisionCombatDetector in a background thread and expose latest snapshot.

    Usage:
        worker = VisionWorker(detector)
        worker.start()
        snapshot = worker.get_latest()
        worker.stop()
    """

    def __init__(self, detector: VisionCombatDetector, poll_fps: float = 20.0):
        self.detector = detector
        self.poll_fps = float(poll_fps) if poll_fps and poll_fps > 0 else 20.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Dict[str, object] = {'reward': int(self.detector.cfg.time_penalty), 'player': [], 'enemy': [], 'bullets_player': [], 'bullets_enemy': [], 'player_hit': [], 'player_hit_animation': False, 'hurt': False, 'bullet_hit': False, 'hit_pairs': [], 'origin': (0, 0), 'ts': time.time()}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def _run(self):
        interval = 1.0 / max(1.0, self.poll_fps)
        step = 0
        while not self._stop.is_set():
            try:
                step += 1
                snapshot = self.detector.process_step(step)
                snapshot['ts'] = time.time()
                with self._lock:
                    self._latest = snapshot
            except Exception as exc:
                logging.debug(f'vision worker loop error: {exc}')
            time.sleep(interval)

    def get_latest(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._latest)

    # compatibility: allow worker to be passed where detector expected
    def process_step(self, step: int, action=None) -> Dict[str, object]:
        return self.get_latest()


def build_vision_worker_from_cfg(screen_cfg: Dict[str, object], fps: float = 20.0) -> Optional[VisionWorker]:
    detector = None
    try:
        detector = VisionCombatDetector(VisionConfig(
            capture_monitor=int(screen_cfg.get('capture_monitor', 1)),
        ))
    except Exception:
        detector = None
    if detector is None:
        return None
    worker = VisionWorker(detector, poll_fps=fps)
    return worker
