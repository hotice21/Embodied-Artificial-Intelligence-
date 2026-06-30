from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import logging
import time

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


Region = Sequence[int]


@dataclass
class CombatSignalConfig:
    player_health_region: Optional[Region] = None
    # 用于检测玩家受击动画（可选）：如果游戏中玩家有明显闪烁/帧差或专门的受击贴图
    player_region: Optional[Region] = None
    player_hit_templates: List[str] = field(default_factory=list)
    # 可选：模板资源目录。目录中的图片文件名可携带标签（如 hurt / normal）。
    player_hit_resource_dir: Optional[str] = None
    # 文件名中命中这些标签时判定为“受伤模板”。
    player_hit_positive_tags: List[str] = field(default_factory=lambda: ['hurt', 'hit', 'injured', 'damage', '受伤'])
    player_hit_template_threshold: float = 0.76
    player_hit_motion_threshold: float = 12.0
    lock_duration_frames: int = 30
    detect_projectiles: bool = True
    enemy_regions: List[Region] = field(default_factory=list)
    enemy_templates: List[str] = field(default_factory=list)
    enemy_template_threshold: float = 0.72
    enemy_motion_threshold: float = 18.0
    enemy_presence_frames: int = 2
    enemy_decay_frames: int = 4
    damage_drop_threshold: float = 5.0
    player_hit_penalty: int = -8
    dodge_window: int = 3
    hit_score_threshold: int = 1
    hit_requires_enemy_visible: bool = True
    hit_reward: int = 18
    dodge_reward: int = 8
    damage_penalty: int = -24
    enemy_visible_penalty: int = 0


class CombatSignalDetector:
    """战斗事件检测层。

    事件定义：
    - enemy_visible: 敌人出现/持续可见
    - damage_taken: 玩家受击（血量下降）
    - hit: 命中敌人（默认用分数跳变或模板闪光作为代理）
    - dodge: 在 threat 窗口内没有受击，且敌人消失或威胁解除
    """

    def __init__(self, cfg: CombatSignalConfig):
        self.cfg = cfg
        self.prev_health = None
        self.prev_score = None
        self.prev_frames: Dict[str, np.ndarray] = {}
        self.prev_player_frame: Optional[np.ndarray] = None
        self.enemy_visible_streak = 0
        self.enemy_last_seen_step = -10_000
        self.threat_start_step = None
        self.threat_damage_flag = False
        self.dodge_claimed = False
        self._templates = self._load_templates(cfg.enemy_templates)
        self._player_templates = self._load_templates(cfg.player_hit_templates)
        if cfg.player_hit_resource_dir:
            self._player_templates.extend(
                self._load_labeled_templates_from_dir(
                    cfg.player_hit_resource_dir,
                    cfg.player_hit_positive_tags,
                )
            )
        self.lock_target = None  # {'x':int,'y':int,'expires':step}
        self.lock_projectiles = []

    def _load_templates(self, template_paths: Sequence[str]):
        templates = []
        if cv2 is None:
            return templates
        for item in template_paths or []:
            try:
                path = Path(item)
                if not path.exists():
                    continue
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates.append((str(path), img))
            except Exception:
                continue
        if templates:
            logging.info(f'Combat detector loaded {len(templates)} enemy templates')
        return templates

    def _load_labeled_templates_from_dir(self, resource_dir: str, positive_tags: Sequence[str]):
        templates = []
        if cv2 is None:
            return templates

        try:
            tags = [str(tag).strip().lower() for tag in (positive_tags or []) if str(tag).strip()]
            root = Path(resource_dir)
            if not root.exists() or not root.is_dir():
                logging.warning(f'player_hit_resource_dir does not exist: {resource_dir}')
                return templates

            image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
            for path in sorted(root.rglob('*')):
                if not path.is_file() or path.suffix.lower() not in image_exts:
                    continue

                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                stem = path.stem.lower()
                label = 'positive' if any(tag in stem for tag in tags) else 'negative'
                templates.append((str(path), img, label))

            if templates:
                pos_count = sum(1 for _, _, lb in templates if lb == 'positive')
                neg_count = len(templates) - pos_count
                logging.info(
                    f'Loaded {len(templates)} player templates from {resource_dir} '
                    f'(positive={pos_count}, negative={neg_count})'
                )
        except Exception as exc:
            logging.warning(f'Failed loading labeled player templates: {exc}')

        return templates

    def _grab_gray(self, region: Region):
        try:
            import mss
            from PIL import Image
        except Exception:
            return None

        x, y, w, h = map(int, region)
        try:
            with mss.mss() as sct:
                img = sct.grab({'left': x, 'top': y, 'width': w, 'height': h})
                im = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
                gray = np.array(im.convert('L'))
                return gray
        except Exception:
            return None

    def _mean_abs_diff(self, a, b):
        if a is None or b is None:
            return 0.0
        if a.shape != b.shape:
            return 0.0
        return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))

    def _template_match(self, region: Region):
        if cv2 is None or not self._templates:
            return False, 0.0
        frame = self._grab_gray(region)
        if frame is None:
            return False, 0.0
        best = 0.0
        for _, tmpl in self._templates:
            if frame.shape[0] < tmpl.shape[0] or frame.shape[1] < tmpl.shape[1]:
                continue
            result = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            if score > best:
                best = float(score)
        return best >= self.cfg.enemy_template_threshold, best

    def _player_hit(self):
        """检测玩家受击：优先使用模板匹配，其次使用帧差（motion）检测。返回 (hit_bool, conf)"""
        if not self.cfg.player_region:
            return False, 0.0
        region = self.cfg.player_region
        frame = self._grab_gray(region)
        if frame is None:
            return False, 0.0

        # 模板匹配优先
        best = 0.0
        best_positive = 0.0
        best_negative = 0.0
        if self._player_templates and cv2 is not None:
            for item in self._player_templates:
                if len(item) == 2:
                    _, tmpl = item
                    label = 'positive'
                else:
                    _, tmpl, label = item
                if frame.shape[0] < tmpl.shape[0] or frame.shape[1] < tmpl.shape[1]:
                    continue
                result = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                if score > best:
                    best = float(score)
                if label == 'positive' and score > best_positive:
                    best_positive = float(score)
                if label == 'negative' and score > best_negative:
                    best_negative = float(score)

            # 若同时有正负模板，要求正样本更强，降低误判。
            if best_positive >= self.cfg.player_hit_template_threshold and best_positive >= (best_negative + 0.03):
                return True, best_positive

            # 兼容仅提供正样本模板的场景。
            if best >= self.cfg.player_hit_template_threshold and best_negative == 0.0:
                return True, best

        # 帧差检测
        prev = self.prev_player_frame
        self.prev_player_frame = frame
        if prev is None:
            return False, 0.0
        diff = self._mean_abs_diff(frame, prev)
        if diff >= self.cfg.player_hit_motion_threshold:
            return True, float(diff)
        return False, float(diff)

    def _find_lock_target(self, step: int):
        """在敌人区域内寻找最显著的运动或模板匹配点，返回屏幕坐标 (x,y) 或 None"""
        best_point = None
        best_score = 0.0
        for region in self.cfg.enemy_regions or []:
            frame = self._grab_gray(region)
            if frame is None:
                continue
            key = f'{int(region[0])}:{int(region[1])}:{int(region[2])}:{int(region[3])}'
            prev = self.prev_frames.get(key)
            if prev is not None and frame.shape == prev.shape:
                diff_map = np.abs(frame.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
                # 找到最大差异点
                idx = int(np.argmax(diff_map))
                r, c = divmod(idx, diff_map.shape[1])
                score = float(diff_map[r, c])
                if score > best_score:
                    best_score = score
                    best_point = (int(region[0] + c), int(region[1] + r))
            # 尝试模板匹配作为补充
            if self._templates and cv2 is not None:
                visible, tscore = self._template_match(region)
                if tscore > best_score and visible:
                    # 近似中心点为 region 中心
                    cx = int(region[0] + region[2] // 2)
                    cy = int(region[1] + region[3] // 2)
                    best_score = tscore
                    best_point = (cx, cy)
        return best_point, best_score

    def _find_projectiles(self, max_hits=6):
        """在敌人区域内基于帧差检测显著小物体（子弹），返回屏幕坐标列表"""
        points = []
        for region in self.cfg.enemy_regions or []:
            frame = self._grab_gray(region)
            if frame is None:
                continue
            key = f'{int(region[0])}:{int(region[1])}:{int(region[2])}:{int(region[3])}'
            prev = self.prev_frames.get(key)
            if prev is None or frame.shape != prev.shape:
                continue
            diff_map = np.abs(frame.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
            # 简单阈值：选取 diff 大于均值上若干倍的位置
            thr = max(20, int(diff_map.mean() + diff_map.std() * 1.5))
            ys, xs = np.where(diff_map >= thr)
            if ys.size:
                pts = list(zip(xs, ys))
                # select top distinct points by value
                vals = diff_map[ys, xs]
                order = np.argsort(vals)[::-1][:max_hits]
                for oi in order:
                    x, y = pts[oi]
                    points.append((int(region[0] + x), int(region[1] + y)))
        return points

    def _enemy_visible(self):
        if self.cfg.enemy_regions:
            scores = []
            for idx, region in enumerate(self.cfg.enemy_regions):
                frame = self._grab_gray(region)
                if frame is None:
                    continue
                key = f'{int(region[0])}:{int(region[1])}:{int(region[2])}:{int(region[3])}'
                prev = self.prev_frames.get(key)
                diff = self._mean_abs_diff(frame, prev)
                self.prev_frames[key] = frame
                scores.append(diff)
            if scores:
                score = max(scores)
                return score >= self.cfg.enemy_motion_threshold, score

        if self._templates and self.cfg.enemy_regions:
            best_visible = False
            best_score = 0.0
            for region in self.cfg.enemy_regions:
                visible, score = self._template_match(region)
                if score > best_score:
                    best_score = score
                best_visible = best_visible or visible
            return best_visible, best_score

        return False, 0.0

    def sample(self, *, step: int, action=None, score=None, score_conf=None, health=None) -> Dict[str, object]:
        events = {
            'enemy_visible': False,
            'enemy_score': 0.0,
            'damage_taken': False,
            'damage_amount': 0.0,
            'hit': False,
            'dodge': False,
            'hit_source': None,
        }

        enemy_visible, enemy_score = self._enemy_visible()
        events['enemy_visible'] = enemy_visible
        events['enemy_score'] = enemy_score

        if enemy_visible:
            self.enemy_visible_streak += 1
            self.enemy_last_seen_step = step
            if self.threat_start_step is None:
                self.threat_start_step = step
            self.dodge_claimed = False
        else:
            self.enemy_visible_streak = 0

        # 受击：优先用显式 health region
        if health is not None and self.prev_health is not None:
            drop = float(self.prev_health) - float(health)
            if drop >= self.cfg.damage_drop_threshold:
                events['damage_taken'] = True
                events['damage_amount'] = drop
                self.threat_damage_flag = True
        self.prev_health = health

        # 击中：默认用 score 跳变做代理，且最好在敌人可见窗口内
        if score is not None and self.prev_score is not None:
            delta = int(score) - int(self.prev_score)
            if delta >= self.cfg.hit_score_threshold:
                if (not self.cfg.hit_requires_enemy_visible) or enemy_visible or (step - self.enemy_last_seen_step) <= self.cfg.enemy_decay_frames:
                    events['hit'] = True
                    events['hit_source'] = 'score_delta'
        self.prev_score = score

        # 躲避：在 threat 窗口内没有受击，且敌人脱离可见状态
        if self.threat_start_step is not None:
            threat_age = step - self.threat_start_step
            recent_threat = (step - self.enemy_last_seen_step) <= self.cfg.enemy_decay_frames
            no_damage = not self.threat_damage_flag
            if (not enemy_visible) and threat_age >= self.cfg.dodge_window and recent_threat and no_damage and not self.dodge_claimed:
                events['dodge'] = True
                self.dodge_claimed = True
                self.threat_start_step = None
                self.threat_damage_flag = False
            elif not recent_threat and not enemy_visible:
                self.threat_start_step = None
                self.threat_damage_flag = False
                self.dodge_claimed = False

        # 玩家受击动画检测与锁定目标逻辑
        player_hit, player_hit_conf = self._player_hit()
        events['player_hit_animation'] = player_hit
        events['player_hit_conf'] = player_hit_conf

        # 如果检测到受击动画，则尝试锁定敌人位置并记录 lock_target
        if player_hit:
            pt, pscore = self._find_lock_target(step)
            if pt:
                self.lock_target = {'x': pt[0], 'y': pt[1], 'expires': step + self.cfg.lock_duration_frames}
            else:
                # 无明确运动点，可用最后被检测到的敌人中心
                if self.enemy_last_seen_step > -1 and self.cfg.enemy_regions:
                    r = self.cfg.enemy_regions[0]
                    self.lock_target = {'x': int(r[0] + r[2]//2), 'y': int(r[1] + r[3]//2), 'expires': step + self.cfg.lock_duration_frames}

        # 如果存在有效锁定且未过期，返回锁定信息
        if self.lock_target and step <= int(self.lock_target.get('expires', -1)):
            events['lock_target'] = {'x': int(self.lock_target['x']), 'y': int(self.lock_target['y']), 'expires': int(self.lock_target['expires'])}
            # 同时尝试检测并返回子弹位置
            if self.cfg.detect_projectiles:
                pts = self._find_projectiles()
                events['projectiles'] = pts
                self.lock_projectiles = pts
        else:
            events['lock_target'] = None
            events['projectiles'] = []

        return events


def build_combat_detector(screen_cfg: Dict[str, object]) -> Optional[CombatSignalDetector]:
    combat_cfg = screen_cfg.get('combat', {}) if screen_cfg else {}
    if not combat_cfg:
        return None

    cfg = CombatSignalConfig(
        player_health_region=combat_cfg.get('player_health_region'),
        player_region=combat_cfg.get('player_region'),
        player_hit_templates=combat_cfg.get('player_hit_templates', []) or [],
        player_hit_resource_dir=combat_cfg.get('player_hit_resource_dir'),
        player_hit_positive_tags=combat_cfg.get('player_hit_positive_tags', []) or ['hurt', 'hit', 'injured', 'damage', '受伤'],
        player_hit_template_threshold=float(combat_cfg.get('player_hit_template_threshold', 0.76)),
        enemy_regions=combat_cfg.get('enemy_regions', []) or [],
        enemy_templates=combat_cfg.get('enemy_templates', []) or [],
        enemy_template_threshold=float(combat_cfg.get('enemy_template_threshold', 0.72)),
        enemy_motion_threshold=float(combat_cfg.get('enemy_motion_threshold', 18.0)),
        enemy_presence_frames=int(combat_cfg.get('enemy_presence_frames', 2)),
        enemy_decay_frames=int(combat_cfg.get('enemy_decay_frames', 4)),
        damage_drop_threshold=float(combat_cfg.get('damage_drop_threshold', 5.0)),
        player_hit_penalty=int(combat_cfg.get('player_hit_penalty', -8)),
        dodge_window=int(combat_cfg.get('dodge_window', 3)),
        hit_score_threshold=int(combat_cfg.get('hit_score_threshold', 1)),
        hit_requires_enemy_visible=bool(combat_cfg.get('hit_requires_enemy_visible', True)),
        hit_reward=int(combat_cfg.get('hit_reward', 18)),
        dodge_reward=int(combat_cfg.get('dodge_reward', 8)),
        damage_penalty=int(combat_cfg.get('damage_penalty', -24)),
        enemy_visible_penalty=int(combat_cfg.get('enemy_visible_penalty', 0)),
        player_hit_motion_threshold=float(combat_cfg.get('player_hit_motion_threshold', 12.0)),
        lock_duration_frames=int(combat_cfg.get('lock_duration_frames', 30)),
        detect_projectiles=bool(combat_cfg.get('detect_projectiles', True)),
    )
    if (
        not cfg.player_health_region
        and not cfg.player_region
        and not cfg.player_hit_templates
        and not cfg.player_hit_resource_dir
        and not cfg.enemy_regions
        and not cfg.enemy_templates
    ):
        return None
    return CombatSignalDetector(cfg)