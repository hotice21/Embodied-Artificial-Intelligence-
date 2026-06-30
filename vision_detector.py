from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


Region = Sequence[int]
Box = Tuple[int, int, int, int]
TemplateItem = Tuple[str, np.ndarray]


@dataclass
class VisionConfig:
    capture_monitor: int = 1
    capture_region: Optional[Region] = None
    display_enabled: bool = True
    display_window_name: str = 'SerpentAI Vision'
    display_scale: float = 1.0
    display_anchor: str = 'top_right'
    analysis_scale: float = 1.0
    template_scales: List[float] = field(default_factory=lambda: [1.0, 0.9, 1.1])
    resource_root: str = 'resources'
    labels_csv: str = 'labels.csv'
    generated_labels_csv: str = 'labels_generated.csv'
    label_properties_file: str = 'label_properties.json'
    label_bank_dir: str = 'label_bank'
    player_label: str = 'player'
    enemy_label: str = 'enemy'
    ground_label: str = 'ground'
    player_bullet_label: str = 'bullets_player'
    enemy_bullet_label: str = 'bullets_enemy'
    player_hit_resource_dir: str = 'resources/player_hit'
    player_hit_positive_tags: List[str] = field(default_factory=lambda: ['hurt', 'hit', 'injured', 'damage', '受伤'])
    player_template_threshold: float = 0.76
    enemy_template_threshold: float = 0.72
    player_bullet_template_threshold: float = 0.70
    enemy_bullet_template_threshold: float = 0.70
    player_hit_template_threshold: float = 0.76
    time_penalty: int = -1
    player_hit_penalty: int = -8
    bullet_hit_reward: int = 18
    contact_margin: int = 4
    contact_cooldown_frames: int = 4
    search_padding: int = 48
    max_enemy_boxes: int = 20
    max_bullet_boxes: int = 30
    use_enemy_motion_candidates: bool = True
    enemy_motion_threshold: int = 18
    enemy_motion_min_area: int = 80
    enemy_motion_max_area: int = 18000
    enemy_motion_max_boxes: int = 10
    enemy_motion_merge_iou: float = 0.35


class VisionCombatDetector:
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.root = Path(__file__).resolve().parent
        self.resource_root = self.root / cfg.resource_root
        self.labels_csv = self.resource_root / cfg.labels_csv
        self.generated_labels_csv = self.resource_root / cfg.generated_labels_csv
        self.label_properties_path = self.resource_root / cfg.label_properties_file
        self.label_bank_dir = self.resource_root / cfg.label_bank_dir
        self.template_scales = [float(scale) for scale in (cfg.template_scales or [1.0]) if float(scale) > 0]
        if not self.template_scales:
            self.template_scales = [1.0]
        self.label_properties = self._load_label_properties()
        self._template_sets = self._load_template_sets()
        self._player_hit_templates = self._load_player_hit_templates()
        self._contact_last_step: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
        self._window_ready = False
        
        # 初始化增强视觉后处理器（新增）
        self._enhanced_processor = None
        try:
            from enhanced_vision_processor import EnhancedVisionPostProcessor
            self._enhanced_processor = EnhancedVisionPostProcessor()
            logging.info("Enhanced vision post-processor initialized")
        except ImportError:
            logging.warning("Enhanced vision post-processor not available")
        self._capture_origin = (0, 0)
        self._last_frame_shape: Optional[Tuple[int, int]] = None
        self._monitor = None
        self._window_positioned = False
        self._prev_gray: Optional[np.ndarray] = None

    def _default_label_properties(self) -> Dict[str, Dict[str, object]]:
        ground_label = str(self.cfg.ground_label or 'ground')
        enemy_label = str(self.cfg.enemy_label or 'enemy')
        player_label = str(self.cfg.player_label or 'player')
        return {
            player_label: {
                'kind': 'positive',
                'trainable': True,
                'match_threshold': self.cfg.player_template_threshold,
            },
            enemy_label: {
                'kind': 'positive',
                'trainable': True,
                'match_threshold': self.cfg.enemy_template_threshold,
            },
            str(self.cfg.player_bullet_label): {
                'kind': 'positive',
                'trainable': True,
                'match_threshold': self.cfg.player_bullet_template_threshold,
            },
            str(self.cfg.enemy_bullet_label): {
                'kind': 'positive',
                'trainable': True,
                'match_threshold': self.cfg.enemy_bullet_template_threshold,
            },
            ground_label: {
                'kind': 'background',
                'trainable': False,
                'match_threshold': 0.68,
                'suppresses': [enemy_label, str(self.cfg.enemy_bullet_label), str(self.cfg.player_bullet_label)],
                'suppress_iou': 0.08,
            },
        }

    def _load_label_properties(self) -> Dict[str, Dict[str, object]]:
        defaults = self._default_label_properties()
        data: Dict[str, Dict[str, object]] = {}
        if self.label_properties_path.exists():
            try:
                with self.label_properties_path.open('r', encoding='utf-8') as handle:
                    loaded = json.load(handle) or {}
                if isinstance(loaded, dict):
                    for key, value in loaded.items():
                        if isinstance(value, dict):
                            data[str(key)] = value
            except Exception as exc:
                logging.warning(f'Failed loading label properties: {exc}')
        merged = defaults.copy()
        merged.update(data)
        return merged

    def _label_property(self, label: str) -> Dict[str, object]:
        prop = self._default_label_properties().get(label, {}).copy()
        prop.update(self.label_properties.get(label, {}))
        return prop

    def _label_match_threshold(self, label: str, fallback: float) -> float:
        prop = self._label_property(label)
        for key in ('match_threshold', 'template_threshold', 'threshold'):
            if key in prop:
                try:
                    return float(prop[key])
                except Exception:
                    continue
        return float(fallback)

    def _load_templates_from_csv(self, csv_path: Path, sets: Dict[str, List[TemplateItem]], seen: set):
        if cv2 is None or not csv_path.exists():
            return
        try:
            with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    image_rel = str(row.get('image', '')).strip().replace('/', '\\')
                    label = str(row.get('label', '')).strip()
                    if not image_rel or not label:
                        continue
                    image_path = self.resource_root / Path(image_rel)
                    if not image_path.exists():
                        continue
                    key = (label, str(image_path.resolve()))
                    if key in seen:
                        continue
                    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    sets[label].append((str(image_path), image))
                    seen.add(key)
        except Exception as exc:
            logging.warning(f'Failed loading templates from {csv_path}: {exc}')

    def _preprocess_template(self, image: np.ndarray) -> np.ndarray:
        """预处理模板图像以提高匹配效果"""
        if cv2 is None:
            return image
        
        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # 边缘增强
        edges = cv2.Canny(enhanced, 50, 150)
        enhanced = cv2.addWeighted(enhanced, 0.7, edges, 0.3, 0)
        
        return enhanced

    def _load_templates_from_directories(self, sets: Dict[str, List[TemplateItem]], seen: set):
        if cv2 is None or not self.resource_root.exists():
            return

        image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        reserved = {
            Path(str(self.cfg.player_hit_resource_dir)).name,
            Path(str(self.cfg.label_bank_dir)).name,
            'collected_labels',
            '__pycache__',
        }

        try:
            for child in sorted(self.resource_root.iterdir()):
                if not child.is_dir():
                    continue
                if child.name in reserved:
                    continue
                label = child.name
                for image_path in sorted(child.rglob('*')):
                    if not image_path.is_file() or image_path.suffix.lower() not in image_exts:
                        continue
                    key = (label, str(image_path.resolve()))
                    if key in seen:
                        continue
                    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    # 预处理模板以提高匹配效果
                    processed_image = self._preprocess_template(image)
                    sets[label].append((str(image_path), processed_image))
                    seen.add(key)
        except Exception as exc:
            logging.warning(f'Failed scanning label directories: {exc}')

    def _load_template_sets(self) -> Dict[str, List[TemplateItem]]:
        sets: Dict[str, List[TemplateItem]] = defaultdict(list)
        if cv2 is None:
            return sets

        seen = set()
        self._load_templates_from_csv(self.labels_csv, sets, seen)
        self._load_templates_from_csv(self.generated_labels_csv, sets, seen)
        self._load_templates_from_directories(sets, seen)

        if sets:
            summary = ', '.join(f'{label}={len(items)}' for label, items in sorted(sets.items()))
            logging.info(f'Loaded vision templates: {summary}')
        return sets

    def _load_player_hit_templates(self) -> List[TemplateItem]:
        templates: List[TemplateItem] = []
        if cv2 is None:
            return templates

        root = self.root / self.cfg.player_hit_resource_dir
        if not root.exists() or not root.is_dir():
            return templates

        positive_tags = [str(tag).strip().lower() for tag in (self.cfg.player_hit_positive_tags or []) if str(tag).strip()]
        image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        try:
            for path in sorted(root.rglob('*')):
                if not path.is_file() or path.suffix.lower() not in image_exts:
                    continue
                image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                stem = path.stem.lower()
                if any(tag in stem for tag in positive_tags):
                    templates.append((str(path), image))
        except Exception as exc:
            logging.warning(f'Failed loading player hit templates: {exc}')

        if templates:
            logging.info(f'Loaded player hit templates: {len(templates)}')
        return templates

    def _resolve_capture_region(self) -> Optional[Region]:
        if self.cfg.capture_region:
            region = tuple(int(value) for value in self.cfg.capture_region)
            if len(region) == 4:
                return region

        # Default to the top-left part of the screen if no explicit region is configured.
        return (0, 0, 1000, 700)

    def _position_window(self, width: int, height: int):
        if cv2 is None or not self.cfg.display_enabled:
            return

        anchor = str(self.cfg.display_anchor or 'top_right').lower()
        try:
            import ctypes
            user32 = ctypes.WinDLL('user32', use_last_error=True)
            screen_width = int(user32.GetSystemMetrics(0))
        except Exception:
            screen_width = width

        if anchor == 'top_right':
            x = max(0, screen_width - int(width))
            y = 0
        else:
            x = 0
            y = 0

        try:
            cv2.resizeWindow(self.cfg.display_window_name, int(width), int(height))
            cv2.moveWindow(self.cfg.display_window_name, x, y)
            self._window_positioned = True
        except Exception as exc:
            logging.debug(f'window positioning skipped: {exc}')

    def _capture_frame(self):
        try:
            import mss
            from PIL import Image
        except Exception:
            return None, None

        region = self._resolve_capture_region()
        try:
            with mss.mss() as sct:
                if region is None:
                    monitor = sct.monitors[min(max(int(self.cfg.capture_monitor), 1), len(sct.monitors) - 1)]
                    self._monitor = monitor
                    left = int(monitor['left'])
                    top = int(monitor['top'])
                    width = int(monitor['width'])
                    height = int(monitor['height'])
                else:
                    left, top, width, height = map(int, region)
                img = sct.grab({'left': left, 'top': top, 'width': width, 'height': height})
                im = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
                bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) if cv2 is not None else np.array(im)
                self._capture_origin = (left, top)
                self._last_frame_shape = (height, width)
                return bgr, (left, top)
        except Exception as exc:
            logging.debug(f'capture failed: {exc}')
            return None, None

    def _resize_for_analysis(self, frame_bgr: np.ndarray):
        scale = float(self.cfg.analysis_scale)
        if cv2 is None or scale == 1.0:
            return frame_bgr, 1.0
        if scale <= 0:
            return frame_bgr, 1.0
        height, width = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _resize_template(self, template: np.ndarray, scale: float):
        if cv2 is None or scale == 1.0:
            return template
        height, width = template.shape[:2]
        new_width = max(2, int(width * scale))
        new_height = max(2, int(height * scale))
        if new_width == width and new_height == height:
            return template
        return cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _iou(self, a: Box, b: Box) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2 = ax1 + aw
        ay2 = ay1 + ah
        bx2 = bx1 + bw
        by2 = by1 + bh
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 0 else 0.0

    def _touch_or_overlap(self, a: Box, b: Box, margin: int = 0) -> bool:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2 = ax1 + aw
        ay2 = ay1 + ah
        bx2 = bx1 + bw
        by2 = by1 + bh
        return not (
            ax2 < bx1 - margin or
            bx2 < ax1 - margin or
            ay2 < by1 - margin or
            by2 < ay1 - margin
        )

    def _quantize_box(self, box: Box) -> Tuple[int, int, int, int]:
        x, y, w, h = box
        return (int(x // 12), int(y // 12), int(max(1, w) // 10), int(max(1, h) // 10))

    def _match_templates(self, frame_gray: np.ndarray, templates: List[TemplateItem], threshold: float, max_results: int) -> List[Dict[str, object]]:
        if cv2 is None or not templates:
            return []

        matches: List[Dict[str, object]] = []
        for template_path, template in templates:
            for scale in self.template_scales:
                resized = self._resize_template(template, scale)
                th, tw = resized.shape[:2]
                if frame_gray.shape[0] < th or frame_gray.shape[1] < tw:
                    continue
                result = cv2.matchTemplate(frame_gray, resized, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= threshold)
                if locations[0].size == 0:
                    continue
                ys, xs = locations
                scores = result[ys, xs]
                order = np.argsort(scores)[::-1]
                limit = min(len(order), max_results * 4)
                for idx in order[:limit]:
                    x = int(xs[idx])
                    y = int(ys[idx])
                    score = float(scores[idx])
                    matches.append({
                        'label': None,
                        'template': template_path,
                        'score': score,
                        'box': (x, y, tw, th),
                    })

        matches.sort(key=lambda item: float(item['score']), reverse=True)
        kept: List[Dict[str, object]] = []
        for candidate in matches:
            if len(kept) >= max_results:
                break
            box = candidate['box']
            if all(self._iou(box, item['box']) < 0.25 for item in kept):
                kept.append(candidate)
        return kept

    def _apply_suppressors(self, detections: Dict[str, List[Dict[str, object]]]) -> Dict[str, List[Dict[str, object]]]:
        suppressors: List[Tuple[str, Box, Dict[str, object]]] = []
        for label, items in detections.items():
            prop = self._label_property(label)
            kind = str(prop.get('kind', 'positive')).lower()
            suppresses = prop.get('suppresses') or []
            if kind in {'background', 'suppressor', 'ignore'} or suppresses:
                for item in items:
                    suppressors.append((label, item['box'], prop))

        if not suppressors:
            return detections

        filtered: Dict[str, List[Dict[str, object]]] = {}
        protected_labels = {self.cfg.player_label, self.cfg.ground_label, 'player_hit'}
        for label, items in detections.items():
            if label in protected_labels:
                filtered[label] = items
                continue
            kept: List[Dict[str, object]] = []
            for item in items:
                box = item['box']
                suppressed = False
                for sup_label, sup_box, sup_prop in suppressors:
                    suppresses = sup_prop.get('suppresses') or []
                    if suppresses and label not in suppresses:
                        continue
                    suppress_iou = float(sup_prop.get('suppress_iou', 0.08))
                    if self._iou(box, sup_box) >= suppress_iou:
                        suppressed = True
                        break
                if not suppressed:
                    kept.append(item)
            filtered[label] = kept
        return filtered

    def _draw_boxes(self, frame_bgr: np.ndarray, detections: Dict[str, List[Dict[str, object]]], reward: int):
        if cv2 is None:
            return frame_bgr

        overlay = frame_bgr.copy()
        color_map = {
            'player': (0, 220, 0),
            'enemy': (0, 0, 255),
            'ground': (120, 120, 120),
            'bullets_player': (255, 220, 0),
            'bullets_enemy': (0, 255, 255),
            'player_hit': (255, 0, 255),
        }

        for label, items in detections.items():
            color = color_map.get(label, (255, 255, 255))
            for item in items:
                x, y, w, h = item['box']
                score = float(item.get('score', 0.0))
                cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                cv2.putText(
                    overlay,
                    f'{label}:{score:.2f}',
                    (int(x), max(12, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(
            overlay,
            f'reward={reward}',
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _show(self, frame_bgr: np.ndarray):
        if cv2 is None or not self.cfg.display_enabled:
            return
        
        try:
            # 检查是否启用仅显示检测框模式
            display_mode = getattr(self.cfg, 'display_mode', 'full')  # full, boxes_only, minimal
            
            if display_mode == 'boxes_only':
                # 仅显示检测框，不显示原图 - 创建空白画布
                display = np.zeros_like(frame_bgr)
            elif display_mode == 'minimal':
                # 极简模式 - 缩小显示尺寸
                display = cv2.resize(frame_bgr, (int(frame_bgr.shape[1] * 0.5), int(frame_bgr.shape[0] * 0.5)))
            else:
                display = frame_bgr
            
            if not self._window_ready:
                cv2.namedWindow(self.cfg.display_window_name, cv2.WINDOW_NORMAL)
                self._window_ready = True
            
            scale = float(self.cfg.display_scale)
            if scale > 0 and scale != 1.0 and display_mode != 'boxes_only':
                display = cv2.resize(display, (max(1, int(display.shape[1] * scale)), max(1, int(display.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            
            if not self._window_positioned:
                self._position_window(display.shape[1], display.shape[0])
            
            cv2.imshow(self.cfg.display_window_name, display)
            
            # 根据显示模式调整waitKey时间，提高帧率
            wait_time = getattr(self.cfg, 'display_wait_ms', 1)
            cv2.waitKey(wait_time)
            
        except Exception as exc:
            logging.debug(f'display skipped: {exc}')
            # 尝试重建窗口
            try:
                cv2.destroyWindow(self.cfg.display_window_name)
                self._window_ready = False
                self._window_positioned = False
            except Exception:
                pass

    def _player_crop(self, frame_gray: np.ndarray, player_box: Optional[Box]) -> Tuple[np.ndarray, Tuple[int, int]]:
        if player_box is None:
            return frame_gray, (0, 0)
        x, y, w, h = player_box
        pad = max(20, int(max(w, h) * 1.5))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_gray.shape[1], x + w + pad)
        y2 = min(frame_gray.shape[0], y + h + pad)
        return frame_gray[y1:y2, x1:x2], (x1, y1)

    def _detect_player_hit(self, frame_gray: np.ndarray, player_box: Optional[Box]) -> Dict[str, object]:
        # 方法1：基于闪烁检测（角色受伤时短暂消失）
        blink_detected = self._detect_blink(frame_gray)
        
        # 方法2：传统模板匹配
        template_matches = []
        if self._player_hit_templates:
            crop, origin = self._player_crop(frame_gray, player_box)
            if crop.size == 0:
                crop = frame_gray
                origin = (0, 0)

            template_matches = self._match_templates(crop, self._player_hit_templates, self.cfg.player_hit_template_threshold, max_results=3)
            for item in template_matches:
                x, y, w, h = item['box']
                item['box'] = (int(x + origin[0]), int(y + origin[1]), int(w), int(h))
        
        # 综合判定：闪烁检测或模板匹配任一触发即判定为受伤
        hit_detected = blink_detected or bool(template_matches)
        
        return {
            'hit': hit_detected, 
            'matches': template_matches,
            'blink_detected': blink_detected
        }
    
    def _detect_blink(self, frame_gray: np.ndarray) -> bool:
        """检测闪烁效果（角色受伤时的短暂消失又快速出现）
        
        原理：比较当前帧与前一帧的差异，如果超过一定比例的像素发生剧烈变化，
        说明可能发生了闪烁（角色消失后又出现）。
        """
        if cv2 is None or self._prev_gray is None:
            self._prev_gray = frame_gray.copy()
            return False
        
        if self._prev_gray.shape != frame_gray.shape:
            self._prev_gray = frame_gray.copy()
            return False
        
        # 计算帧差
        diff = cv2.absdiff(frame_gray, self._prev_gray)
        
        # 计算变化比例
        non_zero_count = np.count_nonzero(diff > 30)  # 阈值30
        total_pixels = frame_gray.shape[0] * frame_gray.shape[1]
        change_ratio = non_zero_count / total_pixels
        
        # 更新前一帧
        self._prev_gray = frame_gray.copy()
        
        # 如果超过25%的像素发生变化，判定为闪烁
        return change_ratio > 0.25
    
    def _find_player_by_heuristics(self, frame_bgr: np.ndarray, frame_gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """使用启发式方法寻找玩家位置
        
        基于以下假设：
        1. 玩家通常在画面中央或下方区域
        2. 玩家有特定的颜色特征（蓝色衣服、肤色等）
        3. 玩家是场景中移动的对象
        """
        if cv2 is None:
            return None
        
        # 初始化历史位置追踪
        if not hasattr(self, '_player_history'):
            self._player_history = []
        
        h, w = frame_gray.shape[:2]
        
        # 方法A：基于颜色特征检测（Isaac角色特征）
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # 蓝色衣服范围
        blue_lower = np.array([90, 50, 50], dtype=np.uint8)
        blue_upper = np.array([130, 255, 255], dtype=np.uint8)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # 肤色范围
        skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        
        # 合并掩码
        combined_mask = cv2.bitwise_or(blue_mask, skin_mask)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 在画面下半部分寻找（玩家通常在下方）
        candidates = []
        lower_half = h * 0.4  # 只搜索下半部分
        
        for contour in contours:
            x, y, w_box, h_box = cv2.boundingRect(contour)
            area = w_box * h_box
            
            # 过滤太小或太大的区域
            if area < 100 or area > 8000:
                continue
            
            # 只考虑画面下半部分
            if y < lower_half:
                continue
            
            # 检查宽高比（角色大致是正方形）
            aspect_ratio = float(w_box) / max(1, h_box)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # 计算置信度
            confidence = min(1.0, area / 2000.0)
            candidates.append((confidence, (x, y, w_box, h_box)))
        
        if candidates:
            candidates.sort(reverse=True)
            best_confidence, best_box = candidates[0]
            if best_confidence > 0.3:
                # 更新历史位置
                self._player_history.append(best_box)
                if len(self._player_history) > 5:
                    self._player_history.pop(0)
                return best_box
        
        # 方法B：基于运动检测（如果有历史帧）
        if hasattr(self, '_prev_frame_for_player') and self._prev_frame_for_player is not None:
            if self._prev_frame_for_player.shape == frame_gray.shape:
                diff = cv2.absdiff(frame_gray, self._prev_frame_for_player)
                blur = cv2.GaussianBlur(diff, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    area = w_box * h_box
                    
                    if area < 100 or area > 5000:
                        continue
                    
                    if y >= lower_half:
                        self._player_history.append((x, y, w_box, h_box))
                        if len(self._player_history) > 5:
                            self._player_history.pop(0)
                        self._prev_frame_for_player = frame_gray.copy()
                        return (x, y, w_box, h_box)
        
        self._prev_frame_for_player = frame_gray.copy()
        return None
    
    def _get_fallback_player_box(self, frame_gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """获取备用玩家位置
        
        当所有检测方法都失败时使用：
        1. 返回历史位置（如果有）
        2. 否则返回画面中心偏下的默认位置
        """
        # 检查历史位置
        if hasattr(self, '_player_history') and self._player_history:
            # 返回最近的历史位置
            return self._player_history[-1]
        
        # 默认位置：画面中心偏下
        h, w = frame_gray.shape[:2]
        player_w, player_h = 80, 100
        x = (w - player_w) // 2
        y = int(h * 0.6)  # 偏下位置
        
        return (x, y, player_w, player_h)

    def _enemy_motion_candidates(self, frame_gray: np.ndarray, player_box: Optional[Box]) -> List[Dict[str, object]]:
        # Motion-based fallback proposals: useful when templates miss unseen monsters.
        if cv2 is None:
            return []
        if not bool(self.cfg.use_enemy_motion_candidates):
            self._prev_gray = frame_gray.copy()
            return []
        if self._prev_gray is None or self._prev_gray.shape != frame_gray.shape:
            self._prev_gray = frame_gray.copy()
            return []

        try:
            diff = cv2.absdiff(frame_gray, self._prev_gray)
            self._prev_gray = frame_gray.copy()
            blur = cv2.GaussianBlur(diff, (5, 5), 0)
            _, mask = cv2.threshold(blur, int(self.cfg.enemy_motion_threshold), 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            return []

        proposals: List[Dict[str, object]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = int(w * h)
            if area < int(self.cfg.enemy_motion_min_area) or area > int(self.cfg.enemy_motion_max_area):
                continue
            box = (int(x), int(y), int(w), int(h))
            if player_box is not None and self._touch_or_overlap(box, player_box, margin=6):
                continue
            proposals.append({
                'label': 'enemy_candidate',
                'template': 'motion',
                'score': 0.55,
                'box': box,
            })

        proposals.sort(key=lambda item: int(item['box'][2]) * int(item['box'][3]), reverse=True)
        kept: List[Dict[str, object]] = []
        max_keep = int(self.cfg.enemy_motion_max_boxes)
        merge_iou = float(self.cfg.enemy_motion_merge_iou)
        for candidate in proposals:
            if len(kept) >= max_keep:
                break
            box = candidate['box']
            if all(self._iou(box, item['box']) < merge_iou for item in kept):
                kept.append(candidate)
        return kept

    def process_step(self, step: int, action=None) -> Dict[str, object]:
        frame_bgr, origin = self._capture_frame()
        if frame_bgr is None:
            return {
                'reward': int(self.cfg.time_penalty),
                'player': [],
                'enemy': [],
                'bullets_player': [],
                'bullets_enemy': [],
                'player_hit': [],
                'player_hit_animation': False,
                'bullet_hit': False,
                'hit_pairs': [],
                'origin': origin,
            }

        analysis_frame, analysis_scale = self._resize_for_analysis(frame_bgr)
        frame_gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY) if cv2 is not None else analysis_frame

        detections: Dict[str, List[Dict[str, object]]] = {
            'player': [],
            'enemy': [],
            'ground': [],
            'bullets_player': [],
            'bullets_enemy': [],
            'player_hit': [],
        }

        player_template_items = self._template_sets.get(self.cfg.player_label, [])
        enemy_template_items = self._template_sets.get(self.cfg.enemy_label, [])
        player_bullet_template_items = self._template_sets.get(self.cfg.player_bullet_label, [])
        enemy_bullet_template_items = self._template_sets.get(self.cfg.enemy_bullet_label, [])

        # ===== 第一步：优先检测玩家，确保有且仅有一个player =====
        player_box = None
        
        # 方法1：模板匹配（严格模式）
        if player_template_items:
            player_matches = self._match_templates(frame_gray, player_template_items, self.cfg.player_template_threshold, max_results=3)
            if player_matches:
                # 选择分数最高的作为玩家
                best_player = max(player_matches, key=lambda x: x['score'])
                detections['player'] = [best_player]
                player_box = best_player['box']
                logging.debug(f"Player detected by template: score={best_player['score']:.3f}, box={player_box}")
        
        # 方法2：如果模板匹配失败，使用启发式方法找最可能的玩家位置
        if player_box is None:
            player_box = self._find_player_by_heuristics(frame_bgr, frame_gray)
            if player_box:
                detections['player'] = [{
                    'label': self.cfg.player_label,
                    'box': player_box,
                    'score': 0.6,
                    'method': 'heuristics'
                }]
                logging.debug(f"Player detected by heuristics: box={player_box}")
        
        # 方法3：如果还是找不到，使用历史位置或默认位置
        if player_box is None:
            player_box = self._get_fallback_player_box(frame_gray)
            if player_box:
                detections['player'] = [{
                    'label': self.cfg.player_label,
                    'box': player_box,
                    'score': 0.3,
                    'method': 'fallback'
                }]
                logging.debug(f"Using fallback player position: box={player_box}")

        if enemy_template_items:
            enemy_matches = self._match_templates(frame_gray, enemy_template_items, self._label_match_threshold(self.cfg.enemy_label, self.cfg.enemy_template_threshold), max_results=self.cfg.max_enemy_boxes)
            # 过滤掉与玩家重叠的敌人检测，避免把玩家识别为敌人
            if player_box:
                enemy_matches = [e for e in enemy_matches if self._iou(e['box'], player_box) < 0.15]
            detections['enemy'] = enemy_matches

        ground_template_items = self._template_sets.get(self.cfg.ground_label, [])
        if ground_template_items:
            ground_matches = self._match_templates(frame_gray, ground_template_items, self._label_match_threshold(self.cfg.ground_label, 0.68), max_results=min(self.cfg.max_enemy_boxes, 16))
            detections['ground'] = ground_matches

        # Add motion-based enemy candidates when template coverage is incomplete.
        motion_candidates = self._enemy_motion_candidates(frame_gray, player_box)
        if motion_candidates:
            for cand in motion_candidates:
                cbox = cand['box']
                if all(self._iou(cbox, e['box']) < 0.25 for e in detections['enemy']):
                    detections['enemy'].append(cand)
                    if len(detections['enemy']) >= int(self.cfg.max_enemy_boxes):
                        break

        # ===== 简化版子弹检测：合并检测所有子弹，然后根据位置分类 =====
        all_bullets = []
        
        # 检测玩家子弹模板
        if player_bullet_template_items:
            bullet_matches = self._match_templates(frame_gray, player_bullet_template_items, self._label_match_threshold(self.cfg.player_bullet_label, self.cfg.player_bullet_template_threshold), max_results=self.cfg.max_bullet_boxes)
            for bm in bullet_matches:
                bm['source'] = 'player_template'
            all_bullets.extend(bullet_matches)
        
        # 检测敌人子弹模板
        if enemy_bullet_template_items:
            bullet_matches = self._match_templates(frame_gray, enemy_bullet_template_items, self._label_match_threshold(self.cfg.enemy_bullet_label, self.cfg.enemy_bullet_template_threshold), max_results=self.cfg.max_bullet_boxes)
            for bm in bullet_matches:
                bm['source'] = 'enemy_template'
            all_bullets.extend(bullet_matches)
        
        # ===== 简化版子弹分类：根据位置判断 =====
        # Isaac 的眼泪（玩家子弹）：从玩家位置发出，向外扩散
        # 敌人子弹：从敌人位置发出，向玩家飞来
        bullets_player = []
        bullets_enemy = []
        
        frame_h, frame_w = frame_gray.shape[:2]
        
        # 初始化子弹历史位置（用于运动方向判断）
        if not hasattr(self, '_bullet_history'):
            self._bullet_history = {}
        
        for bullet in all_bullets:
            bx, by, bw, bh = bullet['box']
            bcx, bcy = bx + bw // 2, by + bh // 2
            
            # 计算到玩家的距离
            player_dist = float('inf')
            if player_box:
                px, py, pw, ph = player_box
                pcx, pcy = px + pw // 2, py + ph // 2
                player_dist = ((bcx - pcx) ** 2 + (bcy - pcy) ** 2) ** 0.5
            
            # 计算到最近敌人的距离
            enemy_dists = []
            for enemy in detections['enemy']:
                ex, ey, ew, eh = enemy['box']
                ecx, ecy = ex + ew // 2, ey + eh // 2
                dist = ((bcx - ecx) ** 2 + (bcy - ecy) ** 2) ** 0.5
                enemy_dists.append(dist)
            
            nearest_enemy_dist = min(enemy_dists) if enemy_dists else float('inf')
            
            # 基于位置的分类逻辑：
            # 1. 靠近敌人且远离玩家 → 玩家子弹
            # 2. 靠近玩家且远离敌人 → 敌人子弹
            # 3. 中间位置 → 基于运动方向判断
            
            bullet_id = (bx // 10, by // 10)  # 简化的子弹ID
            
            if player_dist < nearest_enemy_dist and player_dist < 200:
                # 子弹在玩家附近，远离敌人 → 敌人子弹（正在飞向玩家）
                bullets_enemy.append(bullet)
            elif nearest_enemy_dist < 300:
                # 子弹在敌人附近 → 玩家子弹（正在飞向敌人）
                bullets_player.append(bullet)
            else:
                # 无法判断，检查运动方向
                if bullet_id in self._bullet_history:
                    prev_bx, prev_by = self._bullet_history[bullet_id]
                    dx = bcx - prev_bx
                    dy = bcy - prev_by
                    
                    # 如果子弹向玩家方向移动 → 敌人子弹
                    if player_box:
                        px, py, pw, ph = player_box
                        pcx, pcy = px + pw // 2, py + ph // 2
                        to_player_x = pcx - bcx
                        to_player_y = pcy - bcy
                        
                        # 计算运动方向与玩家方向的点积
                        dot = dx * to_player_x + dy * to_player_y
                        if dot > 0:  # 向玩家方向移动
                            bullets_enemy.append(bullet)
                        else:
                            bullets_player.append(bullet)
                    else:
                        # 默认分类
                        bullets_player.append(bullet)
                else:
                    # 默认玩家子弹
                    bullets_player.append(bullet)
            
            # 更新历史位置
            self._bullet_history[bullet_id] = (bcx, bcy)
            
            # 清理过期的历史记录
            if len(self._bullet_history) > 100:
                self._bullet_history = dict(list(self._bullet_history.items())[-50:])
        
        detections['bullets_player'] = bullets_player
        detections['bullets_enemy'] = bullets_enemy

        detections = self._apply_suppressors(detections)

        # ===== 增强视觉后处理：过滤墙体误识别 + 消失-出现模式受伤检测 =====
        hurt = False
        hurt_source = None
        
        if self._enhanced_processor is not None:
            # 使用增强后处理器处理检测结果
            enhanced_result = self._enhanced_processor.process_detections(frame_bgr, detections)
            
            # 更新检测结果（过滤后的敌人）
            detections['enemy'] = enhanced_result.get('enemy', detections['enemy'])
            
            # 更新受伤状态（三种检测方法）
            hurt = enhanced_result.get('hurt', False)
            hurt_source = enhanced_result.get('hurt_source', None)
        
        reward = int(self.cfg.time_penalty)
        
        # 如果增强处理器检测到受伤，应用惩罚
        if hurt and self._enhanced_processor is not None:
            reward += int(self.cfg.player_hit_penalty)
        
        # 当没有使用增强处理器时，使用原始检测逻辑（备用）
        if not hurt and self._enhanced_processor is None:
            # 方法1：敌人与玩家重叠（近距离接触）
            if player_box:
                for enemy in detections['enemy']:
                    if self._touch_or_overlap(enemy['box'], player_box, margin=self.cfg.contact_margin):
                        hurt = True
                        hurt_source = 'enemy_contact'
                        reward += int(self.cfg.player_hit_penalty)
                        break
            
            # 方法2：敌人子弹与玩家重叠
            if not hurt and player_box:
                for bullet in detections['bullets_enemy']:
                    if self._touch_or_overlap(bullet['box'], player_box, margin=self.cfg.contact_margin):
                        hurt = True
                        hurt_source = 'bullet_contact'
                        reward += int(self.cfg.player_hit_penalty)
                        break
            
            # 方法3：闪烁检测（角色受伤时的短暂消失又快速出现）
            if not hurt:
                blink_detected = self._detect_blink(frame_gray)
                if blink_detected:
                    hurt = True
                    hurt_source = 'blink_animation'
                    reward += int(self.cfg.player_hit_penalty)
            
            # 方法4：受伤动画检测（基于模板匹配）
            if not hurt and player_box and self._player_hit_templates:
                crop, origin = self._player_crop(frame_gray, player_box)
                if crop.size > 0:
                    template_matches = self._match_templates(crop, self._player_hit_templates, 
                                                             self.cfg.player_hit_template_threshold, max_results=3)
                    if template_matches:
                        hurt = True
                        hurt_source = 'hit_animation'
                        reward += int(self.cfg.player_hit_penalty)
        
        hit_pairs = []
        now_hits = 0
        for bullet_item in detections['bullets_player']:
            bullet_box = bullet_item['box']
            for enemy_item in detections['enemy']:
                enemy_box = enemy_item['box']
                if not self._touch_or_overlap(bullet_box, enemy_box, margin=self.cfg.contact_margin):
                    continue
                signature = (self._quantize_box(bullet_box), self._quantize_box(enemy_box))
                last_step = self._contact_last_step.get(signature, -10_000)
                if step - last_step < int(self.cfg.contact_cooldown_frames):
                    continue
                self._contact_last_step[signature] = step
                hit_pairs.append({'bullet': bullet_box, 'enemy': enemy_box})
                now_hits += 1
                reward += int(self.cfg.bullet_hit_reward)

        reward = int(reward)
        annotated = self._draw_boxes(analysis_frame, detections, reward)
        self._show(annotated)

        return {
            'reward': reward,
            'player': detections['player'],
            'enemy': detections['enemy'],
            'ground': detections['ground'],
            'bullets_player': detections['bullets_player'],
            'bullets_enemy': detections['bullets_enemy'],
            'player_hit': [],  # 不使用单独的受伤模板检测
            'player_hit_animation': bool(hurt),  # 受伤即动画
            'hurt': bool(hurt),
            'hurt_source': hurt_source,  # 受伤来源（用于调试）
            'total_hurt_count': self._enhanced_processor.hurt_detector.get_hurt_count() if self._enhanced_processor else 0,
            'bullet_hit': now_hits > 0,
            'hit_pairs': hit_pairs,
            'origin': origin,
        }

    def trigger_manual_hurt(self):
        """手动触发受伤（由外部调用，如按H键）"""
        if self._enhanced_processor is not None:
            self._enhanced_processor.hurt_detector.request_manual_hurt()
            logging.info("Manual hurt triggered")


def build_vision_detector(screen_cfg: Dict[str, object]) -> Optional[VisionCombatDetector]:
    combat_cfg = screen_cfg.get('combat', {}) if screen_cfg else {}
    if not combat_cfg:
        return None

    cfg = VisionConfig(
        capture_monitor=int(combat_cfg.get('capture_monitor', 1)),
        capture_region=combat_cfg.get('capture_region') or [0, 0, 1000, 700],
        display_enabled=bool(combat_cfg.get('display_enabled', True)),
        display_window_name=str(combat_cfg.get('display_window_name', 'SerpentAI Vision')),
        display_scale=float(combat_cfg.get('display_scale', 1.0)),
        display_anchor=str(combat_cfg.get('display_anchor', 'top_right')),
        analysis_scale=float(combat_cfg.get('analysis_scale', 1.0)),
        template_scales=[float(value) for value in (combat_cfg.get('template_scales', [1.0, 0.9, 1.1]) or [1.0, 0.9, 1.1])],
        resource_root=str(combat_cfg.get('resource_root', 'resources')),
        labels_csv=str(combat_cfg.get('labels_csv', 'labels.csv')),
        player_label=str(combat_cfg.get('player_label', 'player')),
        enemy_label=str(combat_cfg.get('enemy_label', 'enemy')),
        player_bullet_label=str(combat_cfg.get('player_bullet_label', 'bullets_player')),
        enemy_bullet_label=str(combat_cfg.get('enemy_bullet_label', 'bullets_enemy')),
        ground_label=str(combat_cfg.get('ground_label', 'ground')),
        player_hit_resource_dir=str(combat_cfg.get('player_hit_resource_dir', 'resources/player_hit')),
        player_hit_positive_tags=combat_cfg.get('player_hit_positive_tags', ['hurt', 'hit', 'injured', 'damage', '受伤']) or ['hurt', 'hit', 'injured', 'damage', '受伤'],
        generated_labels_csv=str(combat_cfg.get('generated_labels_csv', 'labels_generated.csv')),
        label_properties_file=str(combat_cfg.get('label_properties_file', 'label_properties.json')),
        label_bank_dir=str(combat_cfg.get('label_bank_dir', 'label_bank')),
        player_template_threshold=float(combat_cfg.get('player_template_threshold', 0.76)),
        enemy_template_threshold=float(combat_cfg.get('enemy_template_threshold', 0.72)),
        player_bullet_template_threshold=float(combat_cfg.get('player_bullet_template_threshold', 0.70)),
        enemy_bullet_template_threshold=float(combat_cfg.get('enemy_bullet_template_threshold', 0.70)),
        player_hit_template_threshold=float(combat_cfg.get('player_hit_template_threshold', 0.76)),
        time_penalty=int(combat_cfg.get('time_penalty', -1)),
        player_hit_penalty=int(combat_cfg.get('player_hit_penalty', -8)),
        bullet_hit_reward=int(combat_cfg.get('bullet_hit_reward', combat_cfg.get('hit_reward', 18))),
        contact_margin=int(combat_cfg.get('contact_margin', 4)),
        contact_cooldown_frames=int(combat_cfg.get('contact_cooldown_frames', 4)),
        search_padding=int(combat_cfg.get('search_padding', 48)),
        max_enemy_boxes=int(combat_cfg.get('max_enemy_boxes', 20)),
        max_bullet_boxes=int(combat_cfg.get('max_bullet_boxes', 30)),
        use_enemy_motion_candidates=bool(combat_cfg.get('use_enemy_motion_candidates', True)),
        enemy_motion_threshold=int(combat_cfg.get('enemy_motion_threshold', 18)),
        enemy_motion_min_area=int(combat_cfg.get('enemy_motion_min_area', 80)),
        enemy_motion_max_area=int(combat_cfg.get('enemy_motion_max_area', 18000)),
        enemy_motion_max_boxes=int(combat_cfg.get('enemy_motion_max_boxes', 10)),
        enemy_motion_merge_iou=float(combat_cfg.get('enemy_motion_merge_iou', 0.35)),
    )

    if cv2 is None:
        return None
    return VisionCombatDetector(cfg)