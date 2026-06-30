"""
高级视觉检测模块
功能：
1. 基于特征学习的敌人识别（区分敌人与墙体/地面）
2. 三种受伤检测方法：
   - 方法1：玩家消失-出现模式
   - 方法2：玩家位置突然跳跃（瞬移）
   - 方法3：玩家与敌人/子弹重叠
3. 防止墙体误识别的过滤机制
"""

import numpy as np
import cv2
from collections import deque
import time
from typing import Optional, Tuple, Dict, List


class SmartEnemyFilter:
    """
    智能敌人过滤器
    使用颜色、纹理、运动等特征区分敌人与背景（墙体、地面、石头）
    """
    
    def __init__(self):
        # 敌人特征库（从检测结果中动态学习）
        self.enemy_color_signatures = []  # 颜色特征
        self.enemy_texture_signatures = []  # 纹理特征
        self.enemy_size_range = [50, 200]  # 敌人尺寸范围（像素）
        
        # 墙体特征
        self.wall_color = {
            'hue_range': (0, 50),      # 墙体颜色偏暖色调（棕色、灰色）
            'sat_range': (0, 80),      # 饱和度低
            'val_range': (80, 200),    # 明度中等
        }
        
        # 地面特征
        self.ground_color = {
            'hue_range': (20, 70),     # 地面颜色偏黄、棕色
            'sat_range': (30, 150),    # 中等饱和度
            'val_range': (120, 255),   # 较高明度
        }
        
        # 石头特征
        self.stone_color = {
            'hue_range': (0, 20),      # 灰色/棕色
            'sat_range': (0, 60),      # 低饱和度
            'val_range': (60, 180),    # 中等明度
        }
        
        # 学习参数
        self.min_samples_for_learning = 5
        self.learning_enabled = True
    
    def _extract_color_signature(self, roi: np.ndarray) -> Tuple[float, float, float]:
        """提取颜色特征（HSV均值）"""
        if roi.size == 0:
            return (0.0, 0.0, 0.0)
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        return (h_mean, s_mean, v_mean)
    
    def _extract_texture_signature(self, roi: np.ndarray) -> Tuple[float, float]:
        """提取纹理特征（方差和边缘密度）"""
        if roi.size == 0:
            return (0.0, 0.0)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 计算方差（纹理粗糙程度）
        variance = np.var(gray)
        
        # 计算边缘密度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
        
        return (variance, edge_density)
    
    def _matches_wall_color(self, roi: np.ndarray) -> bool:
        """检查是否匹配墙体颜色"""
        h, s, v = self._extract_color_signature(roi)
        return (self.wall_color['hue_range'][0] <= h <= self.wall_color['hue_range'][1] and
                self.wall_color['sat_range'][0] <= s <= self.wall_color['sat_range'][1] and
                self.wall_color['val_range'][0] <= v <= self.wall_color['val_range'][1])
    
    def _matches_ground_color(self, roi: np.ndarray) -> bool:
        """检查是否匹配地面颜色"""
        h, s, v = self._extract_color_signature(roi)
        return (self.ground_color['hue_range'][0] <= h <= self.ground_color['hue_range'][1] and
                self.ground_color['sat_range'][0] <= s <= self.ground_color['sat_range'][1] and
                self.ground_color['val_range'][0] <= v <= self.ground_color['val_range'][1])
    
    def _matches_stone_color(self, roi: np.ndarray) -> bool:
        """检查是否匹配石头颜色"""
        h, s, v = self._extract_color_signature(roi)
        return (self.stone_color['hue_range'][0] <= h <= self.stone_color['hue_range'][1] and
                self.stone_color['sat_range'][0] <= s <= self.stone_color['sat_range'][1] and
                self.stone_color['val_range'][0] <= v <= self.stone_color['val_range'][1])
    
    def _matches_enemy_color(self, roi: np.ndarray) -> bool:
        """检查是否匹配已知敌人颜色特征"""
        if not self.enemy_color_signatures:
            return True  # 如果没有学习到敌人特征，默认通过
        
        h, s, v = self._extract_color_signature(roi)
        threshold = 30  # 颜色匹配阈值
        
        for eh, es, ev in self.enemy_color_signatures:
            if (abs(h - eh) < threshold and
                abs(s - es) < 50 and
                abs(v - ev) < 50):
                return True
        
        return False
    
    def _is_reasonable_size(self, box: Tuple[int, int, int, int]) -> bool:
        """检查尺寸是否在合理范围内"""
        x, y, w, h = box
        size = max(w, h)
        return self.enemy_size_range[0] <= size <= self.enemy_size_range[1]
    
    def _is_reasonable_aspect_ratio(self, box: Tuple[int, int, int, int]) -> bool:
        """检查长宽比是否合理（敌人通常不会太窄或太宽）"""
        x, y, w, h = box
        if w == 0 or h == 0:
            return False
        
        ratio = w / h
        return 0.3 <= ratio <= 3.0  # 排除细长或扁宽的物体（通常是墙体）
    
    def learn_from_detected_enemy(self, roi: np.ndarray):
        """从检测到的敌人中学习特征"""
        if not self.learning_enabled:
            return
        
        color_sig = self._extract_color_signature(roi)
        texture_sig = self._extract_texture_signature(roi)
        
        # 避免重复特征
        if color_sig not in self.enemy_color_signatures:
            self.enemy_color_signatures.append(color_sig)
        
        if texture_sig not in self.enemy_texture_signatures:
            self.enemy_texture_signatures.append(texture_sig)
    
    def is_valid_enemy(self, box: Tuple[int, int, int, int], frame_bgr: np.ndarray) -> bool:
        """
        判断候选区域是否是有效的敌人
        
        Args:
            box: 候选框 (x, y, w, h)
            frame_bgr: 原始帧
            
        Returns:
            bool: 是否是有效敌人
        """
        x, y, w, h = box
        
        # 边界检查
        h_frame, w_frame = frame_bgr.shape[:2]
        if x < 0 or y < 0 or x + w > w_frame or y + h > h_frame:
            return False
        
        # 尺寸检查
        if not self._is_reasonable_size(box):
            return False
        
        # 长宽比检查
        if not self._is_reasonable_aspect_ratio(box):
            return False
        
        # 提取ROI
        roi = frame_bgr[y:y+h, x:x+w]
        
        # 颜色检查（排除墙体）
        if self._matches_wall_color(roi):
            return False
        
        # 颜色检查（排除地面）
        if self._matches_ground_color(roi):
            return False
        
        # 颜色检查（排除石头）
        if self._matches_stone_color(roi):
            return False
        
        # 颜色检查（匹配敌人特征）
        if self.enemy_color_signatures and not self._matches_enemy_color(roi):
            return False
        
        return True


class HurtDetector:
    """
    受伤检测器 - 三种检测方法 + 手动触发
    方法1：玩家消失-出现模式
    方法2：玩家位置突然跳跃（瞬移）
    方法3：玩家与敌人/子弹重叠
    方法4：手动触发（按H键）
    """
    
    def __init__(self):
        # 方法1：消失-出现模式参数（提高灵敏度）
        self.disappearance_count = 0
        self.is_disappearing = False
        self.last_appear_time = 0
        self.min_disappearances = 1  # 降低为1次消失-出现即可检测
        self.max_reset_time = 2.0  # 延长重置时间
        self.min_disappear_duration = 0.02  # 降低最小消失持续时间
        
        # 方法2：位置跳跃检测参数（提高灵敏度）
        self.position_history = deque(maxlen=15)  # 增加历史长度
        self.jump_threshold = 30  # 降低跳跃阈值（30像素）
        
        # 方法3：重叠检测参数（提高灵敏度）
        self.contact_margin = 10  # 增大接触容差
        self.last_overlap_time = 0
        self.overlap_cooldown = 0.3  # 降低冷却时间
        
        # 通用状态
        self.hurt_count = 0
        self.last_hurt_time = -1000.0  # 初始化为负数，表示从未受伤
        self.cooldown_after_hurt = 1.0  # 降低受伤后冷却时间
        self.player_history = deque(maxlen=60)  # 增加历史长度
        
        # 手动触发相关
        self.manual_hurt_requested = False
        self.last_manual_hurt_time = 0
    
    def _detect_disappearance_pattern(self, player_detected: bool, timestamp: float) -> Tuple[bool, str]:
        """
        方法1：检测玩家消失-出现模式
        
        Returns:
            Tuple[bool, str]: (是否受伤, 来源描述)
        """
        if timestamp - self.last_appear_time > self.max_reset_time:
            self.disappearance_count = 0
            self.is_disappearing = False
        
        if not player_detected and not self.is_disappearing:
            self.is_disappearing = True
            self.disappearance_count += 1
            self._disappear_start_time = timestamp
            
        elif player_detected and self.is_disappearing:
            self.is_disappearing = False
            self.last_appear_time = timestamp
            
            # 检查消失持续时间
            disappear_duration = timestamp - getattr(self, '_disappear_start_time', timestamp)
            if disappear_duration < self.min_disappear_duration:
                self.disappearance_count -= 1
                return False, ''
            
            if self.disappearance_count >= self.min_disappearances:
                if self.last_hurt_time == 0 or timestamp - self.last_hurt_time > self.cooldown_after_hurt:
                    return True, 'disappearance_pattern'
        
        return False, ''
    
    def _detect_position_jump(self, player_box: Optional[Tuple[int, int, int, int]], timestamp: float) -> Tuple[bool, str]:
        """
        方法2：检测玩家位置突然跳跃（瞬移）
        
        Args:
            player_box: 玩家检测框 (x, y, w, h)
            
        Returns:
            Tuple[bool, str]: (是否受伤, 来源描述)
        """
        if player_box is None:
            return False, ''
        
        x, y, w, h = player_box
        current_center = (x + w // 2, y + h // 2)
        
        if len(self.position_history) > 0:
            # 计算与最近历史位置的距离
            recent_positions = list(self.position_history)[-3:] if len(self.position_history) >= 3 else list(self.position_history)
            
            for prev_center in recent_positions:
                distance = np.sqrt((current_center[0] - prev_center[0])**2 + 
                                   (current_center[1] - prev_center[1])**2)
                
                if distance > self.jump_threshold:
                    if self.last_hurt_time == 0 or timestamp - self.last_hurt_time > self.cooldown_after_hurt:
                        return True, 'position_jump'
        
        self.position_history.append(current_center)
        return False, ''
    
    def _detect_overlap(self, player_box: Optional[Tuple[int, int, int, int]], 
                        enemies: List[Dict], enemy_bullets: List[Dict], 
                        timestamp: float) -> Tuple[bool, str]:
        """
        方法3：检测玩家与敌人/子弹重叠
        
        Args:
            player_box: 玩家检测框
            enemies: 敌人列表
            enemy_bullets: 敌人子弹列表
            
        Returns:
            Tuple[bool, str]: (是否受伤, 来源描述)
        """
        if player_box is None:
            return False, ''
        
        # 检查冷却时间
        if self.last_overlap_time > 0 and timestamp - self.last_overlap_time < self.overlap_cooldown:
            return False, ''
        
        # 检测与敌人重叠
        for enemy in enemies:
            enemy_box = enemy.get('box')
            if enemy_box and self._touch_or_overlap(player_box, enemy_box, self.contact_margin):
                self.last_overlap_time = timestamp
                if self.last_hurt_time == 0 or timestamp - self.last_hurt_time > self.cooldown_after_hurt:
                    return True, 'enemy_overlap'
        
        # 检测与敌人子弹重叠
        for bullet in enemy_bullets:
            bullet_box = bullet.get('box')
            if bullet_box and self._touch_or_overlap(player_box, bullet_box, self.contact_margin):
                self.last_overlap_time = timestamp
                if self.last_hurt_time == 0 or timestamp - self.last_hurt_time > self.cooldown_after_hurt:
                    return True, 'bullet_overlap'
        
        return False, ''
    
    def _touch_or_overlap(self, box1: Tuple[int, int, int, int], 
                          box2: Tuple[int, int, int, int], 
                          margin: int = 0) -> bool:
        """检查两个框是否重叠或接触"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        left1 = x1 - margin
        top1 = y1 - margin
        right1 = x1 + w1 + margin
        bottom1 = y1 + h1 + margin
        
        left2 = x2 - margin
        top2 = y2 - margin
        right2 = x2 + w2 + margin
        bottom2 = y2 + h2 + margin
        
        return not (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1)
    
    def detect_hurt(self, player_detected: bool, player_box: Optional[Tuple[int, int, int, int]],
                    enemies: List[Dict], enemy_bullets: List[Dict], 
                    timestamp: Optional[float] = None) -> Tuple[bool, str]:
        """
        综合四种方法检测受伤
        
        Args:
            player_detected: 是否检测到玩家
            player_box: 玩家检测框
            enemies: 敌人列表
            enemy_bullets: 敌人子弹列表
            timestamp: 当前时间戳
            
        Returns:
            Tuple[bool, str]: (是否受伤, 受伤来源)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 添加到历史
        self.player_history.append((player_detected, timestamp))
        
        # 检查受伤冷却期
        if self.last_hurt_time >= 0 and timestamp - self.last_hurt_time < self.cooldown_after_hurt:
            return False, ''
        
        # 方法4：手动触发（优先级最高）
        if self.manual_hurt_requested:
            if self.last_manual_hurt_time == 0 or timestamp - self.last_manual_hurt_time > self.cooldown_after_hurt:
                self.manual_hurt_requested = False
                self.last_manual_hurt_time = timestamp
                self.hurt_count += 1
                self.last_hurt_time = timestamp
                return True, 'manual_trigger'
        
        # 方法1：消失-出现模式
        hurt, source = self._detect_disappearance_pattern(player_detected, timestamp)
        if hurt:
            self.hurt_count += 1
            self.last_hurt_time = timestamp
            return True, source
        
        # 方法2：位置跳跃检测
        hurt, source = self._detect_position_jump(player_box, timestamp)
        if hurt:
            self.hurt_count += 1
            self.last_hurt_time = timestamp
            return True, source
        
        # 方法3：重叠检测
        hurt, source = self._detect_overlap(player_box, enemies, enemy_bullets, timestamp)
        if hurt:
            self.hurt_count += 1
            self.last_hurt_time = timestamp
            return True, source
        
        return False, ''
    
    def request_manual_hurt(self):
        """手动触发受伤（由外部调用，如按H键）"""
        self.manual_hurt_requested = True
    
    def get_hurt_count(self) -> int:
        """获取累计受伤次数"""
        return self.hurt_count
    
    def reset(self):
        """重置检测器"""
        self.disappearance_count = 0
        self.is_disappearing = False
        self.last_appear_time = 0
        self.position_history.clear()
        self.last_overlap_time = 0
        self.hurt_count = 0
        self.last_hurt_time = 0
        self.player_history.clear()
        self.manual_hurt_requested = False
        self.last_manual_hurt_time = 0
    
    def get_status(self) -> Dict[str, object]:
        """获取检测器状态"""
        return {
            'hurt_count': self.hurt_count,
            'disappearance_count': self.disappearance_count,
            'is_disappearing': self.is_disappearing,
            'position_history_length': len(self.position_history),
            'last_overlap_time': self.last_overlap_time,
            'last_hurt_time': self.last_hurt_time,
            'history_length': len(self.player_history),
            'manual_hurt_requested': self.manual_hurt_requested,
        }


class EnhancedVisionPostProcessor:
    """
    增强型视觉后处理器
    整合敌人过滤和受伤检测功能
    """
    
    def __init__(self):
        self.enemy_filter = SmartEnemyFilter()
        self.hurt_detector = HurtDetector()
        self.frame_count = 0
        self.last_player_detected = True
    
    def process_detections(self, frame_bgr: np.ndarray, detections: Dict[str, List[Dict]]) -> Dict[str, object]:
        """
        处理检测结果
        
        Args:
            frame_bgr: 原始帧
            detections: 检测结果字典
            
        Returns:
            Dict: 处理后的检测结果
        """
        self.frame_count += 1
        result = {}
        
        # 处理敌人检测（过滤墙体/地面/石头）
        enemies = detections.get('enemy', [])
        filtered_enemies = []
        
        for enemy in enemies:
            box = enemy['box']
            if self.enemy_filter.is_valid_enemy(box, frame_bgr):
                filtered_enemies.append(enemy)
                # 学习敌人特征
                x, y, w, h = box
                roi = frame_bgr[y:y+h, x:x+w]
                self.enemy_filter.learn_from_detected_enemy(roi)
        
        result['enemy'] = filtered_enemies
        
        # 处理玩家检测（用于受伤检测）
        players = detections.get('player', [])
        player_detected = len(players) > 0
        
        # 获取玩家检测框
        player_box = None
        if players:
            best_player = max(players, key=lambda p: p.get('score', 0.0))
            player_box = best_player.get('box')
        
        # 获取敌人子弹列表
        enemy_bullets = detections.get('bullets_enemy', [])
        
        # 检测受伤（三种方法）
        timestamp = time.time()
        hurt_detected, hurt_source = self.hurt_detector.detect_hurt(
            player_detected, player_box, filtered_enemies, enemy_bullets, timestamp
        )
        
        result['player'] = players
        result['hurt'] = hurt_detected
        result['hurt_source'] = hurt_source
        result['total_hurt_count'] = self.hurt_detector.get_hurt_count()
        
        # 保留其他检测结果
        for key in ['bullets_player', 'bullets_enemy', 'ground', 'walls_room']:
            if key in detections:
                result[key] = detections[key]
        
        return result
    
    def get_debug_info(self) -> Dict[str, object]:
        """获取调试信息"""
        return {
            'frame_count': self.frame_count,
            'enemy_filter': {
                'learned_color_signatures': len(self.enemy_filter.enemy_color_signatures),
                'learned_texture_signatures': len(self.enemy_filter.enemy_texture_signatures),
            },
            'hurt_detector': self.hurt_detector.get_status(),
        }
    
    def reset(self):
        """重置处理器"""
        self.frame_count = 0
        self.enemy_filter = SmartEnemyFilter()
        self.hurt_detector.reset()


# ===== 测试代码 =====
if __name__ == '__main__':
    # 测试方法1：消失-出现模式
    hurt_detector = HurtDetector()
    
    simulation = [
        (True, (100, 200, 50, 80), [], [], 0.0), 
        (True, (100, 200, 50, 80), [], [], 0.05),
        (True, (100, 200, 50, 80), [], [], 0.1),
        (False, None, [], [], 0.15),
        (True, (100, 200, 50, 80), [], [], 0.2),
        (False, None, [], [], 0.25),
        (True, (100, 200, 50, 80), [], [], 0.3),
        (True, (100, 200, 50, 80), [], [], 0.35),
    ]
    
    print("=== 测试方法1：消失-出现模式 ===")
    for i, (detected, box, enemies, bullets, ts) in enumerate(simulation):
        is_hurt, source = hurt_detector.detect_hurt(detected, box, enemies, bullets, ts)
        print(f"Frame {i}: detected={detected}, hurt={is_hurt}, source={source}")
        if is_hurt:
            print(f"  >>> HURT DETECTED! <<<")
    
    # 使用新实例测试位置跳跃（避免冷却期影响）
    hurt_detector2 = HurtDetector()
    print("\n=== 测试方法2：位置跳跃 ===")
    jump_simulation = [
        (True, (100, 200, 50, 80), [], [], 0.0),
        (True, (100, 200, 50, 80), [], [], 0.1),
        (True, (100, 200, 50, 80), [], [], 0.2),
        (True, (200, 300, 50, 80), [], [], 0.3),
    ]
    
    for i, (detected, box, enemies, bullets, ts) in enumerate(jump_simulation):
        is_hurt, source = hurt_detector2.detect_hurt(detected, box, enemies, bullets, ts)
        print(f"Frame {i}: position=({box[0]},{box[1]}), hurt={is_hurt}, source={source}")
        if is_hurt:
            print(f"  >>> HURT DETECTED! <<<")
    
    # 使用新实例测试重叠检测
    hurt_detector3 = HurtDetector()
    print("\n=== 测试方法3：重叠检测 ===")
    overlap_simulation = [
        (True, (100, 200, 50, 80), [], [], 0.0),
        (True, (100, 200, 50, 80), [{'box': (140, 230, 40, 60)}], [], 0.1),
        (True, (100, 200, 50, 80), [], [{'box': (120, 240, 10, 10)}], 0.2),
    ]
    
    for i, (detected, box, enemies, bullets, ts) in enumerate(overlap_simulation):
        is_hurt, source = hurt_detector3.detect_hurt(detected, box, enemies, bullets, ts)
        print(f"Frame {i}: enemies={len(enemies)}, bullets={len(bullets)}, hurt={is_hurt}, source={source}")
        if is_hurt:
            print(f"  >>> HURT DETECTED! <<<")
    
    # 测试方法4：手动触发
    hurt_detector4 = HurtDetector()
    print("\n=== 测试方法4：手动触发 ===")
    hurt_detector4.request_manual_hurt()
    is_hurt, source = hurt_detector4.detect_hurt(True, (100, 200, 50, 80), [], [], 0.0)
    print(f"Manual hurt: hurt={is_hurt}, source={source}, count={hurt_detector4.get_hurt_count()}")
    
    print(f"\nTotal hurt count across all tests: {hurt_detector.get_hurt_count() + hurt_detector2.get_hurt_count() + hurt_detector3.get_hurt_count() + hurt_detector4.get_hurt_count()}")
