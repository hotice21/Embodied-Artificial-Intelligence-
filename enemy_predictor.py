"""
敌人路线预测和受伤检测模块
功能：
1. 预测敌人未来位置，提前躲避
2. 多层受伤检测机制
"""

import numpy as np
import cv2
from collections import deque
import time


class EnemyTrajectoryPredictor:
    """
    敌人轨迹预测器
    功能：根据历史轨迹预测敌人未来位置
    """
    
    def __init__(self, history_length=15, prediction_horizon=15):
        """
        初始化轨迹预测器
        
        Args:
            history_length: 保存历史位置的数量
            prediction_horizon: 预测未来多少步
        """
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.enemy_trajectories = {}  # enemy_id -> deque of (x, y, timestamp)
        self.enemy_last_seen = {}  # enemy_id -> last_seen_time
        self.enemy_last_positions = {}  # enemy_id -> (x, y) 最后已知位置
        self.frame_count = 0
        self._next_enemy_id = 1  # 唯一ID计数器
        self._match_threshold = 80  # 最近邻匹配阈值（像素）
    
    def update(self, enemies, timestamp=None):
        """
        更新敌人轨迹
        
        Args:
            enemies: 检测到的敌人列表
            timestamp: 当前时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.frame_count += 1
        
        # 获取当前所有检测到的敌人中心位置
        current_positions = []
        for enemy in enemies:
            bx, by, bw, bh = enemy['box']
            cx, cy = bx + bw // 2, by + bh // 2
            current_positions.append((cx, cy, enemy))
        
        # 获取历史轨迹的最后位置
        available_ids = []
        for enemy_id in self.enemy_trajectories.keys():
            if timestamp - self.enemy_last_seen.get(enemy_id, 0) < 2.0:  # 2秒内见过的才参与匹配
                last_pos = self.enemy_last_positions.get(enemy_id)
                if last_pos is not None:
                    available_ids.append((enemy_id, last_pos[0], last_pos[1]))
        
        # 使用最近邻匹配将当前敌人与历史ID关联
        used_ids = set()
        matched_enemies = []
        
        for cx, cy, enemy in current_positions:
            best_id = None
            best_distance = float('inf')
            
            # 找到最近的历史敌人
            for enemy_id, lx, ly in available_ids:
                if enemy_id in used_ids:
                    continue
                distance = np.sqrt((cx - lx)**2 + (cy - ly)**2)
                if distance < best_distance and distance < self._match_threshold:
                    best_distance = distance
                    best_id = enemy_id
            
            if best_id is None:
                # 没有找到匹配的历史ID，分配新ID
                best_id = f"enemy_{self._next_enemy_id}"
                self._next_enemy_id += 1
            
            used_ids.add(best_id)
            matched_enemies.append((best_id, cx, cy, enemy))
        
        # 更新轨迹和最后位置
        for enemy_id, cx, cy, enemy in matched_enemies:
            if enemy_id not in self.enemy_trajectories:
                self.enemy_trajectories[enemy_id] = deque(maxlen=self.history_length)
            
            self.enemy_trajectories[enemy_id].append((cx, cy, timestamp))
            self.enemy_last_seen[enemy_id] = timestamp
            self.enemy_last_positions[enemy_id] = (cx, cy)
        
        # 清理长时间未出现的敌人轨迹（超过2秒）
        stale_ids = []
        for enemy_id in self.enemy_trajectories.keys():
            if timestamp - self.enemy_last_seen.get(enemy_id, 0) >= 2.0:
                stale_ids.append(enemy_id)
        
        for enemy_id in stale_ids:
            del self.enemy_trajectories[enemy_id]
            del self.enemy_last_seen[enemy_id]
            del self.enemy_last_positions[enemy_id]
    
    def predict_future_position(self, enemy_id, steps=None):
        """
        预测敌人未来位置
        
        Args:
            enemy_id: 敌人ID
            steps: 预测步数，默认使用 self.prediction_horizon
            
        Returns:
            list of (x, y) 预测位置列表
        """
        if steps is None:
            steps = self.prediction_horizon
        
        if enemy_id not in self.enemy_trajectories:
            return None
        
        trajectory = list(self.enemy_trajectories[enemy_id])
        if len(trajectory) < 3:
            return None
        
        # 计算速度和加速度
        velocities = []
        accelerations = []
        
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            dt = max(trajectory[i][2] - trajectory[i-1][2], 0.001)
            
            vx = dx / dt
            vy = dy / dt
            velocities.append((vx, vy))
            
            if i > 1:
                dvx = vx - velocities[i-2][0]
                dvy = vy - velocities[i-2][1]
                accelerations.append((dvx / dt, dvy / dt))
        
        # 计算平均速度（使用加权平均，最近的速度权重更大）
        weights = np.exp(np.linspace(0, 1, len(velocities)))
        weights /= weights.sum()
        
        avg_vx = sum(v[0] * w for v, w in zip(velocities, weights))
        avg_vy = sum(v[1] * w for v, w in zip(velocities, weights))
        
        # 如果有加速度信息，使用更准确的预测
        if accelerations:
            weights_acc = np.exp(np.linspace(0, 1, len(accelerations)))
            weights_acc /= weights_acc.sum()
            
            avg_ax = sum(a[0] * w for a, w in zip(accelerations, weights_acc))
            avg_ay = sum(a[1] * w for a, w in zip(accelerations, weights_acc))
        else:
            avg_ax = 0
            avg_ay = 0
        
        # 预测未来位置（使用匀加速运动模型）
        last_x, last_y, last_time = trajectory[-1]
        predictions = []
        
        for step in range(1, steps + 1):
            dt = step * 0.05  # 假设每步0.05秒（20fps）
            
            # 匀加速运动公式：x = x0 + v0*t + 0.5*a*t^2
            pred_x = last_x + avg_vx * dt + 0.5 * avg_ax * dt * dt
            pred_y = last_y + avg_vy * dt + 0.5 * avg_ay * dt * dt
            
            # 确保预测位置在合理范围内
            pred_x = max(0, min(1000, pred_x))
            pred_y = max(0, min(700, pred_y))
            
            predictions.append((pred_x, pred_y))
        
        return predictions
    
    def get_predicted_threat_zones(self, player_pos, danger_threshold=150):
        """
        获取预测的危险区域
        
        Args:
            player_pos: 玩家位置 (x, y)
            danger_threshold: 危险距离阈值
            
        Returns:
            list of threat zone dicts
        """
        threat_zones = []
        
        for enemy_id, trajectory in self.enemy_trajectories.items():
            predictions = self.predict_future_position(enemy_id)
            
            if predictions:
                for step, (pred_x, pred_y) in enumerate(predictions):
                    # 计算到玩家的距离
                    dist = np.sqrt((pred_x - player_pos[0])**2 + 
                                   (pred_y - player_pos[1])**2)
                    
                    # 如果预测位置接近玩家，标记为危险
                    if dist < danger_threshold:
                        # 危险程度：距离越近、时间越短，危险越高
                        danger_level = (1.0 - dist / danger_threshold) * (1.0 - step / self.prediction_horizon)
                        
                        threat_zones.append({
                            'enemy_id': enemy_id,
                            'position': (pred_x, pred_y),
                            'time': step * 0.05,  # 多少秒后会到达
                            'steps': step,
                            'distance': dist,
                            'danger': danger_level
                        })
        
        # 按危险程度排序
        threat_zones.sort(key=lambda x: x['danger'], reverse=True)
        
        return threat_zones
    
    def get_danger_directions(self, player_pos, num_directions=8):
        """
        获取8个方向的预测危险度
        
        Args:
            player_pos: 玩家位置 (x, y)
            num_directions: 方向数量
            
        Returns:
            list of danger levels for each direction
        """
        danger_levels = [0.0] * num_directions
        
        threat_zones = self.get_predicted_threat_zones(player_pos)
        
        for zone in threat_zones:
            dx = zone['position'][0] - player_pos[0]
            dy = zone['position'][1] - player_pos[1]
            
            # 计算方向索引
            angle = np.arctan2(dy, dx)
            direction_idx = int(((angle + np.pi) / (2 * np.pi)) * num_directions) % num_directions
            
            # 累加危险度
            danger_levels[direction_idx] += zone['danger']
        
        # 归一化
        max_danger = max(danger_levels) if danger_levels else 1.0
        if max_danger > 0:
            danger_levels = [d / max_danger for d in danger_levels]
        
        return danger_levels


class MultiLayerHurtDetector:
    """
    多层受伤检测器
    结合多种方法检测玩家受伤：
    1. 血量变化检测
    2. 屏幕特效检测
    3. 重叠检测
    4. 闪烁检测
    5. 动画检测
    """
    
    def __init__(self):
        self.health_history = deque(maxlen=30)  # 血量历史
        self.last_health = 6  # 假设最大血量为6
        self.current_health = 6
        self.screen_history = deque(maxlen=10)  # 屏幕历史
        self.last_frame = None  # 上一帧
        self.hurt_cooldown = 0  # 受伤冷却时间
        self.hurt_history = deque(maxlen=100)  # 受伤历史
        
        # 受伤检测参数
        self.red_threshold = 0.15  # 红色像素占比阈值
        self.blink_threshold = 0.3  # 闪烁检测阈值
        self.overlap_margin = 5  # 重叠检测容差
        self.cooldown_frames = 10  # 受伤冷却帧数
        
        # 心心检测区域（相对于游戏窗口）
        self.hearts_roi = None  # 会在第一次检测时设置
    
    def detect_hurt(self, frame, player_box, enemies, enemy_bullets, prev_frame=None):
        """
        多层受伤检测
        
        Args:
            frame: 当前帧
            player_box: 玩家位置 (x, y, w, h)
            enemies: 敌人列表
            enemy_bullets: 敌人子弹列表
            prev_frame: 上一帧
            
        Returns:
            tuple: (is_hurt, hurt_penalty, hurt_source)
        """
        hurt_source = None
        hurt_penalty = 0
        
        # 更新冷却时间
        if self.hurt_cooldown > 0:
            self.hurt_cooldown -= 1
            return False, 0, None
        
        # ===== 层1：血量变化检测 =====
        health_hurt, health_penalty = self._detect_by_health(frame)
        if health_hurt:
            return True, health_penalty, 'health_change'
        
        # ===== 层2：屏幕特效检测 =====
        screen_hurt, screen_penalty = self._detect_by_screen_effect(frame, prev_frame)
        if screen_hurt:
            return True, screen_penalty, 'screen_effect'
        
        # ===== 层3：重叠检测 =====
        if player_box:
            # 敌人与玩家重叠
            for enemy in enemies:
                if self._boxes_overlap(player_box, enemy['box']):
                    return True, 80, 'enemy_contact'
            
            # 敌人子弹与玩家重叠
            for bullet in enemy_bullets:
                if self._boxes_overlap(player_box, bullet['box']):
                    return True, 80, 'bullet_contact'
        
        # ===== 层4：闪烁检测 =====
        blink_hurt, blink_penalty = self._detect_by_blink(frame, prev_frame)
        if blink_hurt:
            return True, blink_penalty, 'blink_animation'
        
        # ===== 层5：运动异常检测 =====
        motion_hurt, motion_penalty = self._detect_by_motion_anomaly(frame, prev_frame)
        if motion_hurt:
            return True, motion_penalty, 'motion_anomaly'
        
        return False, 0, None
    
    def _detect_by_health(self, frame):
        """
        基于血量变化的受伤检测
        通过检测心心数量来判断是否受伤
        """
        if self.hearts_roi is None:
            # 第一次检测，设置心心区域
            h, w = frame.shape[:2]
            # 心心通常在屏幕左上角
            self.hearts_roi = (10, 10, 200, 50)
        
        x, y, w, h = self.hearts_roi
        hearts_region = frame[y:y+h, x:x+w]
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(hearts_region, cv2.COLOR_BGR2HSV)
        
        # 红色心心范围
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        # 检测红色像素
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # 计算红色像素占比
        red_ratio = np.sum(red_mask > 0) / (w * h)
        
        # 根据红色占比估算血量
        estimated_health = int(red_ratio * 12)  # 假设最多6颗满心=12个半心
        
        # 更新血量历史
        self.health_history.append(estimated_health)
        self.current_health = estimated_health
        
        # 检测血量下降
        if len(self.health_history) >= 2:
            prev_health = self.health_history[-2]
            if estimated_health < prev_health:
                health_lost = prev_health - estimated_health
                return True, health_lost * 20  # 每失去一个半心惩罚20
        
        return False, 0
    
    def _detect_by_screen_effect(self, frame, prev_frame):
        """
        基于屏幕特效的受伤检测
        受伤时屏幕会出现红色闪烁
        """
        if prev_frame is None:
            self.last_frame = frame.copy()
            return False, 0
        
        # 计算帧差异
        diff = cv2.absdiff(frame, prev_frame)
        
        # 转换到HSV颜色空间
        hsv_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
        
        # 红色范围
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([20, 255, 255])
        
        # 检测红色增加
        red_curr = cv2.inRange(hsv_curr, red_lower, red_upper)
        red_prev = cv2.inRange(hsv_prev, red_lower, red_upper)
        
        # 计算红色变化
        red_change = np.sum(red_curr > 0) - np.sum(red_prev > 0)
        total_pixels = frame.shape[0] * frame.shape[1]
        red_change_ratio = red_change / total_pixels
        
        # 如果红色像素显著增加，认为是受伤
        if red_change_ratio > self.red_threshold:
            return True, 80
        
        # 更新上一帧
        self.last_frame = frame.copy()
        
        return False, 0
    
    def _detect_by_blink(self, frame, prev_frame):
        """
        基于闪烁的受伤检测
        受伤时角色会闪烁（短暂消失）
        """
        if prev_frame is None:
            return False, 0
        
        # 计算帧相似度
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # 计算差异
        diff = cv2.absdiff(gray_curr, gray_prev)
        diff_ratio = np.sum(diff > 50) / (diff.shape[0] * diff.shape[1])
        
        # 如果差异过大（闪烁），认为是受伤
        if diff_ratio > self.blink_threshold:
            return True, 80
        
        return False, 0
    
    def _detect_by_motion_anomaly(self, frame, prev_frame):
        """
        基于运动异常的受伤检测
        受伤时会有短暂的运动停顿或异常
        """
        if prev_frame is None or self.screen_history:
            self.screen_history.append(frame.copy())
            return False, 0
        
        # 添加到历史
        self.screen_history.append(frame.copy())
        
        # 如果历史足够长，检测运动异常
        if len(self.screen_history) >= 5:
            # 计算连续帧的运动
            motions = []
            for i in range(1, len(self.screen_history)):
                gray1 = cv2.cvtColor(self.screen_history[i-1], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(self.screen_history[i], cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray1, gray2)
                motion = np.sum(diff > 30) / (diff.shape[0] * diff.shape[1])
                motions.append(motion)
            
            # 检测是否有运动突然停止（可能表示受伤）
            if len(motions) >= 3:
                avg_motion = np.mean(motions[:-1])
                current_motion = motions[-1]
                
                # 如果运动突然大幅下降
                if current_motion < avg_motion * 0.3 and avg_motion > 0.01:
                    return True, 60
        
        return False, 0
    
    def _boxes_overlap(self, box1, box2, margin=0):
        """
        检测两个矩形是否重叠
        
        Args:
            box1: 第一个矩形 (x, y, w, h)
            box2: 第二个矩形 (x, y, w, h)
            margin: 扩展容差
            
        Returns:
            bool: 是否重叠
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 考虑容差
        x1 -= margin
        y1 -= margin
        w1 += 2 * margin
        h1 += 2 * margin
        
        # 检测重叠
        if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
            return True
        
        return False
    
    def get_hurt_statistics(self):
        """获取受伤统计信息"""
        if not self.hurt_history:
            return {
                'total_hurts': 0,
                'by_source': {},
                'recent_hurts': 0
            }
        
        stats = {
            'total_hurts': len(self.hurt_history),
            'by_source': {},
            'recent_hurts': sum(1 for h in list(self.hurt_history)[-50:])
        }
        
        # 按来源统计
        for hurt_info in self.hurt_history:
            source = hurt_info.get('source', 'unknown')
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
        
        return stats
    
    def reset(self):
        """重置检测器"""
        self.health_history.clear()
        self.screen_history.clear()
        self.hurt_history.clear()
        self.hurt_cooldown = 0
        self.last_health = 6
        self.current_health = 6


# ===== 测试代码 =====
if __name__ == '__main__':
    # 测试轨迹预测器
    predictor = EnemyTrajectoryPredictor()
    
    # 模拟敌人移动
    for t in range(50):
        enemies = [{
            'box': (200 + t * 2, 300 + t, 50, 50)  # 斜向移动
        }]
        predictor.update(enemies, timestamp=t * 0.05)
    
    # 预测未来位置
    predictions = predictor.predict_future_position('4_6')
    print(f"Predicted future positions: {predictions}")
    
    # 测试危险区域
    threat_zones = predictor.get_predicted_threat_zones((500, 350))
    print(f"Threat zones: {threat_zones}")
    
    # 测试受伤检测器
    hurt_detector = MultiLayerHurtDetector()
    print("Hurt detector initialized")
