"""改进版强化学习训练器 - 修复躲避和攻击学习问题"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os

try:
    from training_visualizer import TrainingLogger, TrainingVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False
    print("Warning: training_visualizer.py not found, visualization disabled")

# 导入敌人预测和受伤检测模块
try:
    from enemy_predictor import EnemyTrajectoryPredictor, MultiLayerHurtDetector
    HAS_ENEMY_PREDICTOR = True
except ImportError:
    HAS_ENEMY_PREDICTOR = False
    print("Warning: enemy_predictor.py not found, using basic prediction")

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_AVAILABLE = True


class ActorCriticNetwork(nn.Module):
    """改进版Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 特征提取层（更深的网络）
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Actor - 策略网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic - 价值网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Actor使用较小的初始化
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回动作概率和价值"""
        features = self.feature_net(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
        """采样动作，返回动作、对数概率和价值"""
        with torch.no_grad():
            action_logits, value = self.forward(state)
            probs = F.softmax(action_logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
                log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1))).squeeze(-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作，返回对数概率、价值和熵"""
        action_logits, values = self.forward(states)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """改进版经验缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        """添加经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class ImprovedPPOTrainer:
    """改进版PPO训练器 - 专注于学习躲避和攻击"""
    
    def __init__(self, config: Dict):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU training")
        
        self.config = config
        self.device = DEVICE
        
        # 网络参数
        self.state_dim = config.get('state_dim', 61)  # 更新状态维度（添加敌人预测特征）
        self.action_dim = config.get('action_dim', 9)
        self.hidden_dim = config.get('hidden_dim', 512)  # 增大隐藏层
        
        # PPO参数（优化后）
        self.lr = config.get('lr', 5e-4)  # 提高学习率
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.05)  # 提高探索
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.update_epochs = config.get('update_epochs', 15)  # 增加更新轮数
        self.mini_batch_size = config.get('mini_batch_size', 64)
        
        # 训练参数
        self.episodes = config.get('episodes', 500)
        self.episode_length = config.get('episode_length', 500)
        self.batch_size = config.get('batch_size', 256)  # 增大批次
        
        # 奖励配置（重新设计）
        self.time_penalty = config.get('time_penalty', -0.1)  # 减小时间惩罚
        self.hit_reward = config.get('hit_reward', 30)  # 提高击中奖励
        self.hurt_penalty = config.get('hurt_penalty', -80)  # 大幅提高受伤惩罚
        self.kill_reward = config.get('kill_reward', 100)
        self.survival_bonus = config.get('survival_bonus', 0.5)
        
        # 距离奖励配置（核心改进）
        self.distance_reward_factor = config.get('distance_reward_factor', 0.5)
        self.stay_away_reward = config.get('stay_away_reward', 5.0)
        self.enemy_proximity_penalty = config.get('enemy_proximity_penalty', -10)
        self.safe_distance = config.get('safe_distance', 120)  # 安全距离
        self.danger_distance = config.get('danger_distance', 60)  # 危险距离
        
        # 创建网络
        self.network = ActorCriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-5)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.8  # 更慢衰减
        )
        
        # 经验缓冲区
        self.buffer = RolloutBuffer()
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'entropy': [],
            'kl_divergence': [],
        }
        self.best_score = float('-inf')
        self.total_steps = 0
        
        # 状态历史（用于计算运动方向）
        self.state_history = deque(maxlen=5)
        
        # 训练日志记录器（新增）
        self.training_logger = TrainingLogger() if HAS_VISUALIZER else None
        
        # 敌人预测器和受伤检测器（新增）
        if HAS_ENEMY_PREDICTOR:
            self.enemy_predictor = EnemyTrajectoryPredictor(
                history_length=15,
                prediction_horizon=15
            )
            self.hurt_detector = MultiLayerHurtDetector()
            print("Enemy predictor and hurt detector initialized")
        else:
            self.enemy_predictor = None
            self.hurt_detector = None
        
        # 检查是否需要继续训练
        if config.get('continue_training', False) and config.get('load_path'):
            load_path = config['load_path']
            if os.path.exists(load_path):
                self.load_model(load_path)
                print(f"Successfully loaded model from: {load_path}")
            else:
                print(f"Warning: Model file not found at {load_path}, starting fresh training")
    
    def extract_state(self, vision_result: Dict) -> np.ndarray:
        """改进版状态提取 - 包含躲避关键信息"""
        features = []
        frame_width, frame_height = 1000, 700
        
        # ============ 玩家特征 ============
        player = vision_result.get('player', [])
        if player:
            px, py, pw, ph = player[0]['box']
            pcx, pcy = px + pw // 2, py + ph // 2
            # 玩家位置（归一化）
            features.extend([
                pcx / frame_width,      # 玩家中心X
                pcy / frame_height,     # 玩家中心Y
                pw / frame_width,       # 玩家宽度
                ph / frame_height,      # 玩家高度
            ])
        else:
            features.extend([0.5, 0.5, 0.1, 0.1])  # 默认值
            pcx, pcy = frame_width / 2, frame_height / 2
        
        # ============ 敌人特征（最多3个最近敌人） ============
        enemies = vision_result.get('enemy', [])
        max_enemies = 3
        
        for i in range(max_enemies):
            if i < len(enemies):
                ex, ey, ew, eh = enemies[i]['box']
                ecx, ecy = ex + ew // 2, ey + eh // 2
                
                # 计算相对位置和距离
                dx = ecx - pcx
                dy = ecy - pcy
                distance = np.sqrt(dx**2 + dy**2)
                # 归一化到 [-1, 1]
                features.extend([
                    dx / (frame_width / 2),    # 相对X
                    dy / (frame_height / 2),   # 相对Y
                    min(distance / 500.0, 1.0), # 距离（归一化）
                    ew / frame_width,          # 敌人宽度
                    eh / frame_height,         # 敌人高度
                ])
            else:
                # 没有敌人时填充0
                features.extend([0.0, 0.0, 1.0, 0.0, 0.0])
        
        # 敌人数量
        features.append(min(len(enemies) / 5.0, 1.0))
        
        # ============ 敌人距离特征（核心改进） ============
        # 计算到最近敌人的距离
        if enemies:
            min_enemy_dist = min([
                np.sqrt((e['box'][0] + e['box'][2]//2 - pcx)**2 + 
                        (e['box'][1] + e['box'][3]//2 - pcy)**2)
                for e in enemies
            ])
            # 归一化距离（0-1，值越大表示越远）
            features.append(min(min_enemy_dist / 300.0, 1.0))
            
            # 是否有敌人太近（危险信号）
            has_close_enemy = 1.0 if min_enemy_dist < 60 else 0.0
            features.append(has_close_enemy)
            
            # 安全距离比例（值越大表示越安全）
            safe_ratio = min(min_enemy_dist / 120.0, 1.0)
            features.append(safe_ratio)
        else:
            features.extend([1.0, 0.0, 1.0])  # 没有敌人，最安全
        
        # ============ 敌人路线预测特征（新增） ============
        if self.enemy_predictor and HAS_ENEMY_PREDICTOR:
            # 更新敌人轨迹
            self.enemy_predictor.update(enemies)
            
            # 预测危险区域
            threat_zones = self.enemy_predictor.get_predicted_threat_zones((pcx, pcy))
            
            # 添加预测危险度（1步、5步、10步后的危险）
            danger_1step = 0.0
            danger_5step = 0.0
            danger_10step = 0.0
            
            for zone in threat_zones:
                if zone['steps'] <= 2:
                    danger_1step += zone['danger']
                elif zone['steps'] <= 6:
                    danger_5step += zone['danger']
                else:
                    danger_10step += zone['danger']
            
            features.extend([
                min(danger_1step, 1.0),
                min(danger_5step, 1.0),
                min(danger_10step, 1.0),
            ])
            
            # 添加预测危险方向（8个方向）
            pred_danger_dirs = self.enemy_predictor.get_danger_directions((pcx, pcy))
            features.extend(pred_danger_dirs)
        else:
            # 如果没有预测器，使用默认特征
            features.extend([0.0, 0.0, 0.0])  # 危险度
            features.extend([0.0] * 8)  # 危险方向
        
        # ============ 敌人子弹特征（最多5颗最近子弹）+ 危险方向 ============
        enemy_bullets = vision_result.get('bullets_enemy', [])
        max_bullets = 5
        
        # 危险方向统计（8个方向：上、下、左、右、左上、右上、左下、右下）
        danger_directions = [0.0] * 8  # [上, 下, 左, 右, 左上, 右上, 左下, 右下]
        
        for i in range(max_bullets):
            if i < len(enemy_bullets):
                bx, by, bw, bh = enemy_bullets[i]['box']
                bcx, bcy = bx + bw // 2, by + bh // 2
                
                dx = bcx - pcx
                dy = bcy - pcy
                distance = np.sqrt(dx**2 + dy**2)
                
                # 添加子弹特征
                features.extend([
                    dx / (frame_width / 2),
                    dy / (frame_height / 2),
                    min(distance / 300.0, 1.0),
                ])
                
                # 计算危险方向（只考虑距离小于200的子弹）
                if distance < 200:
                    angle = np.arctan2(dy, dx)  # 弧度
                    # 转换为8个方向
                    direction_idx = int(((angle + np.pi) / (2 * np.pi)) * 8) % 8
                    # 距离越近，危险度越高
                    danger_score = (200 - distance) / 200
                    danger_directions[direction_idx] += danger_score
            else:
                features.extend([0.0, 0.0, 1.0])
        
        # 敌人子弹数量
        features.append(min(len(enemy_bullets) / 10.0, 1.0))
        
        # 添加危险方向特征（归一化）
        max_danger = max(danger_directions) if danger_directions else 1.0
        if max_danger > 0:
            danger_directions = [d / max_danger for d in danger_directions]
        features.extend(danger_directions)
        
        # ============ 玩家子弹特征 ============
        player_bullets = vision_result.get('bullets_player', [])
        features.append(min(len(player_bullets) / 10.0, 1.0))
        
        # ============ 战斗状态 ============
        features.append(1.0 if vision_result.get('hurt', False) else 0.0)
        features.append(1.0 if vision_result.get('bullet_hit', False) else 0.0)
        
        # ============ 填充到固定维度 ============
        features = np.array(features, dtype=np.float32)
        
        if len(features) < self.state_dim:
            # 使用特征平均值填充，避免零填充引入噪声
            if len(features) > 0:
                mean_val = np.mean(features)
                padding = np.full(self.state_dim - len(features), mean_val, dtype=np.float32)
                features = np.concatenate([features, padding])
            else:
                # 如果没有特征，用零填充
                features = np.zeros(self.state_dim, dtype=np.float32)
        elif len(features) > self.state_dim:
            # 智能截断：保留前半部分（玩家和环境信息）和后半部分（敌人和战斗状态）
            # 丢弃中间不太重要的部分
            half = self.state_dim // 2
            features = np.concatenate([features[:half], features[-half:]])
        
        # 更新状态历史
        self.state_history.append(features)
        
        return features
    
    def compute_reward(self, vision_result: Dict) -> float:
        """改进版奖励函数 - 强调躲避和攻击"""
        reward = 0.0
        
        # 1. 时间惩罚（减小，避免AI急于结束）
        reward += self.time_penalty
        
        # 2. 存活奖励（鼓励活着）
        reward += self.survival_bonus
        
        # 3. 击中敌人奖励
        if vision_result.get('bullet_hit', False):
            reward += self.hit_reward
        
        # 4. 击杀奖励
        if vision_result.get('enemy_killed', False):
            reward += self.kill_reward
        
        # 5. 受伤惩罚（大幅提高，强烈不鼓励受伤）
        if vision_result.get('hurt', False):
            reward += self.hurt_penalty
        
        # 6. 躲避奖励（新增：鼓励远离危险子弹）
        player = vision_result.get('player', [])
        enemies = vision_result.get('enemy', [])
        enemy_bullets = vision_result.get('bullets_enemy', [])
        
        if player:
            px, py, pw, ph = player[0]['box']
            pcx, pcy = px + pw // 2, py + ph // 2
            
            # 敌人距离奖励（核心改进：鼓励主动保持距离）
            nearest_enemy_dist = float('inf')
            for enemy in enemies:
                ex, ey, ew, eh = enemy['box']
                ecx, ecy = ex + ew // 2, ey + eh // 2
                distance = np.sqrt((ecx - pcx)**2 + (ecy - pcy)**2)
                nearest_enemy_dist = min(nearest_enemy_dist, distance)
                
                # 根据距离给予不同的奖励/惩罚
                if distance < 60:  # 太近（危险）
                    reward -= 15.0  # 强烈惩罚
                elif distance < 120:  # 接近安全距离
                    reward -= 2.0  # 轻微惩罚
                else:  # 安全距离或更远
                    reward += 5.0  # 保持距离奖励
                    reward += (distance - 120) * 0.1  # 额外距离奖励
            
            # 与敌人子弹的距离奖励（躲避奖励）
            dodge_reward = 0.0
            for bullet in enemy_bullets:
                bx, by, bw, bh = bullet['box']
                bcx, bcy = bx + bw // 2, by + bh // 2
                distance = np.sqrt((bcx - pcx)**2 + (bcy - pcy)**2)
                
                if distance < self.danger_distance:
                    # 距离越近，惩罚越大
                    reward -= (self.danger_distance - distance) * 0.2
                elif distance < self.safe_distance:
                    # 在安全距离内，保持距离有奖励
                    dodge_reward += 0.1
                else:
                    # 超出安全距离，奖励
                    dodge_reward += 0.05
            
            # 躲避奖励（如果有危险子弹但没受伤）
            if enemy_bullets and not vision_result.get('hurt', False):
                reward += dodge_reward
            
            # 8. 基于预测的躲避奖励（新增：鼓励提前躲避）
            if self.enemy_predictor and HAS_ENEMY_PREDICTOR:
                threat_zones = self.enemy_predictor.get_predicted_threat_zones((pcx, pcy))
                
                for zone in threat_zones:
                    # 如果有预测的危险区域，给予奖励
                    if zone['danger'] > 0.5:
                        # 危险度高，保持距离的奖励
                        reward += zone['danger'] * 2.0
                    elif zone['danger'] > 0.2:
                        # 危险度中等
                        reward += zone['danger'] * 1.0
        
        # 7. 攻击方向奖励（鼓励瞄准敌人）
        if player and enemies:
            px, py, pw, ph = player[0]['box']
            pcx, pcy = px + pw // 2, py + ph // 2
            
            # 找到最近的敌人
            min_dist = float('inf')
            nearest_enemy = None
            for enemy in enemies:
                ex, ey, ew, eh = enemy['box']
                ecx, ecy = ex + ew // 2, ey + eh // 2
                dist = np.sqrt((ecx - pcx)**2 + (ecy - pcy)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_enemy = (ecx, ecy)
            
            if nearest_enemy and vision_result.get('bullets_player', []):
                # 如果玩家在射击且有子弹，检查是否朝向敌人
                reward += 5.0  # 射击奖励
        
        # 8. 连续存活奖励（新增：鼓励连续存活）
        if not vision_result.get('hurt', False):
            reward += 0.5  # 没受伤的额外奖励
        
        return reward
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool], next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算广义优势估计(GAE)"""
        advantages = []
        returns = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return np.array(advantages), np.array(returns)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, value = self.network.get_action(state_tensor, deterministic)
        return action, log_prob, value
    
    def update(self) -> Dict[str, float]:
        """执行PPO更新"""
        if len(self.buffer) < self.batch_size:
            return {}
        
        # 获取缓冲区数据
        states = np.array(self.buffer.states)
        actions = np.array(self.buffer.actions)
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        log_probs = np.array(self.buffer.log_probs)
        dones = np.array(self.buffer.dones)
        
        # 计算GAE
        with torch.no_grad():
            last_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
            _, last_value = self.network.forward(last_state)
            advantages, returns = self.compute_gae(rewards, values, dones, last_value.item())
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        
        # 多次更新
        total_loss = 0.0
        total_entropy = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        for _ in range(self.update_epochs):
            # 打乱顺序
            indices = torch.randperm(len(states_tensor))
            
            for start in range(0, len(states_tensor), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                
                # 前向传播
                new_log_probs, new_values, entropy = self.network.evaluate_actions(batch_states, batch_actions)
                
                # PPO裁剪损失
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # 熵损失（鼓励探索）
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_entropy += entropy.mean().item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
        
        # 更新学习率
        self.scheduler.step()
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 记录统计
        avg_loss = total_loss / (self.update_epochs * (len(states_tensor) // self.mini_batch_size))
        avg_entropy = total_entropy / (self.update_epochs * (len(states_tensor) // self.mini_batch_size))
        avg_policy_loss = policy_loss_sum / (self.update_epochs * (len(states_tensor) // self.mini_batch_size))
        avg_value_loss = value_loss_sum / (self.update_epochs * (len(states_tensor) // self.mini_batch_size))
        
        return {
            'loss': avg_loss,
            'entropy': avg_entropy,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
        }
    
    def save_model(self, path: Optional[str] = None):
        """保存模型"""
        save_path = Path(path) if path else self.save_path / 'ppo_model.pth'
        # 确保保存目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, save_path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save(self, path: Optional[str] = None):
        """保存模型（兼容旧接口）"""
        self.save_model(path)
    
    def train(self, env_fn, eval_fn=None):
        """训练主循环"""
        for episode in range(self.episodes):
            env = env_fn()
            state = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(self.episode_length):
                # 选择动作
                action, log_prob, value = self.select_action(state)
                
                # 执行动作
                next_state, reward, done, _ = env.step(action)
                
                # 存储经验
                self.buffer.add(state, action, reward, value, log_prob, done)
                
                # 更新状态
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # 定期更新
                if len(self.buffer) >= self.batch_size:
                    stats = self.update()
                    if stats:
                        self.training_stats['losses'].append(stats['loss'])
                        self.training_stats['entropy'].append(stats['entropy'])
                        if self.training_logger:
                            self.training_logger.log_ppo_loss(
                                self.total_steps + step, 
                                stats.get('policy_loss', 0),
                                stats.get('value_loss', 0),
                                stats['loss'],
                                stats['entropy']
                            )
                
                if done:
                    break
            
            # 处理剩余经验
            if len(self.buffer) > 0:
                stats = self.update()
                if stats:
                    self.training_stats['losses'].append(stats['loss'])
                    self.training_stats['entropy'].append(stats['entropy'])
            
            # 记录统计
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            
            # 保存最佳模型（如果这是第一个episode，或者获得了更好的成绩）
            if episode_reward > self.best_score:
                self.best_score = episode_reward
                self.save_model()
                print(f"New best model saved! Score: {episode_reward:.2f}")
            
            # 每50个episode保存一次checkpoint（确保有备份）
            if (episode + 1) % 50 == 0:
                checkpoint_path = self.save_path / f'checkpoint_ep{episode+1}.pth'
                self.save_model(str(checkpoint_path))
                print(f"Checkpoint saved at episode {episode+1}")
            
            # 日志输出
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-10:])
                print(f"Episode {episode+1}/{self.episodes} | Reward: {episode_reward:.2f} | Avg Reward: {avg_reward:.2f} | Length: {episode_length}")
        
        # 训练结束后保存最终模型
        print("=" * 60)
        print("Training completed!")
        print(f"Best score: {self.best_score:.2f}")
        print(f"Final model saved to: {self.save_path / 'ppo_model.pth'}")
        print("=" * 60)
        
        return self.training_stats


# 默认GPU配置
DEFAULT_GPU_CONFIG = {
    # 网络参数
    'state_dim': 61,           # 状态维度（61维：玩家4+敌人15+敌人数量1+敌人距离特征3+敌人预测特征11+敌人子弹15+子弹数量1+危险方向8+玩家子弹1+战斗状态2）
    'action_dim': 9,           # 动作维度
    'hidden_dim': 512,         # 隐藏层大小（增大）
    
    # PPO参数
    'lr': 5e-4,                # 学习率（提高）
    'gamma': 0.99,             # 折扣因子
    'gae_lambda': 0.95,        # GAE lambda
    'clip_ratio': 0.2,         # PPO裁剪比例
    'value_coef': 0.5,         # 价值损失系数
    'entropy_coef': 0.05,      # 熵系数（提高探索）
    'max_grad_norm': 0.5,      # 梯度裁剪
    'update_epochs': 15,       # 更新轮数（增加）
    'mini_batch_size': 64,     # 小批次大小
    
    # 训练参数
    'episodes': 500,           # 训练轮数
    'episode_length': 500,     # 每轮步数
    'batch_size': 256,         # 批次大小（增大）
    'save_path': './models',   # 模型保存路径
    
    # 奖励配置（关键改进！）
    'time_penalty': -0.1,      # 时间惩罚（减小）
    'hit_reward': 30,          # 击中奖励（提高）
    'hurt_penalty': -80,       # 受伤惩罚（大幅提高）
    'kill_reward': 100,        # 击杀奖励
    'survival_bonus': 0.5,     # 存活奖励
    
    # 距离奖励配置（核心改进）
    'distance_reward_factor': 0.5,  # 距离奖励系数（提高5倍）
    'stay_away_reward': 5.0,        # 保持距离奖励（新增）
    'enemy_proximity_penalty': -10, # 敌人过近惩罚（新增）
    'safe_distance': 120,      # 安全距离（像素）
    'danger_distance': 60,     # 危险距离（像素）
}
