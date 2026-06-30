"""改进的强化学习训练器 - 支持PPO算法和更好的训练流程"""
from __future__ import annotations

import logging
import numpy as np
import os
import pickle
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


class PPOTrainer:
    """近端策略优化训练器"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 lr: float = 3e-4, gamma: float = 0.99, 
                 clip_ratio: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # 简单的策略网络（可替换为PyTorch实现）
        self.policy_weights = np.random.randn(state_dim, action_dim) * 0.1
        self.value_weights = np.random.randn(state_dim, 1) * 0.1
        
        # 优化器状态
        self.policy_momentum = np.zeros_like(self.policy_weights)
        self.value_momentum = np.zeros_like(self.value_weights)
    
    def policy(self, state: np.ndarray) -> np.ndarray:
        """计算动作概率分布"""
        logits = state @ self.policy_weights
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def value(self, state: np.ndarray) -> float:
        """计算状态价值"""
        return float(state @ self.value_weights)
    
    def sample_action(self, state: np.ndarray) -> Tuple[int, float]:
        """采样动作并返回对数概率"""
        probs = self.policy(state)
        action = np.random.choice(len(probs), p=probs)
        
        # 计算对数概率
        log_prob = np.log(probs[action] + 1e-8)
        return action, log_prob
    
    def compute_returns(self, rewards: List[float], values: List[float], 
                       done: bool = False) -> np.ndarray:
        """计算折扣回报"""
        returns = []
        running_sum = 0.0 if done else values[-1]
        
        for r, v in zip(reversed(rewards), reversed(values)):
            running_sum = r + self.gamma * running_sum
            returns.insert(0, running_sum)
        
        # 标准化
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        return returns
    
    def train(self, states: np.ndarray, actions: np.ndarray, 
             log_probs: np.ndarray, returns: np.ndarray, values: np.ndarray):
        """执行一次PPO更新"""
        # 计算优势函数
        advantages = returns - values
        
        # 更新策略网络
        for _ in range(10):
            new_log_probs = []
            entropies = []
            
            for i, state in enumerate(states):
                probs = self.policy(state)
                new_log_probs.append(np.log(probs[actions[i]] + 1e-8))
                entropies.append(-np.sum(probs * np.log(probs + 1e-8)))
            
            new_log_probs = np.array(new_log_probs)
            entropies = np.array(entropies)
            
            # PPO裁剪目标
            ratio = np.exp(new_log_probs - log_probs)
            clipped_ratio = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            policy_loss = -np.mean(
                np.minimum(ratio * advantages, clipped_ratio * advantages)
            )
            
            # 熵正则化
            policy_loss -= self.entropy_coef * np.mean(entropies)
            
            # 策略梯度（简化版）
            grad = np.zeros_like(self.policy_weights)
            for i, state in enumerate(states):
                probs = self.policy(state)
                grad[:, actions[i]] += (ratio[i] - 1) * advantages[i] * state
            
            # Adam优化
            self.policy_momentum = 0.9 * self.policy_momentum + 0.1 * grad
            self.policy_weights -= self.lr * self.policy_momentum
        
        # 更新价值网络
        for _ in range(10):
            value_predictions = np.array([self.value(s) for s in states])
            value_loss = np.mean((returns - value_predictions) ** 2)
            
            grad = np.zeros_like(self.value_weights)
            for i, state in enumerate(states):
                error = (value_predictions[i] - returns[i])
                grad[:, 0] += error * state
            
            self.value_momentum = 0.9 * self.value_momentum + 0.1 * grad
            self.value_weights -= self.lr * self.value_momentum * self.value_coef
    
    def save(self, path: str):
        """保存模型"""
        data = {
            'policy_weights': self.policy_weights,
            'value_weights': self.value_weights,
            'policy_momentum': self.policy_momentum,
            'value_momentum': self.value_momentum,
            'params': {
                'lr': self.lr,
                'gamma': self.gamma,
                'clip_ratio': self.clip_ratio,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.policy_weights = data['policy_weights']
        self.value_weights = data['value_weights']
        self.policy_momentum = data.get('policy_momentum', np.zeros_like(self.policy_weights))
        self.value_momentum = data.get('value_momentum', np.zeros_like(self.value_weights))


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """添加经验"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """采样一批经验"""
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        
        return {
            'states': np.array([s['state'] for s in samples]),
            'actions': np.array([s['action'] for s in samples]),
            'rewards': np.array([s['reward'] for s in samples]),
            'next_states': np.array([s['next_state'] for s in samples]),
            'dones': np.array([s['done'] for s in samples])
        }
    
    def __len__(self):
        return len(self.buffer)


class ImprovedRLTrainer:
    """改进的强化学习训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trainer_type = config.get('trainer_type', 'ppo')
        self.state_dim = config.get('state_dim', 20)
        self.action_dim = config.get('action_dim', 9)
        
        # 训练参数
        self.episodes = config.get('episodes', 200)
        self.episode_length = config.get('episode_length', 300)
        self.batch_size = config.get('batch_size', 64)
        self.update_freq = config.get('update_freq', 10)
        
        # 奖励配置
        self.time_penalty = config.get('time_penalty', -1)
        self.hit_reward = config.get('hit_reward', 18)
        self.hurt_penalty = config.get('hurt_penalty', -8)
        
        # 创建训练器
        if self.trainer_type == 'ppo':
            self.trainer = PPOTrainer(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                lr=config.get('lr', 3e-4),
                gamma=config.get('gamma', 0.99),
                clip_ratio=config.get('clip_ratio', 0.2),
                value_coef=config.get('value_coef', 0.5),
                entropy_coef=config.get('entropy_coef', 0.01)
            )
        
        self.replay_buffer = ReplayBuffer(max_size=config.get('buffer_size', 100000))
        self.best_score = float('-inf')
        self.best_model_path = Path(config.get('save_path', '.')) / 'best_rl_model.pkl'
    
    def extract_state(self, vision_result: Dict) -> np.ndarray:
        """从视觉检测结果中提取状态特征"""
        features = []
        
        # 玩家位置特征
        player = vision_result.get('player', [])
        if player:
            px, py, pw, ph = player[0]['box']
            features.extend([px, py, pw, ph])
        else:
            features.extend([0, 0, 0, 0])
        
        # 敌人数量和距离
        enemies = vision_result.get('enemy', [])
        features.append(len(enemies))
        
        if enemies:
            ex, ey, ew, eh = enemies[0]['box']
            # 相对于玩家的位置
            features.extend([ex - px if player else ex, ey - py if player else ey])
        else:
            features.extend([0, 0])
        
        # 子弹特征
        player_bullets = vision_result.get('bullets_player', [])
        enemy_bullets = vision_result.get('bullets_enemy', [])
        features.append(len(player_bullets))
        features.append(len(enemy_bullets))
        
        # 战斗状态
        features.append(1 if vision_result.get('hurt', False) else 0)
        features.append(1 if vision_result.get('bullet_hit', False) else 0)
        
        # 归一化
        features = np.array(features, dtype=np.float32)
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        # 填充到固定维度
        if len(features) < self.state_dim:
            features = np.pad(features, (0, self.state_dim - len(features)))
        
        return features
    
    def compute_reward(self, vision_result: Dict) -> float:
        """计算奖励"""
        reward = self.time_penalty
        
        if vision_result.get('bullet_hit', False):
            reward += self.hit_reward
        
        if vision_result.get('hurt', False):
            reward += self.hurt_penalty
        
        # 额外奖励：击杀敌人、收集道具等
        # 可以在这里扩展
        
        return reward
    
    def train_step(self, vision_result: Dict, prev_state: Optional[np.ndarray] = None):
        """执行一步训练"""
        state = self.extract_state(vision_result)
        reward = self.compute_reward(vision_result)
        
        # 采样动作
        action, log_prob = self.trainer.sample_action(state)
        
        # 添加到缓冲区
        if prev_state is not None:
            self.replay_buffer.add(prev_state, action, reward, state, False)
        
        # 定期更新
        if len(self.replay_buffer) >= self.batch_size and \
           len(self.replay_buffer) % self.update_freq == 0:
            
            batch = self.replay_buffer.sample(self.batch_size)
            if batch is not None:
                # 简化的PPO更新（实际应该收集完整轨迹）
                returns = batch['rewards']  # 简化处理
                
                try:
                    self.trainer.train(
                        batch['states'],
                        batch['actions'],
                        np.zeros(len(batch['actions'])),  # 简化：无旧log_probs
                        returns,
                        np.zeros(len(batch['actions']))   # 简化：无values
                    )
                except Exception as e:
                    logging.debug(f"Training update failed: {e}")
        
        return action, state, reward
    
    def run_training(self, get_vision_result, get_action_space):
        """运行完整训练循环"""
        logging.info(f"Starting {self.trainer_type} training...")
        
        for episode in range(self.episodes):
            episode_reward = 0
            prev_state = None
            
            for step in range(self.episode_length):
                # 获取视觉结果
                vision_result = get_vision_result()
                
                # 训练步骤
                action, state, reward = self.train_step(vision_result, prev_state)
                prev_state = state
                episode_reward += reward
                
                # 执行动作（由调用者处理）
                
                if step % 50 == 0:
                    logging.debug(f"Episode {episode}, Step {step}, Reward: {reward:.2f}")
            
            # 更新最佳模型
            if episode_reward > self.best_score:
                self.best_score = episode_reward
                self.trainer.save(str(self.best_model_path))
                logging.info(f"New best score: {self.best_score:.2f} (episode {episode})")
            
            if episode % 10 == 0:
                logging.info(f"Episode {episode}/{self.episodes}, Reward: {episode_reward:.2f}, Best: {self.best_score:.2f}")
        
        logging.info("Training completed!")
    
    def load_best_model(self):
        """加载最佳模型"""
        if self.best_model_path.exists():
            self.trainer.load(str(self.best_model_path))
            logging.info(f"Loaded best model from {self.best_model_path}")
    
    def save(self, path: str):
        """保存模型到指定路径"""
        self.trainer.save(path)
        logging.info(f"RL model saved to {path}")


# 示例配置
DEFAULT_CONFIG = {
    'trainer_type': 'ppo',
    'state_dim': 20,
    'action_dim': 9,
    'episodes': 500,
    'episode_length': 500,
    'batch_size': 128,
    'update_freq': 10,
    'lr': 3e-4,
    'gamma': 0.99,
    'clip_ratio': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'time_penalty': -1,
    'hit_reward': 20,
    'hurt_penalty': -10,
    'buffer_size': 50000,
    'save_path': './models'
}


if __name__ == '__main__':
    # 测试示例
    trainer = ImprovedRLTrainer(DEFAULT_CONFIG)
    logging.info(f"RL Trainer initialized with {trainer.trainer_type} algorithm")
    logging.info(f"State dim: {trainer.state_dim}, Action dim: {trainer.action_dim}")