"""训练可视化模块 - 支持实时绘制和数据保存"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional
import time
import cv2


class TrainingLogger:
    """训练数据记录器"""
    
    def __init__(self, log_dir: str = 'training_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练数据
        self.data = {
            'ga_scores': [],
            'ga_best_scores': [],
            'ppo_scores': [],
            'ppo_best_scores': [],
            'ppo_policy_losses': [],
            'ppo_value_losses': [],
            'ppo_total_losses': [],
            'ppo_entropy': [],
            'episode_times': [],
            'hurt_counts': [],
        }
        
        # 临时缓冲区
        self.score_buffer = deque(maxlen=100)
        self.loss_buffer = deque(maxlen=100)
        
    def log_ga_score(self, episode: int, score: float, best_score: float):
        """记录遗传算法得分"""
        self.data['ga_scores'].append({
            'episode': episode,
            'score': score,
            'best_score': best_score,
            'timestamp': time.time()
        })
        
    def log_ppo_score(self, episode: int, score: float, best_score: float):
        """记录PPO得分"""
        self.data['ppo_scores'].append({
            'episode': episode,
            'score': score,
            'best_score': best_score,
            'timestamp': time.time()
        })
        
    def log_ppo_loss(self, step: int, policy_loss: float, value_loss: float, 
                    total_loss: float, entropy: float = 0.0):
        """记录PPO损失"""
        self.data['ppo_policy_losses'].append({
            'step': step,
            'loss': policy_loss,
            'timestamp': time.time()
        })
        self.data['ppo_value_losses'].append({
            'step': step,
            'loss': value_loss,
            'timestamp': time.time()
        })
        self.data['ppo_total_losses'].append({
            'step': step,
            'loss': total_loss,
            'timestamp': time.time()
        })
        self.data['ppo_entropy'].append({
            'step': step,
            'entropy': entropy,
            'timestamp': time.time()
        })
        
    def log_episode_stats(self, episode: int, hurt_count: int, duration: float):
        """记录episode统计信息"""
        self.data['hurt_counts'].append({
            'episode': episode,
            'hurt_count': hurt_count,
            'timestamp': time.time()
        })
        self.data['episode_times'].append({
            'episode': episode,
            'duration': duration,
            'timestamp': time.time()
        })
        
    def save_to_file(self, filename: str = 'training_data.json'):
        """保存数据到文件"""
        filepath = self.log_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"Training data saved to {filepath}")
        
    def load_from_file(self, filename: str = 'training_data.json') -> bool:
        """从文件加载数据"""
        filepath = self.log_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Training data loaded from {filepath}")
            return True
        return False


class TrainingVisualizer:
    """训练可视化器"""
    
    def __init__(self, logger: Optional[TrainingLogger] = None):
        self.logger = logger or TrainingLogger()
        self.output_dir = Path('visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        try:
            self.font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12)
            self.font_title = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=14)
        except Exception:
            self.font = FontProperties(size=12)
            self.font_title = FontProperties(size=14)
        
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_score_comparison(self, show: bool = False, save: bool = True):
        """绘制训练得分曲线对比图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制遗传算法曲线
        ga_scores = self.logger.data.get('ga_scores', [])
        ga_best = self.logger.data.get('ga_best_scores', [])
        
        if ga_scores:
            ga_episodes = [d['episode'] for d in ga_scores]
            ga_values = [d['score'] for d in ga_scores]
            ax.plot(ga_episodes, ga_values, label='遗传算法 - 当前得分', 
                   color='#1f77b4', alpha=0.6, linewidth=1.5)
            
            # 最优得分曲线
            ga_best_episodes = [d['episode'] for d in ga_scores]
            ga_best_values = [d['best_score'] for d in ga_scores]
            ax.plot(ga_best_episodes, ga_best_values, label='遗传算法 - 最优得分',
                   color='#1f77b4', linewidth=2.5, linestyle='--')
        
        # 绘制PPO曲线
        ppo_scores = self.logger.data.get('ppo_scores', [])
        
        if ppo_scores:
            ppo_episodes = [d['episode'] for d in ppo_scores]
            ppo_values = [d['score'] for d in ppo_scores]
            ax.plot(ppo_episodes, ppo_values, label='PPO算法 - 当前得分',
                   color='#ff7f0e', alpha=0.6, linewidth=1.5)
            
            ppo_best_values = [d['best_score'] for d in ppo_scores]
            ax.plot(ppo_episodes, ppo_best_values, label='PPO算法 - 最优得分',
                   color='#ff7f0e', linewidth=2.5, linestyle='--')
        
        ax.set_title('训练得分曲线对比', fontproperties=self.font_title)
        ax.set_xlabel('训练轮数', fontproperties=self.font)
        ax.set_ylabel('单轮得分', fontproperties=self.font)
        ax.legend(prop=self.font)
        ax.grid(True, alpha=0.3)
        
        if save:
            filepath = self.output_dir / 'score_comparison.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Score comparison plot saved to {filepath}")
        
        if show:
            plt.show()
        plt.close()
    
    def plot_ppo_loss(self, show: bool = False, save: bool = True):
        """绘制PPO损失函数变化图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 策略损失
        policy_losses = self.logger.data.get('ppo_policy_losses', [])
        if policy_losses:
            steps = [d['step'] for d in policy_losses]
            losses = [d['loss'] for d in policy_losses]
            ax.plot(steps, losses, label='策略损失', color='#1f77b4', linewidth=1.5)
        
        # 价值损失
        value_losses = self.logger.data.get('ppo_value_losses', [])
        if value_losses:
            steps = [d['step'] for d in value_losses]
            losses = [d['loss'] for d in value_losses]
            ax.plot(steps, losses, label='价值损失', color='#ff7f0e', linewidth=1.5)
        
        # 总损失
        total_losses = self.logger.data.get('ppo_total_losses', [])
        if total_losses:
            steps = [d['step'] for d in total_losses]
            losses = [d['loss'] for d in total_losses]
            ax.plot(steps, losses, label='总损失', color='#2ca02c', linewidth=2.0)
        
        ax.set_title('PPO算法损失函数变化', fontproperties=self.font_title)
        ax.set_xlabel('训练步数', fontproperties=self.font)
        ax.set_ylabel('损失值', fontproperties=self.font)
        ax.legend(prop=self.font)
        ax.grid(True, alpha=0.3)
        
        # 添加平滑曲线
        if total_losses:
            steps = np.array([d['step'] for d in total_losses])
            losses = np.array([d['loss'] for d in total_losses])
            window_size = min(20, len(losses) // 5)
            if window_size > 1:
                smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                smoothed_steps = steps[:len(smoothed)]
                ax.plot(smoothed_steps, smoothed, label='总损失(平滑)', 
                       color='#9467bd', linewidth=2.5, linestyle='--')
        
        if save:
            filepath = self.output_dir / 'ppo_loss.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"PPO loss plot saved to {filepath}")
        
        if show:
            plt.show()
        plt.close()
    
    def plot_hurt_statistics(self, show: bool = False, save: bool = True):
        """绘制受伤统计图表"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        hurt_counts = self.logger.data.get('hurt_counts', [])
        if hurt_counts:
            episodes = [d['episode'] for d in hurt_counts]
            counts = [d['hurt_count'] for d in hurt_counts]
            
            ax.bar(episodes, counts, width=0.6, color='#d62728', alpha=0.7)
            ax.plot(episodes, counts, color='#d62728', linewidth=2, marker='o', markersize=4)
            
            # 计算移动平均
            window_size = min(5, len(counts))
            if window_size > 1:
                moving_avg = np.convolve(counts, np.ones(window_size)/window_size, mode='same')
                ax.plot(episodes, moving_avg, label='移动平均', 
                       color='#ff9896', linewidth=2.5, linestyle='--')
        
        ax.set_title('受伤次数统计', fontproperties=self.font_title)
        ax.set_xlabel('训练轮数', fontproperties=self.font)
        ax.set_ylabel('受伤次数', fontproperties=self.font)
        ax.legend(prop=self.font)
        ax.grid(True, alpha=0.3)
        
        if save:
            filepath = self.output_dir / 'hurt_statistics.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Hurt statistics plot saved to {filepath}")
        
        if show:
            plt.show()
        plt.close()
    
    def generate_all_plots(self):
        """生成所有图表"""
        print("Generating training visualizations...")
        self.plot_score_comparison()
        self.plot_ppo_loss()
        self.plot_hurt_statistics()
        print("All visualizations generated successfully!")


class VisionDetectionVisualizer:
    """视觉检测效果可视化器"""
    
    COLOR_MAP = {
        'player': (0, 255, 0),      # 绿色 - 玩家
        'enemy': (0, 0, 255),        # 红色 - 敌人
        'bullets_player': (255, 0, 0),  # 蓝色 - 玩家子弹
        'bullets_enemy': (0, 255, 255),  # 黄色 - 敌人子弹
        'ground': (255, 165, 0),    # 橙色 - 地面
        'walls_room': (128, 128, 128),  # 灰色 - 墙体
    }
    
    def __init__(self):
        self.output_dir = Path('visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def draw_detection_boxes(self, frame_bgr: np.ndarray, detections: Dict, 
                            output_path: Optional[str] = None) -> np.ndarray:
        """
        在画面上绘制检测框和置信度
        
        Args:
            frame_bgr: 原始帧 (BGR格式)
            detections: 检测结果字典
            output_path: 输出路径
            
        Returns:
            np.ndarray: 绘制后的帧
        """
        result = frame_bgr.copy()
        
        for label, items in detections.items():
            if not items:
                continue
            
            color = self.COLOR_MAP.get(label, (255, 255, 255))
            
            for item in items:
                box = item.get('box')
                if not box:
                    continue
                
                x, y, w, h = box
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # 绘制边框
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                
                # 绘制标签和置信度
                score = item.get('score', 0.0)
                label_text = f"{label}: {score:.2f}"
                
                # 计算文字位置
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x
                text_y = y - 5 if y > 20 else y + h + 15
                
                # 绘制文字背景
                cv2.rectangle(result, (text_x, text_y - text_size[1] - 5),
                            (text_x + text_size[0], text_y), color, -1)
                
                # 绘制文字
                cv2.putText(result, label_text, (text_x, text_y - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 添加标题
        cv2.putText(result, "Vision Detection Result", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if output_path:
            filepath = self.output_dir / output_path
            cv2.imwrite(str(filepath), result)
            print(f"Detection visualization saved to {filepath}")
        
        return result
    
    def create_sample_detection_image(self, frame_width: int = 800, frame_height: int = 600,
                                     output_path: str = 'detection_sample.png'):
        """
        创建示例检测效果图
        
        Args:
            frame_width: 帧宽度
            frame_height: 帧高度
            output_path: 输出路径
        """
        # 创建模拟帧
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # 绘制模拟背景
        cv2.rectangle(frame, (0, 450), (frame_width, frame_height), (139, 119, 101), -1)  # 地面
        cv2.rectangle(frame, (0, 0), (frame_width, 450), (100, 149, 237), -1)  # 天空
        
        # 绘制模拟墙体
        cv2.rectangle(frame, (50, 300), (150, 450), (139, 139, 139), -1)
        cv2.rectangle(frame, (650, 300), (750, 450), (139, 139, 139), -1)
        
        # 绘制模拟玩家
        cv2.rectangle(frame, (350, 380), (410, 450), (50, 205, 50), -1)  # 玩家身体
        
        # 绘制模拟敌人
        cv2.rectangle(frame, (200, 380), (260, 450), (220, 20, 60), -1)  # 敌人1
        cv2.rectangle(frame, (540, 380), (600, 450), (220, 20, 60), -1)  # 敌人2
        
        # 绘制模拟子弹
        cv2.circle(frame, (230, 360), 5, (255, 255, 0), -1)  # 敌人子弹
        cv2.circle(frame, (570, 360), 5, (255, 255, 0), -1)  # 敌人子弹
        cv2.circle(frame, (380, 350), 5, (0, 0, 255), -1)    # 玩家子弹
        
        # 创建模拟检测结果
        detections = {
            'player': [{'box': (350, 380, 60, 70), 'score': 0.95}],
            'enemy': [
                {'box': (200, 380, 60, 70), 'score': 0.92},
                {'box': (540, 380, 60, 70), 'score': 0.88},
            ],
            'bullets_player': [{'box': (378, 348, 10, 10), 'score': 0.99}],
            'bullets_enemy': [
                {'box': (228, 358, 10, 10), 'score': 0.97},
                {'box': (568, 358, 10, 10), 'score': 0.96},
            ],
            'ground': [{'box': (0, 450, frame_width, 150), 'score': 0.94}],
            'walls_room': [
                {'box': (50, 300, 100, 150), 'score': 0.85},
                {'box': (650, 300, 100, 150), 'score': 0.82},
            ],
        }
        
        # 绘制检测框
        self.draw_detection_boxes(frame, detections, output_path)
        print(f"Sample detection image created: {output_path}")


def create_demo_visualizations():
    """创建演示可视化（使用模拟数据）"""
    logger = TrainingLogger()
    
    # 生成模拟数据
    np.random.seed(42)
    
    # 遗传算法模拟数据
    ga_best = 1000
    for ep in range(1, 51):
        ga_best = max(ga_best, ga_best + np.random.randint(50, 200))
        ga_score = ga_best + np.random.randint(-200, 100)
        logger.log_ga_score(ep, ga_score, ga_best)
    
    # PPO算法模拟数据
    ppo_best = 1000
    for ep in range(1, 51):
        if ep < 10:
            ppo_best = max(ppo_best, ppo_best + np.random.randint(20, 80))
        elif ep < 30:
            ppo_best = max(ppo_best, ppo_best + np.random.randint(80, 250))
        else:
            ppo_best = max(ppo_best, ppo_best + np.random.randint(50, 150))
        ppo_score = ppo_best + np.random.randint(-300, 150)
        logger.log_ppo_score(ep, ppo_score, ppo_best)
    
    # PPO损失模拟数据
    policy_loss = 1.0
    value_loss = 0.5
    for step in range(1, 1001):
        policy_loss = max(0.01, policy_loss * 0.995 + np.random.normal(0, 0.02))
        value_loss = max(0.01, value_loss * 0.996 + np.random.normal(0, 0.015))
        total_loss = policy_loss + value_loss
        entropy = max(0.1, 0.8 * 0.999 ** step)
        logger.log_ppo_loss(step, policy_loss, value_loss, total_loss, entropy)
    
    # 受伤统计模拟数据
    for ep in range(1, 51):
        hurt_count = max(0, int(5 - ep * 0.08 + np.random.randint(-2, 3)))
        logger.log_episode_stats(ep, hurt_count, 30.0)
    
    # 创建可视化
    visualizer = TrainingVisualizer(logger)
    visualizer.generate_all_plots()
    
    # 创建示例检测效果图
    vision_vis = VisionDetectionVisualizer()
    vision_vis.create_sample_detection_image()
    
    # 保存数据
    logger.save_to_file()


if __name__ == '__main__':
    create_demo_visualizations()
