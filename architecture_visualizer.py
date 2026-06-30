"""系统整体架构示意图生成器"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path


class ArchitectureVisualizer:
    """系统架构可视化器"""
    
    def __init__(self):
        self.output_dir = Path('visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12)
            self.font_title = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=16)
            self.font_label = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=11)
        except Exception:
            self.font = FontProperties(size=12)
            self.font_title = FontProperties(size=16)
            self.font_label = FontProperties(size=11)
    
    def draw_architecture_diagram(self, output_path: str = 'architecture_diagram.png'):
        """
        绘制系统整体架构示意图
        
        四层架构：
        1. 感知层（Perception Layer）- 视觉检测
        2. 决策层（Decision Layer）- RL算法
        3. 执行层（Execution Layer）- 动作执行
        4. 反馈层（Feedback Layer）- 奖励与受伤检测
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        ax.set_facecolor('white')
        
        layer_height = 2.0
        layer_spacing = 0.5
        
        # ================ 感知层（最底层） ================
        layer_y = 0.8
        
        ax.add_patch(plt.Rectangle((0.5, layer_y), 15, layer_height, 
                                   fill=True, facecolor='#4ECDC4', alpha=0.25,
                                   edgecolor='#4ECDC4', linewidth=2))
        ax.text(8, layer_y + layer_height / 2, '感知层 (Perception Layer)', 
                fontproperties=self.font_title, ha='center', va='center',
                color='#2D5A5A', fontweight='bold')
        
        modules = [
            {'x': 1.5, 'width': 3.2, 'height': 1.5, 'label': '游戏画面\nGame Frame', 'color': '#A8E6CF'},
            {'x': 5.5, 'width': 3.2, 'height': 1.5, 'label': '玩家检测\nPlayer Detector', 'color': '#88D8B0'},
            {'x': 9.5, 'width': 3.2, 'height': 1.5, 'label': '敌人检测\nEnemy Detector', 'color': '#88D8B0'},
            {'x': 13.0, 'width': 2.5, 'height': 1.5, 'label': '增强处理器\nEnhanced\nProcessor', 'color': '#88D8B0'},
        ]
        
        for mod in modules:
            box_y = layer_y + (layer_height - mod['height']) / 2
            ax.add_patch(plt.Rectangle((mod['x'], box_y), mod['width'], mod['height'],
                                       fill=True, facecolor=mod['color'], alpha=0.85,
                                       edgecolor='#2D5A5A', linewidth=1.5))
            ax.text(mod['x'] + mod['width'] / 2, box_y + mod['height'] / 2, mod['label'],
                    fontproperties=self.font_label, ha='center', va='center',
                    color='#2D5A5A')
        
        # 感知层内部数据流
        ax.arrow(4.2, layer_y + layer_height / 2, 1.1, 0, head_width=0.12, head_length=0.15,
                 color='#2D5A5A', alpha=0.7, linewidth=1.5)
        ax.arrow(8.2, layer_y + layer_height / 2, 1.1, 0, head_width=0.12, head_length=0.15,
                 color='#2D5A5A', alpha=0.7, linewidth=1.5)
        ax.arrow(12.2, layer_y + layer_height / 2, 0.6, 0, head_width=0.12, head_length=0.15,
                 color='#2D5A5A', alpha=0.7, linewidth=1.5)
        
        # ================ 决策层 ================
        layer_y = layer_y + layer_height + layer_spacing
        
        ax.add_patch(plt.Rectangle((0.5, layer_y), 15, layer_height,
                                   fill=True, facecolor='#FFE66D', alpha=0.25,
                                   edgecolor='#FFE66D', linewidth=2))
        ax.text(8, layer_y + layer_height / 2, '决策层 (Decision Layer)',
                fontproperties=self.font_title, ha='center', va='center',
                color='#8B7355', fontweight='bold')
        
        modules = [
            {'x': 2.5, 'width': 3.8, 'height': 1.5, 'label': '状态特征提取\nState Feature Extraction', 'color': '#FFF3B0'},
            {'x': 7.0, 'width': 3.8, 'height': 1.5, 'label': 'PPO强化学习\nPPO RL Agent', 'color': '#FFF3B0'},
            {'x': 11.5, 'width': 3.0, 'height': 1.5, 'label': '遗传算法\nGenetic Algorithm', 'color': '#FFF3B0'},
        ]
        
        for mod in modules:
            box_y = layer_y + (layer_height - mod['height']) / 2
            ax.add_patch(plt.Rectangle((mod['x'], box_y), mod['width'], mod['height'],
                                       fill=True, facecolor=mod['color'], alpha=0.85,
                                       edgecolor='#8B7355', linewidth=1.5))
            ax.text(mod['x'] + mod['width'] / 2, box_y + mod['height'] / 2, mod['label'],
                    fontproperties=self.font_label, ha='center', va='center',
                    color='#8B7355')
        
        ax.arrow(5.8, layer_y + layer_height / 2, 1.0, 0, head_width=0.12, head_length=0.15,
                 color='#8B7355', alpha=0.7, linewidth=1.5)
        
        # ================ 执行层 ================
        layer_y = layer_y + layer_height + layer_spacing
        
        ax.add_patch(plt.Rectangle((0.5, layer_y), 15, layer_height,
                                   fill=True, facecolor='#FF9F43', alpha=0.25,
                                   edgecolor='#FF9F43', linewidth=2))
        ax.text(8, layer_y + layer_height / 2, '执行层 (Execution Layer)',
                fontproperties=self.font_title, ha='center', va='center',
                color='#8B4513', fontweight='bold')
        
        modules = [
            {'x': 3.5, 'width': 3.5, 'height': 1.5, 'label': '动作选择\nAction Selection', 'color': '#FFD3A5'},
            {'x': 7.8, 'width': 3.5, 'height': 1.5, 'label': '输入控制器\nInput Controller', 'color': '#FFD3A5'},
            {'x': 12.0, 'width': 3.0, 'height': 1.5, 'label': '游戏API\nGame API', 'color': '#FFD3A5'},
        ]
        
        for mod in modules:
            box_y = layer_y + (layer_height - mod['height']) / 2
            ax.add_patch(plt.Rectangle((mod['x'], box_y), mod['width'], mod['height'],
                                       fill=True, facecolor=mod['color'], alpha=0.85,
                                       edgecolor='#8B4513', linewidth=1.5))
            ax.text(mod['x'] + mod['width'] / 2, box_y + mod['height'] / 2, mod['label'],
                    fontproperties=self.font_label, ha='center', va='center',
                    color='#8B4513')
        
        ax.arrow(6.5, layer_y + layer_height / 2, 1.1, 0, head_width=0.12, head_length=0.15,
                 color='#8B4513', alpha=0.7, linewidth=1.5)
        ax.arrow(10.8, layer_y + layer_height / 2, 1.0, 0, head_width=0.12, head_length=0.15,
                 color='#8B4513', alpha=0.7, linewidth=1.5)
        
        # ================ 反馈层 ================
        layer_y = layer_y + layer_height + layer_spacing
        
        ax.add_patch(plt.Rectangle((0.5, layer_y), 15, layer_height,
                                   fill=True, facecolor='#FF6B6B', alpha=0.25,
                                   edgecolor='#FF6B6B', linewidth=2))
        ax.text(8, layer_y + layer_height / 2, '反馈层 (Feedback Layer)',
                fontproperties=self.font_title, ha='center', va='center',
                color='#8B2323', fontweight='bold')
        
        modules = [
            {'x': 2.0, 'width': 3.0, 'height': 1.5, 'label': '奖励计算\nReward Calculation', 'color': '#FFB3B3'},
            {'x': 5.8, 'width': 3.0, 'height': 1.5, 'label': '受伤检测\nHurt Detection', 'color': '#FFB3B3'},
            {'x': 9.6, 'width': 3.0, 'height': 1.5, 'label': '敌人预测\nEnemy Prediction', 'color': '#FFB3B3'},
            {'x': 13.3, 'width': 2.2, 'height': 1.5, 'label': '训练日志\nTraining Logger', 'color': '#FFB3B3'},
        ]
        
        for mod in modules:
            box_y = layer_y + (layer_height - mod['height']) / 2
            ax.add_patch(plt.Rectangle((mod['x'], box_y), mod['width'], mod['height'],
                                       fill=True, facecolor=mod['color'], alpha=0.85,
                                       edgecolor='#8B2323', linewidth=1.5))
            ax.text(mod['x'] + mod['width'] / 2, box_y + mod['height'] / 2, mod['label'],
                    fontproperties=self.font_label, ha='center', va='center',
                    color='#8B2323')
        
        # ================== 层间数据流 ==================
        
        center_x = 8
        
        # 感知层 -> 决策层
        bottom_layer_y = 0.8
        ax.arrow(center_x, bottom_layer_y + layer_height, 0, layer_spacing, 
                 head_width=0.4, head_length=0.25, color='#2D5A5A', alpha=0.7, linewidth=2)
        ax.text(center_x + 0.5, bottom_layer_y + layer_height + layer_spacing / 2, 
                '检测结果\nDetection Results', fontproperties=self.font_label,
                ha='left', va='center', color='#2D5A5A', rotation=-90)
        
        # 决策层 -> 执行层
        ax.arrow(center_x, bottom_layer_y + 2 * layer_height + layer_spacing, 0, layer_spacing,
                 head_width=0.4, head_length=0.25, color='#8B7355', alpha=0.7, linewidth=2)
        ax.text(center_x + 0.5, bottom_layer_y + 2 * layer_height + layer_spacing * 1.5,
                '动作指令\nAction Command', fontproperties=self.font_label,
                ha='left', va='center', color='#8B7355', rotation=-90)
        
        # 执行层 -> 反馈层
        ax.arrow(center_x, bottom_layer_y + 3 * layer_height + 2 * layer_spacing, 0, layer_spacing,
                 head_width=0.4, head_length=0.25, color='#8B4513', alpha=0.7, linewidth=2)
        ax.text(center_x + 0.5, bottom_layer_y + 3 * layer_height + layer_spacing * 2.5,
                '游戏状态\nGame State', fontproperties=self.font_label,
                ha='left', va='center', color='#8B4513', rotation=-90)
        
        # 游戏 -> 感知层（闭环）
        ax.arrow(center_x, bottom_layer_y - 0.3, 0, -0.8,
                 head_width=0.3, head_length=0.25, color='#2D5A5A', alpha=0.7, linewidth=2)
        
        # 反馈层 -> 决策层（奖励）
        feedback_layer_y = bottom_layer_y + 3 * layer_height + 3 * layer_spacing
        ax.arrow(7.3, feedback_layer_y, 0, -layer_spacing - layer_height,
                 head_width=0.25, head_length=0.2, color='#8B2323', alpha=0.7, linewidth=1.5)
        ax.text(6.8, feedback_layer_y - (layer_spacing + layer_height) / 2,
                '奖励/惩罚\nReward/Penalty', fontproperties=self.font_label,
                ha='right', va='center', color='#8B2323', rotation=90)
        
        # 反馈层 -> 决策层（预测）
        ax.arrow(11.1, feedback_layer_y, 0, -layer_spacing - layer_height,
                 head_width=0.25, head_length=0.2, color='#8B2323', alpha=0.7, linewidth=1.5)
        ax.text(11.6, feedback_layer_y - (layer_spacing + layer_height) / 2,
                '预测信息\nPrediction', fontproperties=self.font_label,
                ha='left', va='center', color='#8B2323', rotation=-90)
        
        # 感知层 -> 反馈层（受伤检测数据）
        ax.arrow(14.1, bottom_layer_y + layer_height, 0, 3 * layer_height + 2 * layer_spacing,
                 head_width=0.15, head_length=0.2, color='#8B2323', alpha=0.5, linestyle='--', linewidth=1.5)
        
        # ================== 外部接口 ==================
        
        # 游戏环境
        ax.add_patch(plt.Rectangle((6, -0.8), 4, 1.3,
                                   fill=True, facecolor='#6C5CE7', alpha=0.9,
                                   edgecolor='#4834A6', linewidth=2))
        ax.text(8, -0.15, '游戏环境\nGame Environment', fontproperties=self.font,
                ha='center', va='center', color='white', fontweight='bold')
        
        # 用户输入（手动受伤触发）
        ax.add_patch(plt.Rectangle((1, feedback_layer_y + 0.8), 1.8, 0.9,
                                   fill=True, facecolor='#00CEC9', alpha=0.9,
                                   edgecolor='#009688', linewidth=1.5))
        ax.text(1.9, feedback_layer_y + 1.25, '手动触发\n(H键)', fontproperties=self.font_label,
                ha='center', va='center', color='#004D40')
        ax.arrow(2.8, feedback_layer_y + 1.25, 0.7, 0, head_width=0.1, head_length=0.12,
                 color='#009688', alpha=0.7, linewidth=1.5)
        
        # 可视化输出
        ax.add_patch(plt.Rectangle((12.2, feedback_layer_y + 0.8), 3.0, 0.9,
                                   fill=True, facecolor='#00CEC9', alpha=0.9,
                                   edgecolor='#009688', linewidth=1.5))
        ax.text(13.7, feedback_layer_y + 1.25, '可视化输出\nVisualization', fontproperties=self.font_label,
                ha='center', va='center', color='#004D40')
        ax.arrow(12.0, feedback_layer_y + 1.25, -0.7, 0, head_width=0.1, head_length=0.12,
                 color='#009688', alpha=0.7, linewidth=1.5)
        
        # ================== 标题 ==================
        ax.text(8, 11.2, 'AI游戏智能体系统架构图', fontproperties=self.font_title,
                ha='center', va='center', color='#2C3E50', fontsize=22, fontweight='bold')
        
        # ================== 图例 ==================
        legend_y = feedback_layer_y + 0.5
        legend_items = [
            ('感知层', '#4ECDC4'),
            ('决策层', '#FFE66D'),
            ('执行层', '#FF9F43'),
            ('反馈层', '#FF6B6B'),
            ('游戏环境', '#6C5CE7'),
            ('外部接口', '#00CEC9'),
        ]
        
        for i, (label, color) in enumerate(legend_items):
            x = 0.8 + i * 2.2
            ax.add_patch(plt.Rectangle((x, legend_y - 0.35), 0.5, 0.35,
                                       fill=True, facecolor=color, alpha=0.5))
            ax.text(x + 0.7, legend_y - 0.17, label, fontproperties=self.font_label,
                    ha='left', va='center', color='#333')
        
        filepath = self.output_dir / output_path
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Architecture diagram saved to {filepath}")
        
        plt.close()
    
    def draw_module_relationship(self, output_path: str = 'module_relationship.png'):
        """
        绘制模块调用关系图
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_facecolor('white')
        
        modules = {
            'train_unsupervised': {'x': 7, 'y': 8.5, 'width': 3.0, 'height': 1.0, 
                                   'label': 'train_unsupervised.py\n训练主脚本', 'color': '#2ECC71'},
            'vision_detector': {'x': 1.5, 'y': 6.5, 'width': 3.2, 'height': 1.0, 
                                'label': 'vision_detector.py\n视觉检测器', 'color': '#3498DB'},
            'rl_trainer': {'x': 5.5, 'y': 6.5, 'width': 3.2, 'height': 1.0, 
                           'label': 'rl_trainer_gpu.py\nPPO训练器', 'color': '#E67E22'},
            'enhanced_processor': {'x': 9.5, 'y': 6.5, 'width': 3.2, 'height': 1.0, 
                                   'label': 'enhanced_vision_processor.py\n增强视觉处理', 'color': '#9B59B6'},
            'enemy_predictor': {'x': 1.5, 'y': 4.5, 'width': 3.2, 'height': 1.0, 
                                'label': 'enemy_predictor.py\n敌人预测器', 'color': '#1ABC9C'},
            'player_detector': {'x': 5.5, 'y': 4.5, 'width': 3.2, 'height': 1.0, 
                                'label': 'player_detector.py\n玩家检测器', 'color': '#1ABC9C'},
            'training_visualizer': {'x': 9.5, 'y': 4.5, 'width': 3.2, 'height': 1.0, 
                                    'label': 'training_visualizer.py\n训练可视化', 'color': '#F39C12'},
            'input_controller': {'x': 3.5, 'y': 2.5, 'width': 3.2, 'height': 1.0, 
                                 'label': 'input_controller\n输入控制器', 'color': '#E74C3C'},
            'game_api': {'x': 8.0, 'y': 2.5, 'width': 3.2, 'height': 1.0, 
                         'label': 'game_api.py\n游戏API', 'color': '#E74C3C'},
        }
        
        for name, mod in modules.items():
            ax.add_patch(plt.Rectangle((mod['x'], mod['y']), mod['width'], mod['height'],
                                       fill=True, facecolor=mod['color'], alpha=0.25,
                                       edgecolor=mod['color'], linewidth=2))
            ax.text(mod['x'] + mod['width'] / 2, mod['y'] + mod['height'] / 2,
                    mod['label'], fontproperties=self.font_label, ha='center', va='center',
                    color=mod['color'], fontsize=10)
        
        connections = [
            ('train_unsupervised', 'vision_detector'),
            ('train_unsupervised', 'rl_trainer'),
            ('train_unsupervised', 'training_visualizer'),
            ('vision_detector', 'enhanced_processor'),
            ('vision_detector', 'player_detector'),
            ('rl_trainer', 'vision_detector'),
            ('rl_trainer', 'enemy_predictor'),
            ('enhanced_processor', 'enemy_predictor'),
            ('train_unsupervised', 'input_controller'),
            ('input_controller', 'game_api'),
            ('vision_detector', 'game_api'),
        ]
        
        for from_mod, to_mod in connections:
            from_pos = modules[from_mod]
            to_pos = modules[to_mod]
            
            from_center_x = from_pos['x'] + from_pos['width'] / 2
            from_bottom_y = from_pos['y']
            to_center_x = to_pos['x'] + to_pos['width'] / 2
            to_top_y = to_pos['y'] + to_pos['height']
            
            ax.arrow(from_center_x, from_bottom_y, to_center_x - from_center_x, to_top_y - from_bottom_y,
                     head_width=0.15, head_length=0.2, color='#7F8C8D', alpha=0.7,
                     linewidth=1.5)
        
        ax.text(7, 9.3, '模块调用关系图', fontproperties=self.font_title,
                ha='center', va='center', color='#2C3E50', fontsize=20, fontweight='bold')
        
        filepath = self.output_dir / output_path
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Module relationship diagram saved to {filepath}")
        
        plt.close()


def generate_architecture_diagrams():
    """生成所有架构图"""
    visualizer = ArchitectureVisualizer()
    print("Generating architecture diagrams...")
    
    visualizer.draw_architecture_diagram()
    visualizer.draw_module_relationship()
    
    print("Architecture diagrams generated successfully!")


if __name__ == '__main__':
    generate_architecture_diagrams()
