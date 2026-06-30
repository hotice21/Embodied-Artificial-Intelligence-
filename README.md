# Embodied Artificial Intelligence - 具身人工智能游戏AI框架

基于视觉检测和强化学习的具身人工智能游戏AI训练框架，支持实时游戏画面分析、敌人识别、受伤检测和自主决策。

---

## 🌟 项目特性

- **视觉检测系统**：实时检测玩家、敌人、子弹、地面、墙体等游戏元素
- **增强玩家检测**：多模态融合检测，提高识别准确率
- **敌人预测器**：预测敌人轨迹和危险区域
- **受伤检测**：支持消失-出现模式、位置跳跃、重叠检测等多种方法
- **PPO强化学习**：GPU加速的深度强化学习训练器
- **遗传算法**：基于动作序列优化的进化算法
- **可视化工具**：训练曲线、架构图、检测效果展示
- **手动触发**：支持H键手动标记受伤事件

---

## 📁 目录结构

```
SerpentAI-unsupervised/
├── config.yml                      # 主配置文件
├── train_unsupervised.py           # 主训练脚本
├── vision_detector.py              # 视觉检测模块
├── vision_worker.py                # 视觉检测异步工作器
├── player_detector.py              # 玩家检测器
├── enemy_predictor.py              # 敌人预测器
├── enhanced_vision_processor.py    # 增强视觉处理器
├── rl_trainer.py                   # RL训练器（CPU版）
├── rl_trainer_gpu.py               # RL训练器（GPU版）
├── combat_signals.py               # 战斗信号处理
├── ocr_score_watch.py              # OCR分数监控
├── control_panel.py                # 控制面板
├── architecture_visualizer.py      # 架构图生成器
├── training_visualizer.py          # 训练可视化器
├── agent.py                        # 智能体基类
├── check_deps.py                   # 依赖检查工具
├── check_image_score.py            # OCR分数检测
├── region_calibrator.py            # 屏幕区域标定工具
├── labeling_tool.py                # 标签标注工具
├── resources/                      # 资源目录
│   ├── label_bank/                 # 模板标签库
│   ├── player/                     # 玩家模板
│   ├── enemy/                      # 敌人模板
│   ├── obstacles/                  # 障碍物模板
│   └── walls_room/                 # 墙体模板
├── SerpentAI-dev/                  # SerpentAI核心模块
│   ├── serpent/                    # 游戏引擎核心
│   ├── cli.py                      # 命令行接口
│   └── config/                     # 配置文件
└── visualizations/                 # 可视化输出（自动生成）
```

---

## 🚀 快速开始

### 1. 环境要求

```bash
Python 3.8+
Windows 10/11
NVIDIA GPU (推荐，用于RL训练加速)
```

### 2. 安装依赖

```bash
# 使用清华源安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 安装PyTorch GPU版本（可选）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 依赖检查

```bash
python check_deps.py
```

### 4. 启动训练

#### 遗传算法模式（默认）
```bash
python train_unsupervised.py --episodes=200 --episode-length=300
```

#### PPO强化学习模式
```bash
python train_unsupervised.py --rl --episodes=500 --episode-length=500
```

#### 测试模式（不发送实际输入）
```bash
python train_unsupervised.py --dry-run --episodes=1 --episode-length=5
```

---

## 🧠 系统架构

### 四层架构

| 层级 | 功能 | 模块 |
|------|------|------|
| **感知层** | 视觉检测 | vision_detector, player_detector |
| **决策层** | 动作选择 | rl_trainer, 遗传算法 |
| **执行层** | 输入控制 | input_controller, game_api |
| **反馈层** | 奖励计算 | enemy_predictor, hurt_detector |

### 数据流

```
游戏画面 → 感知层 → 检测结果 → 决策层 → 动作指令 → 执行层 → 游戏环境
                              ↓
                         反馈层（奖励/惩罚）
                              ↓
                         决策层（优化策略）
```

---

## 🎯 核心功能

### 视觉检测
- 玩家检测：模板匹配 + 颜色识别 + 启发式检测
- 敌人检测：模板匹配 + 运动检测 + 特征学习
- 子弹检测：颜色追踪 + 运动分析
- 地面/墙体检测：颜色过滤 + 形状识别

### 受伤检测（三种方法）
1. **消失-出现模式**：玩家短暂消失后重新出现
2. **位置跳跃**：玩家位置突然变化（瞬移）
3. **重叠检测**：玩家与敌人/子弹碰撞

### 敌人预测
- 轨迹预测：基于历史位置外推
- 危险区域评估：预测敌人攻击范围
- 威胁等级排序：按距离和方向评估

---

## ⚙️ 配置说明

### 主要配置项（config.yml）

```yaml
screen:
  capture_region: [0, 0, 800, 600]  # 捕获区域
  vision_fps: 15                      # 视觉检测帧率

combat:
  hurt_detection_enabled: true        # 启用受伤检测
  min_disappearances: 1               # 消失-出现次数阈值
  jump_threshold: 30                  # 位置跳跃阈值（像素）

rl:
  trainer_type: ppo
  state_dim: 61
  action_dim: 9
  lr: 0.0003
  gamma: 0.99
```

---

## 📊 可视化

### 生成架构图

```bash
python architecture_visualizer.py
```

### 生成训练图表

训练过程中自动生成：
- `visualizations/score_comparison.png` - 训练得分曲线
- `visualizations/ppo_loss.png` - PPO损失曲线
- `visualizations/hurt_statistics.png` - 受伤统计
- `visualizations/architecture_diagram.png` - 系统架构图

---

## 🎮 操作说明

### 训练过程中
- **H键**：手动触发受伤检测
- **ESC键**：停止训练

---

## 📝 命令汇总

```bash
# 依赖检查
python check_deps.py

# 快速测试
python train_unsupervised.py --dry-run --episodes=1 --episode-length=5

# 遗传算法训练
python train_unsupervised.py --episodes=200 --episode-length=300

# PPO强化学习
python train_unsupervised.py --rl --episodes=500 --episode-length=500

# 区域标定
python region_calibrator.py --label player --templates "resources/player/*.png"

# 标签标注
python labeling_tool.py

# 生成架构图
python architecture_visualizer.py
```

---

## 📄 License

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

如有问题，请在 GitHub Issues 中提出。
