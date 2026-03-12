# 灵犀X2人形机器人 Isaac Lab PPO步态训练

## 项目结构

```
sucess_stand train/
├── train_x2_walk_ppo.py          # 主训练脚本（环境配置 + 训练入口）
├── ppo_x2/                       # PPO算法模块
│   ├── __init__.py               # 模块导出
│   ├── actor_critic.py           # Actor-Critic网络 (CNN历史压缩 + 状态估计器)
│   ├── ppo.py                    # PPO核心算法 (GAE + 自适应学习率)
│   ├── runner.py                 # 训练Runner (数据收集 + 日志)
│   └── rollout_storage.py        # 经验回放存储
├── ppo/                          # 旧版PPO（含Transformer策略，备用）
│   ├── transformer_policy.py     # Transformer大模型策略
│   └── ...
├── x2_ultra_simple_collision/    # X2机器人USD模型
└── logs/                         # 训练日志（自动创建）
    └── x2_walk_ppo/
```

## 算法架构

### Actor-Critic网络 (`ppo_x2/actor_critic.py`)
- **Actor**:
  - 短历史观测 (2帧 × 45维 = 90维) → 状态估计器 → 估计线速度(3维)
  - 全历史观测 (10帧 × 45维) → CNN长历史压缩 → 64维特征
  - [短历史 + 估计速度 + CNN特征] → MLP(512→256→12) → 动作
- **Critic**:
  - 特权观测(48维): 真实线速度 + 姿态 + 关节状态 → MLP(768→256→1) → 价值
- **状态估计器**:
  - 短历史 → MLP(256→128→64→3) → 估计线速度
  - 与Critic特权观测中的真实线速度做MSE监督

### 观测空间
| 项目 | 维度 | 说明 |
|------|------|------|
| Actor观测(单帧) | 45 | ang_vel(3)+gravity(3)+cmd(3)+joint_pos(12)+joint_vel(12)+action(12) |
| Actor观测(堆叠10帧) | 450 | 历史帧拼接 |
| Critic特权观测 | 48 | 加入真实lin_vel(3) |

### 动作空间
- 腿部12关节位置控制（髋部×6 + 膝部×2 + 踝部×4）

### 奖励函数
| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `alive` | +1.0 | 存活奖励 |
| `track_lin_vel_xy` | +2.0 | 跟踪线速度命令 |
| `track_ang_vel_z` | +1.0 | 跟踪偏航角速度命令 |
| `terminating` | -200 | 终止惩罚 |
| `lin_vel_z` | -2.0 | 抑制垂直振动 |
| `ang_vel_xy` | -0.05 | 抑制滚转/俯仰抖动 |
| `flat_orientation` | -5.0 | 保持躯干水平 |
| `action_rate` | -0.01 | 动作平滑性 |
| `joint_acc` | -2.5e-7 | 抑制关节加速度 |
| `joint_pos_limits` | -1.0 | 关节限位 |
| `undesired_contact` | -1.0 | 非足底碰撞 |
| `foot_air_time` | +0.5 | 步态质量（腾空时间） |

## 快速开始

### 1. 安装依赖
```bash
# 确保已安装 Isaac Lab
pip install isaaclab

# 安装 tensorboard（用于训练监控）
pip install tensorboard
```

### 2. 开始训练
```bash
cd "/home/xingyun/桌面/X2_URDF-v1.3.0/sucess_stand train"

# 基础训练（4096个并行环境）
python train_x2_walk_ppo.py --num_envs 4096 --max_iterations 5000 --headless

# 小规模调试（减少内存占用）
python train_x2_walk_ppo.py --num_envs 64 --max_iterations 100 --headless

# 恢复训练
python train_x2_walk_ppo.py --resume logs/x2_walk_ppo/YYYYMMDD_HHMMSS/model_1000.pt
```

### 3. 监控训练
```bash
tensorboard --logdir "/home/xingyun/桌面/X2_URDF-v1.3.0/sucess_stand train/logs/x2_walk_ppo"
```

### 4. 使用启动脚本
```bash
chmod +x run_train_ppo.sh
./run_train_ppo.sh
```

## 关键超参数

```python
# train_x2_walk_ppo.py 中的 TRAIN_CFG
TRAIN_CFG = {
    "algorithm": {
        "clip_param":          0.2,      # PPO裁剪系数
        "num_learning_epochs": 2,        # 每次rollout的学习轮数
        "num_mini_batches":    4,        # mini-batch数量
        "gamma":               0.994,    # 折扣因子
        "lam":                 0.9,      # GAE λ
        "learning_rate":       1e-4,     # 初始学习率
        "schedule":            "adaptive", # 自适应学习率（基于KL散度）
        "desired_kl":          0.01,     # 目标KL散度
    },
    "runner": {
        "num_steps_per_env":   24,       # 每个环境每次rollout的步数
        "save_interval":       100,      # 每100次迭代保存一次
    },
}
```

## 训练提示

1. **初始阶段（0-500迭代）**: 机器人学会站立平衡，`flat_orientation`损失下降
2. **中期（500-2000迭代）**: 开始产生步态，`foot_air_time`奖励增加
3. **后期（2000+迭代）**: 速度跟踪精度提升，步态平滑

## 文件说明

- `x2_ultra_simple_collision/x2_ultra_simple_collision.usd`: X2机器人USD模型（简化碰撞体）
- `lingxi_x2_env_cfg.py`: 旧版环境配置（参考用）
- `lingxi_x2_robot_cfg.py`: 旧版机器人配置（参考用）
