# actor_critic.py —— 灵犀X2 Actor-Critic网络
# 移植自 agibot_x1_train/humanoid/algo/ppo/actor_critic_dh.py
#
# 结构:
#   Actor: 短历史观测 + CNN长历史压缩 + 状态估计器速度 → MLP → 动作
#   Critic: 特权观测 → MLP → 价值
#   StateEstimator: 短历史 → MLP → 估计线速度(3维)

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticX2(nn.Module):
    """
    X2行走PPO Actor-Critic网络

    参数说明:
        num_short_obs:     短历史观测维度 (num_single_obs * short_frame_stack)
        num_single_obs:    单步观测维度 (用于CNN输入通道数)
        num_critic_obs:    Critic输入维度 (特权观测)
        num_actions:       动作维度 (12)
        actor_hidden_dims: Actor MLP隐层尺寸
        critic_hidden_dims: Critic MLP隐层尺寸
        state_estimator_hidden_dims: 状态估计器隐层尺寸
        in_channels:       CNN输入通道数 (= frame_stack / 长历史步数)
        kernel_size:       CNN各层卷积核大小
        filter_size:       CNN各层滤波器数量
        stride_size:       CNN各层步长
        lh_output_dim:     CNN输出压缩维度
        init_noise_std:    初始动作噪声标准差
    """

    def __init__(
        self,
        num_short_obs,
        num_single_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(768, 256, 128),
        state_estimator_hidden_dims=(256, 128, 64),
        in_channels=10,
        kernel_size=(6, 4),
        filter_size=(32, 16),
        stride_size=(3, 2),
        lh_output_dim=64,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(f"[ActorCriticX2] 忽略未知参数: {list(kwargs.keys())}")
        super().__init__()

        activation = nn.ELU()

        # ── Actor 输入: 短历史 + 状态估计速度(3) + CNN长历史压缩(lh_output_dim)
        mlp_input_dim_a = num_short_obs + 3 + lh_output_dim
        mlp_input_dim_c = num_critic_obs

        # ── Actor MLP ──────────────────────────────────────────────────────────
        actor_layers = [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), activation]
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
            else:
                actor_layers += [
                    nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]),
                    activation,
                ]
        self.actor = nn.Sequential(*actor_layers)

        # ── Critic MLP ─────────────────────────────────────────────────────────
        critic_layers = [nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]), activation]
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers += [
                    nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]),
                    activation,
                ]
        self.critic = nn.Sequential(*critic_layers)

        # ── 长历史 CNN ─────────────────────────────────────────────────────────
        # 输入形状: (batch, in_channels, num_single_obs)
        cnn_layers = []
        cnn_ch = in_channels
        cnn_len = num_single_obs
        for out_ch, ks, ss in zip(filter_size, kernel_size, stride_size):
            cnn_layers += [
                nn.Conv1d(cnn_ch, out_ch, kernel_size=ks, stride=ss),
                nn.ReLU(),
            ]
            cnn_len = (cnn_len - ks) // ss + 1
            cnn_ch = out_ch
        cnn_out_dim = cnn_len * cnn_ch
        cnn_layers += [
            nn.Flatten(),
            nn.Linear(cnn_out_dim, 128),
            nn.ELU(),
            nn.Linear(128, lh_output_dim),
        ]
        self.long_history_cnn = nn.Sequential(*cnn_layers)

        # ── 状态估计器 MLP ─────────────────────────────────────────────────────
        # 输入: 短历史观测, 输出: 估计线速度(3维)
        se_layers = [nn.Linear(num_short_obs, state_estimator_hidden_dims[0]), activation]
        for i in range(len(state_estimator_hidden_dims)):
            if i == len(state_estimator_hidden_dims) - 1:
                se_layers.append(nn.Linear(state_estimator_hidden_dims[i], 3))
            else:
                se_layers += [
                    nn.Linear(state_estimator_hidden_dims[i], state_estimator_hidden_dims[i + 1]),
                    activation,
                ]
        self.state_estimator = nn.Sequential(*se_layers)

        # ── 动作噪声 ───────────────────────────────────────────────────────────
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        # 保存维度信息
        self.num_short_obs = num_short_obs
        self.num_single_obs = num_single_obs
        self.in_channels = in_channels

        print(f"[ActorCriticX2] Actor: {self.actor}")
        print(f"[ActorCriticX2] Critic: {self.critic}")
        print(f"[ActorCriticX2] CNN长历史: {self.long_history_cnn}")
        print(f"[ActorCriticX2] 状态估计器: {self.state_estimator}")

    # ──────────────────────────────────────────────────────────────────────────
    # 工具属性
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        """兼容接口，无RNN所以不需要重置隐状态"""
        pass

    def forward(self):
        raise NotImplementedError

    # ──────────────────────────────────────────────────────────────────────────
    # Actor 推断
    # ──────────────────────────────────────────────────────────────────────────
    def _build_actor_input(self, observations):
        """
        从完整观测中提取Actor输入:
          短历史 = observations最后 num_short_obs 维
          长历史 = 整个observations reshape为 (B, in_channels, num_single_obs)
        """
        short_hist = observations[..., -self.num_short_obs:]               # (B, num_short_obs)
        est_vel = self.state_estimator(short_hist)                          # (B, 3)
        long_hist = self.long_history_cnn(
            observations.view(-1, self.in_channels, self.num_single_obs)   # (B, C, L)
        )                                                                   # (B, lh_output_dim)
        return torch.cat([short_hist, est_vel, long_hist], dim=-1)

    def update_distribution(self, observations):
        actor_input = self._build_actor_input(observations)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        """训练时采样动作"""
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        """推断时用均值动作（确定性）"""
        actor_input = self._build_actor_input(observations)
        return self.actor(actor_input)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # ──────────────────────────────────────────────────────────────────────────
    # Critic 推断
    # ──────────────────────────────────────────────────────────────────────────
    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
