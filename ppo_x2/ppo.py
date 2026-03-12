# ppo.py —— X2 PPO算法核心
# 移植自 agibot_x1_train/humanoid/algo/ppo/dh_ppo.py
# 主要改动: 适配 Isaac Lab ManagerBasedRLEnv 的观测格式

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic import ActorCriticX2
from .rollout_storage import RolloutStorage


class X2PPO:
    """
    X2 PPO算法
    包含:
      - Actor-Critic参数更新 (surrogate loss + value loss + entropy bonus)
      - 状态估计器更新 (估计线速度 vs Critic特权观测中的真实线速度)
      - 自适应学习率 (KL散度控制)
      - GAE优势估计
    """

    actor_critic: ActorCriticX2

    def __init__(
        self,
        actor_critic: ActorCriticX2,
        # PPO超参数
        clip_param: float = 0.2,
        num_learning_epochs: int = 2,
        num_mini_batches: int = 4,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.001,
        gamma: float = 0.994,
        lam: float = 0.9,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        # 自适应学习率
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        # 状态估计器: 在特权观测中线速度的起始索引
        lin_vel_idx: int = 0,
        device: str = "cpu",
        **kwargs,
    ):
        self.device = device
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.lin_vel_idx = lin_vel_idx

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        # 两个优化器: 一个主优化器, 一个状态估计器专用
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=learning_rate
        )
        self.state_estimator_optimizer = optim.Adam(
            self.actor_critic.state_estimator.parameters(), lr=learning_rate
        )

        self.storage: RolloutStorage = None  # 在init_storage后初始化
        self.transition = RolloutStorage.Transition()

    # ──────────────────────────────────────────────────────────────────────────
    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            device=self.device,
        )

    def train_mode(self):
        self.actor_critic.train()

    def test_mode(self):
        self.actor_critic.eval()

    # ──────────────────────────────────────────────────────────────────────────
    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """前向推断，返回采样动作并缓存过渡数据"""
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # 保存观测用于后续更新
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
    ):
        """处理环境step返回值，添加到存储"""
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # 时间截断的Bootstrap: 超时但未失败的环境继续计算值
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # ──────────────────────────────────────────────────────────────────────────
    def update(self):
        """PPO更新，返回 (value_loss, surrogate_loss, state_estimator_loss)"""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_se_loss = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            _target_values_batch,
            advantages_batch,
            returns_batch,
            old_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _hid,
            _mask,
        ) in generator:

            # ── 前向传播 ────────────────────────────────────────────────────
            self.actor_critic.act(obs_batch)
            actions_log_prob = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # ── 状态估计器: 估计线速度 vs 特权观测中的真实线速度 ─────────────
            short_hist = obs_batch[..., -self.actor_critic.num_short_obs :]
            est_lin_vel = self.actor_critic.state_estimator(short_hist)   # (B, 3)
            ref_lin_vel = critic_obs_batch[
                :, self.lin_vel_idx : self.lin_vel_idx + 3
            ].clone()
            se_loss = nn.functional.mse_loss(est_lin_vel, ref_lin_vel)

            # ── 自适应学习率 (KL散度) ────────────────────────────────────────
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = kl.mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            # ── Surrogate loss (PPO-clip) ────────────────────────────────────
            ratio = torch.exp(
                actions_log_prob - torch.squeeze(old_log_prob_batch)
            )
            adv = torch.squeeze(advantages_batch)
            surrogate = -adv * ratio
            surrogate_clipped = -adv * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ── Value loss ──────────────────────────────────────────────────
            if self.use_clipped_value_loss:
                value_clipped = _target_values_batch + (
                    value_batch - _target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                v_loss = (value_batch - returns_batch).pow(2)
                v_loss_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(v_loss, v_loss_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ── 总损失 & 反向传播 ───────────────────────────────────────────
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + se_loss  # 联合优化状态估计器
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_se_loss += se_loss.item()

        n = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return mean_value_loss / n, mean_surrogate_loss / n, mean_se_loss / n
