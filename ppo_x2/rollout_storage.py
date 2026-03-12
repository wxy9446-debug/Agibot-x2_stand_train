# rollout_storage.py —— PPO经验回放存储
# 移植自 agibot_x1_train/humanoid/algo/ppo/rollout_storage.py

import torch


class RolloutStorage:
    """PPO滚动经验存储器"""

    class Transition:
        def __init__(self):
            self.observations = None        # actor观测
            self.critic_observations = None # critic观测(特权)
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # 核心存储
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=device
        )
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=device
            )
        else:
            self.privileged_observations = None

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        ).byte()

        # PPO专用
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device
        )

        self.step = 0

    # ──────────────────────────────────────────────────────────────────────────
    def add_transitions(self, transition: "RolloutStorage.Transition"):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("RolloutStorage溢出！")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(
                transition.critic_observations
            )
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(
            transition.actions_log_prob.view(-1, 1)
        )
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    # ──────────────────────────────────────────────────────────────────────────
    def compute_returns(self, last_values, gamma, lam):
        """GAE优势估计"""
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = (
                last_values
                if step == self.num_transitions_per_env - 1
                else self.values[step + 1]
            )
            not_done = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + not_done * gamma * next_values
                - self.values[step]
            )
            advantage = delta + not_done * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # 标准化优势
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    # ──────────────────────────────────────────────────────────────────────────
    def mini_batch_generator(self, num_mini_batches, num_epochs=4):
        """随机mini-batch迭代器"""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        obs = self.observations.flatten(0, 1)
        critic_obs = (
            self.privileged_observations.flatten(0, 1)
            if self.privileged_observations is not None
            else obs
        )
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                idx = indices[start:end]
                yield (
                    obs[idx],
                    critic_obs[idx],
                    actions[idx],
                    values[idx],
                    advantages[idx],
                    returns[idx],
                    old_log_prob[idx],
                    old_mu[idx],
                    old_sigma[idx],
                    (None, None),
                    None,
                )
