# runner.py —— X2 On-Policy PPO训练Runner
# 移植自 agibot_x1_train/humanoid/algo/ppo/dh_on_policy_runner.py
# 适配 Isaac Lab ManagerBasedRLEnv 接口

import os
import time
import statistics
from collections import deque
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from .actor_critic import ActorCriticX2
from .ppo import X2PPO


class X2OnPolicyRunner:
    """
    X2 PPO On-Policy训练Runner
    适配 Isaac Lab ManagerBasedRLEnv:
      - env.observation_manager.compute() → 返回 dict
      - env.step(actions) → obs_dict, rewards, terminated, truncated, info
      - env.num_envs, env.num_actions
    """

    def __init__(self, env, train_cfg: dict, log_dir: str = None, device: str = "cpu"):
        self.env = env
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.log_dir = log_dir

        # ── 从env读取维度 ──────────────────────────────────────────────────────
        num_obs = train_cfg["env"]["num_observations"]
        num_privileged_obs = train_cfg["env"].get("num_privileged_obs", num_obs)
        num_actions = train_cfg["env"]["num_actions"]
        num_single_obs = train_cfg["env"]["num_single_obs"]
        num_short_obs = train_cfg["env"]["num_short_obs"]
        in_channels = train_cfg["env"]["frame_stack"]

        # ── 构建Actor-Critic ───────────────────────────────────────────────────
        self.actor_critic = ActorCriticX2(
            num_short_obs=num_short_obs,
            num_single_obs=num_single_obs,
            num_critic_obs=num_privileged_obs,
            num_actions=num_actions,
            in_channels=in_channels,
            **self.policy_cfg,
        ).to(self.device)

        # ── 构建PPO算法 ────────────────────────────────────────────────────────
        self.alg = X2PPO(
            self.actor_critic,
            device=self.device,
            **self.alg_cfg,
        )

        # ── 初始化存储 ─────────────────────────────────────────────────────────
        num_steps = self.cfg["num_steps_per_env"]
        self.num_steps_per_env = num_steps
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            env.num_envs,
            num_steps,
            [num_obs],
            [num_privileged_obs],
            [num_actions],
        )

        # ── 日志 ───────────────────────────────────────────────────────────────
        self.writer: SummaryWriter = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self.it = 0

        # 维度缓存（供log使用）
        self.num_obs = num_obs

    # ──────────────────────────────────────────────────────────────────────────
    # 主训练循环
    # ──────────────────────────────────────────────────────────────────────────
    def learn(self, num_learning_iterations: int):
        # 初始化TensorBoard
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # 获取初始观测 (Isaac Lab reset返回 (obs_dict, extras) tuple)
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs_dict = reset_result[0]
        else:
            obs_dict = reset_result
        obs, privileged_obs = self._extract_obs(obs_dict)
        obs = obs.to(self.device)
        privileged_obs = privileged_obs.to(self.device)

        self.alg.train_mode()

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)
        ep_infos = []

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            self.it = it
            t_start = time.time()

            obs_std, obs_mean = torch.std_mean(obs, dim=0)

            # ── 数据收集 (Rollout) ─────────────────────────────────────────────
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs)

                    # Isaac Lab step: 返回 (obs_dict, reward, terminated, truncated, info)
                    step_result = self.env.step(actions)
                    obs_dict, rewards, terminated, truncated, infos = step_result

                    obs, privileged_obs = self._extract_obs(obs_dict)
                    obs = obs.to(self.device)
                    privileged_obs = privileged_obs.to(self.device)
                    rewards = rewards.to(self.device).squeeze(-1)
                    dones = (terminated | truncated).to(self.device).squeeze(-1)

                    # 时间截断标记（用于bootstrap）
                    infos["time_outs"] = truncated.squeeze(-1).to(self.device)

                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids[:, 0]].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids[:, 0]].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids[:, 0]] = 0.0
                        cur_episode_length[new_ids[:, 0]] = 0.0

            t_collect = time.time() - t_start

            # ── 计算Returns ────────────────────────────────────────────────────
            privileged_obs = privileged_obs.clone()
            self.alg.compute_returns(privileged_obs)

            # ── PPO更新 ────────────────────────────────────────────────────────
            t_learn_start = time.time()
            v_loss, surr_loss, se_loss = self.alg.update()
            t_learn = time.time() - t_learn_start

            # ── 日志 & 保存 ────────────────────────────────────────────────────
            if self.log_dir is not None:
                self._log(
                    it=it,
                    num_learning_iterations=num_learning_iterations,
                    collection_time=t_collect,
                    learn_time=t_learn,
                    mean_value_loss=v_loss,
                    mean_surrogate_loss=surr_loss,
                    mean_se_loss=se_loss,
                    rewbuffer=rewbuffer,
                    lenbuffer=lenbuffer,
                    ep_infos=ep_infos,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                )
            ep_infos.clear()

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────────────────────
    def _extract_obs(self, obs_dict):
        """
        从Isaac Lab observation dict中提取 (actor_obs, critic_obs)
        Isaac Lab通常有 'policy' 和 'critic' 两个key
        """
        if isinstance(obs_dict, dict):
            actor_obs = obs_dict.get("policy", list(obs_dict.values())[0])
            critic_obs = obs_dict.get("critic", actor_obs)
        else:
            # 直接是tensor
            actor_obs = obs_dict
            critic_obs = obs_dict
        return actor_obs, critic_obs

    def _log(
        self,
        it,
        num_learning_iterations,
        collection_time,
        learn_time,
        mean_value_loss,
        mean_surrogate_loss,
        mean_se_loss,
        rewbuffer,
        lenbuffer,
        ep_infos,
        obs_mean,
        obs_std,
    ):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        iteration_time = collection_time + learn_time

        fps = int(
            self.num_steps_per_env * self.env.num_envs / (collection_time + learn_time)
        )
        mean_std = self.alg.actor_critic.std.mean().item()

        # TensorBoard
        self.writer.add_scalar("Loss/value_function", mean_value_loss, it)
        self.writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
        self.writer.add_scalar("Loss/state_estimator", mean_se_loss, it)
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, it)
        self.writer.add_scalar("Policy/mean_noise_std", mean_std, it)
        self.writer.add_scalar("Perf/fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collection_time, it)
        self.writer.add_scalar("Perf/learn_time", learn_time, it)

        if len(rewbuffer) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(rewbuffer), it
            )
            self.writer.add_scalar(
                "Train/mean_episode_length", statistics.mean(lenbuffer), it
            )

        # 逐维度观测统计（前48维）
        log_obs_dim = min(self.num_obs, 48)
        for i in range(log_obs_dim):
            self.writer.add_scalar(f"Obs/mean_{i}", obs_mean[i].item(), it)
            self.writer.add_scalar(f"Obs/std_{i}", obs_std[i].item(), it)

        # episode指标
        ep_string = ""
        if ep_infos:
            for key in ep_infos[0]:
                vals = torch.tensor(
                    [ep[key] for ep in ep_infos if key in ep], dtype=torch.float
                )
                v = vals.mean().item()
                self.writer.add_scalar(f"Episode/{key}", v, it)
                ep_string += f"  {key}: {v:.4f}\n"

        # 控制台输出
        width = 80
        pad = 35
        log_str = (
            f"{'#'*width}\n"
            f"{f' Iter {it}/{self.current_learning_iteration+num_learning_iterations} ':^{width}}\n\n"
            f"{'FPS:':>{pad}} {fps}\n"
            f"{'Value loss:':>{pad}} {mean_value_loss:.4f}\n"
            f"{'Surrogate loss:':>{pad}} {mean_surrogate_loss:.4f}\n"
            f"{'SE loss:':>{pad}} {mean_se_loss:.4f}\n"
            f"{'Noise std:':>{pad}} {mean_std:.3f}\n"
        )
        if len(rewbuffer) > 0:
            log_str += (
                f"{'Mean reward:':>{pad}} {statistics.mean(rewbuffer):.2f}\n"
                f"{'Mean ep length:':>{pad}} {statistics.mean(lenbuffer):.1f}\n"
            )
        log_str += ep_string
        log_str += (
            f"{'-'*width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iter time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.1f}s\n"
            f"{'ETA:':>{pad}} {self.tot_time/(it+1)*(num_learning_iterations-it):.0f}s\n"
        )
        print(log_str)

    # ──────────────────────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "se_optimizer_state_dict": self.alg.state_estimator_optimizer.state_dict(),
                "iter": self.it,
            },
            path,
        )
        print(f"[Runner] 模型已保存: {path}")

    def load(self, path: str, load_optimizer: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(ckpt["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.alg.state_estimator_optimizer.load_state_dict(
                ckpt["se_optimizer_state_dict"]
            )
        self.current_learning_iteration = ckpt.get("iter", 0)
        print(f"[Runner] 模型已加载: {path} (iter={self.current_learning_iteration})")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
