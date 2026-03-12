"""
灵犀X2人形机器人步态训练 - Isaac Lab + PPO + Transformer策略
完整训练脚本：历史帧堆叠 + 特权观测Critic + 状态估计器 + GAE + 自适应学习率
"""

from isaaclab.app import AppLauncher
import argparse
import os

parser = argparse.ArgumentParser(description="X2步态PPO训练")
parser.add_argument("--num_envs", type=int, default=4096, help="并行环境数量")
parser.add_argument("--max_iterations", type=int, default=5000, help="最大训练迭代数")
parser.add_argument("--log_dir", type=str, default="logs/x2_walk_ppo", help="日志目录")
parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")
# parser.add_argument("--headless", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Lab imports（必须在app launch之后）─────────────────────────────────
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.noise import GaussianNoiseCfg

# ── PPO模块 ───────────────────────────────────────────────────────────────────
from ppo_x2 import ActorCriticX2, X2PPO, X2OnPolicyRunner

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 关节定义 ──────────────────────────────────────────────────────────────────
LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

# ── 观测维度 ──────────────────────────────────────────────────────────────────
# 单帧: ang_vel(3) + proj_gravity(3) + cmd(3) + joint_pos(12) + joint_vel(12) + last_action(12) = 45
NUM_SINGLE_OBS = 45
FRAME_STACK = 10          # 历史帧数
NUM_OBS = NUM_SINGLE_OBS * FRAME_STACK       # actor观测 = 450
NUM_SHORT_OBS = NUM_SINGLE_OBS * 2           # 短历史用于状态估计 = 90

# Critic特权观测单帧: lin_vel(3)+ang_vel(3)+proj_gravity(3)+cmd(3)+joint_pos(12)+joint_vel(12)+actions(12) = 48
NUM_PRIVILEGED_OBS = 48
NUM_ACTIONS = 12


# ══════════════════════════════════════════════════════════════════════════════
# 环境配置
# ══════════════════════════════════════════════════════════════════════════════

@configclass
class ObservationsCfg:
    """观测配置（Policy + Critic特权观测）"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor观测 —— 无速度测量（依赖状态估计器推断）"""
        concatenate_terms = True
        # 姿态
        base_ang_vel    = ObsTerm(func=mdp.base_ang_vel,       noise=GaussianNoiseCfg(std=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoiseCfg(std=0.02))
        # 速度命令
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节状态（只取腿部12关节）
        joint_pos       = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoiseCfg(std=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
        )
        joint_vel       = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoiseCfg(std=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
        )
        # 上一步动作
        actions         = ObsTerm(func=mdp.last_action)

    @configclass
    class CriticCfg(ObsGroup):
        """Critic特权观测 —— 包含真实线速度（训练时用）"""
        concatenate_terms = True
        # 真实线速度（训练时可获得）
        base_lin_vel    = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel    = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节状态（只取腿部12关节）
        joint_pos       = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
        )
        joint_vel       = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
        )
        actions         = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    """
    完整奖励函数 —— 参考 agibot_x1_train 奖励设计
    正向奖励: 跟踪速度、存活
    惩罚项: 不平衡、抖动、不对称步态
    """
    # ── 正奖励 ────────────────────────────────────────────────────────────────
    alive           = RewTerm(func=mdp.is_alive,                    weight=1.0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # ── 惩罚 ──────────────────────────────────────────────────────────────────
    terminating     = RewTerm(func=mdp.is_terminated,              weight=-200.0)
    lin_vel_z       = RewTerm(func=mdp.lin_vel_z_l2,               weight=-2.0)
    ang_vel_xy      = RewTerm(func=mdp.ang_vel_xy_l2,              weight=-0.05)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2,       weight=-5.0)
    action_rate     = RewTerm(func=mdp.action_rate_l2,             weight=-0.01)
    joint_acc       = RewTerm(func=mdp.joint_acc_l2,               weight=-2.5e-7)
    joint_vel       = RewTerm(func=mdp.joint_vel_l2,               weight=-0.0)
    # 关节限位惩罚
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits,          weight=-1.0)
    # 足底碰撞以外的接触惩罚
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["pelvis", "torso_link", ".*knee.*"]),
            "threshold": 1.0},
    )
    # 足底腾空时间奖励（鼓励步态）
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])
        }
    )


@configclass
class CommandsCfg:
    """速度命令生成"""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["pelvis", "torso_link"]),
            "threshold": 1.0},
    )
    # 倾斜过大则终止
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0},
    )


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # 随机推力干扰（增强鲁棒性）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
        }},
    )
    # 随机关节初始化
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.0, 0.0),
        },
    )


@configclass
class X2WalkEnvCfg(ManagerBasedRLEnvCfg):
    """X2完整步态训练环境配置"""

    scene:       InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg   = ObservationsCfg()
    actions:     ActionsCfg         = ActionsCfg()
    rewards:     RewardsCfg         = RewardsCfg()
    terminations: TerminationsCfg   = TerminationsCfg()
    events:      EventsCfg          = EventsCfg()
    commands:    CommandsCfg        = CommandsCfg()

    def __post_init__(self):
        self.decimation = 4                  # 200Hz物理 / 4 = 50Hz控制
        self.episode_length_s = 20.0
        self.sim = SimulationCfg(dt=1 / 200, render_interval=self.decimation)

        # 地面
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )

        # X2机器人
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(
                    CURRENT_DIR,
                    "x2_ultra_simple_collision/x2_ultra_simple_collision.usd",
                ),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.95),
                joint_pos={
                    "left_hip_pitch_joint":   -0.15,
                    "right_hip_pitch_joint":  -0.15,
                    "left_knee_joint":         0.35,
                    "right_knee_joint":        0.35,
                    "left_ankle_pitch_joint": -0.20,
                    "right_ankle_pitch_joint":-0.20,
                },
            ),
            actuators={
                "hip_pitch": ImplicitActuatorCfg(
                    joint_names_expr=[".*_hip_pitch_joint"],
                    stiffness=200.0, damping=5.0,
                    effort_limit=150.0,
                ),
                "hip_roll_yaw": ImplicitActuatorCfg(
                    joint_names_expr=[".*_hip_roll_joint", ".*_hip_yaw_joint"],
                    stiffness=150.0, damping=5.0,
                    effort_limit=100.0,
                ),
                "knee": ImplicitActuatorCfg(
                    joint_names_expr=[".*_knee_joint"],
                    stiffness=200.0, damping=5.0,
                    effort_limit=200.0,
                ),
                "ankle": ImplicitActuatorCfg(
                    joint_names_expr=[".*_ankle_.*_joint"],
                    stiffness=40.0, damping=2.0,
                    effort_limit=40.0,
                ),
            },
        )

        # 接触力传感器
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 训练配置
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_CFG = {
    "env": {
        "num_observations":    NUM_OBS,
        "num_privileged_obs":  NUM_PRIVILEGED_OBS,
        "num_actions":         NUM_ACTIONS,
        "num_single_obs":      NUM_SINGLE_OBS,
        "num_short_obs":       NUM_SHORT_OBS,
        "frame_stack":         FRAME_STACK,
    },
    "policy": {
        # Transformer Actor
        "actor_hidden_dims":   [512, 256],
        "critic_hidden_dims":  [512, 256],
        "activation":          "elu",
        "init_noise_std":      1.0,
        # Transformer配置
        "transformer_num_heads":  4,
        "transformer_num_layers": 2,
        "transformer_dim_feedforward": 256,
    },
    "algorithm": {
        "clip_param":          0.2,
        "num_learning_epochs": 2,
        "num_mini_batches":    4,
        "value_loss_coef":     1.0,
        "entropy_coef":        0.001,
        "gamma":               0.994,
        "lam":                 0.9,
        "learning_rate":       1e-4,
        "max_grad_norm":       1.0,
        "use_clipped_value_loss": True,
        "schedule":            "adaptive",
        "desired_kl":          0.01,
        "lin_vel_idx":         0,      # 特权观测中lin_vel的起始索引
    },
    "runner": {
        "num_steps_per_env":   24,
        "save_interval":       100,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 观测历史堆叠Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class FrameStackWrapper:
    """
    历史帧堆叠Wrapper
    将最近 FRAME_STACK 帧的actor观测拼接成一个大向量
    同时保持critic特权观测不变（但补0到同样长度以兼容）
    """

    def __init__(self, env, frame_stack: int, num_single_obs: int, device: str):
        self.env = env
        self.frame_stack = frame_stack
        self.num_single_obs = num_single_obs
        self.device = device
        self.num_envs = env.num_envs
        self.num_actions = NUM_ACTIONS

        # 历史缓冲区 [num_envs, frame_stack, num_single_obs]
        self._buf = torch.zeros(
            self.num_envs, frame_stack, num_single_obs, device=device
        )

    @property
    def unwrapped(self):
        return self.env

    def reset(self):
        obs_dict, extras = self.env.reset()
        single_obs = obs_dict["policy"]          # (num_envs, num_single_obs)
        self._buf.zero_()
        self._buf[:, -1, :] = single_obs
        stacked = self._buf.flatten(1)           # (num_envs, frame_stack*num_single_obs)
        return {"policy": stacked, "critic": obs_dict.get("critic", stacked)}, extras

    def step(self, actions):
        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
        single_obs = obs_dict["policy"]

        # 滚动历史
        self._buf = torch.roll(self._buf, -1, dims=1)
        self._buf[:, -1, :] = single_obs

        # 重置已终止环境的历史
        done = (terminated | truncated).squeeze(-1)
        if done.any():
            self._buf[done] = 0.0
            self._buf[done, -1, :] = single_obs[done]

        stacked = self._buf.flatten(1)
        new_obs = {"policy": stacked, "critic": obs_dict.get("critic", stacked)}
        return new_obs, rewards, terminated, truncated, infos


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    from datetime import datetime

    # ── 构建环境 ──────────────────────────────────────────────────────────────
    env_cfg = X2WalkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    base_env = ManagerBasedRLEnv(cfg=env_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 历史帧堆叠Wrapper ─────────────────────────────────────────────────────
    env = FrameStackWrapper(
        base_env,
        frame_stack=FRAME_STACK,
        num_single_obs=NUM_SINGLE_OBS,
        device=device,
    )

    # ── 日志目录 ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(CURRENT_DIR, args_cli.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[Train] 日志目录: {log_dir}")
    print(f"[Train] 设备: {device}")
    print(f"[Train] 并行环境: {args_cli.num_envs}")
    print(f"[Train] 最大迭代: {args_cli.max_iterations}")
    print(f"[Train] Actor观测维度: {NUM_OBS} (={FRAME_STACK}帧x{NUM_SINGLE_OBS})")
    print(f"[Train] Critic特权观测维度: {NUM_PRIVILEGED_OBS}")
    print(f"[Train] 动作维度: {NUM_ACTIONS}")

    # ── 构建Runner ────────────────────────────────────────────────────────────
    runner = X2OnPolicyRunner(
        env=env,
        train_cfg=TRAIN_CFG,
        log_dir=log_dir,
        device=device,
    )

    # ── 恢复训练 ──────────────────────────────────────────────────────────────
    if args_cli.resume is not None:
        runner.load(args_cli.resume)
        print(f"[Train] 从checkpoint恢复: {args_cli.resume}")

    # ── 开始训练 ──────────────────────────────────────────────────────────────
    runner.learn(num_learning_iterations=args_cli.max_iterations)

    # ── 清理 ──────────────────────────────────────────────────────────────────
    base_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
