# lingxi_x2_env_cfg.py
from isaaclab.utils import configclass
# 假设你有一个H1的基础配置作为父类
from .rough_env_cfg import H1RoughEnvCfg
# 导入我们刚才定义的灵犀X2机器人配置
from .lingxi_x2_robot_cfg import LINGXI_X2_CFG

@configclass
class LingxiX2FlatEnvCfg(H1RoughEnvCfg):
    def __post_init__(self):
        # 调用父类初始化
        super().__post_init__()

        # ====================== 核心：把机器人换成灵犀X2 ======================
        self.scene.robot = LINGXI_X2_CFG  # 这里就把模型地址传进去了！

        # 切换为平地
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # 关闭高度扫描
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # 关闭地形课程学习
        self.curriculum.terrain_levels = None

        # 适配X2的奖励函数（后续填充足部名称）
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6
        self.rewards.feet_air_time.params["foot_names"] = ["left_foot", "right_foot"]

# 推理/演示配置
class LingxiX2FlatEnvCfg_PLAY(LingxiX2FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        # 小场景演示
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # 关闭随机化
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
