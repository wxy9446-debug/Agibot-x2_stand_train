# lingxi_x2_robot_cfg.py
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

# ====================== 核心：这里定义了模型地址 ======================
# 替换成你电脑上的绝对路径！！！
# 例如：/home/xingyun/X2_URDF-v1.3.0/x2_ultra/x2_ultra.usd
X2_USD_ABS_PATH = "/home/你的用户名/X2_URDF-v1.3.0/x2_ultra/x2_ultra.usd"

@configclass
class LingxiX2Cfg(ArticulationCfg):
    """智元灵犀X2机器人资产配置"""
    
    # 机器人在仿真中的Prim路径模板
    prim_path = "/World/envs/env_.*/Robot"
    
    # 这里指定了USD模型的地址！！！
    spawn = ArticulationCfg.SpawnCfg(
        asset_path=X2_USD_ABS_PATH,  # 模型地址就在这里
        articulation_root_xform=True,
    )

    # 初始站立姿态（后续用脚本拿到关节名后补全）
    init_state = ArticulationCfg.InitStateCfg(
        pos=(0.0, 0.0, 0.95),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={},  # 先空着，运行关节检测脚本后填充
        joint_vel={".*": 0.0},
    )

    # 执行器配置
    actuators = {
        "x2_actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=120.0,
            damping=10.0,
            effort_limit=300.0,
            velocity_limit=12.0,
        ),
    }

# 实例化配置
LINGXI_X2_CFG = LingxiX2Cfg()
