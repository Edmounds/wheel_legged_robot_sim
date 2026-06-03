"""控制器参数预设与序列化工具。

所有控制器参数的唯一真值源。launch_mujoco.py、测试文件和 optimize.py
均从此模块引用参数，禁止在其他位置硬编码。

调参顺序 (实机/headless 都建议从上往下逐层冻结):
    1. VMC PD:      kp_motor, kd_motor                 (腿不抖、不漂)
    2. LAND PD:     kp_land, kd_land                   (落地不弹、不软)
    3. FLIGHT damp: flight_pitch_kd                    (空中姿态稳定)
    4. LQR balance: q_diag[0:4], r_diag                (pitch/roll 不发散)
    5. Forward:     pitch_lean_gain, velocity_ki       (跟踪 target_velocity)
    6. Yaw:         yaw_damping → yaw_ki               (P 先, 必要时再 I)
       Heading:     heading_hold_kp                    (航向保持, 串级在 yaw 阻尼外)
    7. Jump:        见 launch_mujoco MANUAL_JUMP_*     (air_height_max, crouch_depth)
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from sim.controllers.combined import CombinedParams
from sim.controllers.phase import JumpPhaseParams
from sim.controllers.vmc import VmcParams


# --- 参数预设 ---

STAND_PARAMS = CombinedParams(
    vmc=VmcParams(
        nominal_height=0.142,  # 站姿腿高 (m). LUT 范围 [0.0784, 0.1537]; 142mm 已贴近伸长上限 (余 ~12mm)
        kp_motor=30.0,
        kd_motor=1.0,
        kp_land=15.0,   # 旧 land_kp_scale=0.5 × kp_motor=30 的等价绝对值
        kd_land=3.5,    # 旧 kd_motor=1 × land_kd_scale=2.5 + landing_damping=1.0
        max_height_rate=0.05,
        flight_pitch_kd=1.5,
        roll_level_kp_height=0.0,
        roll_level_kd_height=0.002,
        roll_level_offset_limit=0.035,  # 单侧腿高差上限; 0.035 可调平 ~65mm 单轮高差 (需 ±32.5mm)
        slope_squat_margin=0.005,       # 上坡降站高: 伸长腿离 h_max 至少留 5mm
    ),
    # 2026-05-16 retune after mass rebalance (10.24 kg → 2.20 kg).
    # B matrix scaled ~7× because B ∝ 1/(M·L) and CoM dropped 36 mm.
    # 2026-05-17 重构到 5D LQR:state 为 [pitch, pitch_rate, roll, roll_rate, wheel_vel],
    # 不再包含 wheel_pos。原 Q[4]=50 (wheel_pos) 已删除,原 Q[5]=500 (wheel_vel) 平移到 Q[4]。
    # 2026-05-18 删除 pitch_trim (改用 equilibrium_pitch_from_geometry 自动计算),
    # 加入位置外环参数 position_kp / position_velocity_limit。
    # 2026-05-21 相位独占重构: LQR 在跳跃全程不输出 (CROUCH/EXTEND/FLIGHT/LAND),
    # VMC 独占腿;轮力矩 = 0。删除 flight_wheel_q、jump_extend_feedforward (轨迹动态 FF 取代)。
    # 2026-05-26 解耦速度跟踪三角:Q[wheel_vel] 500 → 300。原配置下 LQR 拼命驱动轮力矩
    # 追 target_velocity, 与外环 PI (pitch_lean_gain + velocity_ki) 形成并行速度
    # 控制器, 二者互相抵消, 调任一参数都会扰动另一个 (docs/lessons_learned.md:179
    # 记录过的"q_pitch_rate=1307 让 pitch_lean=0.3 只产生 0.001 rad 倾角"就是这个症状)。
    # 把 Q[4] 降低到 300 后, LQR 对 wheel_vel 的反馈强度减弱到 60% (原 R=200 → 比值
    # 2.5→1.5), 外环 PI 的"前倾→重力分量加速"路径获得更明确的主导权, pitch_lean 与
    # LQR 不再争抢同一个目标。同时不破坏 cmd_height 阶跃跟踪 (verify_height_sweep
    # 仍 PASS, base_xy_drift 0.10→0.17m, pitch_peak 0.138→0.112)。完全归零 (q4<100)
    # 在 height_sweep 应力测试下导致 base 漂移失控 → 必须先把外环 position loop
    # 增强 (position_kp / position_velocity_limit) 才能进一步降低 Q[4]。
    q_diag=np.array([1000.0, 200.0, 1000.0, 200.0, 300.0]),
    r_diag=np.array([200.0, 400.0]),
    velocity_ki=0.3,
    pitch_lean_gain=0.15,
    position_kp=3.0,
    position_kd=1.5,
    position_velocity_limit=0.6,
    yaw_damping=0.5,
    yaw_ki=0.0,
    # 航向保持: 未发转向指令 (target_yaw_rate≈0) 且站立接地时锁定当前航向, 抵抗扰动。
    # kp=2.0: 0.1rad(5.7°) 偏航 → 0.2rad/s 回正参考; rate_limit 限幅 0.8rad/s 防猛回正。
    # 保守起点, 实机可在 yaw_damping 冻结后再上调。
    heading_hold_kp=2.0,
    heading_hold_rate_limit=0.8,
    fixed_height=False,
    lqr_height_bin_size=0.02,
    # 2026-05-26 ff_gain 默认 0:旧值 4.0 把 leg-height-rate 信号 (经过 LUT
    # dy_wheel_dh 和 height_dtheta) 注入 LQR target[4], 跟 pitch_lean PI 路径
    # 形成第二条速度控制通路。当前几何下 dy_wheel_dh(0.142)≈0.164 (TODO P0-2),
    # 量级小, 移除后 cmd_height 阶跃的 pitch peak 改变 <5%。仅当 cmd_height 大幅
    # 阶跃 pitch peak 不达标时再启用 (启用值参考 4.0)。
    ff_gain=0.0,
)

# 2026-05-16: same gains as STAND_PARAMS; drive validated up to 0.5 m/s in
# headless. Old Optuna-tuned STAND_THEN_DRIVE_PARAMS (10.24 kg model) replaced.
STAND_THEN_DRIVE_PARAMS = CombinedParams(
    vmc=VmcParams(
        nominal_height=0.142,
        kp_motor=30.0,
        kd_motor=1.0,
        kp_land=15.0,
        kd_land=3.073,  # 旧 kd_motor=1 × land_kd_scale=2.5 + landing_damping=0.628
        flight_pitch_kd=1.5,
        roll_level_kp_height=0.0,
        roll_level_kd_height=0.002,
        roll_level_offset_limit=0.035,
        slope_squat_margin=0.005,
    ),
    q_diag=np.array([1000.0, 200.0, 1000.0, 200.0, 300.0]),
    r_diag=np.array([200.0, 400.0]),
    pitch_lean_gain=0.15,
    velocity_ki=0.3,
    position_kp=3.0,
    position_kd=1.5,
    position_velocity_limit=0.6,
    yaw_damping=0.5,
    yaw_ki=0.0,
    heading_hold_kp=2.0,
    heading_hold_rate_limit=0.8,
)

# 跳跃相位机默认参数
DEFAULT_PHASE_PARAMS = JumpPhaseParams()


# --- 序列化工具 ---

def params_to_dict(params: CombinedParams) -> dict[str, Any]:
    """将 CombinedParams 序列化为可 JSON 存储的字典。"""
    return {
        "vmc": {
            "nominal_height": params.vmc.nominal_height,
            "kp_motor": params.vmc.kp_motor,
            "kd_motor": params.vmc.kd_motor,
            "kp_land": params.vmc.kp_land,
            "kd_land": params.vmc.kd_land,
            "max_height_rate": params.vmc.max_height_rate,
            "flight_pitch_kd": params.vmc.flight_pitch_kd,
            "roll_level_kp_height": params.vmc.roll_level_kp_height,
            "roll_level_kd_height": params.vmc.roll_level_kd_height,
            "roll_level_offset_limit": params.vmc.roll_level_offset_limit,
            "slope_squat_margin": params.vmc.slope_squat_margin,
        },
        "q_diag": params.q_diag.tolist(),
        "r_diag": params.r_diag.tolist(),
        "target_velocity": params.target_velocity,
        "pitch_lean_gain": params.pitch_lean_gain,
        "velocity_ki": params.velocity_ki,
        "position_kp": params.position_kp,
        "position_kd": params.position_kd,
        "position_velocity_limit": params.position_velocity_limit,
        "yaw_damping": params.yaw_damping,
        "yaw_ki": params.yaw_ki,
        "target_yaw_rate": params.target_yaw_rate,
        "heading_hold_kp": params.heading_hold_kp,
        "heading_hold_rate_limit": params.heading_hold_rate_limit,
        "fixed_height": params.fixed_height,
        "lqr_height_bin_size": params.lqr_height_bin_size,
        "ff_gain": params.ff_gain,
    }


def params_from_dict(d: dict[str, Any]) -> CombinedParams:
    """从字典反序列化 CombinedParams。"""
    vmc_data = d["vmc"]
    return CombinedParams(
        vmc=VmcParams(
            nominal_height=vmc_data["nominal_height"],
            kp_motor=vmc_data["kp_motor"],
            kd_motor=vmc_data["kd_motor"],
            kp_land=vmc_data.get("kp_land", 15.0),
            kd_land=vmc_data.get("kd_land", 3.5),
            max_height_rate=vmc_data.get("max_height_rate", 0.1),
            flight_pitch_kd=vmc_data.get("flight_pitch_kd", 1.5),
            roll_level_kp_height=vmc_data.get("roll_level_kp_height", 0.0),
            roll_level_kd_height=vmc_data.get("roll_level_kd_height", 0.0),
            roll_level_offset_limit=vmc_data.get("roll_level_offset_limit", 0.0),
            slope_squat_margin=vmc_data.get("slope_squat_margin", 0.0),
        ),
        q_diag=np.array(d["q_diag"]),
        r_diag=np.array(d["r_diag"]),
        target_velocity=d.get("target_velocity", 0.0),
        pitch_lean_gain=d.get("pitch_lean_gain", 0.02),
        velocity_ki=d.get("velocity_ki", 0.1),
        position_kp=d.get("position_kp", 1.5),
        position_kd=d.get("position_kd", 1.5),
        position_velocity_limit=d.get("position_velocity_limit", 0.3),
        yaw_damping=d.get("yaw_damping", 0.5),
        yaw_ki=d.get("yaw_ki", 0.0),
        target_yaw_rate=d.get("target_yaw_rate", 0.0),
        heading_hold_kp=d.get("heading_hold_kp", 0.0),
        heading_hold_rate_limit=d.get("heading_hold_rate_limit", 1.0),
        fixed_height=d.get("fixed_height", True),
        lqr_height_bin_size=d.get("lqr_height_bin_size", 0.02),
        ff_gain=d.get("ff_gain", 0.0),
    )


def params_to_json(params: CombinedParams) -> str:
    """将 CombinedParams 序列化为 JSON 字符串。"""
    return json.dumps(params_to_dict(params), indent=2)


def params_from_json(json_str: str) -> CombinedParams:
    """从 JSON 字符串反序列化 CombinedParams。"""
    return params_from_dict(json.loads(json_str))
