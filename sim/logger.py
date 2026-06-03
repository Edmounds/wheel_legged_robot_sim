from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path
import numpy as np

from sim.state import SimState

def setup_system_logger(log_path: Path, console: bool = True) -> logging.Logger:
    """初始化标准文本日志，用于记录事件和配置。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("sim")
    logger.setLevel(logging.INFO)
    
    # 清理可能存在的旧 handlers
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    # 防止向 root logger 传递
    logger.propagate = False
    return logger


def _rpy_from_quaternion(q: np.ndarray) -> tuple[float, float]:
    """从四元数提取 roll 和 yaw 角（单位: 弧度）。

    使用 ZYX 内旋（航空）顺序。pitch 已在 SimState 中独立计算，此处不重复。

    Args:
        q: 四元数 [w, x, y, z]。

    Returns:
        (roll, yaw) 元组。
    """
    w, x, y, z = q
    # roll (x 轴旋转)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))
    # yaw (z 轴旋转)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, yaw


class TelemetryLogger:
    """高频物理状态遥测日志，保存为 CSV 文件。"""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_path, mode='w', newline='')
        self.writer = csv.writer(self.file)
        
        # 写入表头
        self.writer.writerow([
            "time",
            "pos_x", "pos_y", "pos_z",
            "quat_w", "quat_x", "quat_y", "quat_z",
            "roll", "pitch", "yaw",
            "vel_x", "vel_y", "vel_z",
            "ang_vel_x", "ang_vel_y", "ang_vel_z",
            "pitch_rate",
            "wheel_pos_l", "wheel_pos_r",
            "wheel_vel_l", "wheel_vel_r",
            "contact_count",
            "target_info",
            "control_output",
        ])
        
    def log_step(self, time: float, state: SimState, target_info: str, control: np.ndarray) -> None:
        """记录每帧的关键数据，包含完整 6-DOF pose。"""
        q = state.base_quaternion
        roll, yaw = _rpy_from_quaternion(q)
        av = state.base_angular_velocity
        row = [
            f"{time:.4f}",
            f"{state.base_position[0]:.4f}", f"{state.base_position[1]:.4f}", f"{state.base_position[2]:.4f}",
            f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}",
            f"{roll:.4f}", f"{state.pitch:.4f}", f"{yaw:.4f}",
            f"{state.base_linear_velocity[0]:.4f}", f"{state.base_linear_velocity[1]:.4f}", f"{state.base_linear_velocity[2]:.4f}",
            f"{av[0]:.4f}", f"{av[1]:.4f}", f"{av[2]:.4f}",
            f"{state.pitch_rate:.4f}",
            f"{state.wheel_positions.get('left', 0.0):.4f}", f"{state.wheel_positions.get('right', 0.0):.4f}",
            f"{state.wheel_velocities.get('left', 0.0):.4f}", f"{state.wheel_velocities.get('right', 0.0):.4f}",
            str(state.contact_count),
            target_info,
            np.array2string(control, precision=3, separator=',', suppress_small=True)
        ]
        self.writer.writerow(row)
        
    def close(self) -> None:
        if not self.file.closed:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def cleanup_old_logs(base_dir: Path, max_keep: int = 4) -> None:
    """清理旧日志目录或文件，只保留最新的 max_keep 个。"""
    if not base_dir.exists():
        return
        
    entries = []
    for p in base_dir.iterdir():
        if p.name.startswith("run_") or p.name.startswith("opt_"):
            entries.append(p)
            
    # 按名称倒序排序（名称包含时间戳，保证最新的在前）
    entries.sort(key=lambda p: p.name, reverse=True)
    
    # 删除超出的部分
    for p in entries[max_keep:]:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception as e:
            print(f"Warning: Failed to cleanup old log {p}: {e}")
