# wheel_legged_robot_sim

轮腿机器人的 MuJoCo 仿真。机器人左右各一条四连杆并联腿，腿端各装一个驱动轮；控制上用 LQR + VMC 组合，能自平衡站立、行走，也能起跳。

## 算法简介

控制器按跳跃相位把平衡和腿部动作分开管，免得两个环在腿电机上抢量程（代码里叫相位独占）。

平衡用 LQR。状态取切空间 5 维 `[pitch, pitch_rate, roll, roll_rate, wheel_vel]`，解出两路虚拟控制：前进轮力矩和左右腿差分。外面再套几层 PI/PD：`pitch_lean` 把速度误差转成目标俯仰，`yaw` 管转向，`heading_hold` 锁航向，零速时位置外环把车拉回原地。

腿高用 VMC。每条腿是四连杆闭链，先查 LUT 把目标腿高换成电机角，再在关节空间做 PD 跟踪。LAND 阶段单独给一组增益，P 更软、D 更硬，专门吸冲击。

跳跃是一台相位机：`STAND → CROUCH → EXTEND → FLIGHT → LAND`，摔倒了进 `FALLEN`。STAND 时 LQR 全开、VMC 管高度；CROUCH、EXTEND、LAND 期间 LQR 只靠轮前进和 yaw 稳住本体，腿整个交给 VMC 轨迹；FLIGHT 把腿和轮都置零，靠角动量在空中保持姿态，这样落地不会反扭。

参数和调参顺序都集中在 `sim/controllers/default_params.py`，注释里写了该从哪项开始、按什么顺序往下冻结。

## 环境搭建

依赖用 [uv](https://docs.astral.sh/uv/) 管理，需要 Python 3.13。

```bash
uv sync
```

> macOS 上 MuJoCo 的交互式查看器必须用 `mjpython` 启动（`uv sync` 会一并装好）。Linux/Windows 用普通 `python` 即可。

## 快速开始

启动带控制器的仿真（默认 `lqr_vmc` 控制器、`stand` 场景）：

```bash
# macOS
uv run mjpython -m sim.launch_mujoco
# Linux / Windows
uv run python -m sim.launch_mujoco
```

常用参数：

```bash
# 跳跃场景
uv run mjpython -m sim.launch_mujoco --scenario jump
# 切换控制器：zero / vmc / lqr_vmc / combined
uv run mjpython -m sim.launch_mujoco --controller combined
# 关闭默认的单轮斜坡地形，使用平地
uv run mjpython -m sim.launch_mujoco --flat-ground
```

场景可选 `stand`（站立）、`jump`（跳跃）、`drive`（行进）、`fall_recover`（摔倒自恢复）。

只看 URDF 模型本身（无控制器）：

```bash
uv run mjpython launch_viewer.py
```

## 手柄控制（可选）

接入实体手柄需要额外依赖（仅 macOS）：

```bash
uv sync --extra gamepad
```

手柄默认启用，左摇杆 + LT/RT 控制行进与跳跃；未安装手柄依赖时会自动跳过，不影响仿真。要显式关闭用 `--no-enable-gamepad`。

## 测试

测试需在仓库根目录运行（用例按相对路径加载 `sim/robot/robot.urdf`）：

```bash
uv run pytest -q
```

## 项目结构

```
sim/
  controllers/     # LQR、VMC、相位机、跳跃轨迹、参数与 LUT
  robot/           # robot.urdf + STL 网格
  launch_mujoco.py # 带控制器的仿真入口
  model_xml.py     # URDF → MJCF 转换与预处理
  rollout.py       # 无头 rollout
  optimize.py      # Optuna 调参
scripts/           # 调参、验证、诊断等工具脚本
tests/             # pytest 用例
launch_viewer.py   # 纯 URDF 查看器
```

## 许可证

[Apache-2.0](LICENSE)
