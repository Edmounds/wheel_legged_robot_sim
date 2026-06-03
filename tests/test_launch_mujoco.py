from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim.controllers.default_params import STAND_PARAMS
from sim.controllers.balance_state import balance_tangent_state_5d
from sim.controllers.jump_trajectory import JumpTrajectory
from sim.controllers.phase import JumpPhase, JumpPhaseMachine
from sim.controllers.vmc import VmcController
from sim.launch_mujoco import (
    MANUAL_JUMP_TRAJECTORY_PARAMS,
    _apply_gamepad_state_to_sliders,
    _can_enable_gamepad,
    _read_cmd_sliders,
    _read_gamepad_state,
    _restore_cmd_sliders,
    _trigger_jump_on_rising_edge,
    build_controlled_model,
    create_controlled_controller,
    parse_args,
    run_controlled_viewer_loop,
    step_controlled_model,
)
from sim.gamepad import GamepadCommandMapper, XboxState
from sim.model_xml import CMD_SLIDER_NAMES, prepare_controlled_mujoco_xml
from sim.model_semantics import MODEL_SEMANTICS
from sim.state import actuator_id, extract_sim_state, model_addresses


class _ControllerWithPhaseMachine:
    def __init__(self, phase_machine: JumpPhaseMachine) -> None:
        self.phase_machine = phase_machine


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.messages.append(message)


class _FiniteGamepad:
    name = "test gamepad"

    def __init__(self, state: XboxState | None) -> None:
        self.state = state

    def poll(self) -> XboxState | None:
        return self.state

    def close(self) -> None:
        pass


def test_gamepad_is_enabled_by_default() -> None:
    args = parse_args([])
    logger = _Logger()

    assert _can_enable_gamepad(logger, args)


def test_gamepad_can_be_disabled_via_no_flag() -> None:
    args = parse_args(["--no-enable-gamepad"])
    logger = _Logger()

    assert not _can_enable_gamepad(logger, args)
    assert any("Gamepad disabled" in message for message in logger.messages)


def test_invalid_gamepad_state_does_not_update_sliders(tmp_path: Path) -> None:
    model, data = build_controlled_model(
        Path("sim/robot/robot.urdf"),
        cache_dir=tmp_path,
        terrain=None,
    )
    mapper = GamepadCommandMapper(
        linear_range=(-1.0, 1.0),
        angular_range=(-3.0, 3.0),
        height_range=(0.07844, 0.142),
    )
    original = _read_cmd_sliders(model, data)

    invalid_state = _read_gamepad_state(_FiniteGamepad(XboxState(float("nan"), 1.0, 0.0, 0.0)))
    _apply_gamepad_state_to_sliders(model, data, invalid_state, mapper)

    assert _read_cmd_sliders(model, data) == original


def test_gamepad_triggers_increment_and_hold_cmd_height(tmp_path: Path) -> None:
    model, data = build_controlled_model(
        Path("sim/robot/robot.urdf"),
        cache_dir=tmp_path,
        terrain=None,
    )
    mapper = GamepadCommandMapper(
        linear_range=(-1.0, 1.0),
        angular_range=(-3.0, 3.0),
        height_range=(0.07844, 0.142),
        height_rate=0.04,
    )
    height_act = actuator_id(model, "cmd_height")
    data.ctrl[height_act] = 0.10

    _apply_gamepad_state_to_sliders(
        model,
        data,
        XboxState(left_x=0.0, left_y=0.0, lt=0.0, rt=1.0),
        mapper,
        dt=0.5,
    )
    raised = float(data.ctrl[height_act])
    assert raised == pytest.approx(0.12)

    _apply_gamepad_state_to_sliders(
        model,
        data,
        XboxState(left_x=0.0, left_y=0.0, lt=0.0, rt=0.0),
        mapper,
        dt=0.5,
    )
    assert float(data.ctrl[height_act]) == pytest.approx(raised)


def test_run_controlled_viewer_loop_polls_gamepad_outside_viewer_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model, data = build_controlled_model(
        Path("sim/robot/robot.urdf"),
        cache_dir=tmp_path,
        terrain=None,
    )
    controller = create_controlled_controller("zero", "stand")

    class _Viewer:
        def __init__(self) -> None:
            self.in_lock = False
            self._running_calls = 0

        def is_running(self) -> bool:
            self._running_calls += 1
            return self._running_calls == 1

        def lock(self) -> "_Viewer":
            return self

        def __enter__(self) -> None:
            self.in_lock = True

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            self.in_lock = False

        def sync(self) -> None:
            pass

    viewer = _Viewer()

    class _LockCheckingGamepad:
        name = "lock checking gamepad"

        def poll(self) -> XboxState:
            assert not viewer.in_lock
            return XboxState(left_x=0.0, left_y=0.0, lt=0.0, rt=0.0)

        def close(self) -> None:
            pass

    times = iter([0.0] + [float(model.opt.timestep)] * 8)
    monkeypatch.setattr("sim.launch_mujoco.time.perf_counter", lambda: next(times))
    monkeypatch.setattr("sim.launch_mujoco.time.sleep", lambda _seconds: None)

    run_controlled_viewer_loop(
        model,
        data,
        controller,
        viewer,
        gamepad=_LockCheckingGamepad(),
        gamepad_mapper=GamepadCommandMapper(
            linear_range=(-1.0, 1.0),
            angular_range=(-3.0, 3.0),
            height_range=(0.07844, 0.142),
        ),
    )


def test_run_controlled_viewer_loop_does_not_fast_forward_after_frame_lag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model, data = build_controlled_model(
        Path("sim/robot/robot.urdf"),
        cache_dir=tmp_path,
        terrain=None,
    )
    controller = create_controlled_controller("zero", "stand")

    class _Viewer:
        def __init__(self) -> None:
            self.now = 0.0
            self.running_calls = 0

        def is_running(self) -> bool:
            self.running_calls += 1
            if self.running_calls == 1:
                self.now = 0.5
            return self.running_calls <= 3

        def lock(self) -> "_Viewer":
            return self

        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def sync(self) -> None:
            pass

    viewer = _Viewer()

    def fake_sleep(seconds: float) -> None:
        viewer.now += max(seconds, 0.0)

    monkeypatch.setattr("sim.launch_mujoco.time.perf_counter", lambda: viewer.now)
    monkeypatch.setattr("sim.launch_mujoco.time.sleep", fake_sleep)

    run_controlled_viewer_loop(model, data, controller, viewer)

    max_steps_per_frame = max(1, int((1.0 / 60.0) / float(model.opt.timestep)) * 2)
    assert data.time <= (max_steps_per_frame + 2.5) * float(model.opt.timestep)
    assert data.time < 0.5


def test_cmd_jump_is_non_physical_command_actuator(tmp_path: Path) -> None:
    model_path = prepare_controlled_mujoco_xml(
        Path("sim/robot/robot.urdf"),
        cache_dir=tmp_path,
    )
    root = ET.parse(model_path).getroot()

    command_body = root.find(".//body[@name='command_slider_body']")
    assert command_body is not None
    assert command_body.find("joint[@name='cmd_jump']") is not None

    actuator = root.find("actuator/motor[@name='cmd_jump']")
    assert actuator is not None
    assert actuator.get("gear") == "0"
    assert actuator.get("ctrlrange") == "0 1"


def test_controlled_model_initializes_cmd_jump_slider(tmp_path: Path) -> None:
    model, data = build_controlled_model(
        Path("sim/robot/robot.urdf"),
        cache_dir=tmp_path,
        terrain=None,
    )

    assert "cmd_jump" in CMD_SLIDER_NAMES
    jump_actuator = actuator_id(model, "cmd_jump")
    assert jump_actuator >= 0
    assert data.ctrl[jump_actuator] == 0.0


def test_leg_actuator_range_allows_dm4310_short_burst_peak(tmp_path: Path) -> None:
    model, _data = build_controlled_model(
        Path("sim/robot/robot.urdf"),
        cache_dir=tmp_path,
        terrain=None,
    )

    leg_ranges = [
        model.actuator_ctrlrange[actuator_id(model, f"act_{joint_name}")]
        for joint_name in ("base_link_旋转-2", "base_link_旋转-1")
    ]

    for low, high in leg_ranges:
        assert abs(low + 12.5) < 1e-9
        assert abs(high - 12.5) < 1e-9


def test_command_slider_read_and_restore_includes_cmd_jump(tmp_path: Path) -> None:
    model, data = build_controlled_model(
        Path("sim/robot/robot.urdf"),
        cache_dir=tmp_path,
        terrain=None,
    )
    jump_actuator = actuator_id(model, "cmd_jump")

    data.ctrl[jump_actuator] = 1.0
    values = _read_cmd_sliders(model, data)
    data.ctrl[:] = 0.0
    _restore_cmd_sliders(model, data, values)

    assert values["cmd_jump"] == 1.0
    assert data.ctrl[jump_actuator] == 1.0


def _make_controller_with_phase_machine() -> tuple[JumpPhaseMachine, "_ControllerWithPhaseMachine"]:
    # 给 controller 一个 fake params.vmc.nominal_height 让 trigger 能读到 h_start
    class _Vmc:
        nominal_height = 0.142

    class _Params:
        vmc = _Vmc()

    phase_machine = JumpPhaseMachine()
    controller = _ControllerWithPhaseMachine(phase_machine)
    controller.params = _Params()  # type: ignore[attr-defined]
    return phase_machine, controller


def test_cmd_jump_rising_edge_triggers_once() -> None:
    phase_machine, controller = _make_controller_with_phase_machine()

    previous = _trigger_jump_on_rising_edge(controller, 0.0, 0.0)
    assert phase_machine.phase == JumpPhase.STAND

    previous = _trigger_jump_on_rising_edge(controller, 1.0, previous)
    assert phase_machine.phase == JumpPhase.CROUCH
    first_time_in_phase = phase_machine.time_in_phase

    previous = _trigger_jump_on_rising_edge(controller, 1.0, previous)
    assert phase_machine.phase == JumpPhase.CROUCH
    assert phase_machine.time_in_phase == first_time_in_phase

    previous = _trigger_jump_on_rising_edge(controller, 0.0, previous)
    phase_machine.phase = JumpPhase.STAND
    phase_machine.trajectory = None
    _trigger_jump_on_rising_edge(controller, 1.0, previous)
    assert phase_machine.phase == JumpPhase.CROUCH


def _start_jump_with_default_traj(phase_machine: JumpPhaseMachine, h_start: float = 0.142) -> None:
    """Helper: 模拟 _trigger_jump_on_rising_edge 的轨迹构造逻辑。"""
    traj = JumpTrajectory(MANUAL_JUMP_TRAJECTORY_PARAMS, h_start=h_start, cmd_jump_amplitude=1.0)
    phase_machine.start_jump(traj)


def test_jump_phase_does_not_enter_flight_while_still_in_contact() -> None:
    phase_machine = JumpPhaseMachine()
    _start_jump_with_default_traj(phase_machine)
    crouch_dur = phase_machine.trajectory.crouch.duration  # type: ignore[union-attr]
    extend_dur = phase_machine.trajectory.extend.duration  # type: ignore[union-attr]

    phase_machine.update(dt=crouch_dur + 0.01, leg_height=0.2, vz=0.0, contact_count=2)
    assert phase_machine.phase == JumpPhase.EXTEND

    # 还在地面接触: extend 跑超过 2x duration 才会去 LAND, 不进 FLIGHT
    # (2x 容忍弹跳期间补推; 见 phase.py 的 EXTEND 转移逻辑)
    phase_machine.update(dt=extend_dur * 2.0 + 0.01, leg_height=0.2, vz=0.2, contact_count=2)
    assert phase_machine.phase == JumpPhase.LAND

    # 重置, 再测离地且足够 vz 的情况
    phase_machine = JumpPhaseMachine()
    _start_jump_with_default_traj(phase_machine)
    crouch_dur = phase_machine.trajectory.crouch.duration  # type: ignore[union-attr]
    extend_dur = phase_machine.trajectory.extend.duration  # type: ignore[union-attr]
    phase_machine.update(dt=crouch_dur + 0.01, leg_height=0.2, vz=0.0, contact_count=2)
    assert phase_machine.phase == JumpPhase.EXTEND
    # vz=0.5 > 0.3 阈值 (避免推地弹跳触发) + ncon=0 → FLIGHT
    phase_machine.update(dt=extend_dur + 0.01, leg_height=0.2, vz=0.5, contact_count=0)
    assert phase_machine.phase == JumpPhase.FLIGHT


def test_jump_phase_accepts_leg_height_and_vz_inputs() -> None:
    phase_machine = JumpPhaseMachine()
    _start_jump_with_default_traj(phase_machine)
    crouch_dur = phase_machine.trajectory.crouch.duration  # type: ignore[union-attr]
    extend_dur = phase_machine.trajectory.extend.duration  # type: ignore[union-attr]

    phase_machine.update(dt=crouch_dur + 0.01, leg_height=0.09, vz=0.0, contact_count=2)
    assert phase_machine.phase == JumpPhase.EXTEND

    # vz=0.5 > 0.3 阈值 → FLIGHT
    phase_machine.update(dt=extend_dur + 0.01, leg_height=0.14, vz=0.5, contact_count=0)
    assert phase_machine.phase == JumpPhase.FLIGHT

    phase_machine.update(dt=0.01, leg_height=0.13, vz=-0.1, contact_count=2)
    assert phase_machine.phase == JumpPhase.LAND


def test_vmc_extend_dynamic_ff_can_exceed_stand_torque_clip(tmp_path: Path) -> None:
    """EXTEND 期间动态 FF 跟随轨迹 ḧ_target,峰值力矩允许 > 3.5 N·m (STAND clip)。"""
    model, data = build_controlled_model(
        Path("sim/robot/robot.urdf"),
        cache_dir=tmp_path,
        terrain=None,
    )
    params = STAND_PARAMS.vmc
    phase_machine = JumpPhaseMachine()
    controller = VmcController(params, phase_machine=phase_machine)
    addresses = model_addresses(model)
    leg_indices = [addresses.actuators[name] for name in MODEL_SEMANTICS.leg_motor_joints]

    # STAND: 仅重力补偿,|τ| <= 3.5
    phase_machine.phase = JumpPhase.STAND
    state = extract_sim_state(model, data)
    stand_control = controller.preview_control(model, data, state)
    assert np.max(np.abs(stand_control[leg_indices])) <= 3.5 + 1e-9

    # 构造轨迹,跳到 EXTEND 中段 (ḧ 接近峰值)
    traj = JumpTrajectory(MANUAL_JUMP_TRAJECTORY_PARAMS, h_start=0.142, cmd_jump_amplitude=1.0)
    phase_machine.trajectory = traj
    phase_machine.phase = JumpPhase.EXTEND
    phase_machine.time_in_phase = traj.extend.duration / 2.0  # 中段加速度峰值附近
    extend_control = controller.preview_control(model, data, state)
    # 动态 FF: τ = m * (g + ḧ_peak) * dh/dθ / 2. 满幅跳 (10cm) 中段 ḧ ~ 60 m/s²
    # → FF 显著大于 STAND 的 1-2 N·m 重力补偿。
    assert np.max(np.abs(extend_control[leg_indices])) > 3.5


def test_flight_phase_zeros_output_and_keeps_lqr_dormant(tmp_path: Path) -> None:
    """相位独占重构后: FLIGHT 时 controller 输出全 0,LQR 不参与 (但实例保留)。

    旧行为 1: FLIGHT 时 LQR 仍输出 forward/roll 力矩,通过 _flight_lqr_weight_schedule
    动态切换 Q,累积 reaction torque 导致空中 pitch 漂移 ~0.3 rad。
    旧行为 2: FLIGHT 时所有输出 = 0,base 姿态由角动量保持 — 但跳得高时 (>=80mm)
    EXTEND 末期累积的 pitch_rate (-3 ~ -6 rad/s) 在 250ms 飞行中累积 -1 rad pitch,
    落地前就已倒下。
    新行为: FLIGHT 时 LQR 仍不参与,但 VMC 在 FLIGHT 用对称腿电机做 pitch_rate 反作用
    阻尼 (左右同 τ = K*pitch_rate),通过 Newton 3rd 给 base 反向力矩抑制空中翻转。
    输出: wheels=0, leg motors = 对称的 pitch_rate 阻尼力矩 (左右严格相等)。
    """
    model, data = build_controlled_model(
        Path("sim/robot/robot.urdf"),
        cache_dir=tmp_path,
        terrain=None,
    )
    controller = create_controlled_controller("lqr_vmc", "stand")
    assert hasattr(controller, "vmc_controller")
    state = extract_sim_state(model, data)
    # STAND 调一次初始化 LQR
    controller(model, data, state)
    lqr = controller.lqr_controller
    assert lqr is not None
    assert lqr.gain.shape == (2, 5)

    phase_machine = controller.vmc_controller.phase_machine
    assert phase_machine is not None
    phase_machine.phase = JumpPhase.FLIGHT
    state = extract_sim_state(model, data)
    control = controller(model, data, state)

    # FLIGHT 输出: wheels = 0, leg motors = 对称 pitch_rate 阻尼 (左右严格相等)。
    # state.pitch_rate ≈ 0 (stand 初始化),所以阻尼力矩 ≈ 0,但代码路径要走通。
    addresses = model_addresses(model)
    left_act = addresses.actuators["base_link_旋转-2"]
    right_act = addresses.actuators["base_link_旋转-1"]
    assert control[left_act] == pytest.approx(control[right_act], abs=1e-9), (
        f"FLIGHT 必须对称: left={control[left_act]} right={control[right_act]}"
    )
    # wheel actuators 严格 0 (LQR 不参与)
    wheel_l = addresses.actuators["link2_left_旋转-13"]
    wheel_r = addresses.actuators["link2_right_旋转-12"]
    assert control[wheel_l] == 0.0 and control[wheel_r] == 0.0
    # tangent state provider 接口仍然是 5D (没改 LQR 状态维度)
    assert balance_tangent_state_5d(model, data, state).shape == (5,)


def test_cmd_jump_rollout_reaches_airborne_contact_state(tmp_path: Path) -> None:
    model, data = build_controlled_model(
        Path("sim/robot/robot.urdf"),
        cache_dir=tmp_path,
        terrain=None,
    )
    controller = create_controlled_controller("lqr_vmc", "stand")

    data.ctrl[actuator_id(model, "cmd_jump")] = 1.0
    _trigger_jump_on_rising_edge(controller, 1.0, 0.0)
    initial_z = float(data.qpos[2])

    min_contact_count = 100
    max_z = initial_z
    for _ in range(int(1.0 / model.opt.timestep)):
        state = extract_sim_state(model, data)
        min_contact_count = min(min_contact_count, state.contact_count)
        max_z = max(max_z, float(data.qpos[2]))
        assert step_controlled_model(model, data, controller)

    final_phase = controller.vmc_controller.phase_machine.phase

    assert min_contact_count == 0
    # 当前 four-bar 几何 + 12.5 N·m 电机扭矩极限下,扭矩饱和后能量主要损耗在
    # pitch 旋转 (~8°) 和轮子滚动上,实际 takeoff 速度约 0.5 m/s,空中升高
    # ~13mm。这是当前硬件参数 + PD/FF 控制能达到的水平; 想要更高的跳跃高度
    # 需要 trajectory optimization 处理 pitch 耦合,或更大的电机。
    assert max_z >= initial_z + 0.010
    assert final_phase != JumpPhase.FALLEN
