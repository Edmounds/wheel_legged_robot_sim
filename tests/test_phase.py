from sim.controllers.jump_trajectory import JumpTrajectory, JumpTrajectoryParams
from sim.controllers.phase import JumpPhase, JumpPhaseMachine


def test_stand_enters_fallen_when_pitch_exceeds_threshold() -> None:
    phase_machine = JumpPhaseMachine()

    phase = phase_machine.update(
        dt=0.01,
        contact_count=2,
        leg_height=0.14,
        vz=0.0,
        pitch=1.5,
    )

    assert phase == JumpPhase.FALLEN
    assert phase_machine.phase == JumpPhase.FALLEN


def test_land_returns_directly_to_stand_after_land_duration() -> None:
    phase_machine = JumpPhaseMachine()
    # 用一个标准 trajectory 让 LAND 有 land_duration 引用
    traj = JumpTrajectory(JumpTrajectoryParams(), h_start=0.142, cmd_jump_amplitude=1.0)
    traj.setup_land(h_contact=0.140, v_contact=-0.5)
    phase_machine.trajectory = traj
    phase_machine._set_phase(JumpPhase.LAND)

    phase = phase_machine.update(
        dt=traj.land.duration,  # type: ignore[union-attr]
        contact_count=2,
        leg_height=0.14,
        vz=0.0,
        pitch=0.0,
    )

    assert phase == JumpPhase.STAND


def test_fallen_is_terminal_and_does_not_recover_when_pitch_returns() -> None:
    phase_machine = JumpPhaseMachine()
    phase_machine._set_phase(JumpPhase.FALLEN)

    phase = phase_machine.update(
        dt=0.5,
        contact_count=2,
        leg_height=0.14,
        vz=0.0,
        pitch=0.0,
    )

    assert phase == JumpPhase.FALLEN
