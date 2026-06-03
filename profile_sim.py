"""Headless profile harness for the lqr_vmc stand controller.

Runs 5 sim seconds (2500 steps @ 500Hz) and prints cumulative time top-30 + wall time.
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mujoco

from src.controllers.combined import CombinedController
from src.controllers.default_params import STAND_PARAMS
from src.mjcf_builder import prepare_controlled_mujoco_xml
from src.state import extract_sim_state


def setup():
    xml = prepare_controlled_mujoco_xml(Path('src/robot/robot.urdf'))
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'stand')
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)
    controller = CombinedController(STAND_PARAMS)
    return model, data, controller


def run(model, data, controller, n_steps: int) -> None:
    for _ in range(n_steps):
        state = extract_sim_state(model, data)
        ctrl = controller(model, data, state)
        data.ctrl[:model.nu] = ctrl
        mujoco.mj_step(model, data)


def main() -> None:
    model, data, controller = setup()
    # warmup 100 steps (first LQR linearization etc.)
    run(model, data, controller, 100)

    N = 2500  # 5 sec sim @ 500 Hz

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    run(model, data, controller, N)
    prof.disable()
    wall = time.perf_counter() - t0

    sim_seconds = N * float(model.opt.timestep)
    print(f'\n===== TIMING =====')
    print(f'sim seconds   = {sim_seconds:.3f}')
    print(f'wall seconds  = {wall:.3f}')
    print(f'realtime ratio = {sim_seconds / wall:.2f}x  (>=1 means real-time-capable)')
    print(f'\n===== cProfile top 30 (cumulative) =====')
    stats = pstats.Stats(prof).strip_dirs().sort_stats('cumulative')
    stats.print_stats(30)


if __name__ == '__main__':
    main()
