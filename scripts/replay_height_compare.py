#!/usr/bin/env python3
"""Replay manual cmd_height profile headlessly and compare FF variants.

读取一次 viewer 手测的 telemetry.csv，把里面的 cmd_height(t) 时程喂给 headless 仿真，
在三种控制器变体下回放同样的高度指令：

- baseline : STAND_PARAMS 当前默认 (fixed_height=False, FF 生效)
- ff_off   : fixed_height=True (FF 直接返回 0)
- ff_flip  : monkey-patch FF 取反 (验证 sign bug)

每个变体跑完都会算出 (cmd_h, pos_x, pos_y, pos_z, pitch) 在原 log 时间轴上的对比，
并对慢扫段输出 dy/dh 比值。
"""
from __future__ import annotations

import argparse
import csv
import importlib
import re
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np

mujoco = cast(Any, importlib.import_module("mujoco"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.controllers.combined import CombinedController
from sim.controllers.default_params import STAND_PARAMS
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import actuator_id, extract_sim_state


SLOW_SEGMENTS = [
    ("warmup_hold",   0.0,  7.4),
    ("slow_lower_1",  7.5, 12.0),
    ("hold_low_1",   12.0, 13.5),
    ("slow_raise_1", 13.5, 20.5),
    ("hold_high_1",  20.5, 25.5),
    ("slow_lower_2", 25.5, 31.0),
]


def parse_log(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, cmd_h, ref_pos_y) from a manual run telemetry.csv."""
    t_list, h_list, y_list = [], [], []
    h_pat = re.compile(r"h=([-\d.]+)")
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            m = h_pat.search(row["target_info"])
            if not m:
                continue
            t_list.append(float(row["time"]))
            h_list.append(float(m.group(1)))
            y_list.append(float(row["pos_y"]))
    if not t_list:
        raise RuntimeError(f"no cmd_height column found in {csv_path}")
    return (np.array(t_list), np.array(h_list), np.array(y_list))


def latest_manual_log() -> Path:
    base = Path("logs/manual")
    runs = sorted([p for p in base.iterdir() if p.name.startswith("run_")])
    if not runs:
        raise RuntimeError("no logs/manual/run_* directories")
    csv_path = runs[-1] / "telemetry.csv"
    if not csv_path.exists():
        raise RuntimeError(f"missing telemetry.csv in {runs[-1]}")
    return csv_path


def make_controller(variant: str) -> CombinedController:
    """Build a fresh CombinedController for the given variant."""
    base_params = STAND_PARAMS
    params = replace(
        base_params,
        vmc=replace(base_params.vmc),
        fixed_height=(variant == "ff_off"),
    )

    controller = CombinedController(params)

    if variant == "ff_flip":
        # Monkey-patch the FF to return its negative
        original = controller._height_wheel_velocity_feedforward

        def flipped(current_leg_height: float) -> float:
            return -original(current_leg_height)

        controller._height_wheel_velocity_feedforward = flipped  # type: ignore[assignment]

    return controller


def run_variant(
    variant: str,
    schedule_t: np.ndarray,
    schedule_h: np.ndarray,
    duration: float,
    sample_times: np.ndarray,
) -> dict:
    """Run a headless rollout following the cmd_height schedule, sample states at sample_times."""
    with TemporaryDirectory(dir=Path("tmp")) as tmp_dir:
        model_path = prepare_controlled_mujoco_xml(
            Path("sim/robot/robot.urdf"), cache_dir=Path(tmp_dir)
        )
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if stand_id < 0:
            raise RuntimeError("missing stand keyframe")
        mujoco.mj_resetDataKeyframe(model, data, stand_id)
        mujoco.mj_forward(model, data)

        height_actuator = actuator_id(model, "cmd_height")

        controller = make_controller(variant)

        dt = float(model.opt.timestep)
        n_steps = int(np.ceil(duration / dt))

        # pre-compute the height command at each sim step (step-hold from schedule)
        sim_t = np.arange(n_steps) * dt
        cmd_idx = np.searchsorted(schedule_t, sim_t, side="right") - 1
        cmd_idx = np.clip(cmd_idx, 0, len(schedule_h) - 1)
        cmd_traj = schedule_h[cmd_idx]

        sample_idx = np.clip((sample_times / dt).astype(int), 0, n_steps - 1)
        sample_set = set(sample_idx.tolist())

        rec: dict[int, dict[str, float]] = {}

        failed = False
        fail_reason = ""
        for step in range(n_steps):
            controller.params.vmc.nominal_height = float(cmd_traj[step])

            state = extract_sim_state(model, data)
            control = np.asarray(controller(model, data, state), dtype=float)
            data.ctrl[:] = np.clip(
                control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
            if height_actuator >= 0:
                data.ctrl[height_actuator] = float(cmd_traj[step])
            mujoco.mj_step(model, data)

            state = extract_sim_state(model, data)
            if (
                not np.all(np.isfinite(data.qpos))
                or not np.all(np.isfinite(data.qvel))
                or abs(float(state.pitch)) > 1.5
                or float(state.base_position[2]) < 0.02
            ):
                failed = True
                fail_reason = f"abort at t={step*dt:.2f}s pitch={state.pitch:.3f} z={state.base_position[2]:.3f}"
                break

            if step in sample_set:
                rec[step] = {
                    "t": step * dt,
                    "cmd_h": float(cmd_traj[step]),
                    "pos_x": float(state.base_position[0]),
                    "pos_y": float(state.base_position[1]),
                    "pos_z": float(state.base_position[2]),
                    "pitch": float(state.pitch),
                    "vy": float(state.base_linear_velocity[1]),
                }

        return {
            "variant": variant,
            "failed": failed,
            "fail_reason": fail_reason,
            "samples": rec,
            "sample_idx": sample_idx,
        }


def summarize(rows_by_variant: dict, ref_t: np.ndarray, ref_y: np.ndarray, ref_h: np.ndarray) -> None:
    variants = list(rows_by_variant.keys())

    print("\n=== timeline comparison (sampled at log timestamps) ===")
    hdr = f"{'t':>5} {'cmd_h':>7} " + " ".join([f"{'py_'+v:>10}" for v in variants]) + f" {'py_log':>10}"
    print(hdr)
    times_to_print = np.arange(0, 32, 1.0)
    for tp in times_to_print:
        idx_ref = int(np.argmin(np.abs(ref_t - tp)))
        row = f"{ref_t[idx_ref]:5.1f} {ref_h[idx_ref]:7.4f} "
        for v in variants:
            samples = rows_by_variant[v]["samples"]
            # find nearest recorded step
            steps = sorted(samples.keys())
            if not steps:
                row += f"{'--':>10} "
                continue
            target_step = min(steps, key=lambda s: abs(samples[s]["t"] - tp))
            row += f"{samples[target_step]['pos_y']:10.4f} "
        row += f"{ref_y[idx_ref]:10.4f}"
        print(row)

    print("\n=== per-segment Δpos_y per variant ===")
    print(f"{'segment':<14}{'Δh':>9} | " + " | ".join([f"{'Δy_'+v+' (dy/dh)':>22}" for v in variants]) + f" | {'Δy_log (dy/dh)':>22}")
    for name, t0, t1 in SLOW_SEGMENTS:
        idx_a_ref = int(np.argmin(np.abs(ref_t - t0)))
        idx_b_ref = int(np.argmin(np.abs(ref_t - t1)))
        dh = ref_h[idx_b_ref] - ref_h[idx_a_ref]
        dy_log = ref_y[idx_b_ref] - ref_y[idx_a_ref]
        ratio_log = dy_log / dh if abs(dh) > 1e-4 else float("nan")
        row = f"{name:<14}{dh:+9.4f} | "
        for v in variants:
            samples = rows_by_variant[v]["samples"]
            steps = sorted(samples.keys())
            if not steps or rows_by_variant[v]["failed"]:
                row += f"{'--':>22} | "
                continue
            sa = min(steps, key=lambda s: abs(samples[s]["t"] - t0))
            sb = min(steps, key=lambda s: abs(samples[s]["t"] - t1))
            dy = samples[sb]["pos_y"] - samples[sa]["pos_y"]
            ratio = dy / dh if abs(dh) > 1e-4 else float("nan")
            row += f"{dy:+8.4f} ({ratio:+7.3f}) | "
        row += f"{dy_log:+8.4f} ({ratio_log:+7.3f})"
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        help="Path to logs/manual/run_*/telemetry.csv. Default: latest run.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "ff_off", "ff_flip"],
        choices=["baseline", "ff_off", "ff_flip"],
    )
    parser.add_argument("--duration", type=float, default=None, help="Override sim duration (s).")
    args = parser.parse_args()

    csv_path = args.log if args.log else latest_manual_log()
    print(f"Replaying cmd_height schedule from: {csv_path}")
    ref_t, ref_h, ref_y = parse_log(csv_path)
    print(f"  {len(ref_t)} rows, t in [{ref_t[0]:.2f}, {ref_t[-1]:.2f}]s")
    print(f"  cmd_h in [{ref_h.min():.4f}, {ref_h.max():.4f}]")

    duration = float(args.duration if args.duration else ref_t[-1])
    sample_times = ref_t

    Path("tmp").mkdir(exist_ok=True)

    results: dict = {}
    for variant in args.variants:
        print(f"\n--- running variant: {variant} ---")
        results[variant] = run_variant(variant, ref_t, ref_h, duration, sample_times)
        r = results[variant]
        if r["failed"]:
            print(f"  FAILED: {r['fail_reason']}")
        else:
            print(f"  OK, {len(r['samples'])} samples recorded")

    summarize(results, ref_t, ref_y, ref_h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
