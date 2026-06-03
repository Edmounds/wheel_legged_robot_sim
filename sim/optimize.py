"""LQR+VMC 参数优化器。

支持多场景（stand/drive/jump）独立评分，对数均匀分布采样，
warm start 从已知最佳参数出发。
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import time
import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import gc

import numpy as np
import optuna

from sim.controllers.combined import CombinedController, CombinedParams
from sim.controllers.default_params import STAND_PARAMS, DEFAULT_PHASE_PARAMS
from sim.controllers.phase import JumpPhaseMachine, JumpPhaseParams
from sim.controllers.vmc import VmcParams
from sim.metrics import RolloutMetrics
from sim.rollout import RolloutConfig, run_rollout


@dataclass(frozen=True)
class OptimizationConfig:
    """优化器配置。

    Attributes:
        trials: 采样轮数。
        duration: 每轮仿真时长 (s)。
        scenario: 仿真场景 (stand/jump/drive)。
        seed: 随机种子。
        cache_dir: 预处理模型缓存目录。
        warm_start: 是否将第一轮设为已知最佳参数。
    """
    trials: int
    duration: float
    scenario: str = "jump"
    seed: int = 0
    cache_dir: Path | None = None
    warm_start: bool = True
    workers: int = 4
    log_path: Path | None = None


@dataclass(frozen=True)
class OptimizationTrial:
    """单轮优化结果。"""
    params: CombinedParams
    score: float
    metrics: RolloutMetrics
    failure_reason: str | None


@dataclass(frozen=True)
class OptimizationResult:
    """完整优化结果。"""
    trials: tuple[OptimizationTrial, ...]
    best_trial: OptimizationTrial


# --- 分场景评分函数 ---

def score_metrics(
    metrics: RolloutMetrics,
    scenario: str = "stand",
    nominal_height: float = -0.026,
) -> float:
    """根据场景计算评分。

    不同场景使用不同权重，确保评分在对应场景下有足够区分度。

    Args:
        metrics: rollout 指标。
        scenario: 场景类型 (stand/drive/jump)。
        nominal_height: 标称站立高度，仅 stand 场景使用。

    Returns:
        标量评分，越大越好。
    """
    if scenario == "stand":
        return _score_stand(metrics, nominal_height)
    elif scenario == "drive":
        return _score_drive(metrics)
    elif scenario == "jump":
        return _score_jump(metrics)
    elif scenario == "stand_then_drive":
        return _score_stand_then_drive(metrics)
    else:
        raise ValueError(f"unsupported scenario for scoring: {scenario}")


def _base_penalty(metrics: RolloutMetrics, finite_penalty: float, fell_penalty: float) -> float:
    """计算通用惩罚项 (饱和、崩溃、无穷大状态)。"""
    score = 0.0
    score -= 2.0 * metrics.saturation_ratio
    if not metrics.finite:
        score -= finite_penalty
    if metrics.fell:
        score -= fell_penalty
    return score


def _score_stand(metrics: RolloutMetrics, nominal_height: float) -> float:
    """stand 评分: 稳定站立 + 高度保持。"""
    score = 0.0
    score -= 100.0 * metrics.max_abs_pitch
    height_deviation = abs(metrics.min_base_height - nominal_height)
    score -= 50.0 * height_deviation
    score -= 0.01 * metrics.control_effort
    score += _base_penalty(metrics, finite_penalty=200.0, fell_penalty=100.0)
    if metrics.finite and not metrics.fell:
        score += 10.0
    return score


def _score_drive(metrics: RolloutMetrics) -> float:
    """drive 评分: 前进距离 + 行驶稳定性。"""
    score = 0.0
    score += 20.0 * metrics.forward_distance
    score -= 50.0 * metrics.max_abs_pitch
    score -= 0.005 * metrics.control_effort
    score += _base_penalty(metrics, finite_penalty=200.0, fell_penalty=100.0)
    if metrics.finite and not metrics.fell:
        score += 5.0
    return score


def _score_jump(metrics: RolloutMetrics) -> float:
    """jump 评分: 跳跃高度 + 着陆恢复。"""
    score = 0.0
    score += 200.0 * metrics.max_jump_height
    score -= 30.0 * metrics.max_abs_pitch
    score += _base_penalty(metrics, finite_penalty=200.0, fell_penalty=50.0)
    if metrics.finite and not metrics.fell:
        score += 5.0
    return score


def _score_stand_then_drive(metrics: RolloutMetrics) -> float:
    """stand_then_drive 评分: 完成10米直行 + 极小漂移 + 极小pitch。"""
    score = 0.0
    score += 100.0 * metrics.forward_distance
    score -= 500.0 * metrics.max_abs_y_drift
    score -= 100.0 * metrics.max_abs_pitch
    score += _base_penalty(metrics, finite_penalty=1000.0, fell_penalty=500.0)
    if metrics.finite and not metrics.fell:
        score += 50.0
        if metrics.forward_distance >= 9.9:
            score += 200.0
    return score


# --- 采样函数 ---

def _sample_combined_params(trial: optuna.Trial) -> CombinedParams:
    """采样 CombinedParams。

    Q/R/增益参数使用对数均匀分布，物理偏移量使用均匀分布。
    """
    return CombinedParams(
        vmc=VmcParams(
            nominal_height=trial.suggest_float("nominal_height", 0.135, 0.148),
            kp_motor=trial.suggest_float("kp_motor", 10.0, 200.0, log=True),
            kd_motor=trial.suggest_float("kd_motor", 1.0, 30.0, log=True),
            kp_land=trial.suggest_float("kp_land", 5.0, 60.0, log=True),
            kd_land=trial.suggest_float("kd_land", 1.0, 10.0, log=True),
        ),
        q_diag=np.array([
            trial.suggest_float("q_pitch", 1000.0, 100000.0, log=True),
            trial.suggest_float("q_pitch_rate", 100.0, 20000.0, log=True),
            trial.suggest_float("q_roll", 1000.0, 100000.0, log=True),
            trial.suggest_float("q_roll_rate", 100.0, 20000.0, log=True),
            trial.suggest_float("q_wheel_vel", 1e-4, 20.0, log=True),
        ]),
        r_diag=np.array([
            trial.suggest_float("r_torque1", 0.1, 20.0, log=True),
            trial.suggest_float("r_torque2", 0.1, 20.0, log=True),
        ]),
        velocity_ki=trial.suggest_float("velocity_ki", 0.005, 0.5, log=True),
        pitch_lean_gain=trial.suggest_float("pitch_lean_gain", 0.005, 0.2, log=True),
        yaw_damping=trial.suggest_float("yaw_damping", 0.1, 3.0, log=True),
        yaw_ki=trial.suggest_float("yaw_ki", 0.01, 1.0, log=True),
    )


def _sample_phase_params(trial: optuna.Trial) -> JumpPhaseParams:
    """采样跳跃相位参数。

    NOTE: 此函数目前 broken — jump scenario 在 _objective 里调用
    machine.start_jump() 时未传 trajectory (API 已改). 调用 jump 场景前需要
    先修这个调用点 + 选定 air_height_max 等 trajectory 参数.
    """
    return JumpPhaseParams(
        flight_timeout=trial.suggest_float("flight_timeout", 0.05, 0.60),
    )


# --- 优化主逻辑 ---

def _serialize_dict(obj):
    """递归将字典中的 ndarray 等转换为可序列化类型"""
    if isinstance(obj, dict):
        return {k: _serialize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _objective(trial: optuna.Trial, xml_path: Path, config: OptimizationConfig) -> float:
    params = _sample_combined_params(trial)

    if config.scenario == "jump":
        phase_params = _sample_phase_params(trial)
        machine = JumpPhaseMachine(phase_params)
        machine.start_jump()
    else:
        machine = None

    controller = CombinedController(params, phase_machine=machine)
    try:
        rollout = run_rollout(
            xml_path,
            RolloutConfig(
                duration=config.duration,
                scenario=config.scenario,
                cache_dir=config.cache_dir,
            ),
            controller=controller,
        )
        score = score_metrics(
            rollout.metrics,
            scenario=config.scenario,
            nominal_height=params.vmc.nominal_height,
        )
        metrics = rollout.metrics
        failure_reason = rollout.failure_reason
    except Exception as exc:
        metrics = RolloutMetrics(
            duration=config.duration,
            finite=False,
            fell=True,
            min_base_height=0.0,
            max_base_height=0.0,
            max_jump_height=0.0,
            max_abs_pitch=0.0,
            max_abs_y_drift=0.0,
            forward_distance=0.0,
            contact_count=0,
            control_effort=0.0,
            saturation_ratio=1.0,
        )
        score = score_metrics(
            metrics,
            scenario=config.scenario,
            nominal_height=params.vmc.nominal_height,
        )
        failure_reason = str(exc)

    status = "OK" if metrics.finite and not metrics.fell else f"FAIL ({failure_reason})"
    msg = (
        f"[Worker PID: {os.getpid()}] Trial {trial.number}: "
        f"score={score:+.4f} "
        f"status={status} "
        f"pitch={metrics.max_abs_pitch:.4f} "
        f"dist={metrics.forward_distance:.3f} "
        f"jump={metrics.max_jump_height:.4f}"
    )
    print(msg)
    if config.log_path:
        from sim.logger import setup_system_logger
        logger = setup_system_logger(config.log_path, console=False)
        logger.info(msg)

    trial.set_user_attr("metrics", _serialize_dict(asdict(metrics)))
    trial.set_user_attr("failure_reason", failure_reason)
    trial.set_user_attr("params_dict", _serialize_dict(asdict(params)))

    # Force garbage collection to prevent MuJoCo objects (MjModel/MjData) from lingering
    if 'rollout' in locals():
        del rollout
    gc.collect()

    return score


def _optimize_worker(storage_url: str, study_name: str, xml_path: Path, config: OptimizationConfig, n_trials: int):
    """单个优化 worker 进程入口。

    捕获 enqueue_trial 并发导致的 ValueError，避免多进程竞争 warm-start trial 时崩溃。
    """
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    try:
        study.optimize(lambda t: _objective(t, xml_path, config), n_trials=n_trials, gc_after_trial=True)
    except ValueError as e:
        if "Cannot tell a COMPLETE trial" in str(e):
            # 多进程并发 enqueue_trial 竞态条件，忽略并继续
            remaining = n_trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            if remaining > 0:
                study.optimize(lambda t: _objective(t, xml_path, config), n_trials=remaining, gc_after_trial=True)
        else:
            raise


def _run_single_process_optimization(
    study: optuna.Study,
    xml_path: Path,
    config: OptimizationConfig,
) -> None:
    study.optimize(lambda t: _objective(t, xml_path, config), n_trials=config.trials, gc_after_trial=True)


def optimize_parameters(xml_path: Path, config: OptimizationConfig) -> OptimizationResult:
    """运行参数优化。

    使用 Optuna 进行贝叶斯优化，支持 warm start 和分场景评分。
    现已修改为通过多进程+SQLite的方案，实现利用多核 CPU 最大加速。

    Args:
        xml_path: MuJoCo 模型 XML 路径。
        config: 优化配置。

    Returns:
        包含所有 trial 和最佳 trial 的优化结果。
    """
    if config.trials <= 0:
        raise ValueError("trials must be positive")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 创建 SQLite storage 用于多进程同步
    db_path = Path("tmp/optuna.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass

    storage_url = f"sqlite:///{db_path.absolute()}"
    study_name = f"opt_{config.scenario}_{int(time.time())}"

    study = optuna.create_study(
        storage=storage_url,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        direction="maximize"
    )

    if config.warm_start:
        study.enqueue_trial({
            "nominal_height": max(0.135, min(0.148, STAND_PARAMS.vmc.nominal_height)),
            "kp_motor": max(10.0, min(200.0, STAND_PARAMS.vmc.kp_motor)),
            "kd_motor": max(1.0, min(30.0, STAND_PARAMS.vmc.kd_motor)),
            "kp_land": max(5.0, min(60.0, STAND_PARAMS.vmc.kp_land)),
            "kd_land": max(1.0, min(10.0, STAND_PARAMS.vmc.kd_land)),
            "q_pitch": max(1000.0, STAND_PARAMS.q_diag[0]),
            "q_pitch_rate": max(100.0, STAND_PARAMS.q_diag[1]),
            "q_roll": max(1000.0, STAND_PARAMS.q_diag[2]),
            "q_roll_rate": max(100.0, STAND_PARAMS.q_diag[3]),
            "q_wheel_vel": max(1e-4, STAND_PARAMS.q_diag[4]),
            "r_torque1": max(0.1, STAND_PARAMS.r_diag[0]),
            "r_torque2": max(0.1, STAND_PARAMS.r_diag[1]),
            "velocity_ki": max(0.005, STAND_PARAMS.velocity_ki),
            "pitch_lean_gain": max(0.005, STAND_PARAMS.pitch_lean_gain),
            "yaw_damping": max(0.1, STAND_PARAMS.yaw_damping),
            "flight_timeout": max(0.05, DEFAULT_PHASE_PARAMS.flight_timeout),
        })

    n_workers = min(os.cpu_count() or 1, config.workers)
    n_workers = max(1, min(n_workers, config.trials))
    if n_workers == 1 or config.trials <= n_workers:
        print("Starting single-process optimization...")
        _run_single_process_optimization(study, xml_path, config)
    else:
        # 将 trials 分配给 CPU 核数或指定的 workers 数量
        trials_per_worker = config.trials // n_workers
        remain = config.trials % n_workers

        print(f"Starting multiprocessing optimization with {n_workers} workers...")

        # 启动进程池
        with multiprocessing.Pool(processes=n_workers) as pool:
            args = []
            for i in range(n_workers):
                k = trials_per_worker + (1 if i < remain else 0)
                if k > 0:
                    args.append((storage_url, study_name, xml_path, config, k))

            pool.starmap(_optimize_worker, args)

    trials = _aggregate_trials(study)
    if not trials:
        raise RuntimeError("Optimization failed, no valid trials completed.")

    best_trial = max(trials, key=lambda t: t.score)
    return OptimizationResult(trials=tuple(trials), best_trial=best_trial)

def _aggregate_trials(study: optuna.Study) -> list[OptimizationTrial]:
    """从 Optuna Study 提取并重建 OptimizationTrial 列表。"""
    trials = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        metrics_dict = t.user_attrs.get("metrics")
        failure_reason = t.user_attrs.get("failure_reason")
        params_dict = t.user_attrs.get("params_dict")

        if metrics_dict is None or params_dict is None:
            continue

        metrics = RolloutMetrics(**metrics_dict)
        vmc_params = VmcParams(**params_dict["vmc"])
        params = CombinedParams(
            vmc=vmc_params,
            q_diag=np.array(params_dict["q_diag"]),
            r_diag=np.array(params_dict["r_diag"]),
            velocity_ki=params_dict["velocity_ki"],
            pitch_lean_gain=params_dict["pitch_lean_gain"],
            yaw_damping=params_dict.get("yaw_damping", 0.5),
        )

        opt_trial = OptimizationTrial(
            params=params,
            score=t.value,
            metrics=metrics,
            failure_reason=failure_reason,
        )
        trials.append(opt_trial)
    return trials



def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="Run parameter optimization over MuJoCo LQR+VMC controller."
    )
    parser.add_argument("xml", type=Path, help="Path to the source MuJoCo XML model.")
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--scenario", default="stand", choices=("stand", "jump", "drive", "stand_then_drive"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use known best params for first trial (default: True).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers (default: 4 to save memory).",
    )
    args = parser.parse_args(argv)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_tuning_base = Path("logs/auto_tuning")
    log_dir = auto_tuning_base / f"opt_{timestamp}"
    log_path = log_dir / "optimization.log"
    
    from sim.logger import setup_system_logger, cleanup_old_logs
    system_logger = setup_system_logger(log_path)
    
    # 清理旧的优化日志记录，只保留最近 4 次
    cleanup_old_logs(auto_tuning_base, max_keep=4)
    
    system_logger.info("="*60)
    system_logger.info(f"Starting parameter optimization: {args.trials} trials, scenario: {args.scenario}")
    
    result = optimize_parameters(
        args.xml,
        OptimizationConfig(
            trials=args.trials,
            duration=args.duration,
            scenario=args.scenario,
            seed=args.seed,
            cache_dir=args.cache_dir or Path("tmp/mujoco_cache").resolve(),
            warm_start=args.warm_start,
            workers=args.workers,
            log_path=log_path,
        ),
    )

    system_logger.info(f"\n{'='*60}")
    system_logger.info("Optimization Finished.")
    system_logger.info(f"Best trial: score={result.best_trial.score:+.6f}")
    system_logger.info(f"  pitch={result.best_trial.metrics.max_abs_pitch:.6f}")
    system_logger.info(f"  dist={result.best_trial.metrics.forward_distance:.4f}")
    system_logger.info(f"  jump={result.best_trial.metrics.max_jump_height:.4f}")
    system_logger.info(f"  fell={result.best_trial.metrics.fell}")
    system_logger.info(f"  params={result.best_trial.params}")
    print(f"[{len(result.trials)}/{args.trials}]")
    print(f"Best trial: score={result.best_trial.score:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
