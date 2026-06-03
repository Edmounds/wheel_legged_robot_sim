from __future__ import annotations

from pathlib import Path

from scripts.verify_single_wheel_ramp import run_single_wheel_ramp


def test_roll_leveling_reduces_single_wheel_ramp_roll(tmp_path: Path) -> None:
    enabled_cache = tmp_path / "enabled"
    disabled_cache = tmp_path / "disabled"
    enabled_cache.mkdir()
    disabled_cache.mkdir()

    enabled = run_single_wheel_ramp(
        duration=4.0,
        target_velocity=0.12,
        terrain_side="left",
        cache_dir=enabled_cache,
        enable_roll_leveling=True,
    )
    disabled = run_single_wheel_ramp(
        duration=4.0,
        target_velocity=0.12,
        terrain_side="left",
        cache_dir=disabled_cache,
        enable_roll_leveling=False,
    )

    assert enabled.finite
    assert not enabled.fell
    assert disabled.finite
    assert not disabled.fell
    assert enabled.max_abs_roll < disabled.max_abs_roll * 0.95


def test_slope_squat_levels_full_platform_single_wheel_step(tmp_path: Path) -> None:
    """单轮完全爬上 65mm 平台时, 前馈找平+上坡降站高应把 base 调到近水平。

    对照: 找平关闭时 base 会大幅侧倾。两者都驱动到同一轮抬升量 (~65mm) 再比较。
    """
    enabled_cache = tmp_path / "enabled"
    disabled_cache = tmp_path / "disabled"
    enabled_cache.mkdir()
    disabled_cache.mkdir()

    enabled = run_single_wheel_ramp(
        duration=8.0,
        target_velocity=0.12,
        terrain_side="left",
        cache_dir=enabled_cache,
        enable_roll_leveling=True,
        full_platform=True,
    )
    disabled = run_single_wheel_ramp(
        duration=8.0,
        target_velocity=0.12,
        terrain_side="left",
        cache_dir=disabled_cache,
        enable_roll_leveling=False,
        full_platform=True,
    )

    # 两个对照都真的把单轮抬到了接近 65mm 平台高度 (否则比较无意义)。
    assert enabled.max_wheel_height_delta > 0.05
    assert disabled.max_wheel_height_delta > 0.05
    assert enabled.finite and not enabled.fell
    assert disabled.finite
    # 找平后 base 近水平; 关闭找平时大幅侧倾。
    assert enabled.final_abs_roll < 0.05
    assert enabled.max_abs_roll < disabled.max_abs_roll * 0.5
