from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutMetrics:
    duration: float
    finite: bool
    fell: bool
    min_base_height: float
    max_base_height: float
    max_jump_height: float
    max_abs_pitch: float
    max_abs_y_drift: float
    forward_distance: float
    contact_count: int
    control_effort: float
    saturation_ratio: float
