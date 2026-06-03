"""Infer and apply HID report mappings for gamepad calibration captures."""

from __future__ import annotations

import json
import statistics
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.gamepad import XboxState, _apply_deadzone


AXIS_ACTIONS: dict[str, tuple[str, str]] = {
    "left_x": ("left_x_pos", "left_x_neg"),
    "left_y": ("left_y_pos", "left_y_neg"),
}

TRIGGER_ACTIONS: dict[str, str] = {
    "lt": "lt_pos",
    "rt": "rt_pos",
}

BUTTON_ACTIONS: dict[str, str] = {
    "a": "button_a",
    "b": "button_b",
    "x": "button_x",
    "y": "button_y",
    "lb": "button_lb",
    "rb": "button_rb",
}

FIELD_FORMATS: dict[str, tuple[str, int]] = {
    "u8": ("<B", 1),
    "i8": ("<b", 1),
    "u16_le": ("<H", 2),
    "i16_le": ("<h", 2),
    "u16_be": (">H", 2),
    "i16_be": (">h", 2),
}

FORMAT_PREFERENCE: dict[str, float] = {
    "u16_le": 1.00,
    "i16_le": 1.00,
    "u16_be": 0.80,
    "i16_be": 0.80,
    "u8": 0.60,
    "i8": 0.60,
}


@dataclass(frozen=True)
class HidSample:
    action: str
    data: bytes
    timestamp: float = 0.0


def read_field(data: bytes, kind: str, offset: int) -> float:
    fmt, size = FIELD_FORMATS[kind]
    if offset < 0 or offset + size > len(data):
        raise ValueError(f"field {kind}@{offset} outside report length {len(data)}")
    return float(struct.unpack_from(fmt, data, offset)[0])


def report_group(data: bytes) -> tuple[int, int]:
    report_id = data[0] if data else -1
    return report_id, len(data)


def select_report_group(samples: Iterable[HidSample]) -> tuple[int, int]:
    sample_list = list(samples)
    counts: dict[tuple[int, int], int] = {}
    for sample in sample_list:
        group = report_group(sample.data)
        counts[group] = counts.get(group, 0) + 1

    best_group: tuple[int, int] | None = None
    best_score: tuple[int, int] = (-1, -1)
    for group, count in counts.items():
        action_count = len({sample.action for sample in sample_list if report_group(sample.data) == group})
        score = (action_count, count)
        if score > best_score:
            best_group = group
            best_score = score
    if best_group is None:
        raise ValueError("no HID reports captured")
    return best_group


def infer_mapping(
    samples: Iterable[HidSample],
    *,
    device_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sample_list = list(samples)
    group = select_report_group(sample_list)
    grouped = [sample for sample in sample_list if report_group(sample.data) == group]
    report_id, report_length = group

    mapping: dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": dict(device_info or {}),
        "report": {"report_id": report_id, "length": report_length},
        "axes": {},
        "triggers": {},
        "buttons": {},
        "warnings": [],
    }

    for name, (pos_action, neg_action) in AXIS_ACTIONS.items():
        candidate = _infer_axis(grouped, pos_action, neg_action)
        if candidate is None:
            mapping["warnings"].append(f"axis {name}: no reliable field found")
        else:
            mapping["axes"][name] = candidate

    for name, action in TRIGGER_ACTIONS.items():
        candidate = _infer_trigger(grouped, action)
        if candidate is None:
            mapping["warnings"].append(f"trigger {name}: no reliable field found")
        else:
            mapping["triggers"][name] = candidate

    for name, action in BUTTON_ACTIONS.items():
        candidate = _infer_button(grouped, action)
        if candidate is not None:
            mapping["buttons"][name] = candidate

    return mapping


def decode_report(
    data: bytes,
    mapping: Mapping[str, Any],
    *,
    deadzone: float = 0.10,
) -> tuple[XboxState, dict[str, bool]]:
    expected = mapping.get("report", {})
    expected_id = expected.get("report_id")
    expected_length = expected.get("length")
    if expected_id is not None and data and data[0] != int(expected_id):
        raise ValueError(f"unexpected report id 0x{data[0]:02x}; expected 0x{int(expected_id):02x}")
    if expected_length is not None and len(data) != int(expected_length):
        raise ValueError(f"unexpected report length {len(data)}; expected {int(expected_length)}")

    axes = mapping.get("axes", {})
    triggers = mapping.get("triggers", {})
    buttons_cfg = mapping.get("buttons", {})

    left_x = _decode_axis(data, axes.get("left_x"), deadzone=deadzone)
    left_y = _decode_axis(data, axes.get("left_y"), deadzone=deadzone)
    lt = _decode_trigger(data, triggers.get("lt"))
    rt = _decode_trigger(data, triggers.get("rt"))
    buttons = {
        name: _decode_button(data, cfg)
        for name, cfg in buttons_cfg.items()
    }
    return XboxState(left_x=left_x, left_y=left_y, lt=lt, rt=rt), buttons


def decode_report_dict(
    data: bytes,
    mapping: Mapping[str, Any],
    *,
    deadzone: float = 0.10,
) -> dict[str, Any]:
    state, buttons = decode_report(data, mapping, deadzone=deadzone)
    return {
        "left_x": state.left_x,
        "left_y": state.left_y,
        "lt": state.lt,
        "rt": state.rt,
        "buttons": buttons,
    }


def write_decoder_script(mapping_path: Path, script_path: Path) -> None:
    text = f'''#!/usr/bin/env python3
"""Generated HID decoder launcher."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAPPING = Path({str(mapping_path.resolve())!r})


def main() -> int:
    return subprocess.call([
        sys.executable,
        str(ROOT / "scripts" / "decode_gamepad_hid.py"),
        "--mapping",
        str(MAPPING),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
'''
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(text)
    script_path.chmod(0o755)


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def save_mapping(path: Path, mapping: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
        f.write("\n")


def _field_candidates(report_length: int) -> Iterable[tuple[str, int]]:
    for kind, (_, size) in FIELD_FORMATS.items():
        for offset in range(1, report_length - size + 1):
            yield kind, offset


def _infer_axis(samples: list[HidSample], pos_action: str, neg_action: str) -> dict[str, Any] | None:
    pos_samples = [sample for sample in samples if sample.action == pos_action]
    neg_samples = [sample for sample in samples if sample.action == neg_action]
    if not pos_samples or not neg_samples:
        return None

    best: dict[str, Any] | None = None
    best_score: tuple[float, float, float] = (0.0, 0.0, 0.0)
    report_length = len(samples[0].data)
    for kind, offset in _field_candidates(report_length):
        pos_values = _values(pos_samples, kind, offset)
        neg_values = _values(neg_samples, kind, offset)
        if not pos_values or not neg_values:
            continue
        pos_mean = statistics.fmean(pos_values)
        neg_mean = statistics.fmean(neg_values)
        separation = abs(pos_mean - neg_mean)
        jitter = _spread(pos_values) + _spread(neg_values) + 1.0
        byte_support = _byte_support(pos_samples, neg_samples, offset, FIELD_FORMATS[kind][1])
        if byte_support < _minimum_byte_support(kind):
            continue
        confidence = min(1.0, separation / _format_range(kind)) / jitter
        rank = (byte_support, FORMAT_PREFERENCE[kind], confidence)
        if rank <= best_score or separation < _minimum_separation(kind):
            continue

        rest_values = _values(
            [sample for sample in samples if sample.action not in (pos_action, neg_action)],
            kind,
            offset,
        )
        center = statistics.median(rest_values) if rest_values else 0.5 * (pos_mean + neg_mean)
        scale = max(abs(pos_mean - center), abs(neg_mean - center), 1.0)
        best_score = rank
        best = {
            "kind": kind,
            "offset": offset,
            "center": center,
            "scale": scale,
            "invert": pos_mean < neg_mean,
            "score": confidence,
            "positive_action": pos_action,
            "negative_action": neg_action,
        }
    return best


def _infer_trigger(samples: list[HidSample], press_action: str) -> dict[str, Any] | None:
    press_samples = [sample for sample in samples if sample.action == press_action]
    rest_samples = [sample for sample in samples if sample.action != press_action]
    if not press_samples or not rest_samples:
        return None

    best: dict[str, Any] | None = None
    best_score: tuple[float, float, float] = (0.0, 0.0, 0.0)
    report_length = len(samples[0].data)
    for kind, offset in _field_candidates(report_length):
        press_values = _values(press_samples, kind, offset)
        rest_values = _values(rest_samples, kind, offset)
        if not press_values or not rest_values:
            continue
        press_mean = statistics.fmean(press_values)
        rest_center = statistics.median(rest_values)
        separation = abs(press_mean - rest_center)
        jitter = _spread(press_values) + _spread(rest_values) + 1.0
        byte_support = _byte_support(press_samples, rest_samples, offset, FIELD_FORMATS[kind][1])
        if byte_support < _minimum_byte_support(kind):
            continue
        confidence = min(1.0, separation / _format_range(kind)) / jitter
        rank = (byte_support, FORMAT_PREFERENCE[kind], confidence)
        if rank <= best_score or separation < _minimum_separation(kind):
            continue
        best_score = rank
        best = {
            "kind": kind,
            "offset": offset,
            "center": rest_center,
            "scale": max(separation, 1.0),
            "invert": press_mean < rest_center,
            "score": confidence,
            "positive_action": press_action,
        }
    return best


def _infer_button(samples: list[HidSample], press_action: str) -> dict[str, Any] | None:
    press_samples = [sample for sample in samples if sample.action == press_action]
    rest_samples = [sample for sample in samples if sample.action != press_action]
    if not press_samples or not rest_samples:
        return None

    best: dict[str, Any] | None = None
    best_score = 0.0
    report_length = len(samples[0].data)
    for offset in range(1, report_length):
        for bit in range(8):
            mask = 1 << bit
            press_ratio = _bit_ratio(press_samples, offset, mask)
            rest_ratio = _bit_ratio(rest_samples, offset, mask)
            score = abs(press_ratio - rest_ratio)
            if score <= best_score or score < 0.80:
                continue
            best_score = score
            best = {
                "offset": offset,
                "mask": mask,
                "pressed_when_set": press_ratio > rest_ratio,
                "score": score,
                "positive_action": press_action,
            }
    return best


def _values(samples: list[HidSample], kind: str, offset: int) -> list[float]:
    values: list[float] = []
    for sample in samples:
        try:
            values.append(read_field(sample.data, kind, offset))
        except ValueError:
            continue
    return values


def _spread(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def _minimum_separation(kind: str) -> float:
    if kind in ("u8", "i8"):
        return 3.0
    return 64.0


def _format_range(kind: str) -> float:
    if kind in ("u8", "i8"):
        return 255.0
    return 65535.0


def _minimum_byte_support(kind: str) -> int:
    if kind in ("u8", "i8"):
        return 1
    return 2


def _byte_support(
    active_samples: list[HidSample],
    inactive_samples: list[HidSample],
    offset: int,
    size: int,
) -> float:
    support = 0
    for byte_offset in range(offset, offset + size):
        active = [sample.data[byte_offset] for sample in active_samples if byte_offset < len(sample.data)]
        inactive = [sample.data[byte_offset] for sample in inactive_samples if byte_offset < len(sample.data)]
        if not active or not inactive:
            continue
        if statistics.median(active) != statistics.median(inactive):
            support += 1
    return float(support)


def _bit_ratio(samples: list[HidSample], offset: int, mask: int) -> float:
    if not samples:
        return 0.0
    active = sum(1 for sample in samples if offset < len(sample.data) and sample.data[offset] & mask)
    return active / len(samples)


def _decode_axis(data: bytes, cfg: Mapping[str, Any] | None, *, deadzone: float) -> float:
    if not cfg:
        return 0.0
    raw = read_field(data, str(cfg["kind"]), int(cfg["offset"]))
    value = (raw - float(cfg["center"])) / float(cfg["scale"])
    if bool(cfg.get("invert", False)):
        value = -value
    value = max(-1.0, min(1.0, value))
    return _apply_deadzone(value, deadzone)


def _decode_trigger(data: bytes, cfg: Mapping[str, Any] | None) -> float:
    if not cfg:
        return 0.0
    raw = read_field(data, str(cfg["kind"]), int(cfg["offset"]))
    value = (raw - float(cfg["center"])) / float(cfg["scale"])
    if bool(cfg.get("invert", False)):
        value = -value
    return max(0.0, min(1.0, value))


def _decode_button(data: bytes, cfg: Mapping[str, Any]) -> bool:
    offset = int(cfg["offset"])
    mask = int(cfg["mask"])
    if offset >= len(data):
        return False
    is_set = bool(data[offset] & mask)
    return is_set if bool(cfg.get("pressed_when_set", True)) else not is_set
