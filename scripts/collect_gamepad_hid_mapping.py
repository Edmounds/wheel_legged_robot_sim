"""Interactive HID capture tool for unknown gamepad reports.

The script records raw HID reports for guided actions, infers field offsets,
then writes:
- a raw capture JSON under tmp/
- an inferred mapping JSON under tmp/
- a small generated decoder launcher under tmp/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.gamepad_hid_mapping import HidSample, infer_mapping, save_mapping, write_decoder_script


READ_SIZE = 128
DEFAULT_DURATION = 1.2
DEFAULT_TIMEOUT_MS = 20
REQUIRED_STEPS: list[tuple[str, str]] = [
    ("neutral", "Release all sticks, triggers, and buttons"),
    ("left_x_pos", "Push left stick fully RIGHT and hold"),
    ("left_x_neg", "Push left stick fully LEFT and hold"),
    ("left_y_pos", "Push left stick fully UP and hold"),
    ("left_y_neg", "Push left stick fully DOWN and hold"),
    ("lt_pos", "Press LT fully and hold"),
    ("rt_pos", "Press RT fully and hold"),
]
OPTIONAL_STEPS: list[tuple[str, str]] = [
    ("button_a", "Press A and hold"),
    ("button_b", "Press B and hold"),
    ("button_x", "Press X and hold"),
    ("button_y", "Press Y and hold"),
    ("button_lb", "Press LB and hold"),
    ("button_rb", "Press RB and hold"),
]


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import hid
    except ImportError:
        print("hidapi is not installed in this environment.")
        return 2

    devices = list(hid.enumerate())
    if not devices:
        print("No HID devices found.")
        return 1

    selected = _select_device(devices, args.device_index)
    if selected is None:
        return 1

    print("\nSelected HID device:")
    print(_device_line(-1, selected))
    print(f"path={_format_path(selected.get('path'))}")

    dev = hid.device()
    try:
        dev.open_path(selected["path"])
    except Exception as exc:
        print(f"Failed to open HID device: {exc}")
        return 1

    if hasattr(dev, "set_nonblocking"):
        try:
            dev.set_nonblocking(False)
        except Exception:
            pass

    steps = list(REQUIRED_STEPS)
    if not args.skip_buttons:
        steps.extend(OPTIONAL_STEPS)

    raw_records: list[dict[str, Any]] = []
    samples: list[HidSample] = []

    print("\nFor each step, set the controller state first, then press Return.")
    print("Keep holding that state until the capture line finishes.")
    print("Use Ctrl+C to stop.\n")

    try:
        for action, instruction in steps:
            input(f"[{action}] {instruction}. Press Return to capture...")
            records = _capture_action(
                dev,
                action=action,
                duration=args.duration,
                timeout_ms=args.timeout_ms,
            )
            raw_records.extend(records)
            unique_reports = {record["hex"] for record in records}
            print(
                f"  captured {len(records)} reports, "
                f"{len(unique_reports)} unique, "
                f"groups={_report_groups(records)}"
            )
            for record in records:
                samples.append(
                    HidSample(
                        action=record["action"],
                        data=bytes.fromhex(record["hex"]),
                        timestamp=float(record["timestamp"]),
                    )
                )
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    finally:
        try:
            dev.close()
        except Exception:
            pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_path = out_dir / f"gamepad_hid_capture_{stamp}.json"
    mapping_path = out_dir / f"gamepad_hid_mapping_{stamp}.json"
    decoder_path = out_dir / f"decode_gamepad_hid_{stamp}.py"

    capture = {
        "created_at": datetime.now().isoformat(),
        "duration": args.duration,
        "timeout_ms": args.timeout_ms,
        "device": _json_device(selected),
        "records": raw_records,
    }
    capture_path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")

    mapping = infer_mapping(samples, device_info=_json_device(selected))
    save_mapping(mapping_path, mapping)
    write_decoder_script(mapping_path, decoder_path)

    print("\nAnalysis result:")
    _print_mapping_summary(mapping)
    print("\nFiles:")
    print(f"  raw capture: {capture_path}")
    print(f"  mapping:     {mapping_path}")
    print(f"  decoder:     {decoder_path}")
    print("\nRun:")
    print(f"  uv run python scripts/decode_gamepad_hid.py --mapping {mapping_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, help="HID device index from the printed list.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Seconds to capture each action.")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="hidapi read timeout per read.")
    parser.add_argument("--out-dir", default="tmp", help="Directory for capture, mapping, and generated decoder.")
    parser.add_argument("--skip-buttons", action="store_true", help="Only calibrate sticks and triggers.")
    return parser.parse_args()


def _select_device(devices: list[dict[str, Any]], device_index: int | None) -> dict[str, Any] | None:
    ranked = sorted(enumerate(devices), key=lambda item: _device_rank(item[1]))
    print("HID devices:")
    for original_index, info in ranked:
        print(_device_line(original_index, info))

    if device_index is not None:
        if device_index < 0 or device_index >= len(devices):
            print(f"Invalid --device-index {device_index}")
            return None
        return devices[device_index]

    likely = [idx for idx, info in ranked if _looks_like_gamepad(info)]
    default = likely[0] if likely else ranked[0][0]
    choice = input(f"\nDevice index [{default}]: ").strip()
    if not choice:
        return devices[default]
    try:
        selected = int(choice)
    except ValueError:
        print("Invalid device index.")
        return None
    if selected < 0 or selected >= len(devices):
        print("Invalid device index.")
        return None
    return devices[selected]


def _capture_action(dev: Any, *, action: str, duration: float, timeout_ms: int) -> list[dict[str, Any]]:
    end_time = time.perf_counter() + duration
    records: list[dict[str, Any]] = []
    while time.perf_counter() < end_time:
        try:
            data = dev.read(READ_SIZE, timeout_ms=timeout_ms)
        except Exception as exc:
            print(f"  read failed: {exc}")
            break
        if not data:
            continue
        raw = bytes(data)
        now = time.time()
        records.append(
            {
                "action": action,
                "timestamp": now,
                "length": len(raw),
                "report_id": raw[0] if raw else None,
                "hex": raw.hex(),
            }
        )
    return records


def _device_rank(info: dict[str, Any]) -> tuple[int, str]:
    likely = 0 if _looks_like_gamepad(info) else 1
    name = f"{info.get('manufacturer_string') or ''} {info.get('product_string') or ''}"
    return likely, name.lower()


def _looks_like_gamepad(info: dict[str, Any]) -> bool:
    text = " ".join(
        str(info.get(key) or "")
        for key in ("manufacturer_string", "product_string")
    ).lower()
    if info.get("vendor_id") == 0x045E:
        return True
    return any(token in text for token in ("xbox", "controller", "gamepad", "joystick", "microsoft"))


def _device_line(index: int, info: dict[str, Any]) -> str:
    prefix = f"[{index:02d}]" if index >= 0 else "    "
    vid = int(info.get("vendor_id") or 0)
    pid = int(info.get("product_id") or 0)
    manufacturer = info.get("manufacturer_string") or "?"
    product = info.get("product_string") or "?"
    usage_page = info.get("usage_page")
    usage = info.get("usage")
    interface = info.get("interface_number")
    return (
        f"{prefix} vid=0x{vid:04x} pid=0x{pid:04x} "
        f"usage=0x{int(usage_page or 0):04x}/0x{int(usage or 0):04x} "
        f"if={interface} {manufacturer} / {product}"
    )


def _format_path(value: Any) -> str:
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _json_device(info: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, bytes):
            result[key] = value.hex()
        else:
            result[key] = value
    return result


def _report_groups(records: list[dict[str, Any]]) -> str:
    counts: dict[tuple[int, int], int] = {}
    for record in records:
        key = (int(record["report_id"] or -1), int(record["length"]))
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"0x{rid:02x}/{length}:{count}" for (rid, length), count in sorted(counts.items()))


def _print_mapping_summary(mapping: dict[str, Any]) -> None:
    print(json.dumps({
        "report": mapping.get("report"),
        "axes": mapping.get("axes"),
        "triggers": mapping.get("triggers"),
        "buttons": mapping.get("buttons"),
        "warnings": mapping.get("warnings"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
