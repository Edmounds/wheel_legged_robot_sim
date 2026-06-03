from __future__ import annotations

import struct

import pytest

from sim.gamepad_hid_mapping import HidSample, decode_report, infer_mapping


def _report(
    *,
    left_x: int = 0,
    left_y: int = 0,
    lt: int = 0,
    rt: int = 0,
    buttons: int = 0,
) -> bytes:
    data = bytearray(16)
    data[0] = 0x31
    struct.pack_into("<h", data, 3, left_x)
    struct.pack_into("<h", data, 5, left_y)
    struct.pack_into("<H", data, 7, lt)
    struct.pack_into("<H", data, 9, rt)
    data[12] = buttons
    return bytes(data)


def _samples() -> list[HidSample]:
    reports: list[tuple[str, bytes]] = [
        ("neutral", _report()),
        ("neutral", _report()),
        ("left_x_pos", _report(left_x=20_000)),
        ("left_x_pos", _report(left_x=21_000)),
        ("left_x_neg", _report(left_x=-20_000)),
        ("left_x_neg", _report(left_x=-21_000)),
        ("left_y_pos", _report(left_y=18_000)),
        ("left_y_neg", _report(left_y=-18_000)),
        ("lt_pos", _report(lt=900)),
        ("rt_pos", _report(rt=950)),
        ("button_a", _report(buttons=0b0000_0001)),
        ("button_b", _report(buttons=0b0000_0010)),
    ]
    return [HidSample(action=action, data=data, timestamp=i * 0.01) for i, (action, data) in enumerate(reports)]


def test_infer_mapping_finds_report_fields() -> None:
    mapping = infer_mapping(_samples(), device_info={"vendor_id": 0x45E, "product_id": 0x0B13})

    assert mapping["report"] == {"report_id": 0x31, "length": 16}
    assert mapping["axes"]["left_x"]["offset"] == 3
    assert mapping["axes"]["left_y"]["offset"] == 5
    assert mapping["triggers"]["lt"]["offset"] == 7
    assert mapping["triggers"]["rt"]["offset"] == 9
    assert mapping["buttons"]["a"]["offset"] == 12
    assert mapping["buttons"]["a"]["mask"] == 1
    assert mapping["buttons"]["b"]["mask"] == 2


def test_decode_report_uses_inferred_mapping() -> None:
    mapping = infer_mapping(_samples())
    state, buttons = decode_report(
        _report(left_x=20_000, left_y=-18_000, lt=450, rt=950, buttons=0b0000_0011),
        mapping,
        deadzone=0.0,
    )

    assert state.left_x == pytest.approx(1.0, abs=0.06)
    assert state.left_y == pytest.approx(-1.0, abs=0.06)
    assert state.lt == pytest.approx(0.5, abs=0.06)
    assert state.rt == pytest.approx(1.0, abs=0.06)
    assert buttons["a"] is True
    assert buttons["b"] is True


def test_decode_report_rejects_unexpected_report_group() -> None:
    mapping = infer_mapping(_samples())
    bad = bytearray(_report())
    bad[0] = 0x30

    with pytest.raises(ValueError, match="unexpected report id"):
        decode_report(bytes(bad), mapping)
