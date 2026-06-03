"""End-to-end smoke test for sim.gamepad (native GameController/hidapi backend).

Move left stick / squeeze LT / squeeze RT for 30 seconds. Each state change
prints a line showing parsed XboxState and mapped cmd_* values.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.gamepad import GamepadCommandMapper, open_gamepad


def main() -> None:
    g = open_gamepad()
    if g is None:
        print("No gamepad opened. Check USB or Bluetooth connection.")
        return
    print(f"Opened: {g.name}")
    mapper = GamepadCommandMapper(
        linear_range=(-1.0, 1.0),
        angular_range=(-3.0, 3.0),
        height_range=(0.07844, 0.142),
    )
    print("Move left stick / press triggers for 30 seconds. Ctrl+C to stop.")
    print("-" * 70)

    t0 = time.perf_counter()
    last = None
    try:
        while time.perf_counter() - t0 < 30.0:
            # Pump the Cocoa event loop for headless CLI environments
            try:
                from Foundation import NSRunLoop, NSDate
                loop = NSRunLoop.currentRunLoop()
                date = NSDate.dateWithTimeIntervalSinceNow_(0.01)
                loop.runUntilDate_(date)
            except ImportError:
                pass

            state = g.poll()
            if state is not None:
                cur = (round(state.left_x, 2), round(state.left_y, 2),
                       round(state.right_x, 2), round(state.right_y, 2),
                       round(state.lt, 2), round(state.rt, 2),
                       bool(state.button_a))
                if cur != last:
                    last = cur
                    lin, ang, h, jump = mapper.map(state)
                    print(
                        f"  lx={state.left_x:+.3f} ly={state.left_y:+.3f} "
                        f"rx={state.right_x:+.3f} ry={state.right_y:+.3f} "
                        f"lt={state.lt:.3f} rt={state.rt:.3f} A={state.button_a}  ->  "
                        f"lin={lin:+.3f} ang={ang:+.3f} h={h:.4f} jump={jump:.0f}",
                        flush=True,
                    )
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        g.close()
    print("-" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
