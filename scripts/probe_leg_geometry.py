#!/usr/bin/env python3
"""Generate sim/controllers/leg_height_lut.json.

The canonical generator is analytic because the dynamic sweep can settle on a
soft-constraint floor near the low posture. Keep this entrypoint for the older
command name used by tests and notes.
"""
from __future__ import annotations

import sys

from probe_leg_geometry_analytic import main


if __name__ == "__main__":
    sys.exit(main())
