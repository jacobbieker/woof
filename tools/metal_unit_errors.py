"""Print each Metal error of one translated unit beside its MSL line.

Usage::

    pixi run -e mlx python tools/metal_unit_errors.py UNIT [--retain] [--limit N]
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("unit")
    ap.add_argument("--retain", action="store_true")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args(argv)
    if args.retain:
        os.environ["GPUWM_WRF_EXACT"] = "1"
    import woof  # noqa: F401
    from woof.backend.metal import ext
    from woof.backend.metal.loader import _translated, contracts, retains_subnormals
    from woof.core import kernels
    src = kernels.module_source(args.unit)
    opts = kernels.module_options(args.unit)
    from woof import numerics
    from woof.backend.translate import option_defines, option_include_dirs
    effective = numerics.effective_options(tuple(opts or ()))
    unit = _translated(src, args.unit, contracts(opts), retains_subnormals(args.unit, src, opts),
                       option_defines(effective), option_include_dirs(effective))
    lines = unit.msl.splitlines()
    try:
        ext.load().compile("errors:" + unit.key, unit.msl, sorted(unit.kernels), 0)
        print("compiles")
        return 0
    except RuntimeError as exc:
        shown = 0
        for line in str(exc).splitlines():
            if "symbol(s) not found" in line or line.startswith("  '"):
                print(line)
            m = re.match(r"program_source:(\d+):(\d+): error: (.*)", line)
            if m and shown < args.limit:
                n = int(m.group(1))
                print(f"{n}: {m.group(3)}\n    {lines[n - 1].strip()[:200]}")
                shown += 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
