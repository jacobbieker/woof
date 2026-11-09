"""Cold Metal compile time of kernel units, soft-float binary64 against df64.

Each (unit, mode) builds in its own interpreter with the pipeline archives
off (``WOOF_METAL_PIPELINE_DIR=off``) and ``WOOF_METAL_F64`` set to the
mode, so neither the archive nor the first build of the other mode serves
it.  The macOS shader cache may still hold a library built earlier from the
same MSL, so each build's MSL carries a fresh nonce comment (a new library
for every cache), unless ``--no-nonce``.

    pixi run -e mlx python tools/df64_compile_bench.py noahmp_water nssl2_qvexcess
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def one(name: str, nonce: bool) -> dict:
    import dataclasses
    import woof  # noqa: F401  -- installs the facade
    from woof.backend.metal.loader import MetalModule, translate_unit
    from woof.core import kernels
    source, options = kernels.module_source(name), kernels.module_options(name)
    t0 = time.perf_counter()
    unit = translate_unit(name, source, options=options)
    if nonce:
        msl = f"// nonce {time.time_ns()}\n" + unit.msl
        unit = dataclasses.replace(unit, msl=msl, key=hashlib.sha256(msl.encode()).hexdigest())
    module = MetalModule(unit, name)
    t1 = time.perf_counter()
    module.build_all()
    t2 = time.perf_counter()
    return {"unit": name, "mode": os.environ.get("WOOF_METAL_F64"), "translate_s": round(t1 - t0, 2),
            "compile_s": round(t2 - t1, 2), "msl_bytes": len(module.unit.msl), "kernels": len(module.unit.kernels)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("units", nargs="+")
    parser.add_argument("--modes", default="soft,df64")
    parser.add_argument("--no-nonce", action="store_true")
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.one:
        print(json.dumps(one(args.units[0], not args.no_nonce)))
        return 0
    status = 0
    for name in args.units:
        for mode in args.modes.split(","):
            env = dict(os.environ, WOOF_METAL_F64=mode, WOOF_METAL_PIPELINE_DIR="off", WOOF_BACKEND="mlx")
            cmd = [sys.executable, __file__, "--one", name] + (["--no-nonce"] if args.no_nonce else [])
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  env=env, cwd=ROOT)
            rows = [l for l in proc.stdout.splitlines() if l.startswith("{")]
            if rows:
                print(rows[-1], flush=True)
            else:
                status = 1
                print(json.dumps({"unit": name, "mode": mode, "error": (proc.stderr or proc.stdout)[-400:]}), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
