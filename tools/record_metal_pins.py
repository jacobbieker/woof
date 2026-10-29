"""Record the MLX backend's measured-word pins (woof/verify/platform_pins.py).

Each family's ``-metal`` file is measured by the same code its test runs,
under the test suite's strict numerics, and written beside the CUDA pin.
Before recording a family, show that its Metal words are CUDA's
``--fmad=false`` words (tools/launch_trace.py on both machines, CUDA with
contraction off), so that the pin records a compiler difference and not a
translation error.

    pixi run -e mlx python tools/record_metal_pins.py momentum prep ...
"""
from __future__ import annotations

import json
import os

import numpy as np
import sys
from pathlib import Path

os.environ["GPUWM_NUMERICS"] = "strict"     # as tests/conftest.py

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, data) -> None:
    import cupy as cp
    data = dict(data, recorded_on={
        "backend": "mlx", "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
        "numerics": "strict", "tool": "tools/record_metal_pins.py"})
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", path.relative_to(ROOT))


def momentum():
    from woof.verify.bigstep_momentum_oracle import (
        MOMENTUM_CASES, MOMENTUM_ORACLE_DIR, MOMENTUM_ROUTINES,
        load_momentum_fixture, measure_momentum_parity, momentum_port_outputs)
    out = {}
    for case in MOMENTUM_CASES:
        fixture = load_momentum_fixture(case)
        out[case] = {r: measure_momentum_parity(fixture.reference[r], momentum_port_outputs(fixture, r))
                     for r in MOMENTUM_ROUTINES}
    _write(MOMENTUM_ORACLE_DIR / "momentum-measurements-metal.json", out)


def prep():
    from woof.verify.bigstep_prep_oracle import (
        ORACLE_DIR, PHY_MAPPING, PREP_CASES, load_prep_fixture, measure_prep, prep_port_outputs)
    pins = json.loads((ORACLE_DIR / "prep-measurements.json").read_text())
    for case in PREP_CASES:
        fixture = load_prep_fixture(case)
        got = prep_port_outputs(fixture)
        pins[case].update(measure_prep(fixture, {"ph_tend": got["ph_tend"]}))
        pins[case].update(measure_prep(fixture, {name: got[name] for name in PHY_MAPPING}))
    _write(ORACLE_DIR / "prep-measurements-metal.json", pins)


def rk():
    from woof.verify.bigstep_rk_oracle import (
        RK_ORACLE_DIR, load_rk_oracle, load_rk_tendency_oracle, measure_rk_case,
        measure_rk_tendency_case)
    pin = json.loads((RK_ORACLE_DIR / "rk-ulp-table.json").read_text())
    arrays, meta = load_rk_oracle()
    for case in meta["cases"]:
        pin["measurements"][f"{case['kind']}:{case['name']}"] = measure_rk_case(arrays, case)
    for filename, key in (("rk-tendency.npz", "rk_tendency_measurements"),
                          ("rk-tendency-stored-theta.npz", "rk_tendency_stored_theta_measurements")):
        arrays, meta = load_rk_tendency_oracle(filename)
        for case in meta["cases"]:
            pin[key][case["name"]] = measure_rk_tendency_case(arrays, case)
    _write(RK_ORACLE_DIR / "rk-ulp-table-metal.json", pin)


def vertical():
    sys.path.insert(0, str(ROOT / "tests"))
    from test_smallstep_vertical_wrf471_parity import measure_vertical, output_sha256
    from woof.verify.smallstep_oracle import ORACLE_DIR
    pin = json.loads((ORACLE_DIR / "vertical-native.json").read_text())
    measured, outputs = measure_vertical()
    pin["cases"] = measured
    pin["native_sha256"] = {name: {field: output_sha256(value) for field, value in got.items()}
                            for name, got in outputs.items()}
    w = outputs["implicit_damper"]["w"].view(np.uint32)
    red = json.loads((ORACLE_DIR / "vertical-pi-red.json").read_text())
    pin["known_deltas"] = {"implicit_damper w": {
        "cause": "the damper's sinf is the vendor function (libdevice on CUDA, Metal precise::sin)",
        "corrected_words": [{"index": r["index"], "fortran_word": r["fortran_word"],
                             "metal_word": int(w[tuple(r["index"])])}
                            for r in red["evidence"]["native"]["repaired"]]}}
    _write(ORACLE_DIR / "vertical-native-metal.json", pin)


def horizontal():
    sys.path.insert(0, str(ROOT / "tests"))
    from test_smallstep_horizontal_wrf471_parity import output_sha256
    from woof.core.fp32_ulp import fp32_ulp_distance
    from woof.verify.smallstep_horizontal_oracle import horizontal_port_outputs, load_horizontal_fixture
    from woof.verify.smallstep_oracle import ORACLE_DIR
    pin = json.loads((ORACLE_DIR / "horizontal-baseline.json").read_text())
    pin["gpu_sha256"] = {}
    for name, data, cfg, reference, full_theta in load_horizontal_fixture(ORACLE_DIR / "horizontal.npz"):
        actual = horizontal_port_outputs(data, cfg, full_theta_probe=full_theta)
        pin["gpu_sha256"][name] = {key: output_sha256(value) for key, value in actual.items()}
        for key, got in actual.items():
            want = reference[key]
            pin[name][key] = {"words": int(got.size),
                              "max_ulp": int(fp32_ulp_distance(got, want).max()),
                              "different_words": int(np.count_nonzero(got.view(np.uint32) != want.view(np.uint32)))}
    _write(ORACLE_DIR / "horizontal-baseline-metal.json", pin)


FAMILIES = {f.__name__: f for f in (momentum, prep, rk, vertical, horizontal)}


def main(argv=None):
    import woof  # noqa: F401  (installs the backend)
    from woof import backend
    if backend.active() != "mlx":
        raise SystemExit("record Metal pins on the MLX backend (pixi run -e mlx)")
    for name in (argv or sys.argv[1:]) or FAMILIES:
        FAMILIES[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
