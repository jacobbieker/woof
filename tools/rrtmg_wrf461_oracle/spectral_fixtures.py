"""Pack the spectral oracle's SWD1 dumps (spectral_build.sh) into the
fixtures tests/test_rrtmg_spectral_oracle.py reads:

    tests/data/oracles/rrtmg_spectral/sw_spectral.npz
        <case>/band  (n1, 14, 6)   float32, [level K, band, quantity]
        <case>/gpt   (n1, 112, 6)  float32, [level K, g-point, quantity]
    tests/data/oracles/rrtmg_spectral/provenance.json

Level K is spcvmc_sw's output index ikl - 1 (the broadband pbbfd's), the
quantity order woof.core.rrtmg_sw.SPECTRAL_QUANTITIES.

    tests/data/oracles/rrtmg_spectral/lw_spectral.npz
        <case>/in/*, <case>/meta/nlayers, <case>/out/{uflx,dflx,uflxc,dflxc}
        <case>/out/lw_band  (nl+1, 16, 4)  [level, band, quantity u d uc dc]
      from the longwave extractor's case files (LW_FIXTURE_DIR).

    python spectral_fixtures.py BUILD_DIR [LW_FIXTURE_DIR]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "tests" / "data" / "oracles" / "rrtmg_spectral"
sys.path.insert(0, str(HERE))
from sw_dumpio import read_swd  # noqa: E402


def main(argv=None):
    build = Path((argv or sys.argv[1:])[0])
    arrays = {}
    for dump in sorted(build.glob("sw_spectral_*.swd")):
        data = read_swd(dump)
        for key, value in data.items():
            case, _, rest = key.partition("/")
            if rest not in ("spectral/band", "spectral/gpt"):
                continue
            width = 14 if rest.endswith("band") else 112
            words = np.asarray(value, np.float32)
            arrays[f"{case}/{rest.split('/')[1]}"] = words.reshape(-1, width, 6, order="F")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "sw_spectral.npz", **arrays)
    provenance = {
        "oracle": "WRF v4.6.1 module_ra_rrtmg_sw.F, spcvmc_sw's inactive per-band accumulation made "
                  "live (tools/rrtmg_wrf461_oracle/spectral_patch.py), driven by sw_fixture_driver on "
                  "sw_fixtures/columns_{real,synth}.txt",
        "compiler": (build / "spectral-compiler.txt").read_text().strip().splitlines(),
        "sha256": (build / "spectral-oracle-sha256sums.txt").read_text().strip().splitlines(),
        "cases": sorted({k.split("/")[0] for k in arrays}),
    }
    args = argv or sys.argv[1:]
    if len(args) > 1:
        from lw_fixtures import read_fixture
        lw = {}
        for path in sorted(Path(args[1]).glob("lw_case_*.bin")):
            fx = read_fixture(path)
            case = f"lw{int(fx['meta/caseid']):04d}"
            for key, value in fx.items():
                if key.startswith("in/") or key in ("meta/nlayers", "out/uflx", "out/dflx",
                                                    "out/uflxc", "out/dflxc", "out/lw_band"):
                    lw[f"{case}/{key}"] = np.asarray(value)
        np.savez_compressed(OUT / "lw_spectral.npz", **lw)
        provenance["lw"] = {
            "oracle": "WRF v4.6.1 module_ra_rrtmg_lw.F, rtrnmc's per-band flux term recorded "
                      "(spectral_patch.py), driven by lw_extract on lw_make_inputs.py's synthetic columns",
            "sha256": (build / "lw-spectral-oracle-sha256sums.txt").read_text().strip().splitlines(),
            "cases": sorted({k.split("/")[0] for k in lw}),
        }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"spectral_fixtures: {len(provenance['cases'])} cases")


if __name__ == "__main__":
    main()
