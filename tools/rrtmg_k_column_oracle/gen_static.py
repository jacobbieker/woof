"""Print RRTMG-K's shortwave liquid cloud tables as woof/core/rrtmgk.py's
generated block.

rrsw_cld_k's extliq1/ssaliq1/asyliq1 are DATA tables of
phys/module_ra_rrtmg_swk.F (swcldpr) whose values differ from the legacy
module's; the oracle writes them as its last three records after
rrtmg_swinit_k.

usage: python gen_static.py WRF_DIR/stock/out_<regime>.bin
"""
from __future__ import annotations

import base64
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "rrtmg_legacy_column_oracle"))
from compare import read_records  # noqa: E402


def main():
    rec = read_records(Path(sys.argv[1]))
    print("# --- BEGIN GENERATED K LIQUID TABLES "
          "(tools/rrtmg_k_column_oracle/gen_static.py) ---")
    print("_K_LIQUID_TABLES = {")
    for name in ("extliq1", "ssaliq1", "asyliq1"):
        a = np.asarray(rec[name], "<f4")
        if a.size != 58 * 14:
            raise SystemExit(f"{name}: {a.size} words, expected 58 x 14")
        b64 = base64.b64encode(a.tobytes()).decode()
        lines = textwrap.wrap(b64, 64)
        print(f'    "{name}": (')
        for line in lines:
            print(f'        "{line}"')
        print("    ),")
    print("}")
    print("# --- END GENERATED K LIQUID TABLES ---")


if __name__ == "__main__":
    main()
