"""The hex radiation helper sources on disk are an admitted source set.

A hex forecast binds woof/core/rrtmg_{lw,sw,mcica,legacy_prep}.py,
woof/ingest/wrf_ozone.py and woof/core/kernels/rrtmg_sw.cu by SHA-256 to a
row of woof/hex/data/profiles/engine-helper-sets.v1.json
(woof.hex.cuda_arwen_helper_scope_v2.bind_arwen_helpers_v2).  An edit to one
of them without its row refused every hex forecast at start, on every
backend ("unknown or mixed woof execution source set": 4e5ca8a54 changed
wrf_ozone.py).  This holds the tree to its rows without a device.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_tree_binds_an_admitted_helper_source_set():
    from woof.hex.cuda_arwen_helper_scope_v2 import bind_arwen_helpers_v2
    bind_arwen_helpers_v2(ROOT)
