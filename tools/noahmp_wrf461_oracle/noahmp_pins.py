"""The two pinned Noah-MP trees an oracle in this directory may be built from.

Same table as ``noahmp_pins.sh`` (which the shell build scripts source);
``tests/test_noahmp_oracle_harness.py`` holds the two in step.  A tool given
``module_sf_noahmplsm.F`` picks the tree whose digest it carries and refuses
any other file.

``patched`` is the module after ``patches/noahmp-lsm-leaf-visibility.patch``
(and ``make_public_radiation.py``, byte for byte): ``private ::`` ->
``public ::``.  ``padded`` is the module after ``visibility_patch_leaves.py``:
``private ::`` -> ``public  ::``, no byte offset moves.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

TREES = {
    "wrf-4.6.1": {
        "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
        "noahmp_commit": "848f54ad3d28c4303151fe5ad83724e232694422",
        "lsm": "bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282",
        "drv": "9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890",
        "glacier": "bf94f3522c3b9c2c9cfbb34fa7e485ff58519106db434520968793409a520579",
        "patched": "3cd3690d6455cfb8549cb41979b7e101e7436464c478f7f7973ab226488ac206",
        "padded": "bfdc0f3632cd30b87208b26a309c533b12d9bc2a39d1a36e9165ecf90d0a12c3",
    },
    "wrf-4.7.1": {
        "wrf_commit": "f52c197ed39d12e087d02c50f412d90d418f6186",
        "noahmp_commit": "e5c0859874407859936739e8be8741f9aed369ee",
        "lsm": "74b475600ad1c999fe43299bcaf4d3e5eaf59b9f861c47fdbff4d69739e3a45b",
        "drv": "29f16622740d7388a17ff9ba6c2f4640487e44f3f26b8c4bdd47aad42e0a4572",
        "glacier": "a35d61ea29f660f4f4c6384bc019d4a57ddc8b0c2ca0f3ef7c6e621b79b4dece",
        "patched": "5e31fa7fbd6aadfdc2e63581b9dc400cf2ec74305859c29a35d9c578a8c7ad06",
        "padded": "072309bc12d64a3e15cf13252ca74411d321e44d3ff9a88bbb8cfcb261ec5ed4",
    },
}

#: Every accepted pristine module_sf_noahmplsm.F digest.
LSM_SHA256 = tuple(t["lsm"] for t in TREES.values())


def tree_for_lsm_digest(digest: str) -> str:
    """The tree name whose LSM digest this is; ValueError for any other."""
    for name, tree in TREES.items():
        if tree["lsm"] == digest:
            return name
    raise ValueError(
        f"module_sf_noahmplsm.F sha256 {digest} is neither pinned tree ("
        + ", ".join(f"{n} {t['lsm']}" for n, t in TREES.items()) + ")")


def tree_for_lsm(path) -> str:
    return tree_for_lsm_digest(hashlib.sha256(Path(path).read_bytes()).hexdigest())
