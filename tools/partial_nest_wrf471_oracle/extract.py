"""Cut WRF v4.7.1's nest coupling operators out of its files verbatim.

    python extract.py <WRF_SOURCE_ROOT> <OUT_DIR>

The partial-depth nest (plans/sorted-mapping-hummingbird.md, Phase D3)
couples its child to the parent with WRF's own operators: ``sint`` for the
interface planes and the top strip, ``copy_fcn`` (and its 1-point siblings)
plus the ``sm121``/``smdsm`` smoother for feedback, and the lateral
boundary arithmetic ``relax_bdytend_core``/``spec_bdytend`` turned on its
side for the top relaxation zone.  This copies those routines, after
checking the sha256 of every file and of every block, so the oracle
compiles WRF's own statements:

* ``share/sint.F`` (whole file) -- ``SINT`` and ``SINTB``;
* ``share/interp_fcn.F:874-993`` -- ``interp_fcn_sint``;
* ``share/interp_fcn.F:1397-1742`` -- ``copy_fcn``;
* ``share/interp_fcn.F:1747-1824`` -- ``copy_fcnm``;
* ``share/interp_fcn.F:1829-1906`` -- ``copy_fcni``;
* ``share/interp_fcn.F:3794-4014`` -- ``smoother``, ``sm121``, ``smdsm``;
* ``share/module_bc.F:1221-1427`` -- ``relax_bdytend_core``;
* ``share/module_bc.F:1430-1549`` -- ``spec_bdytend``.

The interp_fcn.F routines are external subroutines in WRF and stay so;
their ``USE module_configure`` / ``USE module_wrf_error`` resolve to the
stubs in ``stubs.F90`` (the namelist getters the routines call, and
``grid_config_rec_type`` carrying ``periodic_x``).  The two module_bc.F
routines live in WRF's ``MODULE module_bc``; they are wrapped here in
``MODULE wrf_bc`` with the same stub ``USE``.  No statement is edited.

WRF v4.7.1 is tag v4.7.1, commit f52c197ed39d12e087d02c50f412d90d418f6186,
public domain.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"

FILES = {
    "share/sint.F":
        "5df841aea44c53c745a90fe96d2b9b8404097b457cceb1321faed592b727af1c",
    "share/interp_fcn.F":
        "680881db48904396f38856ca1dd2578a84e8eb0a9cdd05d49f4c63cd8ce9bdaa",
    "share/module_bc.F":
        "61b9235004b2a7799faabaa928276af8a7ef2e4672619c8ad120c857461301ad",
}

#: (output, file, first line, last line, sha256 of the lines, LF-joined).
BLOCKS = (
    ("sint", "share/sint.F", 1, 349,
     "5df841aea44c53c745a90fe96d2b9b8404097b457cceb1321faed592b727af1c"),
    ("interp_fcn_sint", "share/interp_fcn.F", 874, 993,
     "fd6fd5028f65a0cd70480db84cd531c98720e78f4c9a1e477d68abb5d5b8f188"),
    ("copy_fcn", "share/interp_fcn.F", 1397, 1742,
     "d10505254ed7382fdaaff7616ab1558f321722a1d6f01a5837fed84d4639771c"),
    ("copy_fcnm", "share/interp_fcn.F", 1747, 1824,
     "f00495d2b35697d90a745f34ffdcdb1fef99419bd781fcb9fc5e055952300b2b"),
    ("copy_fcni", "share/interp_fcn.F", 1829, 1906,
     "7e35d84d50a4e1291dd58e8a2a411aafe8f740d599c8a59870797d6fbfbb97f8"),
    ("smoother", "share/interp_fcn.F", 3794, 4014,
     "ec97ae07805e5de9f110cba05768933114cfe593a2f8f036a77f54b206ec9426"),
    ("relax_bdytend_core", "share/module_bc.F", 1221, 1427,
     "99d336be3a3ed9ecf2540d94cb34c6b764f9d50bf0dfb79b77b611e525cb7bc0"),
    ("spec_bdytend", "share/module_bc.F", 1430, 1549,
     "781ac59d7ddf6522eff88831ff30e0fbbfda257c8014ca611528e67586b97907"),
)

_BC_MODULE = """MODULE wrf_bc
  USE module_configure
  USE module_wrf_error
  IMPLICIT NONE
CONTAINS
{relax}{spec}END MODULE wrf_bc
"""


def _block(root: Path, path: str, name: str, first: int, last: int,
           digest: str) -> str:
    lines = (root / path).read_bytes().split(b"\n")
    text = b"\n".join(lines[first - 1:last]) + b"\n"
    got = hashlib.sha256(text).hexdigest()
    if got != digest:
        raise SystemExit(f"{path}:{first}-{last} ({name}) is not the pinned "
                         f"block: sha256 {got}, expected {digest}")
    return text.decode("ascii")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    root, out = Path(argv[1]), Path(argv[2])
    for path, digest in FILES.items():
        got = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if got != digest:
            raise SystemExit(f"{path} is not WRF v4.7.1's: sha256 {got}")
    blocks = {name: _block(root, path, name, first, last, digest)
              for name, path, first, last, digest in BLOCKS}
    out.mkdir(parents=True, exist_ok=True)
    (out / "wrf_sint.F90").write_text(blocks["sint"], encoding="ascii")
    (out / "wrf_interp_fcn.F90").write_text(
        "".join(blocks[n] for n in ("interp_fcn_sint", "copy_fcn", "copy_fcnm",
                                    "copy_fcni", "smoother")), encoding="ascii")
    (out / "wrf_bc.F90").write_text(
        _BC_MODULE.format(relax=blocks["relax_bdytend_core"],
                          spec=blocks["spec_bdytend"]), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
