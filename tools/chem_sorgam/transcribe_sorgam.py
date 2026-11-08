"""Translate MADE/SORGAM's call tree into chem_sorgam.cu with tools/chem_f2cu.

    pixi run -e default python tools/chem_sorgam/transcribe_sorgam.py WRF_SRC [--check]

WRF_SRC is a WRF v4.6.1 tree; every file read must match
tools/chem_wrf461_oracle/sorgam/SOURCES.sha256.  The block between
``// BEGIN GENERATED`` and ``// END GENERATED`` in
woof/core/kernels/chem_sorgam.cu is rewritten (``--check`` only compares).

Translated: rpmmod3 and everything it calls on the isorop = 0 branch
(module_aerosols_sorgam.F), with aeroproc's one documented change
(line 1580, ``isorop=1`` -> ``isorop=0``: the default build runs RPMARES,
not ISORROPIA), plus aerosols_sorgam_init's module-constant block
(module_aerosols_sorgam.F:7390-7488) as ``sorgam_module_constants``.
sorgam_driver's per-cell glue is hand-written in the .cu.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "chem_f2cu"))
import f2cu  # noqa: E402

KERNEL = ROOT / "woof/core/kernels/chem_sorgam.cu"
PINS = ROOT / "tools/chem_wrf461_oracle/sorgam/SOURCES.sha256"
ROUTINES = ["poly4", "poly6", "cubic", "actcof", "awater", "rpmares_old", "eql4",
            "modpar", "getaf", "coagrate", "klpnuc", "ludcmp", "lubksb", "funcv",
            "fmin", "fdjac", "lnsrch", "newt", "soa_part", "sorgam", "nuclcond",
            "aerostep", "aeroproc", "rpmmod3"]
AEROPROC_ISOROP = 1580
GLUE_V = {"so4aj", "so4ai", "nh4aj", "nh4ai", "no3aj", "no3ai", "naaj", "naai", "claj",
          "clai", "orgaro1j", "orgaro1i", "orgaro2j", "orgaro2i", "orgalk1j", "orgalk1i",
          "orgole1j", "orgole1i", "orgba1j", "orgba1i", "orgba2j", "orgba2i", "orgba3j",
          "orgba3i", "orgba4j", "orgba4i", "orgpaj", "orgpai", "ecj", "eci", "p25aj",
          "p25ai", "antha", "seas", "soila", "nu0", "ac0", "corn", "h2oaj", "h2oai",
          "nu3", "ac3", "cor3", "cvaro1", "cvaro2", "cvalk1", "cvole1", "cvapi1",
          "cvapi2", "cvlim1", "cvlim2", "sulf", "hno3", "nh3", "hcl"}
GLUE_P = {"xyl", "tol", "csl1", "csl2", "hc8", "oli1", "oli2", "oli3", "olt1", "olt2",
          "olt3", "api1", "api2", "api3", "lim1", "lim2", "lim3", "soaaro1", "soaaro2",
          "soaalk1", "soaole1", "soaapi1", "soaapi2", "soalim1", "soalim2"}
#: SORGAM's SOA Newton solver: each routine is headed "Adopted from Numerical
#: Recipes in FORTRAN, 2nd ed.", whose licence forbids redistributing it or
#: anything derived from it.  The committed kernel never contains them: the
#: default build calls WOOF's clean-room sg_newt (chem_sorgam.cu), and the exact
#: path generates them only into a local cache from the user's own WRF tree
#: (woof/core/chem_byo.py).  They are still parsed, for the call conventions.
NUMERICAL_RECIPES = ("newt", "lnsrch", "fmin", "ludcmp", "lubksb")
CONSTANTS = (7390, 7488)
#: WOOF's own partitioning solver (chem_sorgam.cu, CLEAN-ROOM SOLVER), called
#: where soa_part calls newt.
CLEAN_ROOM_PROTOTYPE = (
    "__device__ void sg_newt(SorgamModule &mod, int layer, float *x, int n, bool &check, "
    "const float *ctot, const float *csat, const float *imwcv, float minitw, int &its);"
    "  // clean-room, below")  # xxlsgn = log(sginin) .. ccofm_org = ...


def check_pins(src):
    for line in PINS.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split()
        got = hashlib.sha256((src / rel).read_bytes()).hexdigest()
        if got != digest:
            raise SystemExit(f"{rel}: sha256 {got}, pinned {digest}")


def generate(src: Path, exact: bool = False) -> str:
    text = (src / "chem/module_aerosols_sorgam.F").read_text()
    lines = text.splitlines()
    assert lines[AEROPROC_ISOROP - 1] == "      isorop=1", lines[AEROPROC_ISOROP - 1]
    tr = f2cu.Translator([(src / "chem/module_data_radm2.F").read_text(),
                          (src / "chem/module_data_sorgam.F").read_text()],
                         struct_name="SorgamModule", implicit=False, prefix="sg_")
    # Automatic arrays: every caller passes blksize = numcells = 1
    # (sorgam_driver's PARAMETER blksize=1, rpmmod3's numcells=1), and
    # ncv/nacv are module_data_sorgam's lspcv/lcva all the way down.
    par = {n: tr.module.vars[n].param[0] for n in ("lspcv", "lcva")}
    for routine in ROUTINES:
        tr.dummy_bounds.update({(routine, "blksize"): 1, (routine, "numcells"): 1,
                                (routine, "ncv"): par["lspcv"], (routine, "nacv"): par["lcva"]})
    # SORGAM passes ldrog etc. as dummies that every caller fills with the
    # module PARAMETER of the same name.
    tr.bound_from_module_param = True
    # aeroproc's isorop = 1 branch (ISORROPIA) is excluded from this build.
    tr.unreachable_calls = {"eql3"}
    tr.add_source(text, ROUTINES,
                  patches={"aeroproc": {AEROPROC_ISOROP: "      isorop=0"}},
                  ext_alias={"lnsrch": {"func": "fmin"}})
    lo, hi = CONSTANTS
    body = lines[lo - 1:hi]
    assert body[0].strip() == "xxlsgn = log(sginin)", body[0]
    assert body[-1].strip().startswith("ccofm_org"), body[-1]
    tr.add_synthetic("sorgam_module_constants", body, first_line=lo)
    out = ["// BEGIN GENERATED by tools/chem_sorgam/transcribe_sorgam.py from" +
           (" (EXACT: local cache only, never commit)" if exact else ""),
           "// chem/module_aerosols_sorgam.F, module_data_sorgam.F and",
           "// module_data_radm2.F (WRF v4.6.1).  Do not edit by hand.",
           "// aeroproc runs the isorop = 0 branch (module_aerosols_sorgam.F:1580).",
           "// One more module variable than the routines reference: cw_phase (modpar)."]
    # module_data_sorgam's integer PARAMETERs the hand-written driver glue
    # indexes cblk / drog / condvap with (sorgam_driver, :150-400).
    for name, v in sorted(tr.module.vars.items()):
        if v.param is not None and v.param[1] == "int" and (
                name in ("l1ae", "ldrog", "lspcv", "lcva") or
                (name.startswith("v") and name[1:] in GLUE_V) or
                (name.startswith("p") and name[1:] in GLUE_P)):
            out.append(f"#define SG_{name.upper()} {v.param[0]}")
    for name in ("np", "conmin"):
        v = tr.module.vars[name]
        out.append(f"#define SG_{name.upper()} {f2cu.literal(*v.param)}")
    routines = ["sorgam_module_constants"] + [
        r for r in ROUTINES if exact or r not in NUMERICAL_RECIPES]
    bodies = [tr.emit_routine(tr.routines[n]) for n in routines]
    out += tr.module_struct(extra=("cw_phase",))
    out += f2cu.HELPERS.strip().splitlines()
    out += tr.constant_tables
    for name in routines:
        out.append(tr.prototype(tr.routines[name]) + ";")
    if not exact:
        out.append(CLEAN_ROOM_PROTOTYPE)
    for b in bodies:
        out += b
    out.append("// END GENERATED")
    return "\n".join(out) + "\n"


def main(argv):
    src = Path(argv[1])
    check_pins(src)
    block = generate(src)
    text = KERNEL.read_text()
    start = text.index("// BEGIN GENERATED")
    end = text.index("// END GENERATED") + len("// END GENERATED\n")
    new = text[:start] + block + text[end:]
    if "--check" in argv:
        if new != text:
            raise SystemExit("chem_sorgam.cu's generated block is stale")
        return
    KERNEL.write_text(new)


if __name__ == "__main__":
    main(sys.argv)
