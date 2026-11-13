# WOOF-OPTIN: wrfchem_pnnl
"""Translate phys/module_mixactivate.F's ``activate`` call tree into activate.cu with tools/chem_f2cu.

    pixi run -e default python optional/wrfchem_pnnl/tools/transcribe_activate.py WRF_SRC [--check]

WRF_SRC is a WRF v4.7.1 tree; every file read must match oracle/activate/SOURCES.sha256.  The block between
``// BEGIN GENERATED`` and ``// END GENERATED`` in woof_wrfchem_pnnl/activate.cu is rewritten (``--check``
only compares).

Translated: ``activate``, ``maxsat``, ``maxsat_init`` and ``erf_alt``.  ``erfc_num_recipes`` ("from press et
al, numerical recipes, 1990") is NEVER written to the committed file: its body is generated only into a local
cache from the user's own tree (:func:`erfc_source`, woof_wrfchem_pnnl.activate), the rule woof/core/chem_byo.py
follows.

The oracle (oracle/activate/build.sh) compiles the routines byte for byte.  The kernel is translated from the
same lines with the patches below, which only restate activate's and maxsat's GO TOs as structured Fortran
(the Metal Shading Language has no goto) and its OPTIONAL argument as an explicit flag; the bitwise parity test
holds the restatement to the original on every branch the oracle's cases reach:

* ``goto 30000`` (the modal branch skipping the sectional set-up) -> IF/ELSE;
* ``goto 25000`` (to the DO 25000 terminal) -> CYCLE;
* ``if (sigw <= 1e-5) goto 50000`` ... ``goto 60000`` -> IF (.NOT. ...) / ELSE / END IF;
* the updraft integration's retry of one step (``go to 41000``), its early end (``go to 48000``) and its
  second, diagnostic pass (``goto 40000``) -> two DO WHILE loops with CYCLE / EXIT and a flag;
* ``present(smax_prescribed)`` -> a LOGICAL argument ``woof_has_smax``;
* the array constructor ``(/1/)`` handed to maxsat and maxsat_init -> a local ``woof_one = (/1/)``;
* maxsat's ``go to 1`` out of its two loops -> a flag and EXIT.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "chem_f2cu"))
import f2cu  # noqa: E402

KERNEL = HERE / "woof_wrfchem_pnnl" / "activate.cu"
PINS = HERE / "oracle" / "activate" / "SOURCES.sha256"
SOURCE = "phys/module_mixactivate.F"
ROUTINES = ["maxsat", "maxsat_init", "erfc_num_recipes", "erf_alt", "activate"]
#: Numerical Recipes: parsed for the call conventions, never emitted into the committed file.
NUMERICAL_RECIPES = ("erfc_num_recipes",)
#: The kernel's array extents (activate's automatic arrays need them at compile time): MOSAIC's
#: maxd_asize = 8 sections, two types (module_data_sorgam's maxd_atype = 2; SORGAM's modes fit).  The launcher
#: pads every caller's arrays to (MAXD_ASIZE, MAXD_ATYPE) and passes these as maxd_asize/maxd_atype.
MAXD_ASIZE, MAXD_ATYPE = 8, 2

_HAS = "woof_has_smax"
#: routine -> {line: (the line as WRF v4.7.1 has it, its restatement)}
PATCHES = {
    "activate": {
        1402: ("                      grid_id, ktau, ii, jj, kk,smax_prescribed )!BSINGH - Added smax_prescribed for WRFCuP",
               f"                      grid_id, ktau, ii, jj, kk,smax_prescribed, {_HAS} )"),
        1444: ("      real,intent(in),optional :: smax_prescribed  ! prescribed max. supersaturation for secondary activation !BSINGH - Added for WRFCuP",
               f"      real,intent(in) :: smax_prescribed; logical,intent(in) :: {_HAS}"),
        1531: ("      integer m,n,nw,nwmax", "      integer m,n,nw,nwmax; logical woof_done; integer woof_one(1)"),
        1594: ("      if ( present( smax_prescribed ) ) then", f"      if ( {_HAS} ) then"),
        1629: ("      zero = 0.0", "      zero = 0.0; woof_one(1) = 1"),
        1753: ("              maxd_asize, (/1/), gmlnsig, f1)", "              maxd_asize, woof_one, gmlnsig, f1)"),
        1729: ("            goto 30000", "         else"),
        1730: ("         end if", "!"),
        1795: ("                   goto 25000", "                   cycle"),
        1801: ("                if ( bin_is_narrow(m,n) ) goto 25000", "                if ( bin_is_narrow(m,n) ) cycle"),
        1809: ("                   goto 25000", "                   cycle"),
        1815: ("                   goto 25000", "                   cycle"),
        1837: ("                   goto 25000", "                   cycle"),
        1881: ("30000    continue", "         end if"),
        1893: ("      if(sigw.le.1.e-5) goto 50000", "      if (.not.(sigw.le.1.e-5)) then"),
        1901: ("         ipass_nwloop = 1", "         ipass_nwloop = 1; woof_done = .false."),
        1907: ("40000    continue", "         do while (.true.)"),
        1980: ("41000       wnuc=w+wdiab", "            do while (.true.); wnuc=w+wdiab"),
        2002: ("              if ( present( smax_prescribed ) ) then", f"              if ( {_HAS} ) then"),
        2008: ("                      maxd_asize,(/1/),gmsm,gmlnsig,f1,smax)",
               "                      maxd_asize,woof_one,gmsm,gmlnsig,f1,smax)"),
        2022: ("              if ( present( smax_prescribed ) ) then", f"              if ( {_HAS} ) then"),
        2047: ("                  go to 41000", "                  cycle"),
        2219: ("            if(nw.gt.1.and.(w.gt.wmax.or.fnmin.gt.fmax))go to 48000",
               "            exit; end do; if(nw.gt.1.and.(w.gt.wmax.or.fnmin.gt.fmax)) then; woof_done = .true.; exit; end if"),
        2249: ("         if (ipass_nwloop .eq. 1) then", "         if (woof_done) exit; if (ipass_nwloop .eq. 1) then"),
        2252: ("             goto 40000", "             cycle"),
        2254: ('         call wrf_error_fatal("STOP: activate before 48000")',
               '         call wrf_error_fatal("STOP: activate before 48000"); exit'),
        2256: ("48000    continue", "         end do"),
        2340: ("      goto 60000", "      else"),
        2354: ("50000 continue", "!"),
        2392: ("              if ( present( smax_prescribed ) ) then", f"              if ( {_HAS} ) then"),
        2397: ("                      maxd_asize,(/1/),gmsm,gmlnsig,f1,smax)",
               "                      maxd_asize,woof_one,gmsm,gmlnsig,f1,smax)"),
        2413: ("              if ( present( smax_prescribed ) ) then", f"              if ( {_HAS} ) then"),
        2580: ("60000 continue", "      end if"),
    },
    "maxsat": {
        2621: ("      real thesum", "      real thesum; logical woof_go"),
        2625: ("      do n=1,ntype_aer", "      woof_go = .false.; do n=1,ntype_aer"),
        2633: ("            go to 1", "            woof_go = .true.; exit"),
        2635: ("      end do", "      end do; if (woof_go) exit"),
        2638: ("      return", "      if (.not. woof_go) return"),
        2640: ("  1   continue", "!"),
    },
}


def check_pins(src):
    for line in PINS.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split()
        got = hashlib.sha256((src / rel).read_bytes()).hexdigest()
        if got != digest:
            raise SystemExit(f"{rel}: sha256 {got}, pinned {digest}")


def _translator(src):
    text = (src / SOURCE).read_text()
    lines = text.splitlines()
    patches = {}
    for routine, edits in PATCHES.items():
        for n, (was, now) in edits.items():
            if lines[n - 1] != was:
                raise SystemExit(f"{SOURCE}:{n} is {lines[n - 1]!r}, expected {was!r}")
        patches[routine] = {n: now for n, (_was, now) in edits.items()}
    tr = f2cu.Translator([(src / "share/module_model_constants.F").read_text()],
                         struct_name="ActivateModule", implicit=False, prefix="ma_")
    for routine in ROUTINES:
        tr.dummy_bounds.update({(routine, "maxd_asize"): MAXD_ASIZE, (routine, "maxd_atype"): MAXD_ATYPE})
    tr.add_source(text, ROUTINES, patches=patches)
    return tr


def generate(src: Path) -> str:
    tr = _translator(src)
    routines = [r for r in ROUTINES if r not in NUMERICAL_RECIPES]
    bodies = [tr.emit_routine(tr.routines[n]) for n in routines]
    out = ["// BEGIN GENERATED by optional/wrfchem_pnnl/tools/transcribe_activate.py from",
           "// phys/module_mixactivate.F and share/module_model_constants.F (WRF v4.7.1).  Do not edit by hand.",
           "// erfc_num_recipes (Numerical Recipes) is declared here and defined only in a local build from",
           "// the user's own WRF tree (woof_wrfchem_pnnl.activate.erfc_source)."]
    out += tr.module_struct()
    out += f2cu.HELPERS.strip().splitlines()
    out += tr.constant_tables
    for name in ROUTINES:
        out.append(tr.prototype(tr.routines[name]) + ";")
    for b in bodies:
        out += b
    out.append("// END GENERATED")
    return "\n".join(out) + "\n"


def erfc_source(src: Path) -> str:
    """erfc_num_recipes' body, translated from the user's own tree (local cache only, never committed)."""
    tr = _translator(src)
    return "\n".join(tr.emit_routine(tr.routines["erfc_num_recipes"])) + "\n"


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
            raise SystemExit("activate.cu's generated block is stale")
        return
    KERNEL.write_text(new)


if __name__ == "__main__":
    main(sys.argv)
