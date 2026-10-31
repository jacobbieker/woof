"""Cut WRF v4.7.1's icloud = 3 routines out of the source, byte-unmodified.

    python extract.py WRF_ROOT OUT_DIR

Writes
  cldfra3_cut.F90   cal_cldfra3, find_cloudLayers, adjust_cloudIce,
                    adjust_cloudH2O and adjust_cloudFinal, exactly as they
                    stand in phys/module_radiation_driver.F (:4003-4469),
                    wrapped in a module so they compile alone;
  thompson_cut.F90  RSLF and RSIF exactly as they stand in
                    phys/module_mp_thompson.F, inside a stand-in
                    ``module_mp_thompson`` (cal_cldfra3's own
                    ``USE module_mp_thompson, ONLY : rsif, rslf``).
The routine text between the cut markers is not edited; only the module
wrapper lines around it are written here.
"""
import pathlib
import re
import sys

root, out = map(pathlib.Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)

driver = (root / "phys/module_radiation_driver.F").read_text()
m = re.search(r"(?ms)^      SUBROUTINE cal_cldfra3\(.*?^      END SUBROUTINE adjust_cloudFinal\n",
              driver)
assert m, "cal_cldfra3 .. adjust_cloudFinal block missing"
block = m.group(0)
for name in ("cal_cldfra3", "find_cloudLayers", "adjust_cloudIce",
             "adjust_cloudH2O", "adjust_cloudFinal"):
    assert f"END SUBROUTINE {name}" in block, name
(out / "cldfra3_cut.F90").write_text(
    "module cldfra3_cut\ncontains\n" + block + "end module cldfra3_cut\n")

thompson = (root / "phys/module_mp_thompson.F").read_text()
funcs = []
for name in ("RSLF", "RSIF"):
    f = re.search(rf"(?ms)^      REAL FUNCTION {name}\(P,T\)\n.*?^      END FUNCTION {name}\n",
                  thompson)
    assert f, name
    funcs.append(f.group(0))
(out / "thompson_cut.F90").write_text(
    "module module_mp_thompson\nimplicit none\ncontains\n" + "".join(funcs)
    + "end module module_mp_thompson\n")
print("cut", len(block.splitlines()), "driver lines and",
      sum(len(f.splitlines()) for f in funcs), "thompson lines")
