"""Cut WRF v4.7.1's MAD-WRF routines out of the source, byte-unmodified.

    python extract.py WRF_ROOT OUT_DIR

Writes
  module_madwrf.F90             phys/module_madwrf.F, whole (it is a module);
  module_model_constants.F90    share/module_model_constants.F, whole;
  module_soil_pre_cut.F90       skip_middle_points_t, as it stands in
                                share/module_soil_pre.F, inside a stand-in
                                ``module_soil_pre``;
  module_mp_thompson_cut.F90    RSLF and RSIF, as they stand in
                                phys/module_mp_thompson.F, inside a stand-in
                                ``module_mp_thompson`` (cal_cldfra3_madwrf's
                                ``USE module_mp_thompson, ONLY : rsif, rslf``);
  nudge_cut.F90                 Cloud_tracer_nudge, as it stands in
                                dyn_em/module_big_step_utilities_em.F, in a
                                stand-in module.
Only the stand-in module wrapper lines are written here.
"""
import pathlib
import re
import shutil
import sys

root, out = map(pathlib.Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
shutil.copyfile(root / "phys/module_madwrf.F", out / "module_madwrf.F90")
shutil.copyfile(root / "share/module_model_constants.F", out / "module_model_constants.F90")

soil = (root / "share/module_soil_pre.F").read_text()
m = re.search(r"(?ms)^FUNCTION skip_middle_points_t .*?^END FUNCTION skip_middle_points_t\n", soil)
assert m, "skip_middle_points_t"
(out / "module_soil_pre_cut.F90").write_text(
    "module module_soil_pre\ncontains\n" + m.group(0) + "end module module_soil_pre\n")

thompson = (root / "phys/module_mp_thompson.F").read_text()
funcs = []
for name in ("RSLF", "RSIF"):
    f = re.search(rf"(?ms)^      REAL FUNCTION {name}\(P,T\)\n.*?^      END FUNCTION {name}\n", thompson)
    assert f, name
    funcs.append(f.group(0))
(out / "module_mp_thompson_cut.F90").write_text(
    "module module_mp_thompson\nimplicit none\ncontains\n" + "".join(funcs)
    + "end module module_mp_thompson\n")

big = (root / "dyn_em/module_big_step_utilities_em.F").read_text()
m = re.search(r"(?ms)^  subroutine Cloud_tracer_nudge .*?^  end subroutine Cloud_tracer_nudge\n", big)
assert m, "Cloud_tracer_nudge"
(out / "nudge_cut.F90").write_text("module nudge_cut\ncontains\n" + m.group(0) + "end module nudge_cut\n")
print("cut module_madwrf, constants, skip_middle_points_t, rslf/rsif, Cloud_tracer_nudge")
