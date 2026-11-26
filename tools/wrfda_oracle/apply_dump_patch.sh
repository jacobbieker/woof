#!/bin/bash
# Hook woof_dump.inc into a COPY of a pinned WRFDA 4.7.1 tree (never the
# pinned tree): tools/wrfda_oracle/apply_dump_patch.sh <WRFV4.7.1-copy>
# Then rebuild with ./compile all_wrfvar.  Idempotent.
set -euo pipefail
tree=${1:?usage: apply_dump_patch.sh <WRFV4.7.1 copy>}
here=$(cd "$(dirname "$0")" && pwd)
main=$tree/var/da/da_main
cp "$here/woof_dump.inc" "$main/woof_dump.inc"
# var/build holds symlinks to every source; the include is resolved there.
ln -sfn ../da/da_main/woof_dump.inc "$tree/var/build/woof_dump.inc"
python3 - "$main" <<'PY'
import sys, pathlib
main = pathlib.Path(sys.argv[1])
top = main / "da_wrfvar_top.f90"
s = top.read_text()
missing = [h for h in ('#include "woof_dump.inc"',) if h not in s]
sol_s = (main / "da_solve.inc").read_text()
hooks_all = ("woof_dump_be", "woof_dump_xb", "woof_dump_u", "woof_dump_iv", "woof_dump_final")
missing += [h for h in hooks_all if h not in sol_s]
if missing and len(missing) < len(hooks_all) + 1:
    sys.exit("partially patched tree (missing: " + ", ".join(missing) + "); use a clean copy of WRFV4.7.1")
if '"woof_dump.inc"' not in s:
    s = s.replace('#include "da_solve_dual_res_init.inc"\n',
                  '#include "da_solve_dual_res_init.inc"\n#include "woof_dump.inc"\n', 1)
    top.write_text(s)
sol = main / "da_solve.inc"
s = sol.read_text()
if not any(h in s for h in hooks_all):
    hooks = [
        ("      call da_setup_background_errors (grid, be)\n",
         "      call woof_dump_be(be)\n      call woof_dump_xb(grid)\n"),
        ("      call da_get_innov_vector (it, num_qcstat_conv, ob, iv, grid , config_flags)\n      if ( multi_inc == 1 ) then \n",
         None),
        ("            call da_check (grid, config_flags, cv_size, xbx, be, grid%ep, iv, &\n"
         "                        grid%vv, grid%vp, y)\n",
         "            call woof_dump_u(grid, cv_size, xbx, be)\n"),
        ("      call da_transform_xtoxa (grid)\n\n      ! [8.6]",
         None),
    ]
    # 1: after background errors
    a, b = hooks[0]
    assert s.count(a) == 1, a
    s = s.replace(a, a + b)
    # 2: innovations, before the multi_inc return
    a = hooks[1][0]
    assert s.count(a) == 1, a
    s = s.replace(a, a.replace("      if ( multi_inc", "      if (it == 1) call woof_dump_iv(iv, 'woof_iv.bin')\n      if ( multi_inc"))
    a, b = hooks[2]
    assert s.count(a) == 1, a
    s = s.replace(a, a + b)
    a = hooks[3][0]
    assert s.count(a) == 1, a
    s = s.replace(a, "      call da_transform_xtoxa (grid)\n      call woof_dump_final(grid, cv_size, xhat)\n\n      ! [8.6]")
    sol.write_text(s)
PY
# var4d_detail_out writes the TL perturbation at each slot (tl_d01_* and
# auxhist8_d01_*), the adjoint forcing (af_d01_*) and the 0h gradient
# (gradient_d01_*) through auxhist streams 7 and 8, but registry.var gives
# the a_/g_ state no stream past history ("rh"), so upstream those files are
# a netCDF header with no fields.  Add streams 7 and 8 to the a_/g_ fields
# that carry history; I/O only, the analysis is unchanged.
python3 - "$tree/Registry/registry.var" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
out, n = [], 0
for line in p.read_text().splitlines(keepends=True):
    f = line.split()
    if len(f) > 7 and f[0] == "state" and re.match(r"[ag]_", f[2]) and f[7] in ("rh", "rh078"):
        if f[7] == "rh":
            line = re.sub(r"(\s)rh(\s)", r"\1rh078\2", line, count=1)
        n += 1
    out.append(line)
if n < 14:  # u v w ph t mu p, a_ and g_, plus the moist species
    sys.exit(f"registry.var: only {n} a_/g_ history fields found")
p.write_text("".join(out))
print(f"registry.var: {n} a_/g_ fields on streams 7 and 8")
PY
echo "patched $tree"
