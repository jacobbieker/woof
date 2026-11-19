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
if '"woof_dump.inc"' not in s:
    s = s.replace('#include "da_solve_dual_res_init.inc"\n',
                  '#include "da_solve_dual_res_init.inc"\n#include "woof_dump.inc"\n', 1)
    top.write_text(s)
sol = main / "da_solve.inc"
s = sol.read_text()
if "woof_dump_be" not in s:
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
echo "patched $tree"
