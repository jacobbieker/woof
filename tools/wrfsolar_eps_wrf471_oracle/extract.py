"""Cut WRF v4.7.1's WRF-Solar EPS (multi_perturb) scheme hooks, byte-unmodified.

    python extract.py WRF_ROOT OUT_DIR

Writes four modules.  Three hold the driver module's own ``real, private,
parameter`` statement(s) and the Add_/Remove_ pair exactly as they stand in
the driver:

  eps_rad_cut.F90   phys/module_radiation_driver.F's (ALBEDO_MIN ..
                    QSNOW_MIN) and Add_/Remove_multi_perturb_swrad_perturbations
                    (FARMS, pert_farms);
  eps_shcu_cut.F90  phys/module_shallowcu_driver.F's (QCLOUD_MIN,
                    QVAPOR_MIN) and Add_/Remove_multi_perturb_shcu_perturbations
                    (Deng, pert_deng);
  eps_mp_cut.F90    phys/module_microphysics_driver.F's (QX_MIN, NI_MIN)
                    and Add_/Remove_multi_perturb_mp_perturbations (Thompson,
                    pert_thom);
  eps_cld3_cut.F90  the icloud = 3 input statements of the radiation driver
                    (:1472-1478, the ``if (pert_cld3 .and. multi_perturb == 1)``
                    block that fills t_1d and qv_1d), inside a stand-in loop
                    over i, k, j that only supplies the indices the block reads.

The text between the cut markers is not edited; only the module wrapper
lines (and, for cld3, the loop and declarations around the block) are
written here.
"""
import pathlib
import re
import sys

root, out = map(pathlib.Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)


def cut(path, params, add, remove, name):
    text = (root / path).read_text()
    # The module's ``real, private, parameter`` statement(s) (one per line,
    # or one continued with ``&``), as they stand.
    m = re.search(r"(?m)^  real, private, parameter :: .*\n(?:(?:  real, private, parameter :: |      ).*\n)*",
                  text)
    assert m, path
    header = m.group(0)
    assert header.count("parameter") == params, (path, header)
    blocks = []
    for sub in (add, remove):
        m = re.search(rf"(?msi)^   subroutine\s+{sub} \(.*?^ *end subroutine\s+{sub}\n", text)
        assert m, (path, sub)
        blocks.append(m.group(0))
    (out / f"{name}.F90").write_text(
        f"module {name}\n" + header + "contains\n" + "".join(blocks)
        + f"end module {name}\n")
    return sum(len(b.splitlines()) for b in blocks)


n = cut("phys/module_radiation_driver.F", 1,
        "Add_multi_perturb_swrad_perturbations",
        "Remove_multi_perturb_swrad_perturbations", "eps_rad_cut")
n += cut("phys/module_shallowcu_driver.F", 2,
         "Add_multi_perturb_shcu_perturbations",
         "Remove_multi_perturb_shcu_perturbations", "eps_shcu_cut")
n += cut("phys/module_microphysics_driver.F", 2,
         "Add_multi_perturb_mp_perturbations",
         "Remove_multi_perturb_mp_perturbations", "eps_mp_cut")

rad = (root / "phys/module_radiation_driver.F").read_text().splitlines(keepends=True)
block = "".join(rad[1471:1478])
assert block.lstrip().startswith("if (pert_cld3 .and. multi_perturb == 1) then"), block
assert "t_1d(k) = (1.0 + perts_th(i, k, j) * pert_cld3_t) * t(i, k, j)" in block
assert block.rstrip().endswith("end if"), block
(out / "eps_cld3_cut.F90").write_text(
    "module eps_cld3_cut\ncontains\n"
    "  subroutine cld3_inputs(t, qv, perts_th, perts_qvapor, pert_cld3, multi_perturb, &\n"
    "      pert_cld3_t, pert_cld3_qv, t_out, qv_out, ims, ime, kms, kme, jms, jme)\n"
    "    implicit none\n"
    "    integer, intent(in) :: ims, ime, kms, kme, jms, jme, multi_perturb\n"
    "    logical, intent(in) :: pert_cld3\n"
    "    real, intent(in) :: pert_cld3_t, pert_cld3_qv\n"
    "    real, dimension(ims:ime, kms:kme, jms:jme), intent(in) :: t, qv, perts_th, perts_qvapor\n"
    "    real, dimension(ims:ime, kms:kme, jms:jme), intent(out) :: t_out, qv_out\n"
    "    real :: t_1d(kms:kme), qv_1d(kms:kme)\n"
    "    integer :: i, k, j\n"
    "    do j = jms, jme\n"
    "    do i = ims, ime\n"
    "    do k = kms, kme\n"
    + block +
    "    t_out(i, k, j) = t_1d(k)\n"
    "    qv_out(i, k, j) = qv_1d(k)\n"
    "    end do\n"
    "    end do\n"
    "    end do\n"
    "  end subroutine cld3_inputs\n"
    "end module eps_cld3_cut\n")
print("cut", n, "driver lines and", len(block.splitlines()), "cld3 lines")
