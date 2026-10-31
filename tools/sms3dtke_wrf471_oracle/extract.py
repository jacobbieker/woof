"""Cut WRF v4.7.1's km_opt = 5 routines out of the source, byte-unmodified.

    python extract.py WRF_ROOT OUT_DIR

Writes OUT_DIR/sms_cut.F90: one module holding, exactly as they stand in
the pinned sources (wrf471_sha256.txt),

  dyn_em/module_diffusion_em.F       compute_diff_metrics, cal_deform_and_div,
                                     calculate_km_kh, calculate_N2,
                                     cal_dampkm, isotropic_km, smag_km,
                                     smag2d_km, tke_km, calc_l_scale,
                                     meso_length_scale, free_atmos_length,
                                     nonlocal_flux, pthnl, pthl, pu, phy_bc,
                                     tke_rhs, tke_shear, tke_buoyancy,
                                     tke_dissip, tridiag,
                                     vertical_diffusion_implicit
  share/module_bc.F                  set_physical_bc2d, set_physical_bc3d
  dyn_em/module_big_step_utilities_em.F  phy_prep

The routine text between the cut markers is not edited; only the module
wrapper lines (the USE of the stand-in configuration and WRF's own
module_model_constants, and module_bc's bdyzone parameter) are written
here.  Every cut's line range and sha256 go to OUT_DIR/cut-receipt.json.
"""
import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent

DIFFUSION = ("compute_diff_metrics", "cal_deform_and_div", "calculate_km_kh",
             "calculate_N2", "cal_dampkm", "isotropic_km", "smag_km",
             "smag2d_km", "tke_km", "calc_l_scale", "meso_length_scale",
             "free_atmos_length", "nonlocal_flux", "pthnl", "pthl", "pu",
             "phy_bc", "tke_rhs", "tke_shear", "tke_buoyancy", "tke_dissip",
             "tridiag", "vertical_diffusion_implicit")
CUTS = (("dyn_em/module_diffusion_em.F", DIFFUSION),
        ("share/module_bc.F", ("set_physical_bc2d", "set_physical_bc3d")),
        ("dyn_em/module_big_step_utilities_em.F", ("phy_prep",)))


def pins():
    out = {}
    for line in (HERE / "wrf471_sha256.txt").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        out[rel] = digest
    return out


def cut(raw: bytes, name: str):
    """The one SUBROUTINE/FUNCTION body named ``name``, as bytes."""
    head = re.compile(rb"(?im)^[ \t]*(?:real[ \t]+)?(subroutine|function)[ \t]+"
                      + re.escape(name.encode()) + rb"[ \t]*\(")
    found = list(head.finditer(raw))
    if len(found) != 1:
        raise SystemExit(f"{name}: {len(found)} definitions")
    m = found[0]
    kind = m.group(1)
    end = re.compile(rb"(?im)^[ \t]*end[ \t]*" + kind
                     + rb"(?:[ \t]+" + re.escape(name.encode()) + rb")?[ \t]*(?:!.*)?$")
    e = end.search(raw, m.end())
    if e is None:
        raise SystemExit(f"{name}: no END")
    start = raw.rfind(b"\n", 0, m.start()) + 1
    body = raw[start:e.end()] + b"\n"
    return body, raw[:start].count(b"\n") + 1, raw[:e.end()].count(b"\n") + 1


def main():
    root, out = map(pathlib.Path, sys.argv[1:3])
    out.mkdir(parents=True, exist_ok=True)
    want = pins()
    receipt = {"sources": {}, "cuts": {}}
    bodies = []
    for rel, names in CUTS:
        raw = (root / rel).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != want[rel]:
            raise SystemExit(f"{rel} is not WRF v4.7.1 ({digest})")
        receipt["sources"][rel] = digest
        for name in names:
            body, first, last = cut(raw, name)
            bodies.append(body)
            receipt["cuts"][name] = {"source": rel, "first_line": first,
                                     "last_line": last,
                                     "sha256": hashlib.sha256(body).hexdigest()}
    text = (b"module sms_cut\n"
            b"use module_oracle_config\n"
            b"use module_model_constants\n"
            b"implicit none\n"
            b"INTEGER, PARAMETER :: bdyzone = 4\n"
            b"contains\n" + b"".join(bodies) + b"end module sms_cut\n")
    (out / "sms_cut.F90").write_bytes(text)
    (out / "cut-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print("cut", len(receipt["cuts"]), "routines,",
          sum(v["last_line"] - v["first_line"] + 1
              for v in receipt["cuts"].values()), "lines")


if __name__ == "__main__":
    main()
