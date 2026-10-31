"""Pack the FARMS oracle into tests/data/oracles/farms/.

    python pack.py OUT_DIR BUILD_DIR [BUILD_DIR ...]

The first BUILD_DIR's columns and reference outputs (``<scenario>.out.bin``)
are packed into ``OUT_DIR/farms_wrf471.npz``; every BUILD_DIR's reference
must be byte-identical to the first (the oracle is host-independent: its
libm is woof's ports).  ``OUT_DIR/provenance.json`` records each build's
receipt and, as evidence only, how far the same objects linked against that
host's own C library land from the reference, per output, with the cause.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUTPUTS = ("swdown2", "swddir2", "swddni2", "swddif2", "swdownc2",
           "swddnic2", "aerssa2d", "aerasy2d", "aod5502d", "angexp2d")


def read_out(path, ncol):
    raw = np.fromfile(path, np.float32)
    if raw.size != len(OUTPUTS) * ncol:
        raise SystemExit(f"{path}: {raw.size} words, expected "
                         f"{len(OUTPUTS) * ncol}")
    return {n: raw[i * ncol:(i + 1) * ncol] for i, n in enumerate(OUTPUTS)}


def ulp_distance(a, b):
    """Distance in binary32 units in the last place on the ordered line."""
    def ordered(x):
        i = x.view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7fffffff), i)
    return np.abs(ordered(a) - ordered(b))


def main(out_dir, builds):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first = Path(builds[0])
    cols = np.load(first / "columns.npz")
    scenarios = [str(s) for s in cols["scenarios"]]
    bundle = {k: cols[k] for k in cols.files}
    reference = {}
    for scen in scenarios:
        ncol = cols[f"{scen}/albedo"].size
        reference[scen] = read_out(first / f"{scen}.out.bin", ncol)
        for name, value in reference[scen].items():
            bundle[f"{scen}/out/{name}"] = value
    receipts = []
    for build in map(Path, builds):
        evidence = {}
        for scen in scenarios:
            ncol = cols[f"{scen}/albedo"].size
            ref_bytes = (first / f"{scen}.out.bin").read_bytes()
            if (build / f"{scen}.out.bin").read_bytes() != ref_bytes:
                raise SystemExit(f"{build}: {scen} reference differs from {first}")
            sys_out = read_out(build / f"{scen}.syslibm.out.bin", ncol)
            for name in OUTPUTS[:6]:
                d = ulp_distance(reference[scen][name], sys_out[name])
                e = evidence.setdefault(name, {"words": 0, "differ": 0, "max_ulp": 0})
                e["words"] += int(d.size)
                e["differ"] += int((d > 0).sum())
                e["max_ulp"] = max(e["max_ulp"], int(d.max()))
        receipts.append({
            "receipt": (build / "receipt.txt").read_text().splitlines(),
            "host_libm_evidence": evidence,
        })
    np.savez_compressed(out_dir / "farms_wrf471.npz", **bundle)
    digest = hashlib.sha256((out_dir / "farms_wrf471.npz").read_bytes()).hexdigest()
    provenance = {
        "oracle": ("WRF v4.7.1 (f52c197ed) phys/module_ra_farms.F Farms_driver, "
                   "compiled unmodified, driven one call per scenario by "
                   "tools/farms_wrf471_oracle/run_farms.F90 on make_columns.py's "
                   "columns"),
        "reference_libm": (
            "binary32 expf/powf/log10f (glibc 2.39 algorithms), cosf/sinf/acosf "
            "(glibc 2.43) and correctly rounded sinhf (glibc >= 2.41), bound in "
            "place of the host C library (libm_ports.cpp, libm_shim.c); the "
            "reference is byte-identical on every build below"),
        "host_libm_evidence_cause": (
            "the same Fortran objects linked against the host's own libm: "
            "macOS libSystem's binary32 functions are not glibc's, and glibc "
            "2.44's powf and log10f differ from 2.39's "
            "(tools/host_libm32_check.cpp on the 2.44 host: log10f 29,787,060 of "
            "2^32 arguments, powf 129,511 of 2^28 sampled pairs); evidence only"),
        "npz_sha256": digest,
        "scenarios": scenarios,
        "builds": receipts,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps({b: r["host_libm_evidence"] for b, r in zip(builds, receipts)},
                     indent=1))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: pack.py OUT_DIR BUILD_DIR [BUILD_DIR ...]")
    main(sys.argv[1], sys.argv[2:])
