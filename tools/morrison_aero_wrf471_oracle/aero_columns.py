#!/usr/bin/env python3
"""Column oracle for aerosol-aware Morrison (mp_physics = 40), WRF v4.7.1.

The weather columns are tools/morrison_wrf461_oracle/regime_columns.py's
(twelve regimes x 8 seeded variants plus ten edge columns), each given the
fields the aero driver adds: the droplet number NC, the grid vertical
velocity W and the PBL heat diffusivity KZH on w levels (WVAR =
KZH(K+1)/20 clamped to [0.1, 50] m/s), and the ten prescribed CESM aerosol
mass profiles (aerocu, ug m-3, WRF's species order).  Twelve aerosol edge
columns follow, aimed at the activation and ice-nucleation branches:
no aerosol, subnormal aerosol, one species at a time, a dust plume, a
downdraft (W + WVAR <= 0, no activation), the WVAR cap, subnormal and huge
droplet number, cirrus with fast and slow updrafts (Liu-Penner
homogeneous, heterogeneous and transition regimes), a mixed-phase layer
past the Meyers 150 % ice-supersaturation cap, and every mode loaded
under an updraft of a few ulps (W + WVAR just above zero), where the
maximum-supersaturation solve takes its weak-forcing exit.

``make`` writes run_morrison_aero.F90 inputs, one file per set; ``npz``
packs inputs and WRF's outputs for tests/test_morrison_aero_wrf471_oracle.py;
``compare`` grades WOOF's kernel against WRF's words.

usage:
  aero_columns.py make OUTDIR
  aero_columns.py npz OUTDIR OUT.npz
  aero_columns.py compare OUTDIR|NPZ [--row37-off] [--columns-per-call N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for path in (REPO_ROOT, HERE.parent / "morrison_wrf461_oracle"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import regime_columns as rc  # noqa: E402

#: run_morrison_aero.F90's order: the mp_physics = 10 inputs with NC after NG.
LEVEL_IN = rc.LEVEL_IN[:11] + ("nc",) + rc.LEVEL_IN[11:]
STAGGERED_IN = ("w", "kzh")
SURF = rc.SURF
#: WRF's aerocu order (Registry.EM_COMMON:1288-1297).
SPECIES = ("dust1", "dust2", "dust3", "dust4", "seasalt", "sulfate",
           "phobcar", "phibcar", "phoocar", "phiocar")
LEVEL_OUT = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "ns", "nr",
             "ng", "nc", "refl_10cm", "mskf_refl_10cm", "ccn1", "ccn2",
             "ccn3", "ccn4", "ccn5", "ccn6", "ccn7", "efcg", "efig", "efsg",
             "wact")
AERO_EDGES = ("aero_none", "aero_subnormal", "aero_dust_plume",
              "aero_seasalt_only", "aero_sulfate_only", "aero_carbon_only",
              "w_downdraft", "wvar_capped", "nc_extremes", "cirrus_fast",
              "cirrus_slow", "mixed_meyers_cap", "aero_weak_forcing")

#: name: (morr_rimed_ice, aercu_opt, aercu_fct, dt s, nz, calls, seed)
SETS = {
    "a0_m1_dt20_nz50_x10": (1, 0, 1.0, 20.0, 50, 10, 401),
    "a2_m0_dt72_nz50": (0, 2, 1.0, 72.0, 50, 1, 402),
    "a2_m1_dt20_nz50_x20": (1, 2, 1.5, 20.0, 50, 20, 403),
}

#: Surface mass (ug m-3) of each species by regime family, before a
#: lognormal spread; continental unless named.
_BASE = {
    "continental": (0.3, 0.8, 0.5, 0.2, 0.3, 4.0, 0.2, 0.6, 0.8, 2.0),
    "marine": (0.02, 0.05, 0.03, 0.01, 12.0, 1.2, 0.02, 0.05, 0.1, 0.3),
    "desert": (8.0, 20.0, 15.0, 6.0, 0.2, 1.5, 0.05, 0.1, 0.2, 0.4),
    "clean": (0.05, 0.1, 0.05, 0.02, 0.2, 0.4, 0.01, 0.03, 0.05, 0.1),
}
_FAMILY = {"marine_sc": "marine", "tropical_ocean": "marine",
           "desert_virga": "desert", "high_terrain": "clean",
           "cirrus_anvil": "clean"}


def column_labels():
    return rc.column_labels() + list(AERO_EDGES)


def _heights(dz):
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    return 0.5 * (zf[:-1] + zf[1:]), zf


def _aerosol(rng, z, family, scale=None):
    base = np.asarray(_BASE[family]) * np.exp(rng.normal(0.0, 0.6, 10))
    height = rng.uniform(1200.0, 3000.0) if scale is None else scale
    profile = np.exp(-z / height)
    # an elevated layer in about a third of the columns
    if rng.uniform() < 0.35:
        zl = rng.uniform(2e3, 6e3)
        profile = profile + rng.uniform(0.2, 1.0) * np.exp(
            -((z - zl) / rng.uniform(300.0, 900.0)) ** 2)
    return base[:, None] * profile[None, :]


def _pbl_kzh(rng, zf, strength=None):
    """Heat diffusivity on w levels: a convective-PBL K profile."""
    h = rng.uniform(300.0, 2500.0)
    kmax = rng.uniform(1.0, 150.0) if strength is None else strength
    s = np.clip(zf / h, 0.0, 1.0)
    k = kmax * 6.75 * s * (1.0 - s) ** 2
    k[0] = 0.0
    return np.where(zf < h, k, 0.0)


def _vertical_velocity(rng, name, zf):
    amp = {"supercell": rng.uniform(8.0, 30.0),
           "mcs_stratiform": rng.uniform(0.2, 1.0),
           "cirrus_anvil": rng.uniform(0.1, 2.5),
           "high_terrain": rng.uniform(0.5, 3.0)}.get(name,
                                                      rng.uniform(0.0, 0.6))
    shape = np.sin(np.pi * np.clip(zf / max(zf[-1], 1.0), 0.0, 1.0))
    w = amp * shape + rng.normal(0.0, 0.15 * amp + 0.02, zf.size)
    if rng.uniform() < 0.25:            # a subsiding column
        w = -np.abs(w)
    w[0] = 0.0
    return w


def _droplets(rng, col):
    qc, rho = col["qc"], col["rho"]
    nc = rng.uniform(30.0, 600.0) * 1.0e6 / rho
    nc = np.where(qc > 0.0, nc, 0.0)
    # leftover number where the mass has gone (a previous step's residue)
    stray = (qc == 0.0) & (rng.uniform(size=qc.size) < 0.1)
    return np.where(stray, rng.uniform(1.0, 1.0e6, qc.size), nc)


def _edge_column(name, nz, mode, rng):
    base = {"aero_none": "mcs_stratiform", "aero_subnormal": "marine_sc",
            "aero_dust_plume": "desert_virga",
            "aero_seasalt_only": "tropical_ocean",
            "aero_sulfate_only": "synoptic_snow",
            "aero_carbon_only": "supercell", "w_downdraft": "supercell",
            "wvar_capped": "night_fog", "nc_extremes": "mcs_stratiform",
            "cirrus_fast": "cirrus_anvil", "cirrus_slow": "cirrus_anvil",
            "mixed_meyers_cap": "synoptic_snow",
            "aero_weak_forcing": "marine_sc"}[name]
    col = rc.regime(base, nz, mode, rng)
    return base, col


def make_set(nz, mode, seed):
    fields, surf = rc.make_set(nz, mode, seed)
    rng = np.random.default_rng(seed + 1000)
    labels = rc.column_labels()
    extra = []
    for name in AERO_EDGES:
        extra.append(_edge_column(name, nz, mode, rng))
    ncol = len(labels) + len(extra)
    out = {k: np.zeros((nz, ncol), np.float32) for k in LEVEL_IN}
    for k in rc.LEVEL_IN:
        out[k][:, :len(labels)] = fields[k]
        for i, (_, col) in enumerate(extra):
            out[k][:, len(labels) + i] = np.asarray(col.f[k], np.float32)
    s = {k: np.zeros(ncol, np.float32) for k in SURF}
    for k in SURF:
        s[k][:len(labels)] = surf[k]
        for i, (_, col) in enumerate(extra):
            s[k][len(labels) + i] = np.float32(col.s[k])
    stag = {k: np.zeros((nz + 1, ncol), np.float32) for k in STAGGERED_IN}
    aero = np.zeros((10, nz, ncol), np.float32)
    names = labels + list(AERO_EDGES)
    bases = labels + [b for b, _ in extra]
    for i in range(ncol):
        column = {k: out[k][:, i].astype(np.float64) for k in LEVEL_IN}
        z, zf = _heights(column["dz"])
        base = bases[i]
        family = _FAMILY.get(base, "continental")
        aero[:, :, i] = _aerosol(rng, z, family)
        stag["kzh"][:, i] = _pbl_kzh(rng, zf)
        stag["w"][:, i] = _vertical_velocity(rng, base, zf)
        out["nc"][:, i] = _droplets(rng, column)
        name = names[i]
        if name == "aero_none":
            aero[:, :, i] = 0.0
        elif name == "aero_subnormal":
            aero[:, :, i] = np.float32(1.0e-41) * np.arange(1, nz + 1)
            aero[5, ::3, i] = np.float32(1.4e-45)
        elif name == "aero_dust_plume":
            aero[:4, :, i] *= 8.0
        elif name in ("aero_seasalt_only", "aero_sulfate_only",
                      "aero_carbon_only"):
            keep = {"aero_seasalt_only": [4], "aero_sulfate_only": [5],
                    "aero_carbon_only": [6, 7, 8, 9]}[name]
            mask = np.zeros(10, bool)
            mask[keep] = True
            aero[~mask, :, i] = 0.0
        elif name == "w_downdraft":
            stag["w"][1:, i] = -rng.uniform(0.2, 6.0, nz)
            stag["kzh"][:, i] = 0.0
        elif name == "wvar_capped":
            stag["kzh"][1:, i] = rng.uniform(1.0e3, 5.0e3, nz)
        elif name == "nc_extremes":
            qc = out["qc"][:, i]
            nc = out["nc"][:, i]
            nc[0::4] = np.float32(1.0e-40)
            nc[1::4] = np.float32(5.0e10)
            nc[2::4] = 0.0
            out["nc"][:, i] = np.where(qc > 0.0, nc, nc)
        elif name in ("cirrus_fast", "cirrus_slow"):
            # ice supersaturated above 1/1.2 ice saturation x 1.2 trigger,
            # cold enough for Liu-Penner (T <= -35 C)
            t = out["th"][:, i] * out["pii"][:, i]
            cold = t < 238.0
            qv = out["qv"][:, i]
            out["qv"][:, i] = np.where(cold, qv * np.float32(1.35), qv)
            wv = 3.0 if name == "cirrus_fast" else 0.02
            stag["w"][1:, i] = np.float32(wv)
            stag["kzh"][:, i] = 0.0
            aero[5, :, i] = np.where(cold, 2.0, aero[5, :, i])
            aero[0:4, :, i] = np.where(cold, 0.05, aero[0:4, :, i])
        elif name == "aero_weak_forcing":
            stag["kzh"][:, i] = 0.0             # WVAR = 0.1
            base_w = np.float32(-0.1)
            for k in range(1, nz + 1):
                w = base_w
                for _ in range(1 + k % 4):      # 1..4 ulps above -0.1
                    w = np.nextafter(w, np.float32(0.0))
                stag["w"][k, i] = w
            aero[:, :, i] = np.float32(1.0e4)
        elif name == "mixed_meyers_cap":
            t = out["th"][:, i] * out["pii"][:, i]
            band = (t > 248.0) & (t < 268.0)
            out["qv"][:, i] = np.where(band, out["qv"][:, i] * np.float32(1.7),
                                       out["qv"][:, i])
            out["qc"][:, i] = np.where(band, np.float32(2.0e-4),
                                       out["qc"][:, i])
            out["nc"][:, i] = np.where(band, np.float32(1.0e8),
                                       out["nc"][:, i])
    return out, stag, aero, s, names


def write_input(path, fields, stag, aero, surf, mode, aercu_opt, dt,
                aercu_fct):
    nz, ncol = fields["th"].shape
    with open(path, "wb") as f:
        np.asarray([ncol, nz, mode, aercu_opt], "<i4").tofile(f)
        np.asarray([dt, aercu_fct], "<f4").tofile(f)
        for k in LEVEL_IN:
            fields[k].astype("<f4").tofile(f)
        for k in STAGGERED_IN:
            stag[k].astype("<f4").tofile(f)
        aero.astype("<f4").tofile(f)        # (10, nz, ncol) = F (ncol,nz,10)
        for k in SURF:
            surf[k].astype("<f4").tofile(f)


def read_input(path):
    head = np.fromfile(path, "<i4", count=4)
    ncol, nz, mode, aercu_opt = (int(x) for x in head)
    dt, fct = (np.float32(x) for x in np.fromfile(path, "<f4", count=2,
                                                   offset=16))
    data = np.fromfile(path, "<f4", offset=24)
    pos = 0
    fields, stag, surf = {}, {}, {}
    for k in LEVEL_IN:
        fields[k] = data[pos:pos + nz * ncol].reshape(nz, ncol).copy()
        pos += nz * ncol
    for k in STAGGERED_IN:
        stag[k] = data[pos:pos + (nz + 1) * ncol].reshape(nz + 1, ncol).copy()
        pos += (nz + 1) * ncol
    aero = data[pos:pos + 10 * nz * ncol].reshape(10, nz, ncol).copy()
    pos += 10 * nz * ncol
    for k in SURF:
        surf[k] = data[pos:pos + ncol].copy()
        pos += ncol
    return ncol, nz, mode, aercu_opt, dt, fct, fields, stag, aero, surf


def read_output(path, ncol, nz):
    data = np.fromfile(path, "<f4")
    out, pos = {}, 0
    for k in LEVEL_OUT:
        out[k] = data[pos:pos + nz * ncol].reshape(nz, ncol)
        pos += nz * ncol
    for k in SURF:
        out[k] = data[pos:pos + ncol]
        pos += ncol
    assert pos == data.size, (pos, data.size)
    return out


def export_npz(outdir, out_path):
    meta = json.loads((outdir / "sets.json").read_text())
    arrays = {"labels": np.asarray(meta["labels"])}
    for index, name in enumerate(meta["sets"]):
        info = meta["sets"][name]
        (ncol, nz, mode, aercu_opt, dt, fct, fields, stag, aero,
         surf) = read_input(outdir / f"{name}.in.bin")
        want = read_output(outdir / f"{name}.wrf.bin", ncol, nz)
        arrays[f"s{index}_name"] = np.asarray(name)
        arrays[f"s{index}_meta"] = np.asarray(
            [ncol, nz, mode, info["steps"], aercu_opt], np.int32)
        arrays[f"s{index}_real"] = np.asarray([dt, fct], np.float32)
        for k in LEVEL_IN:
            arrays[f"s{index}_in_{k}"] = fields[k]
        for k in STAGGERED_IN:
            arrays[f"s{index}_in_{k}"] = stag[k]
        arrays[f"s{index}_in_aerocu"] = aero
        for k in SURF:
            arrays[f"s{index}_in_{k}"] = surf[k]
        for k in LEVEL_OUT + SURF:
            arrays[f"s{index}_out_{k}"] = np.ascontiguousarray(want[k])
    np.savez_compressed(out_path, **arrays)


def load_npz(path):
    """[(name, meta dict, fields, stag, aero, surf, want)], labels."""
    data = np.load(path)
    labels = [str(x) for x in data["labels"]]
    sets, index = [], 0
    while f"s{index}_meta" in data:
        ncol, nz, mode, steps, aercu_opt = (int(x) for x in
                                            data[f"s{index}_meta"])
        dt, fct = (np.float32(x) for x in data[f"s{index}_real"])
        meta = {"ncol": ncol, "nz": nz, "mode": mode, "steps": steps,
                "aercu_opt": aercu_opt, "dt": dt, "aercu_fct": fct}
        fields = {k: data[f"s{index}_in_{k}"] for k in LEVEL_IN}
        stag = {k: data[f"s{index}_in_{k}"] for k in STAGGERED_IN}
        aero = data[f"s{index}_in_aerocu"]
        surf = {k: data[f"s{index}_in_{k}"] for k in SURF}
        want = {k: data[f"s{index}_out_{k}"] for k in LEVEL_OUT + SURF}
        sets.append((str(data[f"s{index}_name"]), meta, fields, stag, aero,
                     surf, want))
        index += 1
    return sets, labels


# ------------------------------------------------------------------ GPU side

def run_gpu(meta, fields, stag, aero, surf, columns_per_call=None):
    """Run WOOF's mp_physics = 40 launcher for meta['steps'] calls."""
    import cupy as cp
    from woof.core import morrison_aero

    nz, ncol = meta["nz"], meta["ncol"]

    def dev(a, levels=nz):
        return cp.asarray(np.ascontiguousarray(a, np.float32).reshape(
            levels, 1, ncol))

    d = {k: dev(fields[k]) for k in LEVEL_IN}
    w = dev(stag["w"], nz + 1)
    kzh = dev(stag["kzh"], nz + 1)
    aerocu = cp.asarray(np.ascontiguousarray(aero, np.float32).reshape(
        10, nz, 1, ncol))
    s = {k: cp.asarray(np.ascontiguousarray(surf[k]).reshape(1, ncol))
         for k in SURF}
    outs = {k: cp.zeros((nz, 1, ncol), cp.float32)
            for k in ("refl_10cm", "mskf_refl_10cm", "efcg", "efig", "efsg",
                      "wact")}
    ccn = cp.zeros((7, nz, 1, ncol), cp.float32)
    for _ in range(meta["steps"]):
        morrison_aero.launch_morrison_aero(
            d["th"], d["qv"], d["qc"], d["qr"], d["qi"], d["qs"], d["qg"],
            d["nc"], d["nr"], d["ni"], d["ns"], d["ng"], d["rho"], d["pii"],
            d["p"], d["dz"], w, kzh, s["rainnc"], s["rainncv"], s["snownc"],
            s["snowncv"], s["graupelnc"], s["graupelncv"], s["sr"],
            float(meta["dt"]), aercu_opt=meta["aercu_opt"],
            aercu_fct=float(meta["aercu_fct"]), aerocu=aerocu, ccn=ccn,
            efcg=outs["efcg"], efig=outs["efig"], efsg=outs["efsg"],
            wact=outs["wact"], refl_10cm=outs["refl_10cm"],
            mskf_refl_10cm=outs["mskf_refl_10cm"],
            qrcuten=d["qrcuten"], qscuten=d["qscuten"], qicuten=d["qicuten"],
            morr_rimed_ice=meta["mode"], columns_per_call=columns_per_call)
    got = {k: cp.asnumpy(d[k]).reshape(nz, ncol)
           for k in ("th", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "ns",
                     "nr", "ng", "nc")}
    for k, v in outs.items():
        got[k] = cp.asnumpy(v).reshape(nz, ncol)
    for i in range(7):
        got[f"ccn{i + 1}"] = cp.asnumpy(ccn[i]).reshape(nz, ncol)
    for k in SURF:
        got[k] = cp.asnumpy(s[k]).reshape(ncol)
    return got


def row37_off():
    """Compile the mp_physics = 40 unit with MORR_ROW37_OFF (WRF's
    arithmetic alone); returns the loader to restore."""
    import cupy as cp
    from woof.core import kernels
    original = kernels.load_module

    def load(name):
        if name != "morrison_aero":
            return original(name)
        module = cp.RawModule(
            code="#define MORR_ROW37_OFF 1\n" + kernels.module_source(name),
            options=kernels.module_options(name))
        module.compile()
        return module

    kernels.load_module = load
    kernels.get_kernel.cache_clear()
    return original


def differing(got, want, keys=LEVEL_OUT + SURF):
    """{field: (nz, ncol) or (ncol,) bool} of words that differ (NaN = NaN)."""
    bad = {}
    for k in keys:
        g = np.ascontiguousarray(got[k], np.float32)
        w = np.ascontiguousarray(want[k], np.float32)
        bad[k] = (g.view(np.uint32) != w.view(np.uint32)) & ~(
            np.isnan(g) & np.isnan(w))
    return bad


def grade(got, want, labels):
    from woof.core.fp32_ulp import fp32_ulp_distance
    report = {}
    for k, b in differing(got, want).items():
        g = np.ascontiguousarray(got[k], np.float32).ravel()
        w = np.ascontiguousarray(want[k], np.float32).ravel()
        ulp = fp32_ulp_distance(g, w)
        cols = b.any(axis=0) if b.ndim == 2 else b
        report[k] = {"mismatches": int(b.sum()), "values": int(b.size),
                     "max_ulp": int(ulp[b.ravel()].max()) if b.any() else 0,
                     "columns": sorted({labels[i] for i in np.nonzero(cols)[0]})}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("outdir", type=Path)
    npz = sub.add_parser("npz")
    npz.add_argument("outdir", type=Path)
    npz.add_argument("out", type=Path)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("source", type=Path)
    cmp_.add_argument("--row37-off", action="store_true")
    cmp_.add_argument("--columns-per-call", type=int, default=None)
    args = parser.parse_args(argv)
    if args.cmd == "make":
        args.outdir.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, (mode, opt, fct, dt, nz, steps, seed) in SETS.items():
            fields, stag, aero, surf, labels = make_set(nz, mode, seed)
            write_input(args.outdir / f"{name}.in.bin", fields, stag, aero,
                        surf, mode, opt, dt, fct)
            manifest[name] = {"mode": mode, "aercu_opt": opt,
                              "aercu_fct": fct, "dt": dt, "nz": nz,
                              "steps": steps, "seed": seed,
                              "columns": int(fields["th"].shape[1])}
        (args.outdir / "sets.json").write_text(json.dumps(
            {"sets": manifest, "labels": column_labels()}, indent=2))
        print(json.dumps(manifest, indent=2))
        return 0
    if args.cmd == "npz":
        export_npz(args.outdir, args.out)
        return 0
    if args.source.suffix == ".npz":
        sets, labels = load_npz(args.source)
    else:
        meta = json.loads((args.source / "sets.json").read_text())
        labels, sets = meta["labels"], []
        for name, info in meta["sets"].items():
            (ncol, nz, mode, opt, dt, fct, fields, stag, aero,
             surf) = read_input(args.source / f"{name}.in.bin")
            want = read_output(args.source / f"{name}.wrf.bin", ncol, nz)
            sets.append((name, {"ncol": ncol, "nz": nz, "mode": mode,
                                "steps": info["steps"], "aercu_opt": opt,
                                "dt": dt, "aercu_fct": fct},
                         fields, stag, aero, surf, want))
    if args.row37_off:
        row37_off()
    for name, meta, fields, stag, aero, surf, want in sets:
        got = run_gpu(meta, fields, stag, aero, surf, args.columns_per_call)
        report = grade(got, want, labels)
        total = sum(v["mismatches"] for v in report.values())
        print(f"== {name}: {total} differing words")
        for k, v in report.items():
            if v["mismatches"]:
                print(f"  {k}: {v['mismatches']}/{v['values']} "
                      f"max_ulp={v['max_ulp']} {v['columns'][:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
