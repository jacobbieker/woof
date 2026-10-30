"""OPAC species optical-property libraries for the uvspec oracle.

    pixi run -e mlx python -m tools.uvspec_oracle.opac_library mie  [species ...]
    pixi run -e mlx python -m tools.uvspec_oracle.opac_library woof [species ...]
    pixi run -e mlx python -m tools.uvspec_oracle.opac_library fixture [species ...]

Both write uvspec ``aerosol_species_library`` directories
(``<species>..mie.cdf``, the netCDF layout of libRadtran's ``mie`` tool) under
``$WOOF_OPAC_LIBRARY`` (default ``$XDG_CACHE_HOME/woof/opac-library``):

* ``mie/``: libRadtran's ``mie`` tool run as a black box (separate process,
  pixi ``libradtran`` environment) with ``aerosol_type <species>``,
  ``mie_program MIEV0``, ``output_user netcdf`` on OPAC's wavelengths
  0.45-0.6 um (the end-to-end cases' bands).  ``receipt.json`` records every input text, the conda build,
  the mie binary's sha256, the sha256 of the OPAC data files it read and of
  every library file written.
* ``woof/``: woof's own tables (:mod:`woof.radtran.aerosol`) written in the
  same layout (Legendre moments 0..32; the tabulated phase function, which
  uvspec checks for normalisation but does not use without an intensity
  correction, is Henyey-Greenstein with woof's asymmetry parameter), so
  uvspec can be run on exactly woof's optical properties: that
  isolates uvspec's layer handling (humidity, node selection, wavelength
  interpolation, mixing) from the two Mie codes' size integrations.

Neither directory is stored in woof: libraries are rebuilt from the user's
libRadtran data, and the oracle fixtures record the receipt's digests.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from tools.uvspec_oracle.oracle import ROOT

ORACLE_SPECIES = ("inso", "waso", "soot", "ssam", "sscm", "suso")
#: Wavelength span of the oracle libraries [um] (the oracle cases' bands
#: and their bracketing OPAC nodes; the full 0.25-6 um grid works too).
ORACLE_SPAN_UM = (0.4, 2.0)
#: the mie tool is slow for coarse sea salt (~20 min for ssam over 0.4-2 um):
#: its library covers the end-to-end cases' 500 nm bands only
MIE_SPAN_UM = (0.45, 0.6)
#: moment dimension of the netCDF layout (the mie tool's default nmom)
NMOMMAX = 1000


def library_root() -> Path:
    v = os.environ.get("WOOF_OPAC_LIBRARY")
    if v:
        return Path(v)
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "woof" / "opac-library"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _mie_cmd():
    direct = os.environ.get("WOOF_MIE")
    if direct:
        return [direct]
    env_bin = ROOT / ".pixi" / "envs" / "libradtran" / "bin" / "mie"
    if env_bin.exists():
        return [str(env_bin)]
    pixi = shutil.which("pixi")
    if pixi:
        return [pixi, "run", "--manifest-path", str(ROOT / "pyproject.toml"), "-e", "libradtran", "mie"]
    raise FileNotFoundError("libRadtran's mie tool not found: run `pixi install -e libradtran` or set WOOF_MIE")


def _package():
    cm = ROOT / ".pixi" / "envs" / "libradtran" / "conda-meta"
    for f in sorted(cm.glob("rubin-libradtran-*.json")) if cm.is_dir() else ():
        j = json.loads(f.read_text())
        return {k: j.get(k) for k in ("name", "version", "build", "channel", "url", "sha256")}
    return {}


def _data_files(species):
    from woof.radtran import aerosol, data
    files = {"aerosol/OPAC/size_distr.cfg": data.path("aerosol", "OPAC", "size_distr.cfg")}
    for s in species:
        for nd in aerosol.species_nodes(s):
            rel = f"aerosol/OPAC/refractive_indices/{s}{int(round(nd.rh)):02d}_refr.dat"
            files[rel] = data.data_dir() / rel
    return {k: _sha(v) for k, v in sorted(files.items())}


def _span(name, span):
    from woof.radtran import aerosol
    lam = aerosol.wavelengths_um(name)
    sel = (lam >= span[0] - 1e-9) & (lam <= span[1] + 1e-9)
    return np.nonzero(sel)[0], lam[sel]


def build_mie(species=ORACLE_SPECIES, dest: Path | None = None, span=MIE_SPAN_UM) -> Path:
    """Run libRadtran's mie tool per species; returns the library directory."""
    from woof.radtran import aerosol, data
    dest = Path(dest or library_root() / "mie")
    dest.mkdir(parents=True, exist_ok=True)
    rec_path = dest / "receipt.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"species": {}}
    cmd = _mie_cmd()
    for s in species:
        lam_nm = _span(s, span)[1] * 1e3
        wvl = "".join(f"{i + 1} {w:.1f}\n" for i, w in enumerate(lam_nm))
        inp = (f"data_files_path D\naerosol_type {s}\nmie_program MIEV0\nwavelength wvl.dat\n"
               f"output_user netcdf\nbasename {s}.\n")
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="wmie", dir="/tmp" if os.path.isdir("/tmp") else None) as td:
            d = Path(td)
            (d / "D").symlink_to(data.data_dir())
            (d / "wvl.dat").write_text(wvl)
            p = subprocess.run(cmd, input=inp, capture_output=True, text=True, cwd=td)
            out = d / f"{s}..mie.cdf"
            if p.returncode != 0 or not out.exists():
                raise RuntimeError(f"mie failed for {s} ({p.returncode}): {p.stderr[-800:]}")
            shutil.copy(out, dest / out.name)
        rec["species"][s] = {"input": inp, "wavelength_file": wvl, "span_um": list(span), "seconds": round(time.time() - t0, 1),
                             "sha256": _sha(dest / f"{s}..mie.cdf")}
        print(f"mie {s}: {rec['species'][s]['seconds']} s", flush=True)
    mie_bin = Path(cmd[0]) if len(cmd) == 1 else None
    rec.update({"tool": "libRadtran mie (black box)", "command": " ".join(cmd).replace(str(ROOT), "<woof>"), "package": _package(),
                "mie_sha256": _sha(mie_bin) if mie_bin is not None and mie_bin.exists() else None,
                "platform": f"{platform.system()} {platform.machine()}",
                "data_sha256": _data_files(sorted(rec["species"]))})
    rec_path.write_text(json.dumps(rec, indent=1))
    return dest


def build_woof(species=ORACLE_SPECIES, dest: Path | None = None, span=ORACLE_SPAN_UM) -> Path:
    """Write woof's OPAC tables in the mie tool's netCDF layout."""
    import netCDF4
    from woof.radtran import aerosol
    dest = Path(dest or library_root() / "woof")
    dest.mkdir(parents=True, exist_ok=True)
    rec_path = dest / "receipt.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"species": {}}
    # descending angle (ascending cosine), the mie tool's order
    theta = np.concatenate(([0.0], np.logspace(-3, 1, 200)[:-1], np.linspace(10.0, 180.0, 341)))[::-1].copy()
    mu = np.cos(np.radians(theta))
    L = aerosol.LMAX
    for s in species:
        t0 = time.time()
        ils, lam = _span(s, span)
        nodes = aerosol.species_nodes(s)
        nl, nh = lam.size, len(nodes)
        ext = np.zeros((nl, nh)); ssa = np.zeros((nl, nh)); chi = np.zeros((nl, nh, L + 1))
        for k, il in enumerate(ils):
            for ih in range(nh):
                ext[k, ih], ssa[k, ih], chi[k, ih] = aerosol.node_optics(s, int(il), ih)
        f = dest / f"{s}..mie.cdf"
        with netCDF4.Dataset(f, "w") as ds:
            ds.createDimension("nlam", nl)
            ds.createDimension("nmommax", NMOMMAX)
            ds.createDimension("nphamat", 1)
            ds.createDimension("nhum", nh)
            ds.createDimension("nthetamax", theta.size)
            ds.createDimension("nrho", nh)
            v = ds.createVariable("wavelen", "f8", ("nlam",)); v[:] = lam; v.units = "micrometer"
            v = ds.createVariable("hum", "f8", ("nhum",)); v[:] = [n.rh for n in nodes]; v.units = "per cent"
            v = ds.createVariable("ntheta", "i4", ("nlam", "nhum", "nphamat")); v[:] = theta.size
            v = ds.createVariable("theta", "f4", ("nlam", "nhum", "nphamat", "nthetamax"))
            v[:] = np.broadcast_to(theta, (nl, nh, 1, theta.size))
            g = chi[:, :, 1][..., None]
            ph = (1.0 - g * g) / (1.0 + g * g - 2.0 * g * mu) ** 1.5
            v = ds.createVariable("phase", "f4", ("nlam", "nhum", "nphamat", "nthetamax"))
            v[:] = ph[:, :, None, :]
            v = ds.createVariable("nmom", "i4", ("nlam", "nhum", "nphamat")); v[:] = L + 1
            v = ds.createVariable("pmom", "f4", ("nlam", "nhum", "nphamat", "nmommax"))
            pm = np.zeros((nl, nh, 1, NMOMMAX), dtype=np.float32)
            pm[..., :L + 1] = (chi * (2 * np.arange(L + 1) + 1))[:, :, None, :]
            v[:] = pm
            v.units = "including factor 2*l+1"
            for name, val in (("ext", ext), ("ssa", ssa), ("gg", chi[:, :, 1])):
                v = ds.createVariable(name, "f8", ("nlam", "nhum")); v[:] = val
            refr = np.array([[aerosol.refractive_index(s, n.rh, l) for n in nodes] for l in lam])
            v = ds.createVariable("refre", "f8", ("nlam", "nhum")); v[:] = refr.real
            v = ds.createVariable("refim", "f8", ("nlam", "nhum")); v[:] = refr.imag
            v = ds.createVariable("rho", "f8", ("nlam", "nhum")); v[:] = np.broadcast_to([n.rho for n in nodes], (nl, nh))
            ds.file_info = "woof.radtran OPAC tables (tools/uvspec_oracle/opac_library.py)"
            ds.version = 20110916                     # layout version uvspec reads
            ds.complete_moments = 0
            ds.parameterization = "mie"
            ds.size_distr = "OPAC lognormal (woof.radtran.aerosol)"
            ds.param_alpha = -999.0
            ds.param_step = 0.0
            ds.n_r_max = 5.0
            ds.dx_max = float(aerosol.DX)
        rec["species"][s] = {"seconds": round(time.time() - t0, 1), "span_um": list(span), "sha256": _sha(f)}
        print(f"woof {s}: {rec['species'][s]['seconds']} s", flush=True)
    rec.update({"tool": "woof.radtran.aerosol", "table_version": aerosol.TABLE_VERSION,
                "dlnr": aerosol.DLNR, "dx": aerosol.DX, "lmax": L,
                "data_sha256": _data_files(sorted(rec["species"]))})
    rec_path.write_text(json.dumps(rec, indent=1))
    return dest


FIXTURE = ROOT / "tests" / "data" / "radtran" / "opac_mie_tool.json"
FIXTURE_LAMBDA_UM = (0.5, 0.55)
FIXTURE_MOMENTS = 16


def write_fixture(species=ORACLE_SPECIES, src: Path | None = None) -> Path:
    """The mie tool's optics at selected nodes (tests/data/radtran/opac_mie_tool.json).

    Stores only the mie tool's outputs (extinction per mass, single-scattering
    albedo, moments chi_0..16) with each species' tolerance: twice the
    difference measured against woof's tables now (floor 1e-6)."""
    import netCDF4
    from woof.radtran import aerosol
    src = Path(src or library_root() / "mie")
    rec = json.loads((src / "receipt.json").read_text())
    out = {"tool": "libRadtran mie (black box), tools/uvspec_oracle/opac_library.py",
           "package": rec.get("package"), "species": {}}
    for s in species:
        rows, worst = [], {"ext": 0.0, "ssa": 0.0, "chi": 0.0}
        with netCDF4.Dataset(src / f"{s}..mie.cdf") as ds:
            lam = np.array(ds["wavelen"][:])
            hum = np.array(ds["hum"][:])
            for target in FIXTURE_LAMBDA_UM:
                il = int(np.argmin(np.abs(lam - target)))
                for ih, rh in enumerate(hum):
                    nm = int(ds["nmom"][il, ih, 0])
                    k = min(FIXTURE_MOMENTS + 1, nm)
                    chi = np.array(ds["pmom"][il, ih, 0, :k], dtype=np.float64) / (2 * np.arange(k) + 1)
                    r = {"lambda_um": float(lam[il]), "rh": float(rh), "ext": float(ds["ext"][il, ih]),
                         "ssa": float(ds["ssa"][il, ih]), "chi": chi.tolist()}
                    rows.append(r)
                    jl = int(np.argmin(np.abs(aerosol.wavelengths_um(s) - lam[il])))
                    jh = [n.rh for n in aerosol.species_nodes(s)].index(float(rh))
                    e, w, c = aerosol.node_optics(s, jl, jh)
                    worst["ext"] = max(worst["ext"], abs(e / r["ext"] - 1))
                    worst["ssa"] = max(worst["ssa"], abs(w / r["ssa"] - 1))
                    worst["chi"] = max(worst["chi"], float(np.abs(c[:k] - chi).max()))
        out["species"][s] = {"rows": rows, "measured": worst,
                             "tolerance": {q: max(2.0 * v, 1e-6) for q, v in worst.items()}}
        print(s, {q: f"{v:.1e}" for q, v in worst.items()}, flush=True)
    FIXTURE.write_text(json.dumps(out, indent=1))
    return FIXTURE


def main(argv):
    kind = argv[0] if argv else "mie"
    species = tuple(argv[1:]) or ORACLE_SPECIES
    {"mie": build_mie, "woof": build_woof, "fixture": write_fixture}[kind](species)


if __name__ == "__main__":
    main(sys.argv[1:])
