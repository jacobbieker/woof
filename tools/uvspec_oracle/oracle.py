"""Run libRadtran's uvspec as a black-box oracle on woof.radtran inputs.

LICENCE BOUNDARY.  uvspec (GPL-2) runs only as a separate process from the
pixi ``libradtran`` environment; this module writes its input files, reads
its standard output and never imports, links or reads libRadtran source.
The data directory is the user's installed libRadtran data (read at run
time, never copied into woof).

    from tools.uvspec_oracle.oracle import run
    out = run(options, column, clouds)      # dict of numpy arrays

The same Options object drives woof (woof.radtran.driver.compute) and the
uvspec input (Options.to_uvspec), and the column (levels, temperatures,
pressures, every gas density including CH4/N2O/CO/N2 as explicit mol_files)
is written on exactly woof's levels, so uvspec inserts no levels of its own.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _uvspec_cmd():
    """``pixi run -e libradtran uvspec`` (or $WOOF_UVSPEC for a direct binary)."""
    direct = os.environ.get("WOOF_UVSPEC")
    if direct:
        return [direct]
    pixi = shutil.which("pixi")
    if pixi:
        return [pixi, "run", "--manifest-path", str(ROOT / "pyproject.toml"), "-e", "libradtran", "uvspec"]
    env_bin = ROOT / ".pixi" / "envs" / "libradtran" / "bin" / "uvspec"
    if env_bin.exists():
        return [str(env_bin)]
    raise FileNotFoundError("uvspec not found: run `pixi install -e libradtran` or set WOOF_UVSPEC")


def _fmt(x):
    return repr(float(x))


def write_column(col, d: Path, c: int = 0):
    """Atmosphere file (9 columns, top first) and vmr mol_files for CH4/N2O/CO/N2."""
    z, p, t, na = col.z[c], col.p[c], col.t[c], col.n_air[c]
    g = {k: v[c] for k, v in col.dens.items()}
    lines = ["# z(km) p(hPa) T(K) air O3 O2 H2O CO2 NO2 (cm-3), written by woof tools/uvspec_oracle"]
    for k in range(z.size):
        lines.append(" ".join(_fmt(v) for v in (z[k], p[k], t[k], na[k], g["O3"][k], g["O2"][k],
                                                   g["H2O"][k], g["CO2"][k], g["NO2"][k])))
    (d / "atm.dat").write_text("\n".join(lines) + "\n")
    extra = []
    for gas in ("CH4", "N2O", "CO", "N2"):
        v = g[gas] / na
        (d / f"{gas}.vmr").write_text("".join(f"{_fmt(z[k])} {_fmt(v[k])}\n" for k in range(z.size)))
        extra.append(f"mol_file {gas} {d / (gas + '.vmr')} vmr")
    return d / "atm.dat", extra


def write_cloud(z, content, reff, path: Path):
    """uvspec 1D cloud file on the column's levels: row k = layer above level k."""
    zb = z[1:]
    rows = [f"{_fmt(z[0])} 0 0"]
    for k in range(zb.size):
        rows.append(f"{_fmt(zb[k])} {_fmt(content[k])} {_fmt(reff[k] if content[k] > 0 else 0.0)}")
    path.write_text("# z(km) content(g/m3) reff(um)\n" + "\n".join(rows) + "\n")


def write_aerosol(z, species, mass, path: Path):
    """uvspec ``aerosol_species_file`` on the column's levels: the row at a
    level holds the mass [g m-3] of the layer above it (probed convention)."""
    m = np.asarray(mass, dtype=np.float64)
    rows = [" ".join([_fmt(z[0])] + ["0"] * len(species))]
    for k in range(m.shape[0]):
        rows.append(" ".join([_fmt(z[k + 1])] + [_fmt(v) for v in m[k]]))
    path.write_text("# z(km) " + " ".join(species) + " (g/m3)\n" + "\n".join(rows) + "\n")


def parse(stdout: str, *, nout: int, numu: int, nphi: int, twostr: bool):
    """uvspec's default output: per wavelength, per zout a flux line, then
    (with umu) a header of phi and numu rows 'umu u0u uu(phi...)'."""
    lines = [l for l in stdout.splitlines() if l.strip()]
    i = 0
    lam, flux, uu, u0u = [], [], [], []
    while i < len(lines):
        vals = [float(x) for x in lines[i].split()]
        i += 1
        flux.append(vals)
        if numu:
            i += 1          # phi header
            blk = []
            for _ in range(numu):
                blk.append([float(x) for x in lines[i].split()])
                i += 1
            blk = np.array(blk)
            u0u.append(blk[:, 1])
            uu.append(blk[:, 2:2 + nphi] if nphi else blk[:, 1:2])
    flux = np.array(flux)
    nlam = flux.shape[0] // nout
    flux = flux.reshape(nlam, nout, -1)
    out = {"lambda": flux[:, 0, 0]}
    names = ("edir", "edn", "eup", "uavg") if twostr else ("edir", "edn", "eup", "uavgdir", "uavgdn", "uavgup")
    for k, n in enumerate(names):
        out[n] = flux[:, :, 1 + k]
    if numu:
        out["uu"] = np.array(uu).reshape(nlam, nout, numu, -1)
        out["u0u"] = np.array(u0u).reshape(nlam, nout, numu)
    return out


def run(opts, col, clouds=None, *, aerosol=None, aerosol_library: Path | None = None,
        keep: Path | None = None):
    """Run uvspec on (Options, Column[, CloudLayers][, AerosolLayers]); returns parsed arrays.

    ``aerosol_library``: directory of ``<species>.*.cdf`` Mie tables (libRadtran's
    mie-tool netCDF format; tools/uvspec_oracle/opac_library.py)."""
    from woof.radtran import data as rtdata
    if col.z.ndim == 1:
        from woof.radtran.atmosphere import Column
        col = Column(z=col.z[None], p=col.p[None], t=col.t[None], n_air=col.n_air[None],
                     dens={k: v[None] for k, v in col.dens.items()})
    # uvspec truncates some data file names near 100 characters: run from a
    # short temporary directory holding a link to the data directory.
    with tempfile.TemporaryDirectory(prefix="wuv", dir="/tmp" if os.path.isdir("/tmp") else None) as td:
        d = Path(td)
        (d / "D").symlink_to(rtdata.data_dir())
        atm, mol = write_column(col, d)
        extra = list(mol)
        o = opts
        if clouds is not None and clouds.lwc is not None:
            write_cloud(col.z[0], clouds.lwc[0] if clouds.lwc.ndim > 1 else clouds.lwc,
                        clouds.reff_wc[0] if clouds.reff_wc.ndim > 1 else clouds.reff_wc, d / "wc.dat")
            o = o.with_(wc_file=str(d / "wc.dat"))
        if clouds is not None and clouds.iwc is not None:
            write_cloud(col.z[0], clouds.iwc[0] if clouds.iwc.ndim > 1 else clouds.iwc,
                        clouds.reff_ic[0] if clouds.reff_ic.ndim > 1 else clouds.reff_ic, d / "ic.dat")
            o = o.with_(ic_file=str(d / "ic.dat"))
        if aerosol is not None:
            mass = aerosol.mass[0] if aerosol.mass.ndim > 2 else aerosol.mass
            write_aerosol(col.z[0], aerosol.species, mass, d / "aer.dat")
            (d / "L").symlink_to(Path(aerosol_library).resolve())
            extra += ["aerosol_default", f"aerosol_species_file {d / 'aer.dat'} " + " ".join(aerosol.species),
                      f"aerosol_species_library {d / 'L'}/"]
        text = o.to_uvspec(data_path=str(d / "D"), atmosphere_path=str(atm), extra=tuple(extra) + ("quiet",))
        p = subprocess.run(_uvspec_cmd(), input=text, capture_output=True, text=True, cwd=td)
        if keep is not None:
            keep.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_file():
                    shutil.copy(f, keep / f.name)
            (keep / "uvspec.inp").write_text(text)
            (keep / "uvspec.out").write_text(p.stdout)
            (keep / "uvspec.err").write_text(p.stderr)
        if p.returncode != 0 or "rror" in p.stdout[:200]:
            raise RuntimeError(f"uvspec failed ({p.returncode}): {p.stdout[:400]} {p.stderr[-800:]}")
        out = parse(p.stdout, nout=len(o.zout), numu=len(o.umu), nphi=len(o.phi),
                    twostr=o.rte_solver == "twostr")
        out["input"] = (text.replace(str(d / "D"), "<libRadtran data>").replace(str(d / "L"), "<OPAC library>")
                        .replace(str(d), "<run dir>"))
        return out


def receipt(data_files=()):
    """uvspec version, conda build and hashes of the data files used."""
    from woof.radtran import data as rtdata
    meta = {}
    cm = ROOT / ".pixi" / "envs" / "libradtran" / "conda-meta"
    for f in sorted(cm.glob("rubin-libradtran-*.json")) if cm.is_dir() else ():
        j = json.loads(f.read_text())
        meta = {k: j.get(k) for k in ("name", "version", "build", "channel", "url", "sha256")}
    p = subprocess.run(_uvspec_cmd() + ["-h"], capture_output=True, text=True, input="")
    m = re.search(r"uvspec, version (\S+)", p.stdout + p.stderr)
    hashes = {}
    for rel in data_files:
        fp = rtdata.data_dir() / rel
        if fp.exists():
            hashes[rel] = hashlib.sha256(fp.read_bytes()).hexdigest()
    return {"uvspec_version": m.group(1) if m else None, "package": meta,
            "platform": f"{platform.system()} {platform.machine()}",
            "data_sha256": hashes}
