"""Atmospheric columns for woof.radtran.

A column is a set of levels, top first: altitude above sea level z [km],
pressure p [hPa], temperature T [K], air number density n_air [cm-3] and
number densities [cm-3] of the absorbers O3, O2, H2O, CO2, NO2, CH4, N2O,
CO, N2.

Model columns (WRF, MPAS) cover the model's own levels only.  Above the
model top woof splices a standard AFGL atmosphere (Anderson et al. 1986,
AFGL-TR-86-0110; the U.S. Standard 1976 profile ``afglus`` by default, read
from the user's libRadtran data at run time): its levels strictly above the
model top are appended, and every gas the model does not carry (O2, NO2, CH4,
N2O, CO, N2, and O3/CO2 when absent) is taken from the AFGL profile as a
volume mixing ratio interpolated linearly in altitude onto the model levels.

Layer column amounts follow what uvspec does with the same levels (measured
against it, see docs/dev/radtran.md): within a layer the air density varies
exponentially with altitude and each gas's mixing ratio linearly, so

    N_gas = int_0^dz n_b exp(-t/H) (v_b + (v_t - v_b) t/dz) dt,
    H = dz / ln(n_b / n_t),

which for air itself is the log-mean density times the thickness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from woof.radtran import data

GASES = ("O3", "O2", "H2O", "CO2", "NO2", "CH4", "N2O", "CO", "N2")
AFGL = {"us": "afglus", "tropics": "afglt", "midlatitude_summer": "afglms",
        "midlatitude_winter": "afglmw", "subarctic_summer": "afglss",
        "subarctic_winter": "afglsw"}
#: uvspec's Boltzmann constant check uses p = n k T; woof derives n_air from
#: p and T with the CODATA 2018 value when a model column gives no density.
KB = 1.380649e-23


@dataclass
class Column:
    z: np.ndarray                 # (..., nlev) km, top first
    p: np.ndarray                 # hPa
    t: np.ndarray                 # K
    n_air: np.ndarray             # cm-3
    dens: dict = field(default_factory=dict)   # gas -> (..., nlev) cm-3

    @property
    def nlev(self) -> int:
        return self.z.shape[-1]

    def vmr(self, gas: str) -> np.ndarray:
        return self.dens[gas] / self.n_air


def _read_table(path):
    rows = []
    for line in open(path):
        s = line.split()
        if not s or s[0].startswith("#"):
            continue
        rows.append([float(x) for x in s])
    return np.array(rows, dtype=np.float64)


def afgl(name: str = "us") -> Column:
    """An AFGL standard atmosphere with all nine gases (cm-3), top first."""
    stem = AFGL.get(name, name)
    a = _read_table(data.path("atmmod", f"{stem}.dat"))
    col = Column(z=a[:, 0], p=a[:, 1], t=a[:, 2], n_air=a[:, 3],
                 dens={"O3": a[:, 4], "O2": a[:, 5], "H2O": a[:, 6], "CO2": a[:, 7],
                       "NO2": a[:, 8] if a.shape[1] > 8 else np.zeros(a.shape[0])})
    # CH4, N2O, CO, N2: libRadtran's U.S. standard vmr profiles (afglus_*_vmr.dat)
    for gas in ("CH4", "N2O", "CO", "N2"):
        v = _read_table(data.path("atmmod", f"afglus_{gas.lower()}_vmr.dat"))
        vmr = np.interp(-col.z, -v[:, 0], v[:, 1])
        col.dens[gas] = vmr * col.n_air
    return col


@lru_cache(maxsize=None)
def _afgl_ref(name: str) -> Column:
    """:func:`afgl`, read once per process (the data directory is fixed per
    process, :func:`woof.radtran.data.data_dir`) and read-only."""
    col = afgl(name)
    for a in (col.z, col.p, col.t, col.n_air, *col.dens.values()):
        a.setflags(write=False)
    return col


def interp_vmr(src: Column, gas: str, z_km: np.ndarray) -> np.ndarray:
    """Mixing ratio of ``gas`` from ``src`` at altitudes z (linear in z)."""
    return np.interp(-np.asarray(z_km), -src.z, src.vmr(gas))


def model_column(z_km, p_hpa, t_k, *, h2o_vmr=None, o3_vmr=None, co2_vmr=None,
                 n_air=None, above: str = "us") -> Column:
    """A model column (levels top first, 1-D) spliced onto an AFGL profile.

    The AFGL levels strictly above the model top are prepended unchanged;
    gases the model does not supply come from the AFGL profile's mixing
    ratio at the model levels.
    """
    z = np.asarray(z_km, dtype=np.float64)
    p = np.asarray(p_hpa, dtype=np.float64)
    t = np.asarray(t_k, dtype=np.float64)
    if z.ndim != 1 or not np.all(np.diff(z) < 0):
        raise ValueError("model column altitudes must be 1-D and strictly decreasing (top first)")
    na = p * 100.0 / (KB * t) * 1e-6 if n_air is None else np.asarray(n_air, dtype=np.float64)
    ref = afgl(above)
    top = ref.z > z[0]
    zz = np.concatenate((ref.z[top], z))
    pp = np.concatenate((ref.p[top], p))
    tt = np.concatenate((ref.t[top], t))
    nn = np.concatenate((ref.n_air[top], na))
    dens = {}
    given = {"H2O": h2o_vmr, "O3": o3_vmr, "CO2": co2_vmr}
    for gas in GASES:
        upper = ref.dens[gas][top]
        if given.get(gas) is not None:
            v = np.broadcast_to(np.asarray(given[gas], dtype=np.float64), z.shape)
        else:
            v = interp_vmr(ref, gas, z)
        dens[gas] = np.concatenate((upper, v * na))
    return Column(z=zz, p=pp, t=tt, n_air=nn, dens=dens)


def model_columns(z_km, p_hpa, t_k, *, h2o_vmr=None, o3_vmr=None, co2_vmr=None,
                  above: str = "us", gap_km: float = 1.0) -> Column:
    """Many model columns (ncol, nlev), top first, spliced onto one AFGL profile.

    Every column receives the same AFGL levels: those more than ``gap_km``
    above the highest model top, so each column has the same number of
    levels (the gap layer between a column's top and the first AFGL level
    differs in thickness).  Gases the model does not supply take the AFGL
    mixing ratio at the model levels.
    """
    z = np.asarray(z_km, dtype=np.float64)
    p = np.asarray(p_hpa, dtype=np.float64)
    t = np.asarray(t_k, dtype=np.float64)
    if z.ndim != 2 or not np.all(np.diff(z, axis=1) < 0):
        raise ValueError("model columns must be (ncol, nlev) with altitudes strictly decreasing")
    ncol, nlev = z.shape
    na = p * 100.0 / (KB * t) * 1e-6
    ref = _afgl_ref(above)
    top = ref.z > z[:, 0].max() + gap_km
    nt = int(top.sum())

    def spliced(upper, model=None):
        # (the AFGL levels above, the model levels): written in place, the
        # words of np.concatenate((broadcast upper, model), axis=1)
        out = np.empty((ncol, nt + nlev))
        out[:, :nt] = upper[top]
        if model is not None:
            out[:, nt:] = model
        return out

    dens = {}
    given = {"H2O": h2o_vmr, "O3": o3_vmr, "CO2": co2_vmr}
    mz, mref = -z, -ref.z
    for gas in GASES:
        if given.get(gas) is not None:
            v = np.broadcast_to(np.asarray(given[gas], dtype=np.float64), z.shape)
        else:
            v = np.interp(mz, mref, ref.vmr(gas))
        dens[gas] = spliced(ref.dens[gas])
        np.multiply(v, na, out=dens[gas][:, nt:])
    return Column(z=spliced(ref.z, z), p=spliced(ref.p, p), t=spliced(ref.t, t), n_air=spliced(ref.n_air, na),
                  dens=dens)


def layer_air(z_km, n_air):
    """The air-only factors of :func:`layer_column` for (..., nlev) levels:
    (n_top, n_bottom, n_bottom dz, phi0, phi1).  Every gas column of the same
    levels shares them, so one evaluation serves all of them."""
    z = np.asarray(z_km, dtype=np.float64)
    na = np.asarray(n_air, dtype=np.float64)
    nt, nb = na[..., :-1], na[..., 1:]
    dz = (z[..., :-1] - z[..., 1:]) * 1e5          # cm
    # x = dz/H = ln(n_b/n_t); with u = t/dz the integral is
    # n_b dz [v_b phi0(x) + (v_t - v_b) phi1(x)],
    # phi0 = (1 - e^-x)/x, phi1 = (1 - e^-x - x e^-x)/x^2.
    x = np.log(nb / nt)
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    em = -np.expm1(-xs)
    phi0 = np.where(small, 1.0 - x / 2 + x * x / 6 - x ** 3 / 24, em / xs)
    phi1 = np.where(small, 0.5 - x / 3 + x * x / 8 - x ** 3 / 30,
                    (em - xs * np.exp(-xs)) / (xs * xs))
    return nt, nb, nb * dz, phi0, phi1


def layer_column(z_km, n_air, n_gas=None, *, air=None):
    """Per-layer column [cm-2] (top first), exponential air x linear vmr.

    Works on (..., nlev) arrays; ``n_gas=None`` gives the air column.
    ``air`` is :func:`layer_air` of the same ``z_km`` and ``n_air``, when the
    caller already has it (the same words as evaluating it here).
    """
    nt, nb, nbdz, phi0, phi1 = layer_air(z_km, n_air) if air is None else air
    if n_gas is None:
        vt = np.ones_like(nt)
        vb = np.ones_like(nb)
    else:
        ng = np.asarray(n_gas, dtype=np.float64)
        vt, vb = ng[..., :-1] / nt, ng[..., 1:] / nb
    return nbdz * (vb * phi0 + (vt - vb) * phi1)
