"""Cloud optical properties (parameterisation tables read from libRadtran data).

Water clouds: Hu & Stamnes (1993), J. Climate 6, 728-742.  For each of
three effective-radius ranges (2.5-12.5, 12.5-30, 30-60 um) the tables
``wc/wc.ext``, ``wc/wc.ssa``, ``wc/wc.asy`` give a, b, c at a set of
wavelengths with

    beta_ext / LWC = a r^b + c   [km^-1 per g m^-3],
    1 - omega      = a r^b + c,
    g              = a r^b + c,

each evaluated at the tabulated wavelengths and interpolated linearly in
wavelength.  The phase function is Henyey-Greenstein with that g.

Ice clouds: Fu (1996), J. Climate 9, 2058-2082, eqs. 3.9a-c, constant within
each of its bands (``ic/fu96``):

    beta_ext = IWC (a0 + a1 / D)   [m^-1, IWC in g m^-3, D in um],
    1 - omega = b0 + b1 D + b2 D^2 + b3 D^3,
    g = c0 + c1 D + c2 D^2 + c3 D^3,

with the generalized effective size D = 2 r_Fu and, by default, the effective
radius converted from the volume/projected-area definition (Yang/Key/Baum)
to Fu's: r_Fu = r 4 / (3 sqrt 3) (the 3 sqrt(3)/4 factor for hexagonal
columns).  Henyey-Greenstein phase function with Fu's g; the delta fraction
is not applied (no delta-scaling of the Fu properties).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from woof.radtran import data


def _hu_read(name):
    blocks = []
    cur = None
    for line in open(data.path("wc", name)):
        if line.startswith("!"):
            if "micron < r <" in line:
                s = line.replace("!", "").split()
                cur = {"r": (float(s[0]), float(s[5])), "rows": []}
                blocks.append(cur)
            continue
        s = line.split()
        if len(s) == 4 and cur is not None:
            cur["rows"].append([float(x) for x in s])
    for b in blocks:
        b["rows"] = np.array(b["rows"])[::-1]
    return blocks


@lru_cache(maxsize=None)
def _hu():
    return _hu_read("wc.ext"), _hu_read("wc.ssa"), _hu_read("wc.asy")


def _hu_prop(blocks, lam_um, reff):
    reff = np.asarray(reff, dtype=np.float64)
    out = np.zeros(reff.shape)
    for k, b in enumerate(blocks):
        lo, hi = b["r"]
        sel = (reff >= lo) & (reff <= hi) if k == 0 else (reff > lo) & (reff <= hi)
        if not np.any(sel):
            continue
        rows = b["rows"]
        r = reff[sel]
        vals = rows[:, 1][None, :] * r[:, None] ** rows[:, 2][None, :] + rows[:, 3][None, :]
        out[sel] = np.array([np.interp(lam_um, rows[:, 0], v) for v in vals])
    return out


def hu_stamnes(lam_nm: float, reff_um):
    """(extinction per LWC [km^-1 / (g m^-3)], single-scattering albedo, g)."""
    reff = np.asarray(reff_um, dtype=np.float64)
    if np.any((reff < 2.5) | (reff > 60.0)):
        raise ValueError("Hu & Stamnes (1993) covers effective radii 2.5-60 um")
    ext, co, asy = _hu()
    lam = lam_nm * 1e-3
    return _hu_prop(ext, lam, reff), 1.0 - _hu_prop(co, lam, reff), _hu_prop(asy, lam, reff)


def _fu_read(name):
    rows = []
    for line in open(data.path("ic", "fu96", name)):
        s = line.split()
        if not s or s[0].startswith("#"):
            continue
        rows.append([float(x) for x in s])
    return np.array(rows)


@lru_cache(maxsize=None)
def _fu():
    return _fu_read("fu96.ext"), _fu_read("fu96.ssa"), _fu_read("fu96.asy")


YANG_TO_FU = 4.0 / (3.0 * math.sqrt(3.0))


def fu96(lam_nm: float, reff_um, *, reff_def_fu: bool = False):
    """(extinction per IWC [m^-1 / (g m^-3)], single-scattering albedo, g)."""
    ext, ssa, asy = _fu()
    lam = lam_nm * 1e-3
    i = np.nonzero((ext[:, 0] <= lam) & (ext[:, 1] > lam))[0]
    if i.size == 0:
        raise ValueError(f"Fu (1996) covers 0.25-4.99 um; got {lam_nm} nm")
    i = int(i[0])
    r = np.asarray(reff_um, dtype=np.float64)
    D = 2.0 * (r if reff_def_fu else r * YANG_TO_FU)
    a0, a1 = ext[i, 2], ext[i, 3]
    b = ssa[i, 2:]
    c = asy[i, 2:]
    bext = a0 + a1 / D
    coal = b[0] + b[1] * D + b[2] * D * D + b[3] * D ** 3
    g = c[0] + c[1] * D + c[2] * D * D + c[3] * D ** 3
    return bext, 1.0 - coal, g
