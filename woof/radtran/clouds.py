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
from woof.radtran.optics import _memo


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


# np.interp(x, xp, fp) inside the table evaluates slope * (x - xp[j]) + fp[j]
# with slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j]); whether the multiply-add
# is fused depends on how numpy was compiled (clang on arm64 contracts it, the
# x86-64 wheels do not).  This canary differs between the two, so the mode
# numpy actually uses is read off it once and every row is evaluated the same
# way at once, from the two bracketing table columns only, instead of one
# np.interp call per row over the whole table.
_CANARY = (0.6123, (0.5, 0.7), (-0.3661071783200054, -1.8188992243902193))


@lru_cache(maxsize=None)
def _interp_mode():
    x, xp, fp = _CANARY
    if not hasattr(math, "fma"):                                   # Python < 3.13
        return None
    ref = float(np.interp(x, np.array(xp), np.array(fp)))
    slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
    if ref == math.fma(slope, x - xp[0], fp[0]):
        return "fused"
    if ref == slope * (x - xp[0]) + fp[0]:
        return "separate"
    return None


def _bracket(x, xp):
    """Index j with xp[j] < x < xp[j+1] when the rows can be interpolated
    vectorized (x strictly between two nodes of a strictly increasing table,
    numpy's multiply-add mode known), else None."""
    if _interp_mode() is None or not xp[0] < x < xp[-1] or not np.all(np.diff(xp) > 0):
        return None
    j = int(np.searchsorted(xp, x, side="right")) - 1
    return None if xp[j] == x else j


def _lerp(x, x0, x1, y0, y1):
    """np.interp(x, [.., x0, x1, ..], [.., y0, y1, ..]) for x0 < x < x1, elementwise over y0, y1."""
    slope = (y1 - y0) / (x1 - x0)
    d = x - x0
    if _interp_mode() == "separate":
        return slope * d + y0
    return np.array([math.fma(s, d, y) for s, y in zip(slope.tolist(), y0.tolist())], dtype=np.float64)


def _hu_prop(blocks, lam_um, reff, cache=None):
    """``cache`` (a dict kept for one ``reff``) holds each radius range's
    selection and, per bracketing table pair, the rows' coefficients, across
    wavelengths: the same expressions on the same radii, so the same words."""
    reff = np.asarray(reff, dtype=np.float64)
    out = np.zeros(reff.shape)
    for k, b in enumerate(blocks):
        lo, hi = b["r"]

        def pick():
            sel = (reff >= lo) & (reff <= hi) if k == 0 else (reff > lo) & (reff <= hi)
            return sel, (reff[sel] if np.any(sel) else None)

        sel, r = _memo(cache, (id(blocks), k), pick)
        if r is None:
            continue
        rows = b["rows"]
        xp = rows[:, 0]
        coef = lambda c: c[:, 1][None, :] * r[:, None] ** c[:, 2][None, :] + c[:, 3][None, :]
        j = _bracket(lam_um, xp)
        if j is not None:
            def pair():
                # only the two tabulated wavelengths bracketing lam enter np.interp
                vals = coef(rows[j:j + 2])
                if not np.all(np.isfinite(vals)):
                    return None
                inverse = slice(None)
                if _interp_mode() == "fused":
                    # _lerp is a scalar fma per row here: evaluate it once per
                    # distinct radius (cloud-free layers all carry the default)
                    # when each radius group's rows are the same bits as its
                    # first, and scatter back; the same words either way.
                    _, first, inv = np.unique(r, return_index=True, return_inverse=True)
                    inv = inv.ravel()
                    if np.array_equal(vals.view(np.uint64), vals[first][inv].view(np.uint64)):
                        vals, inverse = vals[first], inv
                return vals, inverse

            got = _memo(cache, (id(blocks), k, j), pair)
            if got is not None:
                vals, inverse = got
                out[sel] = _lerp(lam_um, xp[j], xp[j + 1], vals[:, 0], vals[:, 1])[inverse]
                continue
        out[sel] = np.array([np.interp(lam_um, xp, v) for v in coef(rows)])
    return out


def hu_stamnes(lam_nm: float, reff_um, *, cache: dict | None = None):
    """(extinction per LWC [km^-1 / (g m^-3)], single-scattering albedo, g).

    ``cache``: a dict kept for one ``reff_um`` across wavelengths
    (:func:`_hu_prop`)."""
    reff = np.asarray(reff_um, dtype=np.float64)
    if cache is not None and cache.get("reff_of") is not reff_um:
        # the cached selections belong to one radius array: a different one
        # starts the cache afresh (and is range-checked)
        cache.clear()
        cache["reff_of"] = reff_um
    if cache is None or "checked" not in cache:
        if np.any((reff < 2.5) | (reff > 60.0)):
            raise ValueError("Hu & Stamnes (1993) covers effective radii 2.5-60 um")
        if cache is not None:
            cache["checked"] = True
    ext, co, asy = _hu()
    lam = lam_nm * 1e-3
    return _hu_prop(ext, lam, reff, cache), 1.0 - _hu_prop(co, lam, reff, cache), _hu_prop(asy, lam, reff, cache)


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
