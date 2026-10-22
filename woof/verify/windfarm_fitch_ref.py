"""Float64 NumPy mirror of ``fitch_column`` (woof/core/kernels/windfarm_fitch.cu).

The Fitch et al. (2012) wind-farm drag as WRF v4.7.1 writes it
(``phys/module_wind_fitch.F``): the same level searches, the same table
interpolation, the same energy normalisation and the same three deviations
the kernel documents, in full float64, over the same turbine groups.  For
verification only; the device kernel is graded against it by
``tests/test_windfarm_fitch.py``.
"""
from __future__ import annotations

import math

import numpy as np

#: The density WRF forms the power coefficients and POWER at (dragcof, :175).
WRF_POWER_DENSITY = 1.23


def np_turbine_area(z1, z2, diameter):
    """``turbine_area`` without the wind-farm density: the disk's area
    between heights z1 < z2 above the lower blade tip, m2."""
    r = diameter / 2.0

    def term(zc):
        return zc * math.sqrt(max(r * r - zc * zc, 0.0)) + r * r * math.asin(
            min(zc / r, 1.0))

    a, b = r - z1, r - z2
    if a > 0.0 and b > 0.0:
        return term(abs(a)) - term(abs(b))
    if a < 0.0 and b < 0.0:
        return term(abs(b)) - term(abs(a))
    return term(abs(b)) + term(abs(a))


def np_dragcof(speed, ws, tc, tp, stc, diameter, tke_factor):
    """``dragcof``: (tkecof, powcof, thrcof) at hub speed ``speed``.

    ``ws``/``tc``/``tp`` are one curve's first nval table entries (m s-1,
    Ct, kW).  Cp is linear in the nodes' P/(0.5*1.23*V^3*A), as WRF forms it.
    """
    area = math.pi / 4.0 * diameter ** 2
    cutin, cutout = ws[0], ws[-1]
    above = np.nonzero(ws > speed)[0]
    nu = int(above[0]) if above.size else len(ws) - 1
    nb = nu - 1
    if speed <= cutin:
        thrcof = stc
    elif speed >= cutout:
        thrcof = tc[-1]
    else:
        thrcof = tc[nb] + (tc[nu] - tc[nb]) / (ws[nu] - ws[nb]) * (speed - ws[nb])
    if speed <= cutin or speed >= cutout:
        powcof = 0.0
    else:
        rho = WRF_POWER_DENSITY
        fac1 = 1000.0 / (0.5 * rho * ws[nb] ** 3 * area)
        fac2 = 1000.0 / (0.5 * rho * ws[nu] ** 3 * area)
        powcof = tp[nb] * fac1 + (tp[nu] * fac2 - tp[nb] * fac1) / (
            ws[nu] - ws[nb]) * (speed - ws[nb])
    tkecof = max(tke_factor * (thrcof - powcof), 0.0)
    return tkecof, powcof, thrcof


def _group(out, g, j, i, layout, u, v, zw, rho, cell, dt, tke_factor):
    """One group's contribution to cell (j, i), added into ``out``.

    Returns the group's swept area (count*A, m2), 0 when it is skipped.
    """
    nturb = int(layout.count[g])
    if nturb <= 0:
        return 0.0
    t = int(layout.curve[g])
    nv = int(layout.nval[t])
    ws = np.asarray(layout.speeds_ms[t, :nv], dtype=np.float64)
    tc = np.asarray(layout.thrust_coefficient[t, :nv], dtype=np.float64)
    tp = np.asarray(layout.power_kw[t, :nv], dtype=np.float64)
    hub = float(layout.hub_height_m[g])
    dia = float(layout.rotor_diameter_m[g])
    nz = u.shape[0]
    col = zw[:, j, i]
    dz = np.diff(col)
    speed = np.hypot(u[:, j, i], v[:, j, i])
    blade_l, blade_u = hub - dia / 2.0, hub + dia / 2.0
    # The levels the blades cut, from the cumulative depths.
    k_bot = k_top = None
    zl = 0.0
    for k in range(nz):
        zu = zl + dz[k]
        if zl <= blade_l <= zu:
            k_bot = k
        if zl <= blade_u <= zu:
            k_top = k
            break
        zl = zu
    if k_top is None or k_bot is None:
        return 0.0
    # Hub-height speed between the mass levels about the hub.
    centers = np.cumsum(dz) - 0.5 * dz
    kt = int(np.searchsorted(centers, hub, side="right"))
    if kt >= nz:
        speedhub = speed[-1]
    elif kt == 0:
        speedhub = speed[0] * hub / centers[0]
    else:
        z1, z2 = centers[kt - 1], centers[kt]
        speedhub = speed[kt - 1] + (speed[kt] - speed[kt - 1]) / (
            z2 - z1) * (hub - z1)
    tkecof, powcof, thrcof = np_dragcof(
        speedhub, ws, tc, tp, float(layout.standing_thrust[t]), dia,
        tke_factor)
    area = math.pi / 4.0 * dia ** 2
    power1 = 0.5 * WRF_POWER_DENSITY * speedhub ** 3 * area * powcof
    areas = {}
    for k in range(k_bot, k_top + 1):
        z1 = min(max(col[k] - blade_l - col[0], 0.0), dia)
        z2 = min(max(col[k + 1] - blade_l - col[0], 0.0), dia)
        areas[k] = np_turbine_area(z1, z2, dia)
    power2 = sum(0.5 * powcof * WRF_POWER_DENSITY * speed[k] ** 3 * a
                 for k, a in areas.items())
    ec = 1.0 if power1 == 0.0 or power2 == 0.0 else power1 / power2
    for k, a in areas.items():
        out["overlap"][k, j, i] += nturb * a
        mass = cell * dz[k]
        out["du"][k, j, i] -= nturb * 0.5 * u[k, j, i] * thrcof * speed[k] * a / mass * ec
        out["dv"][k, j, i] -= nturb * 0.5 * v[k, j, i] * thrcof * speed[k] * a / mass * ec
        out["dqke"][k, j, i] += nturb * speed[k] ** 3 * a * tkecof * dt / mass * ec
        if rho is not None:
            out["thrust"][j, i] += (nturb * rho[k, j, i] * 0.5 * thrcof
                                    * speed[k] ** 2 * a * ec)
    out["power"][j, i] += nturb * power1
    out["groups"][g] = dict(ct=thrcof, cp=powcof, ctke=tkecof,
                            speedhub=speedhub, ec=ec, power1=power1)
    return nturb * area


def np_fitch_column(u, v, z_interface, layout, *, dx, dy=None, dt=0.0,
                    tke_factor=0.25, rho=None):
    """The kernel's outputs from zero tendencies, in float64.

    Returns a dict: ``du``, ``dv`` (m s-2), ``dqke`` (the QKE increment over
    ``dt``), ``overlap`` (nz, ny, nx; the fraction of the cell's swept area
    in each level), ``power`` (W) and, with ``rho``, ``thrust`` (N), each
    (ny, nx); and ``groups``, per group index the coefficients ``ct``,
    ``cp``, ``ctke``, the hub speed ``speedhub``, the normalisation ``ec``
    and one turbine's power ``power1``.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    zw = np.asarray(z_interface, dtype=np.float64)
    nz, ny, nx = u.shape
    dy = dx if dy is None else dy
    cell = float(dx) * float(dy)
    out = {name: np.zeros((nz, ny, nx)) for name in ("du", "dv", "dqke",
                                                     "overlap")}
    out["power"] = np.zeros((ny, nx))
    out["thrust"] = np.zeros((ny, nx))
    out["groups"] = {}
    rho = None if rho is None else np.asarray(rho, dtype=np.float64)
    offsets = np.asarray(layout.cell_offsets)
    for c in np.flatnonzero(np.diff(offsets)):
        j, i = divmod(int(c), nx)
        swept = sum(_group(out, g, j, i, layout, u, v, zw, rho, cell, dt,
                           tke_factor)
                    for g in range(int(offsets[c]), int(offsets[c + 1])))
        if swept > 0.0:
            out["overlap"][:, j, i] /= swept
    return out


# ---------------------------------------------------------------------------
# The binary32 CPU mirror: fitch_column statement for statement
# ---------------------------------------------------------------------------

F = np.float32


def _powf(x, y):
    """glibc's powf, host-independent: the transcription gfk_pow mirrors."""
    from woof.core.noahmp_libm import powf
    return F(powf(F(x), F(y)))


# glibc 2.43 e_asinf.c (CORE-MATH, correctly rounded), as glibc_asinf in
# woof/core/kernels/glibc_trig_flt32.cuh transcribes it, in binary64 Python
# arithmetic (every operation there is an explicit round-to-nearest one).
_ASINF_B = tuple(float.fromhex(h) for h in (
    "0x1.0000000000005p+0", "0x1.55557aeca105dp-3", "0x1.3314ec3db7d12p-4",
    "0x1.775738a5a6f92p-5", "0x1.5d5f7ce1c8538p-8", "0x1.605c6d58740fp-2",
    "-0x1.5728b732d73c6p+1", "0x1.f152170f151ebp+3", "-0x1.f962ea3ca992ep+5",
    "0x1.71971e17375ap+7", "-0x1.860512b4ba23p+8", "0x1.26a3b8d4bdb14p+9",
    "-0x1.36f2ea5698b51p+9", "0x1.b3d722aebfa2ep+8", "-0x1.6cf89703b1289p+7",
    "0x1.1518af6a65e2dp+5"))
_ASINCOS_C0 = tuple(float.fromhex(h) for h in (
    "0x1.555555555529cp-3", "0x1.333333337e0ddp-4", "0x1.6db6db3b4465ep-5",
    "0x1.f1c72e13ac306p-6", "0x1.6e89cebe06bc4p-6", "0x1.1c6dcf5289094p-6",
    "0x1.c6dbbcc7c6315p-7", "0x1.8f8dc2615e996p-7", "0x1.a5833b7bf15e8p-8",
    "0x1.43f44ace1665cp-6", "-0x1.0fb17df881c73p-6", "0x1.07520c026b2d6p-5"))
_ASINCOS_C1 = tuple(float.fromhex(h) for h in (
    "0x1.6a09e667f3bcbp+0", "0x1.e2b7dddff2db9p-4", "0x1.b27247ab42dbcp-6",
    "0x1.02995cc4e0744p-7", "0x1.5ffb0276ec8eap-9", "0x1.033885a928decp-10",
    "0x1.911f2be23f8c7p-12", "0x1.4c3c55d2437fdp-13", "0x1.af477e1d7b461p-15",
    "0x1.abd6bdff67dcbp-15", "-0x1.1717e86d0fa28p-16", "0x1.6ff526de46023p-16"))


def _asinf_poly12(z, c):
    z2 = z * z
    z4 = z2 * z2
    c0 = c[0] + z * c[1]
    c2 = c[2] + z * c[3]
    c4 = c[4] + z * c[5]
    c6 = c[6] + z * c[7]
    c8 = c[8] + z * c[9]
    c10 = c[10] + z * c[11]
    c0 = c0 + c2 * z2
    c4 = c4 + c6 * z2
    c8 = c8 + z2 * c10
    return c0 + z4 * (c4 + z4 * c8)


def _asinf(x):
    """glibc_asinf for finite |x| <= 1, word for word."""
    x = F(x)
    ax = (int(np.asarray(x).view(np.uint32)) << 1) & 0xFFFFFFFF
    if ax > (0x7F << 24):
        raise ValueError(f"asinf({x}) is outside the drag's domain")
    xs = float(x)
    if ax < 0x7EC29000:
        if ax < (115 << 24):
            if ax == 0:
                return x
            return F(xs * 2.0 ** -25 + xs)          # fmaf, exact in binary64
        b = _ASINF_B
        z = xs
        z2 = z * z
        z4 = z2 * z2
        z8 = z4 * z4
        z16 = z8 * z8
        r = z * ((((b[0] + z2 * b[1]) + z4 * (b[2] + z2 * b[3]))
                  + z8 * ((b[4] + z2 * b[5]) + z4 * (b[6] + z2 * b[7])))
                 + z16 * (((b[8] + z2 * b[9]) + z4 * (b[10] + z2 * b[11]))
                          + z8 * ((b[12] + z2 * b[13])
                                  + z4 * (b[14] + z2 * b[15]))))
        ub = F(r)
        lb = F(r - z * float.fromhex("0x1.efa8ebp-31"))
        if ub == lb:
            return ub
    if ax < (0x7E << 24):
        z = xs
        z2 = z * z
        r = z + (z * z2) * _asinf_poly12(z2, _ASINCOS_C0)
    else:
        sign = -1.0 if xs < 0 else 1.0
        if ax == 0x7E55688A:
            return F(F(sign * float.fromhex("0x1.75b8a2p-1"))
                     + F(sign * 2.0 ** -26))
        if ax == 0x7E107434:
            return F(F(sign * float.fromhex("0x1.1f4b64p-1"))
                     + F(sign * 2.0 ** -26))
        z = 1.0 - abs(xs)
        r = float.fromhex("0x1.921fb54442d18p+0") - math.sqrt(z) * _asinf_poly12(
            z, _ASINCOS_C1)
        r = math.copysign(r, xs)
    return F(r)


def _area_term(zc, r):
    return zc * np.sqrt(r * r - zc * zc) + r * r * _asinf(zc / r)


def _turbine_area32(z1, z2, tdiameter, wfdensity):
    r = tdiameter / F(2.0)
    z1 = r - z1
    z2 = r - z2
    zc1, zc2 = abs(z1), abs(z2)
    if z1 > F(0.0) and z2 > F(0.0):
        tarea = _area_term(zc1, r) - _area_term(zc2, r)
    elif z1 < F(0.0) and z2 < F(0.0):
        tarea = _area_term(zc2, r) - _area_term(zc1, r)
    else:
        tarea = (_area_term(zc2, r) + zc1 * np.sqrt(r * r - zc1 * zc1)
                 + r * r * _asinf(zc1 / r))
    return tarea * wfdensity


def _layer_area32(zw, k, blade_l, zw0, dia, wfdensity):
    z1 = zw[k] - blade_l - zw0
    z2 = zw[k + 1] - blade_l - zw0
    if z1 < F(0.0):
        z1 = F(0.0)
    if z2 > dia:
        z2 = dia
    if z1 > dia:
        z1 = dia
    if z2 < F(0.0):
        z2 = F(0.0)
    return _turbine_area32(z1, z2, dia, wfdensity)


def _speed32(u, v, k):
    return np.sqrt(_powf(u[k], 2.0) + _powf(v[k], 2.0))


def f32_fitch_column(u, v, z_interface, layout, du, dv, qke=None, *, dx,
                     dy=None, dt=0.0, tke_factor=0.25, rho=None):
    """The kernel in binary32 on the CPU, word for word.

    Adds into float32 ``du``/``dv`` (and ``qke``) in place exactly as
    ``fitch_column`` does, with the C library's powf and asinf
    (:mod:`woof.core.host_libm`) where the kernel calls glibc's, and the
    kernel's order of every sum.  Returns ``(power, thrust, overlap)``.
    """
    u = np.asarray(u, dtype=F)
    v = np.asarray(v, dtype=F)
    z_w = np.asarray(z_interface, dtype=F)
    rho = None if rho is None else np.asarray(rho, dtype=F)
    nz, ny, nx = u.shape
    dy = dx if dy is None else dy
    wfdensity = F(F(1.0) / (F(dx) * F(dy)))
    dt, factor = F(dt), F(tke_factor)
    piconst = F(3.1415926535897932384626433)
    power = np.zeros((ny, nx), F)
    thrust = np.zeros((ny, nx), F)
    overlap = np.zeros((nz, ny, nx), F)
    offsets = np.asarray(layout.cell_offsets)
    maxvals = layout.speeds_ms.shape[1]
    for c in range(ny * nx):
        j, i = divmod(c, nx)
        uc, vc, zw = u[:, j, i], v[:, j, i], z_w[:, j, i]
        zw0 = zw[0]
        pw = th = swept = F(0.0)
        for g in range(int(offsets[c]), int(offsets[c + 1])):
            nturb = int(layout.count[g])
            if nturb <= 0:
                continue
            hub = F(layout.hub_height_m[g])
            dia = F(layout.rotor_diameter_m[g])
            t = int(layout.curve[g])
            ws = layout.speeds_ms[t].astype(F)
            tc = layout.thrust_coefficient[t].astype(F)
            tp = layout.power_kw[t].astype(F)
            nv = int(layout.nval[t])
            cutin, cutout = ws[0], ws[nv - 1]
            stc1, stc2 = F(layout.standing_thrust[t]), tc[nv - 1]
            half = dia / F(2.0)
            blade_l, blade_u = hub - half, hub + half
            k_bot = k_top = -1
            zl = F(0.0)
            for k in range(nz):
                dz = zw[k + 1] - zw[k]
                zu = zl + dz
                if zl <= blade_l <= zu:
                    k_bot = k
                if zl <= blade_u <= zu:
                    k_top = k
                    break
                zl = zu
            if k_top < 0 or k_bot < 0:
                continue
            kb = kt = nz - 1
            hfound = False
            z1 = z2 = F(0.0)
            zl = F(0.0)
            for k in range(nz):
                dz = zw[k + 1] - zw[k]
                z2 = zl + F(0.5) * dz
                if hub >= z2:
                    kb = k
                else:
                    kt = k
                    hfound = True
                    break
                z1 = z2
                zl = z2 + F(0.5) * dz
            if not hfound:
                speedhub = _speed32(uc, vc, nz - 1)
            elif kt == 0:
                z1 = z2
                speedhub = _speed32(uc, vc, 0) * hub / z1
            else:
                s1, s2 = _speed32(uc, vc, kb), _speed32(uc, vc, kt)
                speedhub = s1 + ((s2 - s1) / (z2 - z1)) * (hub - z1)
            area = (piconst / F(4.0)) * _powf(dia, 2.0)
            nu = nb = 0
            for k in range(maxvals):
                if ws[k] > speedhub:
                    nu, nb = k, k - 1
                    break
            if speedhub <= cutin:
                thrcof = stc1
            elif speedhub >= cutout:
                thrcof = stc2
            else:
                thrcof = tc[nb] + (tc[nu] - tc[nb]) / (ws[nu] - ws[nb]) * (
                    speedhub - ws[nb])
            if speedhub <= cutin or speedhub >= cutout:
                powcof = F(0.0)
            else:
                fac1 = F(1000.0) / (F(0.5) * F(1.23) * _powf(ws[nb], 3.0) * area)
                fac2 = F(1000.0) / (F(0.5) * F(1.23) * _powf(ws[nu], 3.0) * area)
                powcof = tp[nb] * fac1 + (tp[nu] * fac2 - tp[nb] * fac1) / (
                    ws[nu] - ws[nb]) * (speedhub - ws[nb])
            tkecof = factor * (thrcof - powcof)
            if tkecof < F(0.0):
                tkecof = F(0.0)
            power1 = F(0.5) * F(1.23) * _powf(speedhub, 3.0) * area * powcof
            power2 = F(0.0)
            for k in range(k_bot, k_top + 1):
                tarea = _layer_area32(zw, k, blade_l, zw0, dia, wfdensity)
                speed = _speed32(uc, vc, k)
                power2 = power2 + F(0.5) * powcof * F(1.23) * _powf(
                    speed, 3.0) * tarea / wfdensity
            ec = (F(1.0) if power1 == F(0.0) or power2 == F(0.0)
                  else power1 / power2)
            for _ in range(nturb):
                pw = power1 + pw
            for k in range(k_bot, k_top + 1):
                tarea = _layer_area32(zw, k, blade_l, zw0, dia, wfdensity)
                speed = _speed32(uc, vc, k)
                dz = zw[k + 1] - zw[k]
                dq = _powf(speed, 3.0) * tarea * tkecof * dt / dz * ec
                fu = F(0.5) * uc[k] * thrcof * speed * tarea / dz * ec
                fv = F(0.5) * vc[k] * thrcof * speed * tarea / dz * ec
                for _ in range(nturb):
                    if qke is not None:
                        qke[k, j, i] = qke[k, j, i] + dq
                    du[k, j, i] = du[k, j, i] - fu
                    dv[k, j, i] = dv[k, j, i] - fv
                overlap[k, j, i] = overlap[k, j, i] + F(nturb) * (tarea / wfdensity)
                if rho is not None:
                    f = rho[k, j, i] * (F(0.5) * thrcof * speed * speed * tarea
                                        * ec) / wfdensity
                    for _ in range(nturb):
                        th = th + f
            swept = swept + F(nturb) * area
        if swept > F(0.0):
            overlap[:, j, i] = overlap[:, j, i] / swept
        power[j, i] = pw
        thrust[j, i] = th
    return power, thrust, overlap
