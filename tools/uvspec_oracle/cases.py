"""The uvspec oracle case matrix for woof.radtran.

Each case is (name, options, column builder, clouds builder).  The same
objects drive uvspec (tools/uvspec_oracle/oracle.py) and woof
(woof.radtran.driver.compute).  Columns are built at run time from the
user's libRadtran data (AFGL profiles are never stored in woof); the
fixtures in tests/data/oracles/uvspec/ hold a digest of the built inputs.
"""

from __future__ import annotations

import numpy as np

from woof.radtran import aerosol as opac
from woof.radtran import atmosphere
from woof.radtran.driver import CloudLayers
from woof.radtran.options import Options


def afgl(name="us"):
    return lambda: atmosphere.afgl(name)


def afgl_rh(name="us", rh=50.0, ztop=12.0):
    """An AFGL atmosphere whose water vapour gives relative humidity ``rh`` [%]
    (over water, woof.radtran.aerosol's saturation law) at every level up to
    ``ztop`` km; the AFGL profile above."""
    def f():
        col = atmosphere.afgl(name)
        n = rh / 100.0 * opac.saturation_pressure_hpa(col.t) * 100.0 / (opac.KB * col.t) * 1e-6
        col.dens["H2O"] = np.where(col.z <= ztop, n, col.dens["H2O"])
        return col
    return f


def opac_mixture(name):
    """An OPAC mixture (libRadtran's standard_aerosol_files) on the column's layers."""
    def f(col):
        z = col.z if col.z.ndim == 1 else col.z[0]
        return opac.mixture_layers(name, z[1:] - z[-1])
    return f


def model_column():
    """A synthetic WRF-like column (41 levels, 0.35-17 km) spliced on afglus."""
    z = np.linspace(17.0, 0.35, 41)
    t = 288.0 - 6.5 * (z - 0.35)
    t = np.maximum(t, 216.0)
    # hydrostatic pressure from the surface (1000 hPa at 0.35 km)
    p = np.empty_like(z)
    p[-1] = 1000.0
    for k in range(z.size - 2, -1, -1):
        tm = 0.5 * (t[k] + t[k + 1])
        p[k] = p[k + 1] * np.exp(-9.80665 * (z[k] - z[k + 1]) * 1000.0 / (287.05 * tm))
    rh_q = 0.012 * np.exp(-(z - 0.35) / 2.2)
    h2o = rh_q * 28.964 / 18.015
    o3 = 3e-8 + 6e-8 * np.exp(-((z - 17.0) / 4.0) ** 2)
    return atmosphere.model_column(z, p, t, h2o_vmr=h2o, o3_vmr=o3, co2_vmr=4.1e-4)


def wc_layers(col, ztop=4.0, zbot=2.0, lwc=0.2, reff=10.0):
    z = col.z if col.z.ndim == 1 else col.z[0]
    zb = z[1:]
    sel = (zb >= zbot - 1e-9) & (zb < ztop - 1e-9)
    return CloudLayers(lwc=np.where(sel, lwc, 0.0), reff_wc=np.where(sel, reff, 0.0))


def ic_layers(col, ztop=10.0, zbot=8.0, iwc=0.02, reff=30.0):
    z = col.z if col.z.ndim == 1 else col.z[0]
    zb = z[1:]
    sel = (zb >= zbot - 1e-9) & (zb < ztop - 1e-9)
    return CloudLayers(iwc=np.where(sel, iwc, 0.0), reff_ic=np.where(sel, reff, 0.0))


def wc_ic_layers(col):
    w = wc_layers(col, lwc=0.01, reff=8.0)
    i = ic_layers(col, iwc=0.005, reff=40.0)
    return CloudLayers(lwc=w.lwc, reff_wc=w.reff_wc, iwc=i.iwc, reff_ic=i.reff_ic)


def O(**kw):
    base = dict(number_of_streams=8, albedo=0.2, zout=("sur", "toa"))
    base.update(kw)
    return Options(**base).validate()


UMU = (-0.9, -0.5, -0.1, 0.1, 0.5, 0.9)
PHI = (0.0, 45.0, 180.0)

CASES = {
    # clear-sky Rayleigh, one band, absorption off
    "rayleigh_mono_sza0": (O(wavelength=(450.0, 450.0), sza=0.0, no_absorption=True), afgl(), None),
    "rayleigh_mono_sza30": (O(wavelength=(450.0, 450.0), sza=30.0, no_absorption=True), afgl(), None),
    "rayleigh_mono_sza60_alb0": (O(wavelength=(450.0, 450.0), sza=60.0, albedo=0.0, no_absorption=True), afgl(), None),
    # clear-sky REPTRAN (Rayleigh + all gases)
    "reptran_clear_400": (O(wavelength=(400.0, 405.0), sza=30.0), afgl(), None),
    "reptran_clear_tropics_550": (O(wavelength=(548.0, 552.0), sza=45.0), afgl("tropics"), None),
    # gas absorption
    "o3_huggins": (O(wavelength=(318.0, 322.0), sza=40.0), afgl(), None),
    "o3_chappuis": (O(wavelength=(598.0, 602.0), sza=40.0), afgl(), None),
    "h2o_940": (O(wavelength=(935.0, 940.0), sza=40.0), afgl("midlatitude_summer"), None),
    "o2_aband": (O(wavelength=(759.0, 768.0), sza=40.0), afgl(), None),
    "co2_ch4_1600": (O(wavelength=(1600.0, 1610.0), sza=40.0), afgl(), None),
    # clouds
    "wc_hu_500": (O(wavelength=(500.0, 502.0), sza=30.0), afgl(), lambda c: wc_layers(c)),
    "wc_hu_1600": (O(wavelength=(1600.0, 1603.0), sza=30.0), afgl(), lambda c: wc_layers(c, lwc=0.1, reff=15.0)),
    "ic_fu_550": (O(wavelength=(550.0, 552.0), sza=30.0), afgl(), lambda c: ic_layers(c)),
    "ic_fu_1600": (O(wavelength=(1600.0, 1603.0), sza=30.0), afgl(), lambda c: ic_layers(c, reff=40.0)),
    # pseudo-spherical, SZA 0/60/85/89
    "sph_sza0": (O(wavelength=(450.0, 452.0), sza=0.0, pseudospherical=True), afgl(), None),
    "sph_sza60": (O(wavelength=(450.0, 452.0), sza=60.0, pseudospherical=True), afgl(), None),
    "sph_sza85": (O(wavelength=(450.0, 452.0), sza=85.0, pseudospherical=True), afgl(), None),
    "sph_sza89": (O(wavelength=(450.0, 452.0), sza=89.0, pseudospherical=True), afgl(), None),
    "pp_sza85": (O(wavelength=(450.0, 452.0), sza=85.0), afgl(), None),
    # two-stream
    "twostr_clear": (O(wavelength=(400.0, 405.0), sza=30.0, rte_solver="twostr"), afgl(), None),
    "twostr_wc_thin": (O(wavelength=(500.0, 502.0), sza=30.0, rte_solver="twostr"), afgl(),
                       lambda c: wc_layers(c, lwc=0.002)),
    "twostr_wc": (O(wavelength=(500.0, 502.0), sza=30.0, rte_solver="twostr"), afgl(), lambda c: wc_layers(c)),
    "twostr_sph85": (O(wavelength=(450.0, 452.0), sza=85.0, rte_solver="twostr", pseudospherical=True), afgl(), None),
    # radiances, 16 streams
    "rad16_rayleigh": (O(wavelength=(400.0, 401.0), sza=30.0, number_of_streams=16, umu=UMU, phi=PHI),
                       afgl(), None),
    "rad16_wc_intcor_off": (O(wavelength=(500.0, 501.0), sza=30.0, number_of_streams=16, umu=UMU, phi=PHI,
                              disort_intcor="off"), afgl(), lambda c: wc_layers(c, lwc=0.002)),
    "rad16_wc_moments": (O(wavelength=(500.0, 501.0), sza=30.0, number_of_streams=16, umu=UMU, phi=PHI,
                           disort_intcor="moments"), afgl(), lambda c: wc_layers(c, lwc=0.002)),
    "rad16_sph85": (O(wavelength=(450.0, 451.0), sza=85.0, number_of_streams=16, umu=UMU, phi=PHI,
                      pseudospherical=True), afgl(), None),
    # Nakajima & Tanaka IMS (disort_intcor moments): directions across uvspec's
    # 10-degree window around the sun; water + ice clouds at several levels
    "rad16_ims_window_sza60": (O(wavelength=(500.0, 501.0), sza=60.0, number_of_streams=16,
                                 umu=(-0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35), phi=(0.0, 90.0),
                                 disort_intcor="moments", zout=("sur",)),
                               afgl(), lambda c: wc_layers(c, lwc=0.01)),
    "rad16_ims_wc_ic_levels": (O(wavelength=(500.0, 501.0), sza=30.0, number_of_streams=16,
                                 umu=(-0.9, -0.85, -0.8), phi=(0.0, 30.0, 120.0), disort_intcor="moments",
                                 zout=("sur", 3.0, 6.0, 9.0)),
                               afgl(), lambda c: wc_ic_layers(c)),
    # a model column spliced onto afglus
    "model_clear": (O(wavelength=(400.0, 403.0), sza=35.0), model_column, None),
    "model_h2o": (O(wavelength=(935.0, 938.0), sza=35.0), model_column, None),
    "model_wc": (O(wavelength=(550.0, 552.0), sza=35.0), model_column, lambda c: wc_layers(c, ztop=3.2, zbot=1.5)),
    # Sun-Earth distance
    "day_of_year_180": (O(wavelength=(450.0, 451.0), sza=30.0, day_of_year=180), afgl(), None),
}


#: OPAC aerosol cases: name -> (options, column, aerosol builder, library).
#: Library "woof": uvspec reads woof's own OPAC tables (written in the mie
#: tool's netCDF layout by tools/uvspec_oracle/opac_library.py), isolating
#: uvspec's aerosol layer handling; "mie": uvspec reads tables made by
#: libRadtran's mie tool (end to end, both Mie codes' size integrations).
#: Radiances with disort_intcor off: uvspec's "moments" correction refuses
#: Mie tables without complete moments.
_RAD = dict(number_of_streams=16, umu=UMU, phi=PHI, disort_intcor="off")
_IRR = dict(disort_intcor="off")
AEROSOL_CASES = {
    "opac_continental_afglus": (O(wavelength=(500.0, 502.0), sza=30.0, **_IRR), afgl(),
                                opac_mixture("continental_average"), "woof"),
    "opac_continental_rh90": (O(wavelength=(500.0, 502.0), sza=30.0, **_IRR), afgl_rh(rh=90.0),
                              opac_mixture("continental_average"), "woof"),
    "opac_maritime_rh70": (O(wavelength=(500.0, 502.0), sza=45.0, **_IRR), afgl_rh(rh=70.0),
                           opac_mixture("maritime_clean"), "woof"),
    "opac_maritime_rh97": (O(wavelength=(500.0, 502.0), sza=45.0, **_IRR), afgl_rh(rh=97.0),
                           opac_mixture("maritime_clean"), "woof"),
    "opac_urban_rh10": (O(wavelength=(500.0, 502.0), sza=30.0, **_IRR), afgl_rh(rh=10.0),
                        opac_mixture("urban"), "woof"),
    "opac_urban_rh80": (O(wavelength=(500.0, 502.0), sza=30.0, **_IRR), afgl_rh(rh=80.0),
                        opac_mixture("urban"), "woof"),
    "opac_maritime_1600_rh90": (O(wavelength=(1600.0, 1603.0), sza=30.0, **_IRR), afgl_rh(rh=90.0),
                                opac_mixture("maritime_clean"), "woof"),
    "opac_urban_rad16_rh80": (O(wavelength=(500.0, 501.0), sza=30.0, **_RAD), afgl_rh(rh=80.0),
                              opac_mixture("urban"), "woof"),
    "opac_maritime_rad16_rh95": (O(wavelength=(500.0, 501.0), sza=45.0, **_RAD), afgl_rh(rh=95.0),
                                 opac_mixture("maritime_clean"), "woof"),
    # end to end on the mie tool's tables
    "opac_continental_mie_rh50": (O(wavelength=(500.0, 502.0), sza=30.0, **_IRR), afgl_rh(rh=50.0),
                                  opac_mixture("continental_average"), "mie"),
    "opac_maritime_mie_rh90": (O(wavelength=(500.0, 502.0), sza=45.0, **_IRR), afgl_rh(rh=90.0),
                               opac_mixture("maritime_clean"), "mie"),
    "opac_urban_mie_rh80": (O(wavelength=(500.0, 502.0), sza=30.0, **_IRR), afgl_rh(rh=80.0),
                            opac_mixture("urban"), "mie"),
    "opac_maritime_mie_rad16_rh95": (O(wavelength=(500.0, 501.0), sza=45.0, **_RAD), afgl_rh(rh=95.0),
                                     opac_mixture("maritime_clean"), "mie"),
}
for _n, (_o, _c, _a, _l) in AEROSOL_CASES.items():
    CASES[_n] = (_o, _c, None)


def build(name):
    opts, colf, clf = CASES[name]
    col = colf()
    cl = clf(col) if clf is not None else None
    return opts, col, cl


def aerosol(name, col):
    """(AerosolLayers, library kind) of an aerosol case, else (None, None)."""
    if name not in AEROSOL_CASES:
        return None, None
    _, _, af, lib = AEROSOL_CASES[name]
    return af(col), lib
