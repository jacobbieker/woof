"""The uvspec oracle case matrix for woof.radtran.

Each case is (name, options, column builder, clouds builder).  The same
objects drive uvspec (tools/uvspec_oracle/oracle.py) and woof
(woof.radtran.driver.compute).  Columns are built at run time from the
user's libRadtran data (AFGL profiles are never stored in woof); the
fixtures in tests/data/oracles/uvspec/ hold a digest of the built inputs.
"""

from __future__ import annotations

import numpy as np

from woof.radtran import atmosphere
from woof.radtran.driver import CloudLayers
from woof.radtran.options import Options


def afgl(name="us"):
    return lambda: atmosphere.afgl(name)


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
    # a model column spliced onto afglus
    "model_clear": (O(wavelength=(400.0, 403.0), sza=35.0), model_column, None),
    "model_h2o": (O(wavelength=(935.0, 938.0), sza=35.0), model_column, None),
    "model_wc": (O(wavelength=(550.0, 552.0), sza=35.0), model_column, lambda c: wc_layers(c, ztop=3.2, zbot=1.5)),
    # Sun-Earth distance
    "day_of_year_180": (O(wavelength=(450.0, 451.0), sza=30.0, day_of_year=180), afgl(), None),
}


def build(name):
    opts, colf, clf = CASES[name]
    col = colf()
    cl = clf(col) if clf is not None else None
    return opts, col, cl
