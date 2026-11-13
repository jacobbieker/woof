"""The clean-room SORGAM modal optics (WOOF P9, woof/core/chem_modal_optics.py).

The exact port of WRF-Chem's optics (the ``wrfchem_pnnl`` opt-in, P7) does
not exist yet, so the clean room is graded against its own references and
against physics:

1. WOOF's Mie code (woof.radtran.mie, which the tables are built with)
   against an independent evaluation of the Mie series straight from
   scipy's spherical Bessel functions (Bohren and Huffman eqs. 4.53, 4.61,
   4.62);
2. the lognormal table against the Rayleigh limits (E_abs = 6 pi Im K,
   E_sca = 4 pi |K|^2 x_g^3 exp(13.5 ln^2 sigma), K = (m^2-1)/(m^2+2)) and
   the large-particle limit (Q_ext -> 2: E_ext -> 3 pi exp(-2.5 ln^2 sigma)
   / x_g);
3. the table's interpolation against direct lognormal integration of the
   Mie efficiencies at off-node sizes and refractive indices (TOLERANCE
   below);
4. the device kernel against its float64 host twin (CUDA and Metal);
5. the mixing rules and SORGAM's row mapping, and a plausibility check of
   the mass extinction of an ammonium sulfate accumulation mode.

TOLERANCE of the table (3): extinction per volume within 5 %, single-
scattering albedo within 0.015 and asymmetry within 0.01 absolute.  Measured
at 300 random (sigma, x_g in 1e-3..30, n in 1.1..2, k in 1e-5..1) points off
the nodes: extinction 4.3 % at most (median 0.24 %; the largest errors sit
at n < 1.2, where extinction goes as (n - 1)^2, which only longwave water
reaches), albedo 0.010 (median 7e-5), asymmetry 0.008 (median 4e-4).  The interpolation error is well below
the spread of the inputs: aerosol refractive indices are uncertain by
0.02-0.05 in n and a factor 2 in k, which moves extinction by 5-10 %.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core import chem_modal_optics as MO

EXT_TOL, SSA_TOL, G_TOL = 0.05, 0.015, 0.01


def test_data_files_match_the_manifest_and_name_their_sources():
    manifest, n, k, wl, table = MO.host_data()
    assert table.shape == (3, 3, *(int(manifest[a][2]) for a in ("ln_xg", "n", "ln_k")))
    assert np.isfinite(table).all()
    ri = json.loads((MO.data_dir() / "refractive_index.v1.json").read_text())
    assert ri["wavelength_um"][:4] == [0.3, 0.4, 0.6, 0.999]
    for name in MO.CLASSES:
        src = ri["sources"][name]
        assert src["citation"]
        if not src.get("wavelength_independent"):
            assert src["database"].endswith("(CC0-1.0)") and len(src["sha256"]) == 64
    assert (k >= 0).all() and (n > 0).all()
    assert tuple(manifest["sigma_g"]) == MO.SIGMA_G


def _bessel_mie(x, m):
    """Q_ext, Q_sca, g from the series with scipy's spherical Bessel functions."""
    from scipy.special import spherical_jn, spherical_yn
    nmax = int(x + 4 * x ** (1 / 3) + 2)
    n = np.arange(1, nmax + 1)
    mx = m * x

    def psi(z, nn):
        return z * spherical_jn(nn, z)

    def dpsi(z, nn):
        return spherical_jn(nn, z) + z * spherical_jn(nn, z, derivative=True)

    xi = psi(x, n) + 1j * x * spherical_yn(n, x)
    dxi = dpsi(x, n) + 1j * (spherical_yn(n, x) + x * spherical_yn(n, x, derivative=True))
    p_mx, dp_mx = psi(mx, n), dpsi(mx, n)
    p_x, dp_x = psi(x, n), dpsi(x, n)
    a = (m * p_mx * dp_x - p_x * dp_mx) / (m * p_mx * dxi - xi * dp_mx)
    b = (p_mx * dp_x - m * p_x * dp_mx) / (p_mx * dxi - m * xi * dp_mx)
    qe = 2 / x ** 2 * ((2 * n + 1) * (a + b).real).sum()
    qs = 2 / x ** 2 * ((2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2)).sum()
    t1 = (n[:-1] * (n[:-1] + 2) / (n[:-1] + 1) * (a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:])).real).sum()
    t2 = ((2 * n + 1) / (n * (n + 1)) * (a * np.conj(b)).real).sum()
    return qe, qs, 4 / x ** 2 * (t1 + t2) / qs


def test_mie_code_against_the_spherical_bessel_series():
    from tools.chem_modal_optics.make_tables import mie_on_grid
    for m in (1.33 + 1e-8j, 1.55 + 0.0j, 1.52 + 0.01j, 1.95 + 0.79j, 0.6 + 0.5j, 2.5 + 0.1j):
        x = np.array([0.05, 0.3, 1.0, 3.0, 8.0, 20.0])
        qe, qs, gq = mie_on_grid(x, m)
        for i, xi in enumerate(x):
            if abs(m.imag) * xi > 15:
                continue        # the direct Bessel evaluation overflows
            be, bs, bg = _bessel_mie(xi, m)
            assert qe[i] == pytest.approx(be, rel=1e-8), (m, xi)
            assert qs[i] == pytest.approx(bs, rel=1e-8), (m, xi)
            assert gq[i] / qs[i] == pytest.approx(bg, abs=1e-8), (m, xi)


def _k_factor(m):
    return (m * m - 1) / (m * m + 2)


def test_table_rayleigh_and_geometric_limits():
    manifest = MO.host_data()[0]
    lnx0, lnx1 = manifest["ln_xg"][0], manifest["ln_xg"][1]
    for m in (1.45 + 0.01j, 1.52 + 0.001j, 1.95 + 0.79j):
        K = _k_factor(m)
        ea, es, _g = MO.table_lookup(0, lnx0, m)
        assert ea == pytest.approx(6 * math.pi * K.imag, rel=0.01)
        s = math.log(MO.SIGMA_G[0])
        assert es == pytest.approx(4 * math.pi * abs(K) ** 2 * math.exp(3 * lnx0 + 13.5 * s * s), rel=0.03)
        for si, sg in enumerate(MO.SIGMA_G):
            ea, es, g = MO.table_lookup(si, lnx1, m)
            s = math.log(sg)
            limit = 3 * math.pi * math.exp(-2.5 * s * s - lnx1)
            assert (ea + es) == pytest.approx(limit, rel=0.06), (m, sg)
            assert 0.5 < g < 1.0


def _direct(si, lnxg, m, dlnx=0.002):
    """E_abs, E_sca, g by direct lognormal integration (float64, a finer
    grid than the table's, efficiencies held beyond x = 3000 as the table)."""
    from tools.chem_modal_optics.make_tables import X_MIE_MAX, mie_on_grid
    s = math.log(MO.SIGMA_G[si])
    lnx = np.arange(lnxg - 5 * s, lnxg + 3 * s * s + 5 * s, dlnx)
    x = np.exp(lnx)
    inside = x <= X_MIE_MAX
    qe, qs, gq = mie_on_grid(x[inside], m)
    pad = (~inside).sum()
    qe, qs, gq = (np.concatenate([q, np.full(pad, q[-1])]) for q in (qe, qs, gq))
    w = np.exp(-0.5 * ((lnx - lnxg) / s) ** 2)
    vol = (w * x ** 3).sum()
    ext = 1.5 * math.pi * (w * qe * x ** 2).sum() / vol
    sca = 1.5 * math.pi * (w * qs * x ** 2).sum() / vol
    return ext - sca, sca, (w * gq * x ** 2).sum() / (w * qs * x ** 2).sum()


def test_table_interpolation_against_direct_integration():
    rng = np.random.default_rng(3)
    errs = []
    for _ in range(60):
        si = int(rng.integers(0, 3))
        # sizes the SORGAM modes reach at the four shortwave wavelengths and
        # in the longwave: x_g from 1e-3 to 30
        lnxg = rng.uniform(math.log(1e-3), math.log(30.0))
        m = complex(rng.uniform(1.1, 2.0), math.exp(rng.uniform(math.log(1e-5), math.log(1.0))))
        ea, es, g = _direct(si, lnxg, m)
        ta, ts, tg = MO.table_lookup(si, lnxg, m)
        e_ext = abs((ta + ts) - (ea + es)) / (ea + es)
        e_ssa = abs(ts / (ta + ts) - es / (ea + es))
        e_g = abs(tg - g)
        errs.append((e_ext, e_ssa, e_g))
    errs = np.array(errs)
    assert errs[:, 0].max() < EXT_TOL, errs[:, 0].max()
    assert errs[:, 1].max() < SSA_TOL, errs[:, 1].max()
    assert errs[:, 2].max() < G_TOL, errs[:, 2].max()


def test_mixing_rules():
    n = np.array([1.33, 1.52, 1.54, 1.45, 1.95, 1.47])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.79, 0.0])
    v = np.array([1.0, 2.0, 0.0, 1.0, 0.0, 0.0])
    assert MO.mixture_index(v, n, k, 2) == MO.mixture_index(v, n, k, 1)
    v_ec = v.copy()
    v_ec[4] = 0.2
    vol, mg = MO.mixture_index(v_ec, n, k, 1), MO.mixture_index(v_ec, n, k, 2)
    # Maxwell Garnett with absorbing inclusions absorbs less than the volume
    # average (Bohren and Huffman sec. 8.5; Chylek et al. 1988).
    assert 0 < mg.imag < vol.imag
    tiny = v.copy()
    tiny[4] = 1e-9
    host = MO.mixture_index(v, n, k, 1)
    assert abs(MO.mixture_index(tiny, n, k, 2) - host) < 1e-8


def test_sorgam_rows_are_mapped_once():
    from woof.core.chem_sorgam import AEROSOLS, AUX
    names = [r[0] for r in MO.ROWS]
    assert len(names) == len(set(names))
    assert set(names) | set(MO.NUMBERS) == set(AEROSOLS)
    assert {w for w, _m in MO.WATER} <= set(AUX)


def test_ammonium_sulfate_mass_extinction_is_plausible():
    """A dry (NH4)2SO4 accumulation mode, D_g 0.15 um: about 3 m2/g at 550-600
    nm (Seinfeld and Pandis 2006, fig. 15.7 range 2-5 m2/g)."""
    v = np.zeros((3, 6))
    mass = 10e-9                       # kg m-3, 10 ug m-3
    v[1, 1] = mass / 1800.0
    dg = 0.15e-6
    num = np.array([0.0, v[1, 1] / (math.pi / 6 * dg ** 3 * math.exp(4.5 * math.log(2.0) ** 2)), 0.0])
    out = MO.reference_cell(v, num, 1000.0)
    mee = out["tau"][2] / 1000.0 / (mass * 1e3)          # m2 per g at 600 nm
    assert 2.0 < mee < 5.0, mee
    assert out["ssa"][2] > 0.999
    assert 0.5 < out["g"][2] < 0.75
    assert out["tau"][0] > out["tau"][2] > out["tau"][3]   # Angstrom exponent > 0


def _random_cells(rng, ncell):
    vol = np.zeros((3, 6, ncell))
    num = np.zeros((3, ncell))
    for m, (dlo, dhi) in enumerate(((0.01, 0.06), (0.08, 0.6), (0.5, 5.0))):
        for c in range(6):
            vol[m, c] = 10 ** rng.uniform(-13, -9, ncell) * (rng.random(ncell) < 0.8)
        dg = 10 ** rng.uniform(math.log10(dlo), math.log10(dhi), ncell) * 1e-6
        vt = vol[m].sum(axis=0)
        num[m] = vt / (math.pi / 6 * dg ** 3 * math.exp(4.5 * math.log(MO.SIGMA_G[m]) ** 2))
    num[:, rng.random(ncell) < 0.05] = 0.0
    dz = rng.uniform(20, 800, ncell)
    return vol, num, dz


@requires_gpu
@pytest.mark.parametrize("mixing", (1, 2))
def test_kernel_matches_the_float64_twin(mixing):
    import cupy as cp
    rng = np.random.default_rng(11 + mixing)
    vol, num, dz = _random_cells(rng, 400)
    out = MO.launch_modal_optics(cp.asarray(vol.reshape(18, -1)), cp.asarray(num), cp.asarray(dz),
                                 mixing=mixing)
    host = {k: cp.asnumpy(v) for k, v in out.items()}
    for c in range(dz.size):
        ref = MO.reference_cell(vol[:, :, c], num[:, c], dz[c], mixing=mixing)
        np.testing.assert_allclose(host["tau"][:, c], ref["tau"], rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(host["tau_lw"][:, c], ref["tau_lw"], rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(host["ssa"][:, c], ref["ssa"], atol=2e-5)
        np.testing.assert_allclose(host["g"][:, c], ref["g"], atol=2e-5)
    assert (host["ssa"] >= 0).all() and (host["ssa"] <= 1).all()
    assert (np.abs(host["g"]) <= 1).all() and (host["tau"] >= 0).all()


def test_refusals():
    from types import SimpleNamespace
    assert MO.refusal(SimpleNamespace(aer_ra_feedback=0)) is None
    assert "aer_op_opt=3" in MO.refusal(SimpleNamespace(aer_ra_feedback=1, aer_op_opt=3))


@pytest.mark.gpu
@requires_gpu
def test_radm2sorg_chain_with_feedback_feeds_fast_tuv():
    """The process chain (tests/test_chem_radm2sorg_chain_gpu.py's) with
    aer_ra_feedback = 1: Fast-TUV now runs with the SORGAM optics where it
    used to refuse, the optics are physical, and only the photolysis (and
    what follows from it) moves."""
    import cupy as cp
    import test_chem_radm2sorg_chain_gpu as chain

    _t, off, _o = chain._run(2, feedback=0)
    table, on, _o = chain._run(2, feedback=1)
    assert np.isfinite(on["ph_no2"]).all() and (on["ph_no2"] > 0).all()
    ratio = on["ph_no2"] / off["ph_no2"]
    assert not np.array_equal(on["ph_no2"], off["ph_no2"])
    assert 0.7 < ratio.min() and ratio.max() < 1.3
    # The optics on the final state, through the same entry point.
    from types import SimpleNamespace
    from woof.core.chem_context import ChemClock, ChemContext
    met = {k: cp.asarray(v) for k, v in chain._column().items()}
    state = SimpleNamespace(chem=SimpleNamespace())
    for row in table.rows:
        setattr(state, row.state_attr, cp.asarray(on[row.name]))
    diag = {n: cp.asarray(on[n]) for n in ("sorgam_h2oai", "sorgam_h2oaj")}
    ctx = ChemContext(state, SimpleNamespace(aer_op_opt=1), table,
                      prep=SimpleNamespace(get=met.get), physics_fields=met, diag=diag,
                      clock=ChemClock(gmt=12.0, julday=172, curr_secs=120.0, ktau=3, dt=60.0))
    optics = {k: cp.asnumpy(v) for k, v in MO.sorgam_optics(ctx).items()}
    tau = optics["tauaer"]
    assert (tau >= 0).all() and tau[:, :4].sum() > 0
    aod600 = tau[2].sum(axis=0)
    assert 1e-4 < aod600.min() and aod600.max() < 2.0
    assert ((optics["waer"] > 0.8) & (optics["waer"] <= 1.0)).all()
    assert ((optics["gaer"] >= 0.0) & (optics["gaer"] < 0.9)).all()
    assert (optics["tauaerlw"] >= 0).all()


def _radm2sorg_config(**kwargs):
    from woof.config import RunConfig
    values = dict(nx=12, ny=10, nz=20, dx=3000., dy=3000., ztop=20000., dt=18.,
                  run_seconds=3600., moist=True, chem_sets="radm2sorg", bl_pbl_physics=1,
                  sf_sfclay_physics=1, sf_surface_physics=2, ra_lw_physics=4,
                  ra_sw_physics=4, aer_ra_feedback=1, aer_op_opt=1,
                  ra_rrtmg_variant="rrtmg_legacy")
    values.update(kwargs)
    return RunConfig(**values)


def test_the_door_admits_sorgam_feedback_on_legacy_rrtmg_only():
    from woof.config import validate_chem_config
    from woof.core.chem_ftuv import refusal as ftuv_refusal
    for op in (1, 2):
        cfg = _radm2sorg_config(aer_op_opt=op)
        validate_chem_config(cfg)
        assert ftuv_refusal(cfg) is None
    with pytest.raises(ValueError, match="legacy RRTMG"):
        validate_chem_config(_radm2sorg_config(ra_rrtmg_variant="rte-rrtmgp"))
    with pytest.raises(ValueError, match="aer_op_opt=3"):
        validate_chem_config(_radm2sorg_config(aer_op_opt=3))
    # RADM2 alone has no aerosol to hand Fast-TUV or radiation.
    with pytest.raises(ValueError, match="no SORGAM aerosol"):
        validate_chem_config(_radm2sorg_config(chem_sets="radm2"))


def test_sorgam_feedback_memory_is_priced():
    from woof.core.chem_radiation import feedback_row_count, memory_shapes
    cfg = _radm2sorg_config()
    held, work = memory_shapes(cfg, 5000., 64)
    assert held["chem_modal_optics/bands"] == (28, cfg.nz, cfg.ny, cfg.nx)
    assert held["chem_modal_optics/table"][0] == MO.host_data()[4].size
    assert "chem_modal_optics/volumes" in work
    assert feedback_row_count(cfg) == len(MO.ROWS)
    assert memory_shapes(_radm2sorg_config(aer_ra_feedback=0), 5000., 64) == ({}, {})


@pytest.mark.gpu
@requires_gpu
def test_sorgam_feedback_reaches_legacy_rrtmg_fluxes():
    """The live legacy RRTMG with the SORGAM optics: a polluted accumulation
    mode dims the surface shortwave and changes the longwave, finite."""
    from datetime import datetime
    import cupy as cp
    from test_chem_transport import _bubble_case
    from woof.core.physics import _prepare_atmosphere
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
    cfg, state = _bubble_case(chem_sets="radm2sorg", aer_ra_feedback=1, aer_op_opt=1,
                              ra_rrtmg_variant="rrtmg_legacy", ra_lw_physics=4,
                              ra_sw_physics=4, mp_physics=1, icloud=1)
    shape = state.qv.shape[1:]
    fields = {name: cp.full(shape, value, cp.float32) for name, value in
              (("tsk", 290.), ("albedo", .15), ("emiss", .96),
               ("xland", 1.), ("xice", 0.), ("snow", 0.))}
    lat, lon = (cp.full(shape, value, cp.float32) for value in (35., -5.))
    rad = RRTMGLegacyRadiation(datetime(2022, 3, 15, 12), lat, lon, column_chunk=97,
                               p_top=float(state.p_top))
    atmosphere = _prepare_atmosphere(state)
    clean = rad(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    table = state.chem.table
    # 40 ug/kg of accumulation-mode sulfate, 4 of elemental carbon, with the
    # number of a D_g = 0.15 um mode.
    getattr(state, table.row("so4aj").state_attr).fill(40.)
    getattr(state, table.row("ecj").state_attr).fill(4.)
    v = 40e-9 / 1800. + 4e-9 / 2200.
    getattr(state, table.row("ac0").state_attr).fill(
        v / (math.pi / 6 * 0.15e-6 ** 3 * math.exp(4.5 * math.log(2.0) ** 2)))
    # Same model time: drop both caches (radiation's and the shared optics').
    state.chem.radiation_aerosols_time = None
    state.chem.modal_optics_time = None
    polluted = rad(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    assert bool(cp.all(cp.isfinite(polluted.swdown)))
    assert bool(cp.all(cp.isfinite(polluted.rthratensw)))
    dimming = float((clean.swdown - polluted.swdown).mean())
    # AOD ~1.5 at 600 nm from this heavy whole-column loading.
    assert 1. < dimming < 600.
    assert float(cp.abs(clean.glw - polluted.glw).mean()) > 1e-4
    print("SORGAM legacy RRTMG surface dimming W m-2:", dimming)
