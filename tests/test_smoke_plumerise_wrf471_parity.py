"""Freitas plume rise: the CPU reference and the device port against WRF.

The fixture is WRF v4.7.1's own chem/module_chem_plumerise_scalar.F,
byte-unmodified and pinned by sha256, compiled at -O0 by
``tools/smoke_wrf471_oracle/build.sh`` and run by ``run_plumerise.F90`` on
eight synthetic soundings (every biome, the grassland shortcut, dry and
moist columns with condensation, rain, ice and melting, calm and sheared
wind, weak and strong stability, a fire too small to lift its plume, a
column holding several biomes), one fresh process per column.  Compared
word for word (max ULP 0), per case:

* ``eburn_out`` -- plumerise's output, the injection profile;
* ``ztopmax``, ``k1``, ``k2`` -- the plume tops and injection layers it
  keeps local (from a second fresh process that repeats its calls);
* the plume model's state as every makeplume returns -- w, t, qv, qc, qh,
  qi, rho, radius, vel_p, rad_p, wc, qsat on all 200 plume levels and the
  scalars time, dt, wbar, dqsdz, mintime.

The CPU reference (``plumerise_np``) is Python-speed; the two deep moist
cases (about 20 s and 35 s) are ``slow``.  The device runs all eight.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.smoke import plume_rise as pr
from woof.verify.smoke_oracle import (assert_words_equal, case_names,
                                      load_case)

SLOW = {"case02", "case07"}
CASES = case_names("plumerise")
FAST = [c for c in CASES if c not in SLOW]


def _case(name):
    return load_case(f"plumerise/{name}"), load_case(f"plumerise/{name}/tops")


def _column(c):
    m1 = c["u"].size

    def pad(a):
        out = np.zeros(pr.NKP + 3, dtype=np.float32)
        out[1:m1 + 1] = a
        return out

    return {"m1": m1, "ucon": pad(c["u"]), "vcon": pad(c["v"]),
            "thtcon": pad(c["theta"]), "picon": pad(c["pi"]),
            "rvcon": pad(c["qv"]), "zcon": pad(c["zt"]),
            "zzcon": pad(c["zm"]), "firesize": c["firesize"],
            "mean_fct": c["mean_fct"], "eburn_in": np.float32(c["eburn_in"])}


def _domain(cases):
    """The cases side by side as one (m1, ncase) domain."""
    def stack(key):
        return np.stack([c[key] for c in cases], axis=-1)
    fields = [stack(k) for k in ("u", "v", "theta", "pi", "qv", "zt", "zm",
                                 "firesize", "mean_fct")]
    ein = np.array([c["eburn_in"] for c in cases], dtype=np.float32)
    return fields + [ein]


def _assert_column(label, out, tops, k1, k2, trace, c, t):
    assert_words_equal(out, c["eburn_out"], f"{label} eburn_out")
    assert_words_equal(tops, np.ascontiguousarray(t["ztopmax"].T),
                       f"{label} ztopmax")
    assert np.array_equal(k1, t["k1"]) and np.array_equal(k2, t["k2"]), label
    for j, name in enumerate(pr.STATE_FIELDS):
        assert_words_equal(trace["state"][:, j, :], t["state"][:, j, :],
                           f"{label} {name}")
    for j, name in enumerate(pr.STATE_SCALARS):
        assert_words_equal(trace["scalars"][j], t["scalars"][j],
                           f"{label} {name}")
    assert np.array_equal(trace["mintime"], t["mintime"]), label


def test_every_case_is_published():
    assert CASES == [f"case{i:02d}" for i in range(1, 9)]


def test_literals_and_folded_constants_are_gfortrans_words():
    """Each real(4) literal and folded constant the port spells is the word
    gfortran made of it (a decimal parsed through binary64 can round twice)."""
    words = load_case("plumerise_literals")
    ours = np.array([pr._c(text) for text in pr.LITERALS], dtype=np.float32)
    assert_words_equal(ours, words["literals"], "literals")
    assert_words_equal(np.array(pr.folded_constants(), dtype=np.float32),
                       words["folded"], "folded constants")


@pytest.mark.parametrize("name", [pytest.param(
    n, marks=pytest.mark.slow) if n in SLOW else n for n in CASES])
def test_cpu_reference_is_wrfs_word_for_word(name):
    c, t = _case(name)
    trace = {}
    out, tops, k1, k2 = pr._plumerise_column(_column(c), pr.HEAT_FLUX, trace)
    _assert_column(f"cpu {name}", out, tops, k1, k2, trace, c, t)


def test_cpu_domain_entry_matches_the_column_runs():
    """plumerise_np over a domain: burning columns are the column runs,
    skipped columns keep only their smouldering emission."""
    cases = [_case(n)[0] for n in ("case03", "case06")]
    fields = _domain(cases)
    out = pr.plumerise_np(*fields)
    for i, c in enumerate(cases):
        assert_words_equal(out.eburn_out[:, i], c["eburn_out"], f"col {i}")
    # A column with no flaming fraction, one with no fire size and one with
    # no smouldering emission are skipped as module_plumerise1.F skips them.
    fields[8] = np.zeros_like(fields[8])
    skipped = pr.plumerise_np(*fields)
    assert np.array_equal(skipped.eburn_out[1:], 0 * skipped.eburn_out[1:])
    assert_words_equal(skipped.eburn_out[0], fields[9], "smouldering")
    assert not skipped.ztopmax.any() and not skipped.k1.any()


def test_bad_inputs_are_refused():
    c, _ = _case("case06")
    fields = _domain([c])
    with pytest.raises(ValueError, match="heat_flux"):
        pr.plumerise_np(*fields, heat_flux=np.ones((4, 2)))
    with pytest.raises(ValueError, match="firesize"):
        pr.plumerise_np(*(fields[:7] + [fields[7][:3]] + fields[8:]))
    bad = list(fields)
    bad[1] = bad[1][:-1]
    with pytest.raises(ValueError, match="v has shape"):
        pr.plumerise_np(*bad)


# --------------------------------------------------------------------------
# device


@pytest.mark.gpu
@requires_gpu
def test_gpu_is_wrfs_word_for_word():
    """All eight cases in one launch, every output and the whole plume
    state against the oracle."""
    import cupy as cp

    pairs = [_case(n) for n in CASES]
    trace = {}
    r = pr.plumerise(*(cp.asarray(a) for a in _domain([c for c, _ in pairs])),
                     trace=trace)
    host = {k: v.get() for k, v in trace.items()}
    for i, (name, (c, t)) in enumerate(zip(CASES, pairs)):
        col_trace = {k: v[..., i] for k, v in host.items()}
        _assert_column(f"gpu {name}", r.eburn_out[:, i].get(),
                       r.ztopmax[:, :, i].get(), r.k1[:, i].get(),
                       r.k2[:, i].get(), col_trace, c, t)


@pytest.mark.gpu
@requires_gpu
def test_gpu_equals_cpu_on_a_mixed_domain_and_another_heat_table():
    """GPU and CPU word for word on a (m1, 2, 3) domain with burning,
    skipped and multi-biome columns, under WRF's table and a 50 kW m-2
    forest table, and the domain-wide outputs' shapes."""
    import cupy as cp

    base = [_case(n)[0] for n in ("case01", "case03", "case04", "case06",
                                  "case08", "case06")]
    fields = _domain(base)
    fields[9][3] = 0.0                    # no smouldering emission: skipped
    fields[8][:, 5] = 0.0                 # no flaming fraction: skipped
    shaped = [a.reshape(a.shape[:-1] + (2, 3)) for a in fields]
    table = pr.HEAT_FLUX.copy()
    table[:, 0] = 50.0
    for heat in (None, table):
        cpu = pr.plumerise_np(*shaped, heat_flux=heat)
        gpu = pr.plumerise(*(cp.asarray(a) for a in shaped), heat_flux=heat)
        for name in pr.PlumeriseResult._fields:
            got, want = getattr(gpu, name).get(), getattr(cpu, name)
            assert got.shape == want.shape, name
            assert_words_equal(got, want, f"gpu vs cpu {name}")
    assert cpu.eburn_out.shape == (shaped[0].shape[0], 2, 3)


@pytest.mark.gpu
@requires_gpu
def test_gpu_with_nothing_burning_returns_the_smouldering_profile():
    import cupy as cp

    fields = [cp.asarray(a) for a in _domain([_case("case03")[0]])]
    fields[9] = cp.zeros_like(fields[9])
    r = pr.plumerise(*fields)
    assert not bool(r.eburn_out.any()) and not bool(r.k2.any())
