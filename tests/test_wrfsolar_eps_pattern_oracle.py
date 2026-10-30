"""The multi_perturb pattern generator against WRF v4.7.1's module_stoch.F.

tests/data/oracles/wrfsolar_eps/stoch_* is tools/wrfsolar_eps_wrf471_oracle/
build_stoch.sh's record: SETUP_RAND_PERTURB('R'), UPDATE_STOCH's AR(1)
step, RAND_PERT_UPDATE's fill, vertical structure, FFTPACK5 backward
transforms and clip, unmodified, gfortran -O0 -ffp-contract=off on glibc
2.43 / x86-64, seven fields over three updates each with supplied
Gaussian draws (make_stoch_inputs.py).

Graded word for word: SP_AMP and ALPH, the coefficients after every AR(1)
step, RAND_REAL/RAND_IMAG after the vertical structure, and the clip of
WRF's own transformed field.  The backward FFT is cuFFT or pocketfft, not
FFTPACK5: graded to a tolerance of the field's scale.  The draws
themselves are woof's Philox (woof/core/wrfsolar_eps.py), checked here for
the distribution WRF's gauss_noise loop draws from.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core import wrfsolar_eps as eps

DATA = Path(__file__).parent / "data" / "oracles" / "wrfsolar_eps"


def _read():
    receipt = (DATA / "stoch_receipt.txt").read_text()
    for name in ("stoch_inputs.bin", "stoch.out.bin"):
        digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert f"{digest}  {name}" in receipt, name
    raw = np.fromfile(DATA / "stoch_inputs.bin", dtype=np.uint8)
    out = np.fromfile(DATA / "stoch.out.bin", dtype="<f4")
    (ncase,) = raw[:4].view("<i4")
    pos, opos = 4, 0
    cases = []
    for _ in range(ncase):
        lmax, kmax, nz, itime_step, vert, nstep = (int(v) for v in raw[pos:pos + 24].view("<i4"))
        pos += 24
        dx, dy, std, length, tau, cutoff = (np.float32(v) for v in raw[pos:pos + 24].view("<f4"))
        pos += 24
        plane = lmax * kmax
        draws = []
        for _s in range(nstep):
            a = raw[pos:pos + 8 * plane].view("<f4").reshape(2, kmax, lmax)
            pos += 8 * plane
            draws.append((a[0], a[1]))
        alph = out[opos]
        sp_amp = out[opos + 1:opos + 1 + plane].reshape(kmax, lmax)
        opos += 1 + plane
        steps = []
        for _s in range(nstep):
            take = lambda n: out[opos:opos + n]
            c = out[opos:opos + plane].reshape(kmax, lmax); opos += plane
            s = out[opos:opos + plane].reshape(kmax, lmax); opos += plane
            lev = lambda: out[opos:opos + nz * plane].reshape(nz, kmax, lmax)
            rr = lev(); opos += nz * plane
            ri = lev(); opos += nz * plane
            fft = lev(); opos += nz * plane
            pert = lev(); opos += nz * plane
            steps.append(dict(c=c, s=s, rr=rr, ri=ri, fft=fft, pert=pert))
        cases.append(SimpleNamespace(
            lmax=lmax, kmax=kmax, nz=nz, itime_step=itime_step, draws=draws,
            alph=alph, sp_amp=sp_amp, steps=steps,
            row=eps.StochpertRow("QVAPOR", float(std), float(length), float(tau),
                                 float(cutoff), 21, vert),
            dx=dx, dy=dy))
    assert pos == raw.size and opos == out.size
    return cases


@pytest.fixture(scope="module")
def cases():
    return _read()


def _words(got, want, what):
    got = np.asarray(cp.asnumpy(got), dtype=np.float32).ravel()
    want = np.asarray(want, dtype=np.float32).ravel()
    assert got.size == want.size, what
    bad = np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))
    assert bad.size == 0, (what, bad.size, bad[:5], got[bad[:5]], want[bad[:5]])
    return want.size


def _generator(c):
    return eps.FieldGenerator("QVAPOR", c.row, lmax=c.lmax, kmax=c.kmax, dx=c.dx,
                              dy=c.dy, itime_step=c.itime_step, nens=1, nz=c.nz)


@pytest.mark.parametrize("exact", ["0", "1"])
def test_pattern_arithmetic_every_word(cases, monkeypatch, exact):
    monkeypatch.setenv("GPUWM_WRF_EXACT", exact)
    words = 0
    worst = 0.0
    for i, c in enumerate(cases):
        gen = _generator(c)
        assert np.float32(gen.alph).view(np.uint32) == np.float32(c.alph).view(np.uint32), i
        words += 1 + _words(gen.sp_amp, c.sp_amp, f"case {i} SP_AMP")
        spc = cp.zeros((c.kmax, c.lmax), dtype=cp.float32)
        sps = cp.zeros_like(spc)
        for s, (step, (noss, nosc)) in enumerate(zip(c.steps, c.draws)):
            gen.ar1(spc, sps, cp.asarray(noss), cp.asarray(nosc))
            words += _words(spc, step["c"], f"case {i} step {s} SPFORCC")
            words += _words(sps, step["s"], f"case {i} step {s} SPFORCS")
            real, imag = gen.fill(spc, sps)
            words += _words(real, step["rr"][:gen.nlev], f"case {i} step {s} RAND_REAL")
            words += _words(imag, step["ri"][:gen.nlev], f"case {i} step {s} RAND_IMAG")
            if gen.nlev == 1:
                # vertstruc = 0: every WRF level holds the same words
                assert np.array_equal(step["rr"], np.broadcast_to(step["rr"][:1], step["rr"].shape))
            # the FFT: cuFFT/pocketfft against FFTPACK5
            got = cp.asnumpy(eps.backward_fft2(real, imag))
            scale = float(np.abs(step["fft"]).max())
            err = float(np.abs(got - step["fft"][:gen.nlev]).max()) / scale
            worst = max(worst, err)
            assert err < 2e-6, (i, s, err)
            # the clip of WRF's own transformed field, on the mass points
            pert = cp.empty((gen.nlev, c.kmax - 1, c.lmax - 1), dtype=cp.float32)
            gen.clip(cp.asarray(np.ascontiguousarray(step["fft"][:gen.nlev])), pert)
            words += _words(pert, step["pert"][:gen.nlev, :-1, :-1], f"case {i} step {s} RAND_PERT")
            if gen.nlev == 1:
                # vertstruc = 0: WRF's levels are copies of level 1
                assert np.array_equal(step["pert"], np.broadcast_to(step["pert"][:1], step["pert"].shape))
    assert words > 300_000
    assert worst > 0.0                      # FFTPACK5 and cuFFT/pocketfft do differ


def test_the_record_covers_both_structures_and_cadences(cases):
    assert {c.row.vertstruc for c in cases} == {0, 1}
    assert len({(c.lmax % 2, c.kmax % 2) for c in cases}) >= 3
    assert len({c.itime_step for c in cases}) >= 5


def test_philox_draws_are_wrf_shaped():
    """UPDATE_STOCH draws standard normals truncated at |z| < 3; the
    Philox draws keep that law and depend only on (seed, step, stream)."""
    row = eps.StochpertRow("QVAPOR", 0.05, 100000.0, 3600.0, 3.0, 21, 1)
    gen = eps.FieldGenerator("QVAPOR", row, lmax=161, kmax=121, dx=3000.0, dy=3000.0,
                             itime_step=18, nens=1, nz=4)
    a, b = (cp.asnumpy(v) for v in gen.draws(7))
    a2, _ = (cp.asnumpy(v) for v in gen.draws(7))
    c, _ = (cp.asnumpy(v) for v in gen.draws(8))
    assert np.array_equal(a.view(np.uint32), a2.view(np.uint32))
    assert not np.array_equal(a, c) and not np.array_equal(a, b)
    z = np.concatenate([a.ravel(), b.ravel()])
    assert np.abs(z).max() < 3.0
    assert abs(z.mean()) < 0.02 and abs(z.std() - 0.9866) < 0.02   # sd of N(0,1) | |z|<3
    other = eps.FieldGenerator("QVAPOR", row, lmax=161, kmax=121, dx=3000.0, dy=3000.0,
                               itime_step=18, nens=2, nz=4)
    d, _ = (cp.asnumpy(v) for v in other.draws(7))
    assert not np.array_equal(a, d)          # nens moves the stream, as in WRF
