"""woof.core.host_libm is glibc 2.39's words, element by element, with
NumPy's domain answers.

Its callers (the projection transcriptions, beside the static-fields crate
that computes the same values in Rust) need the words of the C library the
WRF references link, glibc 2.39, on every machine; NumPy 2.5's AVX-512
loops, macOS's libm and other glibc releases round differently.  These
tests hold each function to glibc 2.39's recorded words, to the scalar
entries of the same transcription and, on a glibc host, to :mod:`math`;
to NumPy's NaN and infinity answers outside the domain, and to the input's
shape.
"""

from __future__ import annotations

import hashlib
import math
import platform

import numpy as np
import pytest

from woof.core import host_libm

#: The transcription, not the C library, is what gives glibc 2.39's words
#: on a host whose libm is not glibc's (macOS, Windows) or not 2.39's.
REQUIRES_TRANSCRIPTION = pytest.mark.skipif(
    host_libm.implementation() != host_libm.GLIBC239,
    reason="the installed CPU bridge does not carry the glibc 2.39 "
           "transcription (build tools/grib1_bridge)")
#: math.* is glibc 2.39's word on these arguments only on a glibc host
#: (glibc 2.28 through 2.44 agree with 2.39 away from huge sin/cos/tan
#: arguments, which _CASES does not use).
_GLIBC_HOST = platform.libc_ver()[0] == "glibc"

_RNG = np.random.default_rng(29)
_ANGLES = _RNG.uniform(-1.5, 1.5, 2000)
_POSITIVE = _RNG.uniform(1e-6, 1e4, 2000)
_UNIT = _RNG.uniform(-1.0, 1.0, 2000)

_CASES = (
    (host_libm.exp, math.exp, _RNG.uniform(-700.0, 700.0, 2000)),
    (host_libm.log, math.log, _POSITIVE),
    (host_libm.log10, math.log10, _POSITIVE),
    (host_libm.sin, math.sin, _ANGLES * 4.0),
    (host_libm.cos, math.cos, _ANGLES * 4.0),
    (host_libm.tan, math.tan, _ANGLES),
    (host_libm.arctan, math.atan, _RNG.uniform(-1e3, 1e3, 2000)),
    (host_libm.arcsin, math.asin, _UNIT),
    (host_libm.arccos, math.acos, _UNIT),
)


@pytest.mark.skipif(not _GLIBC_HOST, reason="math.* is glibc's words only on a glibc host")
@pytest.mark.parametrize("ours,libm,values", _CASES,
                         ids=[c[0].__name__ for c in _CASES])
def test_each_function_is_the_glibc_c_library_element_by_element(ours, libm, values):
    got = ours(values.reshape(40, 50))
    assert got.shape == (40, 50) and got.dtype == np.float64
    want = np.array([libm(float(v)) for v in values]).reshape(40, 50)
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))


def _word_arguments(name: str) -> np.ndarray:
    """4096 arguments per function: the usual ranges, every binade, and for
    sin, cos and tan the branred range, where glibc 2.44 differs from 2.39.
    pow and atan2 take interleaved (first, second) pairs."""
    rng = np.random.default_rng(2039)
    def binades(lo, hi, n):
        a, b = np.float64(lo).view(np.uint64), np.float64(hi).view(np.uint64)
        return rng.integers(a, b, n, dtype=np.uint64, endpoint=True).view(np.float64)
    def signed(x):
        return x * rng.choice([-1.0, 1.0], x.size)
    if name == "exp":
        return np.concatenate([rng.uniform(-745.0, 709.0, 2048), signed(binades(1e-300, 700.0, 2048))])
    if name in ("log", "log10"):
        return np.concatenate([binades(5e-324, 1.7e308, 2048), rng.uniform(0.5, 2.0, 2048)])
    if name in ("sin", "cos", "tan"):
        return np.concatenate([rng.uniform(-7.0, 7.0, 1024), signed(binades(1e-300, 1e5, 1024)),
                               signed(binades(1.1e8, 1e300, 2048))])
    if name == "atan":
        return np.concatenate([rng.uniform(-30.0, 30.0, 2048), signed(binades(1e-300, 1e300, 2048))])
    if name in ("asin", "acos"):
        return np.concatenate([rng.uniform(-1.0, 1.0, 2048), signed(binades(1e-300, 1.0, 2048))])
    if name == "pow":
        base = np.concatenate([rng.uniform(0.0, 4.0, 2048), binades(1e-300, 1e300, 2048)])
        exponent = np.concatenate([rng.uniform(-10.0, 10.0, 2048), signed(binades(1e-6, 1.0, 2048))])
        return np.stack([base, exponent], 1).reshape(-1)
    y = np.concatenate([rng.uniform(-3.0, 3.0, 2048), signed(binades(1e-300, 1e300, 2048))])
    x = np.concatenate([rng.uniform(-3.0, 3.0, 2048), signed(binades(1e-300, 1e300, 2048))])
    return np.stack([y, x], 1).reshape(-1)


#: SHA-256 of the output words glibc 2.39 itself gives for _word_arguments:
#: Ubuntu 24.04's libc6 2.39-0ubuntu8.9 on x86-64 with FMA (the variant the
#: references ran), recorded in docker ubuntu:24.04 with the C library's
#: own functions (tools/glibc239_libm_proof).  The functions are the C
#: functions, without host_libm's domain handling, which is tested below.
_GLIBC239_SHA256 = {
    "exp": "d5667ed6c6ff37e4fabada8fe6fb1b9e03f8e32e9c482ddd2c75056290dab068",
    "log": "c03870b804f9285efa0cfcc5944d3fde55d52923d4ea2d9270665b23f325c0f0",
    "log10": "b35557f36b2fa3d303d669fc6fd5fd2e88e39c664c3c5455cd8ab03603337eba",
    "sin": "2c4cd0e6e121c73f988806ee282221c54ffe4c25e44ec829d38f937b6ab5b065",
    "cos": "2ec0b66ff78e23bfc8942de417da2e44ae46d551fb0a011c2397800e062a4456",
    "tan": "aba4782c0f894664a1d3d69fc1172afbbb3189352855e0008a66991001431972",
    "atan": "eeeac2a591dc887a91549f2541fa9d34ebcc4ce0afb5b4536395c5c5a13e298d",
    "asin": "6ea574f5a86aa2bcdeb13029578fa68a7eb7d4fe40a79e1e04f28c4a219da046",
    "acos": "9e83782d7bee6e3bb07d96762ab14fa8f596e75cb5cdca7f11fc104f331561ac",
    "pow": "805c247eafd80274d9857c0a31ebfad89116f77d056a9d4329ae696b65b0760f",
    "atan2": "ba5d0f8afed22be229c2483bec46c3dd30458ca611f9ef3012f5820cd315dd64",
}

@REQUIRES_TRANSCRIPTION
@pytest.mark.parametrize("name", sorted(_GLIBC239_SHA256))
def test_the_transcription_is_glibc_2_39s_words(name):
    function = host_libm.scalar(name)
    values = _word_arguments(name)
    if name in ("pow", "atan2"):
        words = np.array([function(float(a), float(b)) for a, b in values.reshape(-1, 2)])
    else:
        words = np.array([function(float(v)) for v in values])
    assert hashlib.sha256(words.tobytes()).hexdigest() == _GLIBC239_SHA256[name]


@REQUIRES_TRANSCRIPTION
def test_huge_arguments_get_2_39s_words_where_later_glibc_does_not():
    """glibc 2.44 reduces these differently and returns the word one below
    or above; macOS's libm differs on many more.  2.39's words, measured in
    tools/glibc239_libm_proof/verification.log."""
    pins = (
        (host_libm.sin, 0x440d614d83c29e1e, 0x3fb79580dbb00036),
        (host_libm.sin, 0x6c644e503274dc17, 0x3fae5fe0e5309805),
        (host_libm.cos, 0x6cad82c6d91fdea8, 0x3fbea09312471b6c),
        (host_libm.cos, 0xe6926d3be9db3170, 0x3fbf86dcd6d51676),
        (host_libm.tan, 0xdf2ed6efde49e977, 0xc0193c938344cc4f),
        (host_libm.tan, 0x443620d5c530dedb, 0x3ffd8c768168e053),
    )
    for function, argument, word in pins:
        got = function(np.array([argument], np.uint64).view(np.float64))
        assert int(got.view(np.uint64)[0]) == word, (function.__name__, hex(argument))


def test_power_is_pow_element_for_element_and_broadcasts():
    pow_word = host_libm.scalar("pow")  # the transcription's, or the C library's
    base = _POSITIVE.reshape(40, 50)
    got = host_libm.power(base, 0.7159)
    want = np.array([pow_word(float(v), 0.7159) for v in _POSITIVE])
    assert np.array_equal(got.reshape(-1).view(np.uint64),
                          want.view(np.uint64))
    exponents = _RNG.uniform(-3.0, 3.0, 50)
    got = host_libm.power(base, exponents)
    assert got.shape == (40, 50)
    assert got[3, 7] == pow_word(float(base[3, 7]), float(exponents[7]))


def test_arctan2_is_atan2_element_for_element_and_broadcasts():
    """The Lambert inverse's longitude takes ``arctan2``, whose NumPy 2.5
    AVX-512 float64 loop differs from the C library's atan2 on about 8% of
    random argument pairs, so it is held to :func:`math.atan2` here, on
    ordinary pairs, on Lambert-inverse-shaped offsets and on the signed
    zeros, infinities and NaNs."""
    atan2_word = host_libm.scalar("atan2")  # the transcription's, or the C library's
    y = _RNG.uniform(-5.0, 5.0, 2000).reshape(40, 50)
    x = _RNG.uniform(-3.0, 3.0, 2000).reshape(40, 50)
    got = host_libm.arctan2(y, x)
    assert got.shape == (40, 50) and got.dtype == np.float64
    want = np.array([atan2_word(float(a), float(b))
                     for a, b in zip(y.reshape(-1), x.reshape(-1))])
    assert np.array_equal(got.reshape(-1).view(np.uint64),
                          want.view(np.uint64))
    offsets = _RNG.uniform(-600.0, 600.0, 2000)
    got = host_libm.arctan2(offsets, 2400.0)
    want = np.array([atan2_word(float(a), 2400.0) for a in offsets])
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))
    specials = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan])
    a, b = np.meshgrid(specials, specials)
    ours = host_libm.arctan2(a, b).reshape(-1)
    theirs = np.array([math.atan2(float(u), float(v))
                       for u, v in zip(a.reshape(-1), b.reshape(-1))])
    number = ~np.isnan(theirs)
    assert np.array_equal(np.isnan(ours), ~number)
    assert np.array_equal(ours[number].view(np.uint64),
                          theirs[number].view(np.uint64))
    assert host_libm.arctan2(1.0, 2.0).shape == ()


def test_a_scalar_argument_gives_a_zero_dimensional_answer():
    assert host_libm.tan(0.3).shape == ()
    assert float(host_libm.tan(0.3)) == host_libm.scalar("tan")(0.3)
    assert host_libm.power(2.0, 0.5).shape == ()


def test_outside_the_domain_the_answers_are_numpys():
    with np.errstate(all="ignore"):
        cases = (
            (host_libm.log, np.log, [0.0, -0.0, -1.0, np.inf, np.nan]),
            (host_libm.log10, np.log10, [0.0, -1.0, np.inf, np.nan]),
            (host_libm.exp, np.exp, [1000.0, -1000.0, np.inf, -np.inf,
                                     np.nan]),
            (host_libm.tan, np.tan, [np.inf, -np.inf, np.nan]),
            (host_libm.arcsin, np.arcsin, [1.5, -1.5, np.inf, np.nan]),
            (host_libm.arccos, np.arccos, [1.5, -1.5, -np.inf, np.nan]),
            (host_libm.arctan, np.arctan, [np.inf, -np.inf, np.nan]),
        )
        for ours, numpy_function, values in cases:
            values = np.array(values)
            np.testing.assert_array_equal(ours(values), numpy_function(values))
        np.testing.assert_array_equal(
            host_libm.power(np.array([0.0, -2.0, 1e300]),
                            np.array([-1.0, 0.5, 2.5])),
            np.power(np.array([0.0, -2.0, 1e300]), np.array([-1.0, 0.5, 2.5])))


@pytest.mark.parametrize("workers", [1, 8, 32])
def test_native_host_math_preserves_scalar_bits_and_special_values(workers):
    """The native entries against the Python fallback (math, with the same
    domain handling), special values and NaN payloads included.  That
    fallback is glibc 2.39's words only on a glibc host; elsewhere the
    transcription is held to its own scalar entries below."""
    from woof.core import portable_math as pm
    library = pm._load()
    if not hasattr(library, "gpuwm_host_binary_f64"):
        pytest.skip("CPU bridge predates the separate host-libm ABI")
    if host_libm.implementation() == host_libm.GLIBC239 and not _GLIBC_HOST:
        pytest.skip("the Python fallback is this host's libm, not glibc 2.39")
    rng = np.random.default_rng(719)
    values = rng.uniform(-700, 700, 65_539)
    specials = np.array([
        0, 0x8000000000000000, 1, 0x8000000000000001,
        0x7ff0000000000000, 0xfff0000000000000,
        0x7ff8000000000042, 0xfff8000000000019,
        0x7ff0000000000031, 0xfff0000000000027,
    ], dtype=np.uint64).view(np.float64)
    values[:len(specials)] = specials
    references = (
        (host_libm.exp, host_libm._exp), (host_libm.log, host_libm._LOG),
        (host_libm.log10, host_libm._LOG10), (host_libm.tan, host_libm._tan),
        (host_libm.sin, host_libm._SIN), (host_libm.cos, host_libm._COS),
        (host_libm.arctan, math.atan), (host_libm.arcsin, host_libm._ASIN),
        (host_libm.arccos, host_libm._ACOS),
    )
    with pm.worker_limit(workers):
        for function, reference in references:
            expected = np.array([reference(float(value)) for value in values])
            assert function(values).tobytes() == expected.tobytes(), function.__name__
        bases = rng.integers(0, np.iinfo(np.uint64).max, 65_539, dtype=np.uint64).view(np.float64)
        exponents = rng.uniform(-4, 4, bases.size)
        bases[:len(specials)] = specials
        for power in (-3.0, -0.5, 0.0, 0.5, 2.0, 3.0, np.inf, -np.inf, np.nan):
            exponents[:len(specials)] = power
            expected = np.array([host_libm._pow(float(x), float(y))
                                 for x, y in zip(bases, exponents)])
            assert host_libm.power(bases, exponents).tobytes() == expected.tobytes()
        for y, x in ((values, 0.0), (values, values[::-1])):
            a, b = np.broadcast_arrays(y, x)
            expected = np.array([math.atan2(float(u), float(v))
                                 for u, v in zip(a.flat, b.flat)])
            assert host_libm.arctan2(y, x).tobytes() == expected.tobytes()


@REQUIRES_TRANSCRIPTION
@pytest.mark.parametrize("workers", [1, 8, 32])
def test_the_array_entries_are_the_scalar_transcription(workers):
    """On every host: the bridge's threaded array entries and its scalar
    entries are one implementation, word for word, wherever no domain rule
    applies (the rules are graded against NumPy above)."""
    from woof.core import portable_math as pm
    rng = np.random.default_rng(311)
    cases = (
        (host_libm.exp, "exp", rng.uniform(-700, 700, 65_539)),
        (host_libm.log, "log", rng.uniform(1e-300, 1e300, 65_539)),
        (host_libm.log10, "log10", rng.uniform(1e-6, 1e6, 65_539)),
        (host_libm.sin, "sin", rng.uniform(-1e9, 1e9, 65_539)),
        (host_libm.cos, "cos", rng.uniform(-10, 10, 65_539)),
        (host_libm.tan, "tan", rng.uniform(-10, 10, 65_539)),
        (host_libm.arctan, "atan", rng.uniform(-1e3, 1e3, 65_539)),
        (host_libm.arcsin, "asin", rng.uniform(-1, 1, 65_539)),
        (host_libm.arccos, "acos", rng.uniform(-1, 1, 65_539)),
    )
    with pm.worker_limit(workers):
        for function, name, values in cases:
            scalar = host_libm.scalar(name)
            expected = np.array([scalar(float(v)) for v in values])
            assert function(values).tobytes() == expected.tobytes(), name
        bases, exponents = rng.uniform(0, 1e4, 65_539), rng.uniform(-4, 4, 65_539)
        scalar = host_libm.scalar("pow")
        expected = np.array([scalar(float(x), float(y)) for x, y in zip(bases, exponents)])
        assert host_libm.power(bases, exponents).tobytes() == expected.tobytes()
        y, x = rng.uniform(-1e6, 1e6, 65_539), rng.uniform(-1e6, 1e6, 65_539)
        scalar = host_libm.scalar("atan2")
        expected = np.array([scalar(float(u), float(v)) for u, v in zip(y, x)])
        assert host_libm.arctan2(y, x).tobytes() == expected.tobytes()


def test_native_host_math_broadcast_blocks_and_older_bridge_fallback(monkeypatch):
    from woof.core import portable_math as pm
    native = host_libm.implementation() == host_libm.GLIBC239
    pow_word = host_libm.scalar("pow") if native else math.pow
    exp_word = host_libm.scalar("exp") if native else math.exp
    base = np.linspace(0.01, 10, 513 * 257).reshape(513, 257)[:, ::2]
    exponents = np.linspace(-0.75, 1.25, base.shape[1])
    pairs = list(zip(base.flat, np.broadcast_to(exponents, base.shape).flat))
    expected = np.array([pow_word(float(x), float(y)) for x, y in pairs]).reshape(base.shape)
    assert host_libm.power(base, exponents).tobytes() == expected.tobytes()
    assert host_libm.power(np.empty((0, 3)), 2).shape == (0, 3)
    unaligned = np.ndarray((1003,), dtype=np.float64,
                           buffer=bytearray(1003 * 8 + 1), offset=1)
    unaligned[:] = np.linspace(0.01, 1.0, 1003)
    assert not unaligned.flags.aligned
    assert host_libm.exp(unaligned).tobytes() == np.array([exp_word(x) for x in unaligned]).tobytes()
    assert host_libm.power(unaligned, 0.7159).tobytes() == np.array([pow_word(x, 0.7159) for x in unaligned]).tobytes()
    # The fallback is the host C library through math.
    expected = np.array([host_libm._pow(float(x), float(y)) for x, y in pairs]).reshape(base.shape)
    monkeypatch.setattr(pm, "_load", lambda: None)
    assert host_libm.power(base, exponents).tobytes() == expected.tobytes()
    assert host_libm.exp(np.array([1.0, 2.0])).tobytes() == np.array([math.exp(1), math.exp(2)]).tobytes()


def test_geometry_callers_keep_reference_without_native_assets(monkeypatch):
    from woof.core import portable_math as pm
    def absent():
        raise FileNotFoundError("CPU bridge is not installed")
    monkeypatch.setattr(pm, "_load", absent)
    values = np.array([0.1, 1.0, 2.0])
    assert host_libm.log(values).tobytes() == np.array([math.log(x) for x in values]).tobytes()
    assert host_libm.power(values, 0.7159).tobytes() == np.array([math.pow(x, 0.7159) for x in values]).tobytes()
    def broken():
        raise OSError("native ABI cannot be loaded")
    monkeypatch.setattr(pm, "_load", broken)
    with pytest.raises(OSError, match="native ABI"):
        host_libm.power(values, 0.7159)
