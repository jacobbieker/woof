"""Fixed solver inputs and the host build's output words (tests/data/radtran/kernel_words.npz).

    pixi run -e mlx python -m tools.uvspec_oracle.kernel_words

The inputs are synthetic binary64 optical properties (no libRadtran data):
a 30-layer Rayleigh + absorber atmosphere with a Henyey-Greenstein cloud,
solved as plane-parallel and pseudo-spherical DISORT (4, 8, 16 streams,
fluxes and radiances with the TMS correction) and as two-stream.  Every
backend (host C++, CUDA, Metal) must reproduce the stored output words
bit for bit; that is how CUDA and MLX are held to each other.
"""

from __future__ import annotations

import math

import numpy as np

from woof.radtran import solver



def inputs():
    rng = np.random.default_rng(20261030)
    L = 30
    z = np.linspace(60.0, 0.0, L + 1)
    tau_r = 0.25 * np.exp(-z[1:] / 8.0) * 0.4
    tau_a = 0.02 * rng.random(L) * np.exp(-((z[1:] - 22.0) / 6.0) ** 2)
    cases = []
    for k, (tc, g, reff_ssa) in enumerate(((0.0, 0.0, 1.0), (2.0, 0.85, 0.99999), (25.0, 0.86, 0.9999),
                                           (0.3, 0.7, 0.98))):
        cloud = np.zeros(L)
        cloud[24:27] = tc / 3.0
        tsca = tau_r + cloud * reff_ssa
        dtau = tau_r + tau_a + cloud
        ssa = tsca / dtau
        pm = np.zeros((L, 17))
        pm[:, 0] = 1.0
        for l in range(17):
            pm[:, l] = (cloud * reff_ssa * g ** l + (tau_r * (1.0 if l == 0 else (0.0956 if l == 2 else 0.0)))) / tsca
        cases.append((dtau, ssa, pm))
    dtau = np.array([c[0] for c in cases])
    ssa = np.array([c[1] for c in cases])
    pmom = np.array([c[2] for c in cases])
    umu0 = np.array([0.95, 0.5, 0.2, math.cos(math.radians(85.0))])
    albedo = np.array([0.0, 0.2, 0.6, 0.1])
    zlev = np.tile(z, (4, 1))
    umu = (-0.9, -0.4, 0.2, 0.8)
    phi = (0.0, 60.0, 180.0)
    # exact phase function values for TMS (HG + Rayleigh), any fixed numbers serve
    ptrue = np.abs(rng.normal(1.0, 0.3, size=(4, L, len(umu), len(phi))))
    return dict(dtau=dtau, ssa=ssa, pmom=pmom, umu0=umu0, albedo=albedo, zlev=zlev,
                umu=np.array(umu), phi=np.array(phi), ptrue=ptrue)


#: (label, nstr, flags, with angles)
RUNS = (
    ("pp4", 4, solver.DELTAM, False),
    ("pp8", 8, solver.DELTAM, False),
    ("sph8", 8, solver.DELTAM | solver.SPHERICAL, False),
    ("rad16_tms", 16, solver.DELTAM | solver.TMS, True),
    ("rad8_sph", 8, solver.DELTAM | solver.SPHERICAL, True),
    ("twostr", 2, solver.DELTAM | solver.TWOSTREAM, False),
    ("twostr_sph", 2, solver.DELTAM | solver.TWOSTREAM | solver.SPHERICAL, False),
)


def run(inp, label, nstr, flags, angles, backend):
    pm = inp["pmom"][:, :, :nstr + 1]
    kw = dict(nstr=nstr, flags=flags, dtau=inp["dtau"], ssa=inp["ssa"], pmom=pm, umu0=inp["umu0"],
              fbeam=1.0, albedo=inp["albedo"], zlev=inp["zlev"], radius=6370.0,
              outlev=[0, 10, 26, 30], backend=backend)
    if angles:
        kw.update(umu=tuple(inp["umu"]), phi=tuple(inp["phi"]),
                  ptrue=inp["ptrue"] if flags & solver.TMS else None)
    return solver.solve(**kw)


def main():
    from pathlib import Path
    inp = inputs()
    out = {}
    for label, nstr, flags, angles in RUNS:
        f, r = run(inp, label, nstr, flags, angles, "host")
        out[f"{label}_flux"] = f
        out[f"{label}_rad"] = r
    dest = Path(__file__).resolve().parents[2] / "tests" / "data" / "radtran" / "kernel_words.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **{f"in_{k}": v for k, v in inp.items()}, **out)
    print("wrote", dest)


if __name__ == "__main__":
    main()
