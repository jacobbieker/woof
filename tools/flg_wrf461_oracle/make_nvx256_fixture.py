"""Write the 256-layer FLG oracle fixture from the patched-nvx Fortran build.

Run after build_nvx256.sh, on the platform whose libm the strict CUDA column
reproduces (glibc 2.39, the GPUWM_WRF_GLIBC default):

    python tools/flg_wrf461_oracle/make_nvx256_fixture.py BUILD_DIR \
        tests/fixtures/flg/nvx256-columns.npz

Before writing anything it proves the patch inert where stock WRF runs: the
pristine (nvx=100) and patched (nvx=256) libraries return the same words for
every output of a 24-layer and a 100-layer deck. The 256-layer deck is
synthetic (no captured forecast data): a standard-atmosphere column to 1 hPa
with clear, overcast, broken and mixed-phase cloud, day and night columns.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from tools.flg_wrf461_oracle.check_column import native
except ImportError:  # run as a script from this directory
    from check_column import native
from woof.core.flg import FLGColumn, SOURCE_SHA256

PATCHED_SHA256 = 'a6b183cbc1ba103ff585ea9a5f80291c1cdfafb0879b6644b3613f00263e5a48'
PATCH = Path(__file__).with_name('nvx256.patch')


def deck(nv):
    """Six columns of ``nv`` layers, top to surface, in native FLG units."""
    n = 6
    p = np.geomspace(1., 1000., nv + 1).astype(np.float32)
    z = np.float32(7.)*np.log(np.float32(1013.25)/p)  # km, scale height 7 km
    t = np.where(z < 11, 288.15 - 6.5*z, np.where(z < 20, 216.65, np.where(
        z < 32, 216.65 + (z - 20), np.minimum(228.65 + 2.8*(z - 32), 270.65))))
    h = np.maximum(0.012*np.exp(-z/2.2), 3e-6)
    o = 8e-6*np.exp(-((z - 25.)/8.)**2) + 3e-8
    tile = lambda v: np.tile(np.asarray(v, dtype=np.float32), (n, 1))
    zl = 0.5*(z[1:] + z[:-1])
    zero = np.zeros((n, nv), dtype=np.float32)
    cloud, ice, liquid, rain, graupel = zero.copy(), zero.copy(), zero.copy(), zero.copy(), zero.copy()
    high = (zl > 8) & (zl < 11)
    mid = (zl > 3) & (zl < 6)
    low = (zl > 0.6) & (zl < 2)
    for c, fraction in ((1, 1.), (2, .5), (3, 1.), (5, 1.)):
        cloud[c, high | mid | low] = fraction
        ice[c, high] = .012
        liquid[c, low] = .25
        ice[c, mid] = .004
        liquid[c, mid] = .06
    rain[3, low] = .05
    graupel[3, mid] = .02
    coszen = np.array([.85, .6, .35, .1, -.2, -.7], dtype=np.float32)
    return dict(pressure=tile(p), temperature=tile(t), humidity=tile(h), ozone=tile(o),
                ice=ice, ice_size=np.full_like(zero, 60.), liquid=liquid,
                liquid_radius=np.full_like(zero, 9.), rain=rain, graupel=graupel,
                cloud_fraction=cloud, inhomogeneity=np.ones_like(zero),
                albedo=np.tile(np.array([.12, .14, .16, .2, .22, .25], dtype=np.float32), (n, 1)),
                emissivity=np.full((n, 12), .985, dtype=np.float32), coszen=coszen,
                solar_constant=np.full(n, 1361., dtype=np.float32),
                skin_temperature=np.full(n, 292., dtype=np.float32))


def words(result):
    return {k: v.view(np.uint32).copy() for k, v in result.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('build', type=Path)
    parser.add_argument('fixture', type=Path)
    args = parser.parse_args()
    build = args.build.resolve()
    patched = (build/'nvx256'/'phys'/'module_ra_flg.F').read_bytes()
    pristine = (build/'pristine'/'phys'/'module_ra_flg.F').read_bytes()
    if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256 or hashlib.sha256(pristine).hexdigest() != SOURCE_SHA256:
        raise SystemExit('FLG oracle sources differ from the pinned and patched hashes')
    stock_lib, deep_lib = build/'libflg_oracle_pristine.so', build/'libflg_oracle_nvx256.so'
    inert = {}
    for nv in (24, 100):
        fields = deck(nv)
        stock = words(native(stock_lib, fields, nvx=100))
        deep = words(native(deep_lib, fields, nvx=256))
        for name in FLGColumn.output_names:
            if not np.array_equal(stock[name], deep[name]):
                raise SystemExit(f'nvx256 patch changes {name} at {nv} layers')
        inert[str(nv)] = sum(v.size for v in stock.values())
    fields = deck(256)
    result = native(deep_lib, fields, nvx=256)
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise SystemExit(f'patched oracle {name} is not finite')
    arrays = {**{'in_'+k: np.asarray(v) for k, v in fields.items()},
              **{'out_'+k: v for k, v in result.items()}}
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fixture, **arrays)
    receipt = {
        'kind': 'patched-nvx WRF FLG rad_all oracle, synthetic 256-layer deck',
        'source': 'https://github.com/wrf-model/WRF/blob/d66e442fccc04111067e29274c9f9eaccc3cef28/phys/module_ra_flg.F',
        'source_sha256': SOURCE_SHA256,
        'patch': 'tools/flg_wrf461_oracle/nvx256.patch',
        'patch_sha256': hashlib.sha256(PATCH.read_bytes()).hexdigest(),
        'patched_source_sha256': PATCHED_SHA256,
        'compiler': (build/'compiler.txt').read_text().strip(),
        'compiler_flags': (build/'compiler-flags.txt').read_text().strip(),
        'libc': (build/'libc.txt').read_text().strip(),
        'columns': int(fields['coszen'].size), 'layers': 256,
        'sunlit_columns': int(np.sum(fields['coszen'] > 0)),
        'cloudy_columns': int(np.sum(np.any(fields['cloud_fraction'] > 0, axis=1))),
        'patch_inert_words': inert,
        'output_words': int(sum(v.size for v in result.values())),
        'fixture_sha256': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        'fields': {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()},
    }
    args.fixture.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'fields'}, indent=2))


if __name__ == '__main__':
    main()
