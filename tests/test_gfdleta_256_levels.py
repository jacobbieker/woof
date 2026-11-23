"""GFDL Eta radiation (ra_lw/sw_physics = 99) on columns of up to 256 layers.

Stock WRF has no 81-layer ceiling for this scheme: ``module_ra_gfdleta.F``
sizes its model-level arrays from ``kte``/``kme`` and its ``NL=81`` is the
ozone climatology's fixed pressure grid.  WOOF's transcription fixes the
per-column storage; the shipped units are generated at 81 and the deep tier
regenerates the identical lowering at 256 (woof.core.gfdleta_levels,
tools/gfdleta_wrf461_oracle/build_tier.py).  Claims:

1. The ladder, the contract bound, the registry and both tiers' layouts
   agree; the climatology stays on its 81 levels in the deep layout (no GPU).
2. Preflight prices the column workspace and the modules of the tier a grid
   runs at (no GPU).
3. Forcing the 256 tier onto the WRF fixtures of 12 to 60 layers reproduces
   every initialization word of the shipped units and every ETARA output
   word of WRF.
4. A 256-layer column against byte-unmodified WRF 4.6.1 at 256 layers
   (run_column_seed.F90; fixture oracle-256.npz): in strict numerics
   (GPUWM_WRF_EXACT=1, run in a child process so the referee glibc binding
   is the fixture's) every CO2/ozone/table word of the initialization and
   every LW/SW output word is WRF's.  The default numerics stay finite and
   within a small relative distance of the same words.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures/gfdleta/column"
DEEP, SHIPPED = 256, 81
NCOL = 8
gpu = pytest.mark.skipif(os.environ.get("GPUWM_NO_LOCAL_GPU") == "1", reason="CPU-only check")

# ETARA output names in run_column_seed's fields.csv -> column ABI field.
OUTPUTS = {'RTHRATEN': 'THRATEN', 'RTHRATENLW': 'THRATENLW', 'RTHRATENSW': 'THRATENSW',
           'CLDFRA': 'CLDFRA', 'GLW': 'GLW', 'GSW': 'GSW', 'SWDOWN': 'RSWIN',
           'SWDOWNC': 'RSWINC', 'RLWTOA': 'RLWTOA', 'RSWTOA': 'RSWTOA', 'CFRACL': 'CFRACL',
           'CFRACM': 'CFRACM', 'CFRACH': 'CFRACH', 'CZMEAN': 'CZMEAN', 'ACFRST': 'ACFRST',
           'ACFRCV': 'ACFRCV', 'HTOPR': 'HTOPR', 'HBOTR': 'HBOTR', 'CUPPT': 'CUPPT'}


def _shape_count(shape):
    return int(np.prod([int(v) for v in shape.split(',')])) if shape else 1


# --------------------------------------------------------------------------- CPU

def test_the_ladder_contract_registry_and_layouts_agree():
    from woof.core import gfdleta_levels as lv
    from woof.physics_vertical_contract import GFDLETA_VERTICAL_LEVEL_BOUNDS
    from woof.config import DEFAULT_COLUMN_CHUNK

    assert [lv.maxl_tier(n) for n in (1, 60, 81, 82, 256)] == [81, 81, 81, 256, 256]
    for bad in (0, 257):
        with pytest.raises(ValueError):
            lv.maxl_tier(bad)
    assert GFDLETA_VERTICAL_LEVEL_BOUNDS == lv.VERTICAL_LEVEL_BOUNDS == (1, DEEP)
    registry = json.loads((ROOT / "woof/physics_registry_v2.json").read_text())
    warning = " ".join(registry["components"]["radiation"]["options"]["wrf-gfdleta"]["warnings"])
    assert "tables allow at most 81" not in warning and "256-layer storage tier" in warning
    shipped, deep = lv.column_layout(SHIPPED), lv.column_layout(DEEP)
    assert lv.WORK_BUDGET_BYTES == DEFAULT_COLUMN_CHUNK * shipped['scratch_bytes']
    assert [f['name'] for f in shipped['fields']] == [f['name'] for f in deep['fields']]
    for a, b in zip(shipped['fields'], deep['fields']):
        assert (a['count'], b['count']) in ((1, 1), (SHIPPED + 1, DEEP + 1)), a['name']
    for name, (kind, shape) in shipped['tables'].items():
        other = deep['tables'][name][1]
        if shape == f'{SHIPPED + 1},{SHIPPED + 1}':
            assert other == f'{DEEP + 1},{DEEP + 1}', name
        elif shape == str(SHIPPED + 1):
            assert other == str(DEEP + 1), name
        elif shape == str(SHIPPED) and name != 'PRGFDL':
            assert other == str(DEEP), name
        else:  # The ozone climatology (XDUO3N..., PRGFDL) and fixed tables.
            assert other == shape, name
    for maxl, layout in ((SHIPPED, shipped), (DEEP, deep)):
        words = sum(_shape_count(s) for _, s in layout['tables'].values())
        assert lv.init_layout(maxl)['table_bytes'] == 4 * words
    assert lv.init_layout(DEEP)['state_bytes'] == lv.init_layout(SHIPPED)['state_bytes']


def test_the_deep_co2_launch_unit_is_the_counted_storage_substitution(monkeypatch):
    import importlib.util
    monkeypatch.setenv('GFDLETA_MAXL', str(DEEP))
    spec = importlib.util.spec_from_file_location(
        "gfdleta_build_tier_256", ROOT / "tools/gfdleta_wrf461_oracle/build_tier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernels = ROOT / "woof/core/kernels"
    assert module.co2_launch_unit((kernels / "gfdleta_co2_init.cu").read_text()) == \
        (kernels / "gfdleta_co2_init_l256.cu").read_text()


@pytest.mark.parametrize("nz,maxl", [(60, 81), (81, 81), (82, 256), (256, 256)])
def test_preflight_prices_the_tier_the_grid_runs_at(nz, maxl):
    from woof.config import RunConfig
    from woof.core import gfdleta_levels as lv
    from woof.core import preflight

    cfg = RunConfig(nx=40, ny=50, nz=nz, dx=1000.0, dy=1000.0, ztop=30000.0,
                    dt=1.0, run_seconds=10.0, ra_physics=0, ra_lw_physics=99,
                    ra_sw_physics=99)
    shapes = preflight.gfdleta_column_shapes(cfg)
    assert preflight.physics_array_shapes(cfg)['radiation/gfdleta_tables'] == \
        (lv.init_layout(maxl)['table_bytes'] // 4,)
    layout = lv.column_layout(maxl)
    ncol = 40 * 50
    assert shapes['gfdleta/packed'][0] == (ncol, layout['float_count'] + layout['int_count'])
    assert shapes['gfdleta/column_work'][0] == (lv.column_slots(ncol, 3125, maxl), layout['scratch_bytes'])
    modules = preflight._radiation_kernel_modules(cfg, 1)
    suffix = '' if maxl == SHIPPED else '_l256'
    assert {'gfdleta_column' + suffix, 'gfdleta_co2_init' + suffix} <= modules
    assert not {'gfdleta_column_l256', 'gfdleta_co2_init_l256'} & modules if maxl == SHIPPED \
        else not {'gfdleta_column', 'gfdleta_co2_init'} & modules


def test_the_config_admits_256_layers_and_refuses_257():
    from woof.config import ConfigurationRefused, RunConfig, validate_run_config

    def cfg(nz):
        return RunConfig(nx=8, ny=8, nz=nz, dx=1000.0, dy=1000.0,
                         ztop=30000.0, dt=1.0, run_seconds=10.0,
                         ra_physics=0, ra_lw_physics=99, ra_sw_physics=99)

    validate_run_config(cfg(DEEP))
    with pytest.raises(ConfigurationRefused, match="GFDL Eta radiation requires 1 to 256"):
        validate_run_config(cfg(DEEP + 1))


def test_the_deep_slab_stays_within_the_shipped_budget():
    from woof.core import gfdleta_levels as lv
    deep = lv.column_layout(DEEP)['scratch_bytes']
    assert lv.column_slots(10**6, 3125, SHIPPED) == 3125
    assert lv.column_slots(10, 3125, DEEP) == 10
    slots = lv.column_slots(10**6, 3125, DEEP)
    assert slots * deep <= lv.WORK_BUDGET_BYTES < (slots + 1) * deep
    assert lv.column_slots(10**6, 7, DEEP) == 7


# --------------------------------------------------------------------------- GPU helpers

def _tier_table_words(tables, n):
    """Initialized words of every table, Fortran order, trimmed to n layers."""
    import cupy as cp
    words = cp.asnumpy(tables.words)
    maxl = tables.maxl
    out = {}
    for name, (kind, shape) in tables.tier.layout['tables'].items():
        off, count = tables.tier.offsets[name]
        v = words[off:off + count]
        if shape == f'{maxl + 1},{maxl + 1}':
            v = v.reshape(maxl + 1, maxl + 1)[:n + 1, :n + 1].reshape(-1)
        elif shape == str(maxl + 1):
            v = v[:n + 1]
        elif shape == str(maxl) and name != 'PRGFDL':
            v = v[:n]
        if kind == 'INTEGER':
            v = v.view(np.int32).astype(np.float32).view(np.uint32)
        out[name] = v
    return out


def _deep_from_shipped(fixture):
    """A shipped-ABI column fixture repacked into the deep ABI."""
    from woof.core import gfdleta_levels as lv
    a, b = lv.column_layout(SHIPPED), lv.column_layout(DEEP)
    fa = fixture['floats'].view(np.uint32).reshape(NCOL, -1)
    ia = fixture['integers'].reshape(NCOL, -1)
    f = np.zeros((NCOL, b['float_count']), np.uint32)
    ints = np.zeros((NCOL, b['int_count']), np.int32)
    for x, y in zip(a['fields'], b['fields']):
        src, dst = (fa, f) if x['kind'] == 'real' else (ia, ints)
        dst[:, y['offset']:y['offset'] + x['count']] = src[:, x['offset']:x['offset'] + x['count']]
    from woof.core.gfdleta import _table_offsets
    oa, _ = _table_offsets(a)
    ob, total = _table_offsets(b)
    table = np.zeros(total, np.uint32)
    for name, (kind, shape) in a['tables'].items():
        (sa, ca), (sb, cb) = oa[name], ob[name]
        v = fixture['table'][sa:sa + ca]
        if shape == f'{SHIPPED + 1},{SHIPPED + 1}':
            t = table[sb:sb + cb].reshape(DEEP + 1, DEEP + 1)
            t[:SHIPPED + 1, :SHIPPED + 1] = v.reshape(SHIPPED + 1, SHIPPED + 1)
        else:
            table[sb:sb + ca] = v
    return f.view(np.float32), ints, table


def _run_columns(module, table, floats, ints, scratch_bytes):
    import cupy as cp
    from woof.core.kernels import get_kernel
    f = cp.asarray(floats)
    i = cp.asarray(ints)
    work = cp.empty((NCOL, scratch_bytes), cp.uint8)
    get_kernel(module, 'gd_radiation_columns')((1,), (32,), (cp.asarray(table), f, i, work, np.int32(NCOL)))
    return cp.asnumpy(f).view(np.uint32)


# --------------------------------------------------------------------------- GPU: identity

@gpu
@pytest.mark.parametrize("n", [12, 60])
@pytest.mark.parametrize("co2tf", [0, 1])
def test_the_deep_tier_reproduces_every_initialization_word(n, co2tf):
    import cupy as cp
    from woof.core.gfdleta import GFDLTables
    with np.load(FIX / f'init-{n}.npz') as z:
        args = (cp.asarray(z['sigma_full'][:n + 1]), cp.asarray(z['sigma_half'][:n]), 1000.)
        kw = dict(co2tf=co2tf, transmission=z['transmission'], start_time=datetime(2001, 3, 21, 12))
        shipped = GFDLTables(*args, **kw)
        deep = GFDLTables(*args, maxl=DEEP, **kw)
    assert (shipped.maxl, deep.maxl) == (SHIPPED, DEEP)
    a, b = _tier_table_words(shipped, n), _tier_table_words(deep, n)
    assert a.keys() == b.keys()
    bad = {k: int(np.count_nonzero(a[k] != b[k])) for k in a if not np.array_equal(a[k], b[k])}
    assert not bad, bad
    assert cp.asnumpy(shipped.solar_distance).tobytes() == cp.asnumpy(deep.solar_distance).tobytes()


@gpu
@pytest.mark.parametrize("n", [12, 24, 42, 60])
def test_the_deep_tier_reproduces_every_wrf_column_word(n):
    """Every real word the deep unit writes equals the shipped unit's on this
    backend and numerics; in the WRF-exact build both are also WRF's words."""
    from woof.core import gfdleta_levels as lv
    from woof.core import kernels
    a, b = lv.column_layout(SHIPPED), lv.column_layout(DEEP)
    with np.load(FIX / f'columns-{n}.npz') as z:
        fixture = {k: z[k] for k in z.files}
    floats, ints, table = _deep_from_shipped(fixture)
    got = _run_columns('gfdleta_column_l256', table, floats, ints, b['scratch_bytes'])
    ref = _run_columns('gfdleta_column', fixture['table'],
                       fixture['floats'].view(np.float32).reshape(NCOL, -1),
                       fixture['integers'].reshape(NCOL, -1), a['scratch_bytes'])
    # Map each shipped-ABI real word to the same field and level in the deep ABI.
    deep_index = np.full(a['float_count'], -1, np.int64)
    for x, y in zip(a['fields'], b['fields']):
        if x['kind'] == 'real':
            deep_index[x['offset']:x['offset'] + x['count']] = y['offset'] + np.arange(x['count'])
    real = np.flatnonzero(deep_index >= 0)
    differ = np.argwhere(got[:, deep_index[real]] != ref[:, real])
    assert not len(differ), [(int(c) + 1, int(real[k]), f'{got[c, deep_index[real[k]]]:08X}',
                              f'{ref[c, real[k]]:08X}') for c, k in differ[:12]]
    if "-DGPUWM_GFDLETA_STRICT=1" in kernels.module_options('gfdleta_column_l256'):
        words = got[fixture['columns'], deep_index[fixture['indices']]]
        bad = np.flatnonzero(words != fixture['reference'])
        assert not len(bad), [(str(fixture['fields'][k]), int(fixture['columns'][k]) + 1,
                               f'{words[k]:08X}', f'{fixture["reference"][k]:08X}') for k in bad[:12]]


# --------------------------------------------------------------------------- GPU: 256 layers

def _oracle_run(path):
    """Initialize from sigma and run the oracle's eight 256-layer columns."""
    import cupy as cp
    from woof.core import gfdleta_levels as lv
    from woof.core.gfdleta import GFDLTables
    with np.load(path) as z:
        fx = {k: z[k] for k in z.files}
    n = int(fx['n'])
    tables = GFDLTables(cp.asarray(fx['sigma_full'].view(np.float32)),
                        cp.asarray(fx['sigma_half'].view(np.float32)), 1000., co2tf=1,
                        start_time=datetime(2001, 3, 21, 12))
    layout = tables.tier.layout
    fields = {x['name'].upper(): x for x in layout['fields']}
    f = np.zeros((NCOL, layout['float_count']), np.uint32)
    ints = np.zeros((NCOL, layout['int_count']), np.int32)
    names = fx['input_names']
    for field, column, idx, bits in zip(fx['input_field'], fx['input_column'],
                                        fx['input_index'], fx['input_bits']):
        d = fields[str(names[field])]
        target = f if d['kind'] == 'real' else ints.view(np.uint32)
        target[column - 1, d['offset'] + max(0, int(idx) - 1)] = bits
    got = _run_columns(tables.tier.column_module, tables.words, f.view(np.float32), ints,
                       layout['scratch_bytes'])
    table_words = _tier_table_words(tables, n)
    outputs = []
    onames = fx['output_names']
    for field, column, level, bits in zip(fx['output_field'], fx['output_column'],
                                          fx['output_level'], fx['output_bits']):
        name = str(onames[field])
        d = fields[OUTPUTS[name]]
        outputs.append((name, int(column), int(level),
                        int(got[column - 1, d['offset'] + max(0, int(level) - 1)]), int(bits)))
    digests = {}
    for name, size, want in zip(fx['table_names'], fx['table_sizes'], fx['table_sha256']):
        name = str(name)
        if name not in table_words:  # init-only state (PA) and copies the column never reads
            continue
        v = table_words[name]
        digests[name] = (int(size), len(v), str(want),
                         hashlib.sha256(np.asarray(v, '<u4').tobytes()).hexdigest())
    return n, tables.maxl, outputs, digests


_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[2])
from test_gfdleta_256_levels import _oracle_run
n, maxl, outputs, digests = _oracle_run(sys.argv[1])
print('RESULT' + json.dumps({'n': n, 'maxl': maxl, 'outputs': outputs, 'digests': digests}))
"""


@gpu
def test_a_256_layer_column_is_stock_wrf_word_for_word_in_strict_numerics():
    path = FIX / 'oracle-256.npz'
    with np.load(path) as z:
        glibc = str(z['referee_glibc'])
    env = dict(os.environ, GPUWM_WRF_EXACT='1', GPUWM_NUMERICS='strict', GPUWM_WRF_GLIBC=glibc)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    proc = subprocess.run([sys.executable, '-c', _CHILD, str(path), str(ROOT / 'tests')],
                          env=env, capture_output=True, text=True, timeout=3600)
    line = [x for x in proc.stdout.splitlines() if x.startswith('RESULT')]
    assert proc.returncode == 0 and line, proc.stderr[-4000:]
    result = json.loads(line[0][len('RESULT'):])
    assert (result['n'], result['maxl']) == (DEEP, DEEP)
    bad_tables = {k: v for k, v in result['digests'].items() if v[0] != v[1] or v[2] != v[3]}
    assert not bad_tables, sorted(bad_tables)
    assert {'CO251', 'CDT58', 'C2DM51', 'STEMP', 'XDUO3N', 'PRGFDL', 'TABLE1'} <= result['digests'].keys()
    misses = [x for x in result['outputs'] if x[3] != x[4]]
    assert not misses, (len(misses), [(a, c, l, f'{g:08X}', f'{w:08X}') for a, c, l, g, w in misses[:12]])
    assert len(result['outputs']) == NCOL * (4 * DEEP + 15)


@gpu
def test_a_256_layer_column_in_the_running_numerics_is_finite_and_near_wrf():
    n, maxl, outputs, _ = _oracle_run(FIX / 'oracle-256.npz')
    assert (n, maxl) == (DEEP, DEEP)
    got = np.array([x[3] for x in outputs], np.uint32).view(np.float32)
    want = np.array([x[4] for x in outputs], np.uint32).view(np.float32)
    assert np.isfinite(got).all()
    scale = {}
    for (name, *_), w in zip(outputs, want):
        scale[name] = max(scale.get(name, 0.0), abs(float(w)))
    for (name, column, level, *_), g, w in zip(outputs, got, want):
        assert abs(float(g) - float(w)) <= 1e-3 * max(scale[name], 1e-30), (name, column, level, g, w)
