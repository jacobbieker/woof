"""The mp=8 finish clamp and mp=28 against WRF with only the rain-graupel
slab index corrected, on the live production adapters."""
import os
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/thompson_mp8_column_oracle'))


def test_mp8_finish_clamps_with_the_real_product_wrf_forms():
    """module_big_step_utilities_em.F:5706-5707, :5743-5745, through the
    production mp=8 launch (woof.core.thompson.launch_adapter_finish).

    Breakage prevented: WRF clamps the theta increment at the REAL product
    mp_tend_lim*dt, 0x3ca3d70b for 0.001 K/s at dt = 20 s.  The classic
    finish formed the product in double and rounded it (0x3ca3d70a), so a
    clamped cell's heating rate came out one float32 unit off WRF's
    +/-REAL 0.001 (0x3a83126e instead of 0x3a83126f).
    """
    cp = pytest.importorskip('cupy')
    from woof.core.thompson import launch_adapter_finish
    shape = (1, 1, 2)
    temperature = cp.asarray([301.0, 299.0], cp.float32).reshape(shape)
    pii = cp.ones(shape, cp.float32)
    th = cp.empty(shape, cp.float32)
    thp = cp.full(shape, 0.25, cp.float32)
    h_diabatic = cp.full(shape, 300.0, cp.float32)   # the saved theta
    surface = [cp.zeros(shape[1:], cp.float32) for _ in range(4)]
    launch_adapter_finish(temperature, pii, th, thp, h_diabatic, *surface,
                          SimpleNamespace(mp_tend_lim=0.001, no_mp_heating=0),
                          20.0)
    assert np.array_equal(h_diabatic.get().ravel().view(np.uint32),
                          [0x3a83126f, 0xba83126f])
    assert np.array_equal(thp.get().ravel(),
                          np.asarray([0.27, 0.23], np.float32))


def _corrected_mp28():
    """corrected_mp28 imports the aerosol tool's oracle_io and compare by bare
    name; four oracle tools ship an oracle_io.py (and several a compare.py),
    so an xdist worker that ran another oracle file first handed it theirs.
    Load it with those bare names cleared, as test_wdm6_wrf461_oracle does,
    and put the caller's entries back afterwards."""
    if 'corrected_mp28' in sys.modules:
        return sys.modules['corrected_mp28']
    siblings = ('oracle_io', 'compare')
    saved = {key: sys.modules.pop(key) for key in siblings if key in sys.modules}
    try:
        import corrected_mp28
    finally:
        for key in siblings:
            sys.modules.pop(key, None)
        sys.modules.update(saved)
    return corrected_mp28


@pytest.mark.parametrize('dt', [5, 20])
def test_aerosol_all_columns_against_only_index_corrected_wrf(tmp_path, dt):
    build = os.environ.get('THOMPSON_MP28_ORACLE_BUILD')
    pytest.importorskip('cupy')
    corrected_mp28 = _corrected_mp28()
    pin = json.loads((ROOT / 'tests/data/thompson_slab1_corrected_mp28.json').read_text())
    assert pin['source']['new'] == corrected_mp28.NEW
    assert pin['source']['stock_module_sha256'] == corrected_mp28.PINNED
    assert pin['source']['replacement_count'] == 8
    assert hashlib.sha256((ROOT / 'tests/data/mp28_column_oracle_wrf461.npz').read_bytes()).hexdigest() == pin['input_fixture_sha256']
    if not build:
        cols, gpu, _deleted = corrected_mp28.gpu_answers(tmp_path, dt)
        assert cols['p'].shape == (pin['columns'], pin['levels'])
        assert corrected_mp28.field_hashes(gpu) == pin['dts'][str(dt)]['sha256']
        return
    receipt = corrected_mp28.check(build, tmp_path, dt)
    assert receipt['columns'] == 157
    assert receipt['cells_differ'] == 0, receipt
    assert receipt['rain_meets_graupel_columns'] > 0
    assert receipt['rain_meets_graupel_differing_words'] == 0
    assert receipt['full_field_sha256']['wrf'] == receipt['full_field_sha256']['woof']
    assert receipt['full_field_sha256']['wrf'] == pin['dts'][str(dt)]['sha256']
