"""The optional diagnostic modules' fields are 'produced' before the writer is built.

streams_output_v842.PhysicsStreams checks every enabled output stream's fields
against ``diagnostics.produced`` in its constructor, before it binds the
diagnostics object.  These fields used to join ``produced`` only in bind(), so
every live run with config_nwp_diagnostics, p/z_lev, prec_acc/acc_phy_tend,
AFWA or the CLWRF/mean/diurnal/surface-energy reductions stopped before step 1
with "MPAS output fields lack computed authority".
"""
from types import SimpleNamespace

import pytest

from woof.hex.mpasdiag import acc_diag_v842 as acc, afwa_diag_v842 as afwa
from woof.hex.mpasdiag import nwp_diag_v842 as nwp, plevel_diag_v842 as plevel
from woof.hex.mpasdiag import reduction_diag_v842 as reduce
from woof.hex.mpasphys import resident_output_diagnostics_v842 as rod


def _owner(packages):
    session = SimpleNamespace(layout=SimpleNamespace(packages=set(packages), fields=[]),
                              device_pools=SimpleNamespace(cp=None))
    return rod.ResidentOutputDiagnostics(session, dycore=None)


def test_no_optional_package_adds_nothing():
    assert _owner(()).produced == rod.PRODUCED


@pytest.mark.parametrize("packages, names", [
    ({nwp.PACKAGE}, nwp.NAMES),
    ({acc.PREC_PACKAGE}, acc.NAMES),
    ({acc.TEND_PACKAGE}, acc.NAMES),
    ({next(iter(reduce.PACKAGES))}, reduce.NAMES),
    ({next(iter(afwa.SELECTORS))}, afwa.NAMES),
    ({plevel.P_PACKAGE}, plevel.NAMES),
    ({plevel.Z_PACKAGE}, plevel.NAMES),
])
def test_an_active_package_s_fields_are_produced_at_construction(packages, names):
    produced = _owner(packages).produced
    assert set(names) <= produced and rod.PRODUCED <= produced
