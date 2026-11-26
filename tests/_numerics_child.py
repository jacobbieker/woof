"""Run a test module's identity tests in a child process under one numerics mode.

``tests/conftest.py`` pins ``GPUWM_NUMERICS=strict`` before woof loads, and a
process fixes its mode at the first compile, so a byte-identity claim checked
only by the battery is a strict-numerics claim.  Production runs fast: fast
mode drops ``--fmad=false`` (CUDA) and lets Metal's ``fp contract(fast)``
stand, so a restructured kernel can contract a product into a sum its
reference kept apart (#234's stored w stresses moved a quarter of a 3 km
step's w words for 8 days with every strict test green; #498).

``run(test_file, names, numerics, exact)`` starts a child, selects the mode
after the module's import and before its first compile, then calls each named
test once per combination of its own ``pytest.mark.parametrize`` marks, with
``monkeypatch`` / ``tmp_path`` supplied when the test asks for them.  The child
prints one line per failing case; the parent asserts on its output.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

#: (GPUWM_NUMERICS = WOOF_HEX_NUMERICS, GPUWM_WRF_EXACT) per child: the production default, the
#: battery's strict, and WRF-exact (strict numerics plus WRF's controls).
MODES = ("numerics,exact", [
    pytest.param("fast", "0", id="fast"),
    pytest.param("strict", "0", id="strict"),
    pytest.param("strict", "1", id="wrf_exact"),
])

_CHILD = r'''
import inspect, itertools, os, sys, tempfile, traceback
from pathlib import Path
sys.path.insert(0, {here!r})
import pytest
import conftest  # the battery's setup: backend install (WOOF_BACKEND) before any cupy import
import woof  # the automatic facade, before the module's importorskip("cupy")
import {module} as t
if conftest._MPAS_PARITY_MODULE.search({module!r}):  # the MPAS oracle parity modules run in exact mode
    from woof import correct_mode
    correct_mode.PHYSICS_MODE_DEFAULT = correct_mode.PHYSICS_MODE_EXACT
os.environ["GPUWM_NUMERICS"] = os.environ["WOOF_NUMERICS"] = {numerics!r}
from woof import numerics
assert numerics.mode() == {numerics!r}, numerics.mode()
# The hex units read their own knob (woof.hex.numerics, WOOF_HEX_NUMERICS, default fast); without
# it every hex "strict" child compiled fast.  set_mode refuses if a hex kernel already bound the other mode.
from woof.hex import numerics as hex_numerics
hex_numerics.set_mode({numerics!r}, source="tests/_numerics_child")
assert hex_numerics.mode() == {numerics!r}, hex_numerics.mode()

def cases(fn):
    marks = [m for m in getattr(fn, "pytestmark", []) if m.name == "parametrize"]
    axes = []
    for m in marks:
        names = [n.strip() for n in (m.args[0].split(",") if isinstance(m.args[0], str) else m.args[0])]
        values = []
        for v in m.args[1]:
            v = v.values if hasattr(v, "values") and hasattr(v, "marks") else v
            values.append(dict(zip(names, v if len(names) > 1 else (v,))))
        axes.append(values)
    for combo in itertools.product(*axes):
        kw = {{}}
        for part in combo:
            kw.update(part)
        yield kw

ran = failed = skipped = 0
for name in {names!r}:
    fn = getattr(t, name)
    wants = inspect.signature(fn).parameters
    for kw in cases(fn):
        with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory() as tmp:
            if "monkeypatch" in wants:
                kw["monkeypatch"] = mp
            if "tmp_path" in wants:
                kw["tmp_path"] = Path(tmp)
            label = name + str({{k: v for k, v in kw.items() if k not in ("monkeypatch", "tmp_path")}})
            try:
                fn(**kw)
                ran += 1
            except pytest.skip.Exception as exc:
                skipped += 1
                print("SKIP", label, exc, flush=True)
            except BaseException:
                failed += 1
                print("FAIL", label, flush=True)
                traceback.print_exc(limit=6, file=sys.stdout)
print("SUMMARY ran=%d failed=%d skipped=%d" % (ran, failed, skipped), flush=True)
'''


def run(test_file, names, numerics, exact, *, env=None, timeout=1800):
    """Run ``names`` from ``test_file`` in a child under ``numerics``/``exact``."""
    if numerics == "fast" and exact == "1":
        pytest.skip("GPUWM_WRF_EXACT=1 is strict numerics")
    path = Path(test_file).resolve()
    child_env = dict(os.environ)
    # Both spellings: importing woof copies GPUWM_X onto an unset WOOF_X only,
    # so an inherited WOOF_WRF_EXACT would otherwise win.
    child_env["GPUWM_WRF_EXACT"] = child_env["WOOF_WRF_EXACT"] = exact
    # woof.hex.numerics reads WOOF_HEX_NUMERICS, not GPUWM_NUMERICS: name it too.
    child_env["WOOF_HEX_NUMERICS"] = numerics
    # A diagnostic contraction override would make the fast row compile with
    # --fmad=false and test nothing production runs.
    for name in ("WOOF_FMAD_OFF", "GPUWM_FMAD_OFF"):
        child_env.pop(name, None)
    # Inherited WRF-exact sub-controls (woof/wrf_exact.py) would make the
    # exact row test an incomplete control set; a caller sets one through env.
    for name in [n for n in child_env if n.startswith(("WOOF_WRF_EXACT_", "GPUWM_WRF_EXACT_"))]:
        child_env.pop(name)
    child_env.update(env or {})
    code = _CHILD.format(here=str(path.parent), module=path.stem, numerics=numerics,
                         names=list(names))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env=child_env, timeout=timeout, cwd=str(path.parent.parent))
    out = result.stdout + result.stderr
    summary = [line for line in result.stdout.splitlines() if line.startswith("SUMMARY ")]
    assert result.returncode == 0 and summary, out[-8000:]
    fields = dict(part.split("=") for part in summary[-1].split()[1:])
    assert fields["failed"] == "0", out[-8000:]
    if fields["ran"] == "0":
        pytest.skip("every case skipped in the child: " + out[-500:])
