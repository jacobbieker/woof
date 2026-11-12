"""The opt-in glibc 2.39 binary64 sin, cos and atan for GPU kernels.

``tools/glibc239_libm_gpu/glibc239_trig_flt64.cuh`` is LGPL code: a source
checkout uses it only with ``WOOF_GLIBC239_LIBM64=1``, and no wheel carries
it. Its words are graded in full by ``tools/glibc239_libm_gpu/grade_trig.py``
(``verification.log``); these tests hold the parts that can break quietly.

THE BREAKAGES THESE PREVENT
---------------------------
1. Tables that drift from the Rust transcription the words were proven on.
2. A transcription that no longer gives glibc 2.39's pinned words, on the
   host or on the device (CUDA, or Metal's soft binary64).
3. The LGPL code reaching a wheel or a default build: it lives outside the
   package, the switch is off unless set, and the NOTICE says so.
"""
from __future__ import annotations

import ctypes
import re
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "tools" / "glibc239_libm_gpu"
HEADER = HERE / "glibc239_trig_flt64.cuh"
RUST = ROOT / "tools" / "rustwx" / "crates" / "glibc239-libm" / "src"
sys.path.insert(0, str(HERE))
import grade_trig  # noqa: E402


def _words(text: str) -> list[str]:
    return re.findall(r"0x[0-9a-f]{16}", text)


def test_the_tables_are_the_rust_crates_words():
    header = HEADER.read_text(encoding="utf-8")
    sincos = header.split("SINCOSTAB[440] = {")[1].split("};")[0]
    cij = header.split("CIJ[241][7] = {")[1].split("};")[0]
    assert _words(sincos) == _words((RUST / "sincostab.rs").read_text(encoding="utf-8").split("SINCOSTAB")[1])
    rust_cij = (RUST / "uatan_tbl.rs").read_text(encoding="utf-8").split("CIJ")[1]
    assert _words(cij) == _words(rust_cij)
    assert len(_words(sincos)) == 440 and len(_words(cij)) == 241 * 7


def test_the_header_keeps_its_lgpl_notice_and_the_notice_names_the_opt_in():
    head = "".join(HEADER.read_text(encoding="utf-8").splitlines(True)[:30])
    assert "THIRD-PARTY NOTICE: LGPL-2.1-or-later" in head and "IBM Accurate Mathematical Library" in head
    assert "Modified: transcribed to CUDA C++" in HEADER.read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    section = notice.split("glibc 2.39 binary64 sin, cos and atan for the GPU")[1].split("-" * 74)[0]
    for token in ("THE TRANSCRIPTION IS OPT-IN.", "WOOF_GLIBC239_LIBM64=1", "licenses/LICENSE-glibc-LGPL-2.1.txt",
                  "tools/glibc239_libm_gpu/glibc239_trig_flt64.cuh", "metal-woof"):
        assert token in section, token
    assert (ROOT / "licenses" / "LICENSE-glibc-LGPL-2.1.txt").is_file()


def test_the_transcription_is_not_package_data():
    assert not list((ROOT / "woof").rglob("glibc239_trig*"))
    assert not (HERE / "__init__.py").exists()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = config["tool"]["setuptools"]
    assert not any(name.startswith("tools.") for name in setuptools["packages"]["find"]["include"])
    assert not any("glibc239" in pattern
                   for patterns in setuptools["package-data"].values() for pattern in patterns)
    import fnmatch
    for path in HERE.iterdir():
        relative = path.relative_to(ROOT / "tools").as_posix()
        assert not any(fnmatch.fnmatch(relative, pattern) for pattern in setuptools["package-data"].get("tools", []))


def test_the_switch_is_off_unless_set(monkeypatch):
    from woof.core import glibc239_gpu
    monkeypatch.delenv(glibc239_gpu.VARIABLE, raising=False)
    assert not glibc239_gpu.enabled()
    for value in ("0", "off", ""):
        monkeypatch.setenv(glibc239_gpu.VARIABLE, value)
        assert not glibc239_gpu.enabled()
    monkeypatch.setenv(glibc239_gpu.VARIABLE, "1")
    assert glibc239_gpu.enabled()
    assert glibc239_gpu.header_text().startswith(f"#define {glibc239_gpu.MACRO} 1\n")
    monkeypatch.setenv(glibc239_gpu.VARIABLE, "maybe")
    with pytest.raises(ValueError, match=glibc239_gpu.VARIABLE):
        glibc239_gpu.enabled()
    monkeypatch.setenv(glibc239_gpu.VARIABLE, "1")
    monkeypatch.setenv(glibc239_gpu.SOURCE_VARIABLE, str(HERE / "missing.cuh"))
    with pytest.raises(RuntimeError, match="source checkout"):
        glibc239_gpu.header_text()


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    if not (shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")):
        pytest.skip("no C++ compiler")
    return grade_trig.build(tmp_path_factory.mktemp("glibc239_trig"))


def test_the_host_transcription_gives_glibc_2_39s_words(host):
    assert grade_trig.pinned_check(grade_trig._host_evaluate(host)) == 0


def test_the_host_transcription_equals_this_hosts_glibc_below_the_branred_range(host):
    if grade_trig.glibc_version() is None or not host.trig_bind_host_libm():
        pytest.skip("needs a glibc host")
    for f in range(3):
        counts, first = np.zeros(3, dtype=np.uint64), ctypes.c_uint64()
        host.trig_sweep(f, 0, 1 << 20, counts.ctypes.data, ctypes.byref(first))
        assert counts[0] == 0, (grade_trig.FUNCTIONS[f], int(first.value))


@requires_gpu
def test_the_device_equals_the_host_transcription(host, capsys):
    # one leg of grade_trig.py's gpu run, on 2^16 arguments per function
    assert grade_trig.gpu(host, 1 << 16) == 0, capsys.readouterr().out
