"""Build JCSDA CRTM v2.4+ (CC0) and woof's C shim into one shared library (opt-in).

    python -m woof.rtm.crtm_build --source CRTM_SRC --out DIR [--fc gfortran] [--jobs N]

``CRTM_SRC`` is a checkout of https://github.com/JCSDA/crtm at a v2.4 or later tag
(the tree with ``LICENSE`` and ``src/CMakeLists.txt``).  Nothing is downloaded here
and nothing is vendored: the user supplies the source tree, and the library lands in
``DIR`` (``libwoof_crtm.so`` or ``.dylib`` plus ``build-receipt.json``).  The default
woof wheel carries neither CRTM nor this library (docs/dev/WRFDA-3DVAR-PLAN.md 4.11).

The build needs a Fortran compiler and netCDF-Fortran (``nf-config``): CRTM's
coefficient readers include netCDF variants.  Without a system netCDF-Fortran::

    pixi exec -s netcdf-fortran -s gfortran -- python -m woof.rtm.crtm_build ...

This builder reads CRTM's own source list from ``src/CMakeLists.txt`` (CRTM's CMake
build needs ecbuild), orders the files by their Fortran module dependencies and
compiles them, so the library holds exactly the sources CRTM's CMake would build.

Refusals: a tree whose ``LICENSE`` is not the CC0 dedication (the CRTM 2.3.0 copy
inside WRF carries GPL-2.0-or-later files, and woof never compiles it), and a
version below 2.4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHIM = Path(__file__).with_name("woof_crtm_shim.f90")
LIBRARY_STEM = "libwoof_crtm"
RECEIPT_SCHEMA = "woof.rtm.crtm-build/v1"

_MODULE = re.compile(r"^\s*module\s+(?!procedure\b)(\w+)", re.I | re.M)
_USE = re.compile(r"^\s*use\s*(?:,\s*\w+\s*)?(?:::)?\s*(\w+)", re.I | re.M)


class CrtmBuildError(RuntimeError):
    """The source tree or toolchain cannot give a CC0 CRTM v2.4+ library."""


def library_name() -> str:
    return LIBRARY_STEM + (".dylib" if sys.platform == "darwin" else ".so")


def check_tree(source: Path) -> dict:
    """The tree's licence and version, or a refusal."""

    licence = source / "LICENSE"
    cmake = source / "src" / "CMakeLists.txt"
    if not licence.is_file() or not cmake.is_file():
        raise CrtmBuildError(f"{source}: not a JCSDA CRTM tree (needs LICENSE and src/CMakeLists.txt)")
    text = licence.read_text(errors="replace")
    if "public domain" not in text.lower() or "creativecommons.org/publicdomain/zero" not in text:
        raise CrtmBuildError(f"{licence}: not the CC0 dedication of JCSDA CRTM v2.4+; woof links no other CRTM")
    for path in (source / "src").rglob("*.[fF]90"):
        head = path.read_text(errors="replace")[:4000]
        if "GNU General Public License" in head:
            raise CrtmBuildError(f"{path}: carries a GPL header; this is not a CC0 CRTM tree")
    match = re.search(r"v(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        raise CrtmBuildError(f"{licence}: names no CRTM version")
    version = tuple(int(g or 0) for g in match.groups())
    if version < (2, 4, 0):
        raise CrtmBuildError(f"{licence}: CRTM v{'.'.join(map(str, version))} is below 2.4")
    return dict(licence_sha256=hashlib.sha256(text.encode()).hexdigest(),
                version=".".join(map(str, version)))


def source_list(source: Path) -> list[Path]:
    """CRTM's own library source list, as its CMake build declares it."""

    text = (source / "src" / "CMakeLists.txt").read_text()
    block = re.search(r"list\(\s*APPEND\s+crtm_src_files(.*?)\)", text, re.S)
    if block is None:
        raise CrtmBuildError("src/CMakeLists.txt: no crtm_src_files list")
    files = [source / "src" / name for name in block.group(1).split()]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise CrtmBuildError(f"listed sources missing: {missing[:3]}")
    return files


def dependency_order(files: list[Path]) -> list[list[Path]]:
    """Waves of files whose used modules are all provided by earlier waves."""

    provides, uses = {}, {}
    for path in files:
        text = path.read_text(errors="replace")
        for name in _MODULE.findall(text):
            provides[name.lower()] = path
        uses[path] = {u.lower() for u in _USE.findall(text)}
    waves, done = [], set()
    remaining = list(files)
    while remaining:
        ready = [p for p in remaining
                 if all(provides.get(u) in done or provides.get(u) in (None, p) for u in uses[p])]
        if not ready:
            raise CrtmBuildError(f"module dependency cycle among {[p.name for p in remaining][:5]}")
        waves.append(ready)
        done.update(ready)
        remaining = [p for p in remaining if p not in done]
    return waves


def _netcdf_flags() -> tuple[list[str], list[str]]:
    exe = shutil.which("nf-config")
    if exe is None:
        raise CrtmBuildError("nf-config not found: CRTM needs netCDF-Fortran "
                             "(e.g. `pixi exec -s netcdf-fortran -s gfortran -- python -m woof.rtm.crtm_build ...`)")
    flags = subprocess.run([exe, "--fflags"], capture_output=True, text=True, check=True).stdout.split()
    libs = subprocess.run([exe, "--flibs"], capture_output=True, text=True, check=True).stdout.split()
    return flags, libs


def build(source: Path, out: Path, *, fc: str | None = None, jobs: int = 4) -> dict:
    source, out = Path(source).resolve(), Path(out).resolve()
    tree = check_tree(source)
    files = source_list(source)
    # The list also names .inc files, which the compiled sources include.
    waves = dependency_order([f for f in files if f.suffix in (".f90", ".F90")])
    fc = fc or os.environ.get("FC") or shutil.which("gfortran")
    if not fc:
        raise CrtmBuildError("no Fortran compiler (set --fc or FC)")
    nc_flags, nc_libs = _netcdf_flags()
    objdir = out / "obj"
    objdir.mkdir(parents=True, exist_ok=True)
    include = sorted({str(f.parent) for f in files} | {str(source / "src")})
    flags = ["-O2", "-fPIC", "-fopenmp", "-ffree-line-length-none", "-fno-range-check", f"-J{objdir}",
             *(f"-I{d}" for d in include), *nc_flags]

    def compile_one(path: Path) -> Path:
        obj = objdir / (path.stem + ".o")
        extra = ["-cpp"] if path.suffix == ".F90" or path.name == "CRTM_K_Matrix_Module.f90" else []
        run = subprocess.run([fc, *flags, *extra, "-c", str(path), "-o", str(obj)],
                             capture_output=True, text=True)
        if run.returncode:
            raise CrtmBuildError(f"{path.name}: compile failed\n{run.stderr[-2000:]}")
        return obj

    objects = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for wave in waves:
            objects.extend(pool.map(compile_one, wave))
    shim_obj = compile_one(SHIM)
    library = out / library_name()
    shared = ["-dynamiclib", f"-Wl,-install_name,@rpath/{library.name}"] if sys.platform == "darwin" else ["-shared"]
    run = subprocess.run([fc, *shared, "-fopenmp", "-o", str(library), *map(str, objects), str(shim_obj), *nc_libs],
                         capture_output=True, text=True)
    if run.returncode:
        raise CrtmBuildError(f"link failed\n{run.stderr[-2000:]}")
    compiler = subprocess.run([fc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    receipt = dict(schema=RECEIPT_SCHEMA, crtm_version=tree["version"], licence="CC0-1.0",
                   licence_sha256=tree["licence_sha256"], source=str(source), files=len(files),
                   compiler=compiler, library=library.name,
                   library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                   shim_sha256=hashlib.sha256(SHIM.read_bytes()).hexdigest())
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m woof.rtm.crtm_build", description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, help="a JCSDA CRTM v2.4+ checkout (CC0)")
    parser.add_argument("--out", required=True, help="directory for the library and its receipt")
    parser.add_argument("--fc", default=None, help="Fortran compiler (default $FC or gfortran)")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        receipt = build(Path(args.source), Path(args.out), fc=args.fc, jobs=args.jobs)
    except CrtmBuildError as error:
        print(f"crtm_build: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
