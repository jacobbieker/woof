"""Every module that DECODES NetCDF does it in Rust, and none quietly doesn't.

The first pass converted one module and left a dozen. This pins the
result of finishing the sweep, and it pins it by PARSING rather than by
grepping -- a grep counts the comment that explains an absence, which is
how a removal can look like a no-op.

Four lists, and the point is that all four are explicit:

* :data:`_DECODERS` must not import a Python NetCDF library. These read
  meteorological field arrays, so they go through ``rw_netcdf``.
* :data:`_PLUMBING_OR_BLOCKED` may. Each entry carries the reason in the
  table below.
* :data:`_WORKAROUND_ENGINE` may, and has to prove the netCDF4 half is an
  ESCAPE: the default engine is the project's Rust writer and the environment
  variable that reaches netCDF4 is named in the module.
* :data:`_OPEN_FINDINGS` may, and is the accurate one: a data path that has
  NOT been converted, named with the finding against it.  It is EMPTY
  today -- F1 (wrfinput/wrfbdy) and F2 (`woof spectral score`) of the
  2026-08-18 hidden-scope audit were both closed, by separate lanes, and
  each module moved to the list its new state belongs in.  The list stays
  because the next unconverted read needs somewhere to be DECLARED rather
  than excused, and an empty list says "none open" out loud.

A module may not join any of them silently: the test fails if a module
appears in none, so a new ``import netCDF4`` anywhere in the package has
to be argued for here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from woof import netcdf_bridge

_PACKAGE = Path(netcdf_bridge.__file__).resolve().parent

#: Modules that decode meteorological data. None may import netCDF4.
_DECODERS = {
    "woof/mapped_source.py",
    "woof/era5_direct.py",
    "woof/obs/goes_grid.py",
    "woof/ingest/nest_init.py",
    "woof/ingest/water_overlay.py",
    "woof/obs/target_grid.py",
    "woof/verify/field_metrics.py",
    "woof/verify/profiles.py",
    "woof/verify/spectral_io.py",
    "woof/verify/t0_state_digest.py",
}

#: Modules that may still import netCDF4, each with the reason.
#:
#: "plumbing"  -- reads only attributes/dimensions/identity.
#:
#: "writes" is NOT a reason here.  It used to be -- the estate said
#: "writing is not decoding" -- and the law says "NetCDF read/write", both
#: words.  A module that WRITES gridded NetCDF belongs in
#: :data:`_WORKAROUND_ENGINE` below, which demands the Rust writer be the
#: default and the netCDF4 one be a named escape.
#: "char"      -- needs character/string data; netcrust exposes no
#:                character read at all (upstream gap 3).
#: "char+attrs" -- "char", plus the file's global attributes, because the
#:                bridge cannot see an HDF5 attribute that landed in an
#:                object-header continuation block and answers None for one
#:                instead of failing (upstream gap 4, measured 2026-08-18:
#:                seven global attributes with one large among them is
#:                enough to spill).  Metadata only; every VALUE still
#:                decodes on the bridge.
#: "bytes"     -- needs the STORED representation; the bridge promotes
#:                every numeric type to f64, so a byte-identity or FP32
#:                digest cannot round-trip through it.
#: "tables"    -- sweeps a STATIC asset table variable by variable (the
#:                RRTMGP k-distribution and cloud-optics files) for a
#:                finiteness screen.  Not meteorological data, and the
#:                bridge costs one process launch plus one full f64 temp
#:                file PER VARIABLE, on a door a user runs before a run.
#: "independent-oracle" -- verification-only field reads of the external
#:                reference, kept independent of the production decoder so
#:                a shared decoder defect cannot validate itself.
_PLUMBING_OR_BLOCKED = {
    "woof/certify/pins.py": "plumbing",
    "woof/downscale.py": "plumbing",
    "woof/ensemble/wrfout_inventory.py": "char",
    # The source-orography FIELD read moved to the bridge (audit HIT-5);
    # what is left on netCDF4 here is the RRTMGP asset-table sweep.
    "woof/ingest/preflight.py": "tables",
    # Writes on the Rust classic writer since the 2026-08-18 audit.  What
    # is left is ONE netCDF4 open for the file's metadata: `radar_id` /
    # `radar_valid_time`, and the global attributes, which the bridge
    # silently drops from a legacy HDF5 product (see "char+attrs").
    "woof/obs/radar_grid.py": "char+attrs",
    # The process-wide netCDF4 session lock and the deferred import every
    # other user here takes since 0b680cac1 (D-10): it opens no dataset.
    "woof/io/netcdf_serialization.py": "plumbing",
    # merge_boundary rewrites a wrfbdy_d01 with a longer forcing tail
    # (2.9.2, the forcing-extension door).  Blocked on the reader, not
    # the writer: rw_netcdf's inventory hands attribute values back
    # untyped (an NC_INT FieldType of 104 arrives as the float 104.0),
    # and the rebuilt file must carry WRF's int/float attribute types
    # as the source did.  Converts once the inventory names each
    # attribute's stored type (named gap; owner: the NetCDF bridge).
    "woof/io/wrf_boundary_extension.py": "attrs",
    "woof/offline_child.py": "plumbing",
    "woof/offline_child_run.py": "plumbing",
    "woof/render.py": "plumbing",
    # woof/run_stamp.py and woof/core/rrtmgp.py left this list at D-10
    # (2.8.8: 0b680cac1 and 1a90daab3): both read through the Rust bridge
    # now.
    "woof/verify/cases/real74_d01.py": "plumbing",
    "woof/verify/cases/real74_d02.py": "bytes",
    "woof/verify/cases/real74_n5b.py": "plumbing",
    "woof/verify/cases/real74_n5s.py": "char",
    "woof/verify/obs/cross_reader.py": "char",
    "woof/verify/metem_differential.py": "independent-oracle",
    "woof/wrf_backend_parity.py": "bytes",
    # Declared when the hosted CPU lanes first ran this sweep on the Recast
    # WOOF tree, each read for what its import is used for.
    # Earth2 member admission compares two global attributes per frame.
    "woof/earth2_members.py": "plumbing",
    # Typed global and per-variable attributes, the reason
    # woof/io/wrf_boundary_extension.py gives above.
    "woof/moving_downscale.py": "attrs",
    # "foreign-env": launched in Earth2Studio's own environment (torch,
    # earth2studio, xarray), not in woof's; the bridge is not on that
    # interpreter.
    "woof/earth2_diagnose_runner.py": "foreign-env",
    "woof/earth2_diagnostic_runner.py": "foreign-env",
    # "xarray-api": xarray in and out; the caller hands a Dataset/DataArray
    # or the module opens a Zarr/Icechunk store.  No NetCDF file is decoded.
    "woof/energy/icing_turbine.py": "xarray-api",
    "woof/energy/ramps.py": "xarray-api",
    "woof/ml_export.py": "xarray-api",
}

#: Modules whose netCDF4 import IS an announced workaround engine: the
#: DEFAULT writes through the project's Rust classic writer, and netCDF4 is
#: reachable only by naming an environment variable.
#:
#: This is the category "plumbing" was hiding.  ``module -> the env var
#: that selects the netCDF4 engine``.
_WORKAROUND_ENGINE = {
    "woof/io/wrfout.py": "WOOF_WRFOUT_WRITER",
    "woof/obs/grid_product.py": "WOOF_OBS_GRID_WRITER",
    "woof/wrf_direct.py": "WOOF_WRFINPUT_WRITER",
}

#: Modules with an OPEN, named finding against them: a real data path that
#: has not been converted yet.  Listed so the module is declared as
#: UNCONVERTED rather than excused, and so this suite fails loudly if one
#: appears that nobody has named.
#: Empty is the truthful state as of 2026-08-18: `woof/wrf_direct.py`
#: (audit F1) now writes every wrfinput and wrfbdy field on the Rust
#: classic writer and sits in :data:`_WORKAROUND_ENGINE`, and
#: `woof/verify/spectral_io.py` (audit F2) reads its planes through
#: :mod:`woof.netcdf_bridge` and sits in :data:`_DECODERS`.
#:
#: Not empty in the Recast WOOF tree: the modules below arrived after the
#: sweep (fork commits and the 1.0.x-1.1.x exports) with a netCDF4 field
#: read or write that never went through rw_netcdf or the Rust writer.
#: They are listed as found when the hosted CPU lanes first ran this file,
#: so the open work is named rather than excused.
_OPEN_FINDINGS: dict[str, str] = {
    "woof/core/hub_wind_diag.py": "writes the hub-height wind NetCDF4 product with netCDF4 (e98bbafd5)",
    "woof/core/windfarm_power_diag.py": "writes the per-turbine power NetCDF4 product with netCDF4 (eb0eb03b1)",
    "woof/earth2_run.py": "reads Earth2 member frames with netCDF4 (9c5ebdda9)",
    "woof/energy/contracts.py": "reads woof-energy product files with netCDF4",
    "woof/energy/extract.py": "writes the extracted forecast with netCDF4 and xarray",
    "woof/energy/icechunk_store.py": "reads wrfout frames with netCDF4 into the site store",
    "woof/energy/power_farm.py": "writes and reads the farm power product with netCDF4",
    "woof/energy/products.py": "writes and reads woof-energy products with netCDF4",
    "woof/energy/sample.py": "reads wrfout history fields with netCDF4 (ba5aa9ad3)",
    "woof/energy/sample_hex.py": "reads MPAS history cells with netCDF4",
    "woof/radtran/reptran.py": "reads the REPTRAN gas-optics tables with netCDF4 (cc980b540)",
    "woof/regional_crop.py": "reads and writes the cropped global source with netCDF4 (30a4a3f56)",
    "woof/setup_targets.py": "reads frame fields (XLAT, the target field) with netCDF4",
    "woof/track/splice.py": "reads and rewrites history frames with netCDF4 (d97400bcc)",
    "woof/ukv_source.py": "reads and writes UKV fields with netCDF4 (8929f6032)",
}

#: The hex model folded into the package (woof/hex, from components/hex at
#: b0c71f0a8) reads and writes MPAS meshes and histories with netCDF4
#: throughout; the sweep was written for the regional engine before the
#: fold.  One finding names the whole subtree as unconverted.
_OPEN_FINDING_PREFIXES: dict[str, str] = {
    "woof/hex/": "the folded-in hex model's MPAS mesh and history I/O is netCDF4 throughout",
}


def _declared_open(name: str) -> bool:
    return name in _OPEN_FINDINGS or any(
        name.startswith(prefix) for prefix in _OPEN_FINDING_PREFIXES)

_BANNED = {"netCDF4", "xarray", "h5py", "h5netcdf"}


#: The deferred netCDF4 proxy 0b680cac1 (2.8.8, D-10) made the one
#: spelling for a module that uses netCDF4: ``from
#: woof.io.netcdf_serialization import netCDF4``.  Its call sites still read
#: ``netCDF4.Dataset(...)``, so importing the proxy IS importing netCDF4 for
#: this sweep; read as an import of ``woof`` alone, every user of the
#: library went invisible to it at once.
_DEFERRED_PROXY = ("woof.io.netcdf_serialization", "netCDF4")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
            if node.module == _DEFERRED_PROXY[0] and any(
                    alias.name == _DEFERRED_PROXY[1] for alias in node.names):
                found.add(_DEFERRED_PROXY[1])
    return found


def _relative(path: Path) -> str:
    return path.relative_to(_PACKAGE.parent).as_posix()


def _python_netcdf_importers() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        hits = _imports(path) & _BANNED
        if hits:
            result[_relative(path)] = hits
    return result


def test_no_decoder_module_imports_a_python_netcdf_library():
    """The conversions are real: these modules decode through Rust only."""

    offenders = {
        name: sorted(hits)
        for name, hits in _python_netcdf_importers().items()
        if name in _DECODERS
    }
    assert not offenders, (
        "module(s) that decode meteorological NetCDF still import a Python "
        f"NetCDF library: {offenders}")


def test_every_python_netcdf_importer_is_declared_with_a_reason():
    """A new netCDF4 import has to be argued for, not just added."""

    importers = set(_python_netcdf_importers())
    undeclared = sorted(name for name in importers - _DECODERS
                        - set(_PLUMBING_OR_BLOCKED) - set(_WORKAROUND_ENGINE)
                        if not _declared_open(name))
    assert not undeclared, (
        f"{len(undeclared)} module(s) import a Python NetCDF library and are "
        f"in none of the lists; classify each as a decoder (convert it), an "
        f"announced workaround engine, or give it a reason in "
        f"_PLUMBING_OR_BLOCKED: {undeclared}")


def test_the_declared_exemptions_have_not_gone_stale():
    """An exemption must not outlive the import it excuses."""

    from _woof_snapshot import left_out
    importers = set(_python_netcdf_importers())
    # An exempted module the Recast WOOF tree leaves out by a named rule
    # (tests/_woof_left_out.txt: the real74 verification cases) is absent,
    # not stale; where it ships it is read as usual.
    stale = sorted(name for name in (set(_PLUMBING_OR_BLOCKED)
                                     | set(_WORKAROUND_ENGINE)
                                     | set(_OPEN_FINDINGS)) - importers
                   if not left_out(name))
    assert not stale, (
        f"{len(stale)} module(s) are exempted but no longer import a Python "
        f"NetCDF library; drop them from _PLUMBING_OR_BLOCKED: {stale}")


def test_the_sweep_actually_converted_something():
    """Counts, not a boolean: the population really did shrink.

    v2.3.3 had 29 importers. If this number climbs back up, the sweep is
    being undone one module at a time.

    That 29 counted a tree without the folded-in hex model, whose MPAS I/O
    is an open finding by prefix (73 importers at 1.1.6), and the tree
    outside it has since grown declared importers too (21 at upstream
    1.1.0, 38 now), so the count is held at today's population, every one
    declared: it may fall, and a new importer has to raise it here with its
    reason.
    """

    importers = _python_netcdf_importers()
    declared = (len(_PLUMBING_OR_BLOCKED) + len(_WORKAROUND_ENGINE)
                + len(_OPEN_FINDINGS)
                + sum(1 for name in importers
                      if name not in _OPEN_FINDINGS and _declared_open(name)))
    assert len(importers) <= declared, (
        f"{len(importers)} modules import a Python NetCDF library; only "
        f"{declared} are declared exempt")
    hex_tree = sum(1 for name in importers
                   if any(name.startswith(prefix) for prefix in _OPEN_FINDING_PREFIXES))
    assert hex_tree <= 73 and len(importers) - hex_tree <= 38, (
        f"{len(importers) - hex_tree} importers outside the hex tree (38 when "
        f"this was pinned) and {hex_tree} in it (73): the population climbed")


@pytest.mark.parametrize("module", sorted(_DECODERS))
def test_each_decoder_reaches_the_bridge(module):
    """Each converted module actually references the Rust front door.

    Absence of ``netCDF4`` is only half the claim -- a module that decodes
    nothing at all would also pass that. This asserts the positive.
    """

    path = _PACKAGE.parent / module
    source = path.read_text(encoding="utf-8", errors="replace")
    assert "netcdf_bridge" in source, (
        f"{module} imports no Python NetCDF library, but does not reach "
        f"woof.netcdf_bridge either")


@pytest.mark.parametrize("module", sorted(_WORKAROUND_ENGINE))
def test_a_workaround_engine_module_proves_the_rust_default(module):
    """Its netCDF4 import must be an ESCAPE, not the shipped path.

    "It writes, so it is plumbing" is the sentence that kept two gridded
    observation writers on the C library through a whole release line.  A
    module in this list has to name the environment variable that selects
    netCDF4, resolve to the Rust engine with nothing set, and actually
    reach the Rust writer seam -- otherwise the exemption is just the old
    sentence with a new label.
    """

    path = _PACKAGE.parent / module
    source = path.read_text(encoding="utf-8", errors="replace")
    env = _WORKAROUND_ENGINE[module]
    assert f'"{env}"' in source, (
        f"{module} is exempted as a workaround engine but never names "
        f"{env}, so nothing tells a reader how the netCDF4 half is reached")
    assert '"rust"' in source, (
        f"{module} does not name a 'rust' engine, so it cannot be defaulting "
        f"to one")
    assert "nc_writer_bridge" in source, (
        f"{module} never reaches woof.io.nc_writer_bridge, so its default "
        f"is not the Rust writer at all")


@pytest.mark.parametrize("module,env", sorted(_WORKAROUND_ENGINE.items()))
def test_a_workaround_engine_resolves_to_rust_with_nothing_set(module, env):
    """The default is proved by RUNNING the resolver, not by reading it.

    The test above reads the source; a module can name "rust", name its
    environment variable and still resolve to netCDF4 with nothing set --
    that is a default-on defect a grep cannot see, and "fixed means
    default" is the ruling it would break.  This one imports the module
    and asks its own resolver, with the escape hatch removed from the
    environment.
    """

    import importlib
    import os

    loaded = importlib.import_module(module[:-3].replace("/", "."))
    resolvers = [value for key, value in vars(loaded).items()
                 if key.startswith("resolve_") and key.endswith("_engine")]
    assert resolvers, (
        f"{module} declares {env} but exposes no resolve_*_engine, so "
        f"nothing here can ask it what a bare default does")
    previous = os.environ.pop(env, None)
    try:
        for resolve in resolvers:
            assert resolve() == "rust", (
                f"{module}: with {env} unset the engine must be rust")
    finally:
        if previous is not None:
            os.environ[env] = previous
