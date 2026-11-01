"""WRF's ``windfarm_opt = 1`` in the running model: ``wind_farm_scheme =
"fitch_wrf"``, the Fitch scheme word for word behind woof.core.wind_farm.

Upstream's wind-farm design owns the switch (``wind_farm_scheme``), the
turbine input (``wind_farms`` / ``wind_farm_table``), the TKE factor
(``windfarm_tke_factor``) and the driver seam (``PhysicsDriver.wind_farm``,
its ``apply_mynn`` in the MYNN slot); ``fitch_wrf`` is the scheme value whose
implementation is this module and :mod:`woof.core.windfarm_fitch`.
:func:`woof.core.wind_farm.attach_wind_farm` builds it through
:func:`build_windfarm` and wraps it as
:class:`woof.core.wind_farm.WrfFitchFarm`.

WRF v4.7.1 calls the Fitch wind-farm drag from its PBL driver, right after
the PBL scheme and before the orographic drag
(``module_pbl_driver.F:2045-2125``)::

    CALL dragforce(ID=id, Z_AT_W=z_at_w, u=u_phy, v=v_phy, DX=dx, DZ=dz8w,
                   DT=dt, QKE=qke, DU=rublten, DV=rvblten,
                   WINDFARM_OPT=windfarm_opt, POWER=power, ...)

so on every PBL call the turbines subtract their thrust from the PBL
momentum tendencies RUBLTEN/RVBLTEN (A-grid, before mass coupling), add
their TKE source into MYNN's QKE as MYNN left it, and overwrite POWER.  This
module holds what that call needs and nothing it computes:

* :class:`WindFarm`, one domain's turbines as a
  :class:`woof.core.windfarm_fitch.FitchLayout` on the device (uploaded
  once, at physics initialisation) plus the ``windfarm_tke_factor``;
  :meth:`WindFarm.apply` is the ``dragforce`` call itself, through
  :func:`woof.core.windfarm_fitch.launch_fitch`, whose kernel is held word
  for word to WRF's Fortran by ``tools/windfarm_fitch_wrf471_oracle``.
* :func:`fitch_layout_for_file`, the turbine file read and placed on the
  domain: a woof turbine table through :mod:`woof.windfarm.layout`
  (library curves, WRF's own placement code), or WRF's own
  ``windturbines.txt`` (``windfarm_ij = 0``) / ``windturbines-ij.txt``
  (``windfarm_ij = 1``) with its ``wind-turbine-N.tbl`` tables, read as
  ``init_module_wind_fitch`` reads them (``module_wind_fitch.F:341-517``).
* :func:`bind_windfarm_identity`, the identity rule for ``fitch_wrf``: the
  turbine file's and its curves' SHA-256 bind beside the path (off, every
  wind-farm key leaves the document through
  :func:`woof.wind_table.drop_inert_config`).
* :func:`windfarm_array_shapes`, the device arrays the preflight prices.

THE ARGUMENTS WRF PASSES.  ``u_phy``/``v_phy`` are phy_prep's mass-point
winds, the PBL slot's ``atmosphere["u"]``/``["v"]``; ``z_at_w`` is
``(phb + ph)/g`` and ``dz8w`` its first difference
(``module_big_step_utilities_em.F:4869-4877``), ``atmosphere
["z_interface"]`` and ``["dz"]`` (the kernel forms the difference itself,
in the same order); ``dx`` is the domain's grid spacing, the cell area
``dx*dx``.  ``DT=dt`` is the MODEL time step, not the PBL interval
``dtbl`` the PBL schemes are handed: with ``bldt > 0`` WRF adds one model
step's worth of TKE per PBL call while the momentum tendencies are held for
the whole interval, and woof does the same.  WRF's ``rho`` is not an
argument: the scheme divides the thrust by the layer's air mass, so the
tendencies need no density.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping

import numpy as np

from woof.config import (WINDFARM_TABLE_SUFFIXES, WRF_FITCH_SCHEME,
                         wrf_turbine_list_ij)

F = np.float32

#: The identity keys :func:`bind_windfarm_identity` adds beside the path.
TURBINE_FILE_IDENTITY_KEY = "windfarm_turbine_sha256"
CURVES_IDENTITY_KEY = "windfarm_curves_sha256"
#: module_wind_fitch.F:54-55: at most 100 turbine types of 100 rows each.
MAXVALS = 100
MAXVALS2 = 100


class WindFarmRefused(ValueError):
    """A turbine file cannot be read or placed on this domain."""


# --------------------------------------------------------------------------
# WRF's list-directed reads


def _check_token(token: str, where: str) -> None:
    """Refuse the list-directed forms this reader does not interpret.

    Only tokens a READ consumes reach here: the rest of a record, which
    Fortran skips (a trailing comment, a units note), is never looked at.
    """
    if "*" in token:
        raise WindFarmRefused(
            f"{where}: {token!r} is a repeat count (r*c); write every value "
            "out in full")
    if "/" in token:
        raise WindFarmRefused(
            f"{where}: {token!r} holds a slash, which ends a list-directed "
            "READ before all its values are read")
    if "_" in token:
        raise WindFarmRefused(
            f"{where}: {token!r} is not a Fortran number (no digit "
            "separators)")


def _f32_decimal(token: str, where: str) -> np.float32:
    """A decimal as gfortran's list-directed READ of a REAL makes it.

    libgfortran converts a REAL(4) item with ``strtof``, which glibc
    rounds correctly (to nearest, ties to even) from the decimal itself.
    Going through binary64 first can round twice, so the nearest float32
    is chosen here from the exact rational value.
    """
    _check_token(token, where)
    text = token.strip().replace("d", "e").replace("D", "e")
    try:
        exact = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise WindFarmRefused(f"{where}: {token!r} is not a number") from None
    try:
        guess = F(float(exact))
    except OverflowError:
        guess = F(np.inf)
    if not np.isfinite(guess):
        raise WindFarmRefused(f"{where}: {token!r} overflows a REAL")
    best = guess
    for candidate in (np.nextafter(guess, F(-np.inf)),
                      np.nextafter(guess, F(np.inf))):
        if not np.isfinite(candidate):
            continue
        here = abs(Fraction(float(candidate)) - exact)
        there = abs(Fraction(float(best)) - exact)
        if here < there or (here == there and
                            int(candidate.view(np.uint32)) % 2 == 0):
            best = candidate
    return F(best)


def _records(path: Path) -> list[list[str]]:
    """The file's records as list-directed value tokens."""
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError:
        raise WindFarmRefused(f"{path} is not an ASCII text file") from None
    return [line.replace(",", " ").split() for line in text.splitlines()]


class _ListRead:
    """Sequential list-directed READs over a file's records.

    Each READ starts at a new record and may continue across records until
    it has its items; the rest of its last record is skipped -- Fortran's
    rule, which is how ``init_module_wind_fitch`` reads the tables.
    """

    def __init__(self, path: Path):
        self.path = path
        self.records = _records(path)
        self.at = 0

    def read(self, count: int, what: str) -> list[str]:
        out: list[str] = []
        while len(out) < count:
            if self.at >= len(self.records):
                raise WindFarmRefused(
                    f"{self.path}: the file ends before {what}")
            record = self.records[self.at]
            self.at += 1
            out.extend(record[:count - len(out)])
        return out


def _int(token: str, where: str) -> int:
    _check_token(token, where)
    try:
        return int(token)
    except ValueError:
        raise WindFarmRefused(f"{where}: {token!r} is not an integer") from None


@dataclass(frozen=True)
class WrfTurbineTable:
    """One ``wind-turbine-N.tbl``: header and rows, as REAL words."""

    kind: int
    hub_height_m: np.float32
    diameter_m: np.float32
    standing_thrust: np.float32
    nominal_power_mw: np.float32
    speeds_ms: np.ndarray
    thrust_coefficient: np.ndarray
    power_kw: np.ndarray


def read_wrf_table(path: Path, kind: int) -> WrfTurbineTable:
    """``module_wind_fitch.F:497-504``: nval, the header, nval rows."""
    reader = _ListRead(path)
    nval = _int(reader.read(1, "nval")[0], f"{path} nval")
    if not 2 <= nval <= MAXVALS2:
        raise WindFarmRefused(
            f"{path}: nval = {nval}; WRF holds 2..{MAXVALS2} rows "
            "(MAXVALS2) and the drag interpolates between two of them")
    header = [_f32_decimal(t, f"{path} header") for t in
              reader.read(4, "the hubheight/diameter/stc/npower header")]
    rows = np.array([[_f32_decimal(t, f"{path} row {k + 1}")
                      for t in reader.read(3, f"row {k + 1}")]
                     for k in range(nval)], dtype=F)
    hub, dia, stc, npower = header
    if not (np.isfinite(rows).all() and (np.diff(rows[:, 0]) > 0).all()
            and rows[0, 0] > 0):
        raise WindFarmRefused(
            f"{path}: wind speeds must be positive and ascend: dragcof "
            "divides by the bracketing speeds cubed and searches them in "
            "order")
    if not (dia > 0 and hub - dia / F(2.0) >= 0):
        raise WindFarmRefused(
            f"{path}: hub height {hub} m and rotor diameter {dia} m put the "
            "blade tip below the ground (hub - D/2 < 0) or the rotor has no "
            "size; WRF's level search would leave the bottom level unset")
    return WrfTurbineTable(kind, hub, dia, stc, npower, rows[:, 0].copy(),
                           rows[:, 1].copy(), rows[:, 2].copy())


def turbine_file_kind(path) -> str:
    """``"table"`` for a woof turbine table, ``"wrf"`` for WRF's list.

    A ``.csv``, ``.geojson`` or ``.json`` is a woof table; any other file
    is WRF's list when its first record is all numbers (WRF's lists carry
    no header), and a woof CSV otherwise (``read_turbines`` reads a
    ``.txt`` as CSV with a header row).
    """
    path = Path(path)
    if path.suffix.lower() in WINDFARM_TABLE_SUFFIXES:
        return "table"
    try:
        first = next((r for r in _records(path) if r), [])
    except (OSError, WindFarmRefused):
        return "table"
    if len(first) < 3:
        return "table"
    # The three values a WRF row READ consumes; the rest of the record (a
    # comment, a name) is skipped by Fortran and so is not looked at.
    try:
        [float(t.replace("d", "e").replace("D", "e")) for t in first[:3]]
    except ValueError:
        return "table"
    return "wrf"


def _wrf_rows(path: Path, ij: int) -> list[tuple]:
    """``init_module_wind_fitch``'s turbine list: one READ per record."""
    records = _records(path)
    if any(not r for r in records):
        # WRF counts records to size its arrays (:98-104) and then reads
        # list-directed past a blank one, so a blank line ends the read in
        # an end-of-file error.
        raise WindFarmRefused(
            f"{path} has a blank line; WRF counts every record as a turbine "
            "and would stop at end of file.  Remove it")
    if not records:
        raise WindFarmRefused(f"{path} lists no turbines")
    rows = []
    for n, record in enumerate(records, start=1):
        where = f"{path} turbine {n}"
        if len(record) < 3:
            raise WindFarmRefused(
                f"{where}: {'i j' if ij else 'lat lon'} type needs three "
                "values on its record")
        if ij:
            rows.append((_int(record[0], where), _int(record[1], where),
                         _int(record[2], where)))
        else:
            rows.append((_f32_decimal(record[0], where),
                         _f32_decimal(record[1], where),
                         _int(record[2], where)))
    return rows


def _list_kinds(path: Path) -> list[int]:
    """The turbine types a WRF list names, in ascending order: the third
    value of every row, as the row READ takes it."""
    kinds = set()
    for n, record in enumerate(_records(path), start=1):
        if len(record) >= 3:
            kinds.add(_int(record[2], f"{path} turbine {n}"))
    return sorted(kinds)


def wrf_table_paths(path) -> dict[int, Path]:
    """``{type: wind-turbine-<type>.tbl}`` for a WRF list, beside it."""
    path = Path(path)
    return {kind: path.parent / f"wind-turbine-{kind}.tbl"
            for kind in _list_kinds(path)}


def _is_cells(grid) -> bool:
    """True for an unstructured mesh's columns
    (:class:`woof.windfarm.cells.CellPlacement`), not a projected grid."""
    from woof.windfarm.cells import CellPlacement

    return isinstance(grid, CellPlacement)


def _ij_on_cells(path) -> WindFarmRefused:
    return WindFarmRefused(
        f"windfarm_ij = 1 with {path}: i/j turbine rows index a WRF grid, "
        "and an unstructured mesh's columns have no i/j.  List the turbines "
        "by lat/lon (windfarm_ij = 0); each goes to the cell whose centre "
        "is nearest")


def _place(rows, ij: int, *, grid, shape, path) -> list[tuple[int, int]]:
    """0-based (j, i) of each turbine; refused outside or on the edge."""
    ny, nx = shape
    cells = []
    if grid is not None and _is_cells(grid):
        # An unstructured mesh: the nearest cell centre (woof.windfarm.cells),
        # refused outside the mesh and in its boundary columns.
        from woof.windfarm.cells import CellPlacementRefused, place_on_cells
        try:
            columns = place_on_cells(
                grid, np.array([r[0] for r in rows], F).astype(np.float64),
                np.array([r[1] for r in rows], F).astype(np.float64),
                [f"turbine {n}" for n in range(1, len(rows) + 1)])
        except CellPlacementRefused as error:
            raise WindFarmRefused(f"{path}: {error}") from error
        return [(0, int(c)) for c in columns]
    if ij:
        cells = [(j - 1, i - 1) for i, j, _ in rows]
    else:
        if grid is None:
            raise WindFarmRefused(
                f"{path} places turbines by lat/lon (windfarm_ij = 0) and "
                "this route hands the physics no projected grid to place "
                "them on")
        from woof.windfarm.layout import wrf_cells
        from woof.windfarm.wrf_llij import (WrfProjection,
                                            WrfProjectionRefused)
        try:
            projection = WrfProjection.for_grid(grid)
        except WrfProjectionRefused as error:
            raise WindFarmRefused(
                f"{path}: {error}; WRF's wind-farm placement has no other "
                "projection") from error
        _, _, jj, ii = wrf_cells(
            projection, np.array([r[0] for r in rows], F),
            np.array([r[1] for r in rows], F))
        cells = [(int(j), int(i)) for j, i in zip(jj, ii)]
    outside = [n + 1 for n, (j, i) in enumerate(cells)
               if not (0 <= i < nx and 0 <= j < ny)]
    edge = [n + 1 for n, (j, i) in enumerate(cells)
            if n + 1 not in outside and (i in (0, nx - 1) or j in (0, ny - 1))]
    if outside or edge:
        names = ", ".join(str(n) for n in (outside + edge)[:12])
        raise WindFarmRefused(
            f"{path}: turbine(s) {names} lie outside this {nx} x {ny} domain "
            "or in its outermost row or column.  Drop them, give this domain "
            "its own [[domain]] wind_farms, or set wind_farm_scheme = "
            "\"off\" on it.  (WRF skips a turbine outside a domain; woof refuses "
            "it, so a mistyped coordinate cannot drop a turbine unseen, and "
            "the edge cells are boundary-forced, not modelled.)")
    return cells


def wrf_fitch_layout(path, *, ij: int, shape, grid=None):
    """WRF's own turbine list and tables as a :class:`FitchLayout`.

    The list is ``windturbines.txt`` (``lat lon type``, ``windfarm_ij = 0``)
    or ``windturbines-ij.txt`` (``i j type``, ``windfarm_ij = 1``); each
    type ``N`` reads ``wind-turbine-N.tbl`` from the list's directory.  A
    cell's turbines act in list order, so each cell's groups are its runs of
    consecutive turbines of one type.  Returns ``(layout, table_paths)``.
    """
    from woof.core.windfarm_fitch import FitchLayout

    path = Path(path)
    if not path.is_file():
        raise WindFarmRefused(f"turbine list {path} does not exist")
    if ij and grid is not None and _is_cells(grid):
        raise _ij_on_cells(path)
    rows = _wrf_rows(path, ij)
    paths = wrf_table_paths(path)
    kinds = sorted(paths)
    bad = [k for k in kinds if not 1 <= k <= MAXVALS]
    if bad:
        raise WindFarmRefused(
            f"{path}: turbine type(s) {bad} outside 1..{MAXVALS}; WRF indexes "
            "its tables by type (turbws(nkind, :), MAXVALS)")
    tables = {}
    for kind in kinds:
        table_path = paths[kind]
        if not table_path.is_file():
            raise WindFarmRefused(
                f"{path} lists type {kind} and {table_path} does not exist")
        tables[kind] = (table_path, read_wrf_table(table_path, kind))
    cells = _place(rows, ij, grid=grid, shape=shape, path=path)
    ny, nx = shape
    curve_of = {kind: n for n, kind in enumerate(kinds)}
    group_cell, group_count, group_kind = [], [], []
    last_in_cell: dict[int, int] = {}
    for (j, i), (*_, kind) in zip(cells, rows):
        cell = j * nx + i
        g = last_in_cell.get(cell)
        if g is not None and group_kind[g] == kind:
            group_count[g] += 1
            continue
        last_in_cell[cell] = len(group_cell)
        group_cell.append(cell)
        group_count.append(1)
        group_kind.append(kind)
    width = max(t.speeds_ms.size for _, t in tables.values())
    speeds = np.zeros((len(kinds), width), dtype=F)
    thrust = np.zeros_like(speeds)
    power = np.zeros_like(speeds)
    for kind, (_, t) in tables.items():
        n = t.speeds_ms.size
        c = curve_of[kind]
        speeds[c, :n] = t.speeds_ms
        thrust[c, :n] = t.thrust_coefficient
        power[c, :n] = t.power_kw
    layout = FitchLayout(
        ny=ny, nx=nx, cell=np.array(group_cell, np.int64),
        count=np.array(group_count, np.int32),
        curve=np.array([curve_of[k] for k in group_kind], np.int32),
        hub_height_m=np.array([tables[k][1].hub_height_m
                               for k in group_kind], F),
        rotor_diameter_m=np.array([tables[k][1].diameter_m
                                   for k in group_kind], F),
        speeds_ms=speeds, thrust_coefficient=thrust, power_kw=power,
        nval=np.array([tables[k][1].speeds_ms.size for k in kinds], np.int32),
        standing_thrust=np.array([tables[k][1].standing_thrust
                                  for k in kinds], F))
    return layout, [tables[k][0] for k in kinds]


def fitch_layout_from_layout(layout):
    """A :class:`woof.windfarm.layout.WindFarmLayout` as the kernel's
    :class:`FitchLayout` (same groups, same WRF tables, same words)."""
    from woof.core.windfarm_fitch import FitchLayout

    curves = layout.kernel_curves()
    return FitchLayout(
        ny=layout.ny, nx=layout.nx, **layout.kernel_groups(),
        **{name: curves[name] for name in (
            "speeds_ms", "thrust_coefficient", "power_kw", "nval",
            "standing_thrust")})


def fitch_layout_for_file(path, *, ij: int = 0, shape, grid=None,
                          grid_id: int = 1):
    """The kernel layout of one domain's turbine file.

    ``shape`` is the domain's mass grid (ny, nx); ``grid`` its projected
    grid (needed for lat/lon placement), or an unstructured mesh's columns
    (:class:`woof.windfarm.cells.CellPlacement`, shape ``(1, n_columns)``),
    where each turbine goes to the nearest cell centre.  Every refusal is
    the reader's or the layout's own, by name.
    """
    path = Path(path)
    if not path.is_file():
        raise WindFarmRefused(f"wind_farms {path} does not exist")
    if turbine_file_kind(path) == "wrf":
        layout, _ = wrf_fitch_layout(path, ij=ij, shape=shape, grid=grid)
        return layout
    if ij:
        if grid is not None and _is_cells(grid):
            raise _ij_on_cells(path)
        raise WindFarmRefused(
            f"{path}: i/j rows (WRF windfarm_ij = 1) are read from a WRF "
            "windturbines-ij.txt only; a woof turbine table is lat/lon")
    if grid is not None and _is_cells(grid):
        from woof.windfarm.layout import (WindFarmLayoutRefused,
                                          build_cell_layout, read_turbines)
        try:
            built = build_cell_layout(read_turbines(path), grid,
                                      grid_id=grid_id)
        except WindFarmLayoutRefused as error:
            raise WindFarmRefused(str(error)) from error
        if (built.ny, built.nx) != tuple(shape):
            raise WindFarmRefused(
                f"{path} was placed on {built.nx} columns and this state "
                f"carries {shape[1]}")
        return fitch_layout_from_layout(built)
    if grid is None:
        raise WindFarmRefused(
            f"{path} places turbines by lat/lon and this route hands the "
            "physics no projected grid to place them on")
    from woof.windfarm.layout import (WindFarmLayoutRefused, build_layout,
                                      read_turbines)
    try:
        built = build_layout(read_turbines(path), grid, grid_id=grid_id)
    except WindFarmLayoutRefused as error:
        hint = ""
        if "outside" in str(error) or "outermost" in str(error):
            hint = ("  Give this domain its own [[domain]] "
                    "wind_farms, or set wind_farm_scheme = \"off\" on it.")
        raise WindFarmRefused(str(error) + hint) from error
    if (built.ny, built.nx) != tuple(shape):
        raise WindFarmRefused(
            f"{path} was placed on a {built.nx} x {built.ny} grid and this "
            f"state's columns are {shape[1]} x {shape[0]}")
    return fitch_layout_from_layout(built)


# --------------------------------------------------------------------------
# identity


#: The digests of the bytes each layout was built from, by resolved path:
#: what the running model holds, whatever the file on disk says later.
_BUILT: dict[str, tuple[str, str]] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def windfarm_input_digests(path) -> tuple[str, str]:
    """``(turbine file sha256, curves sha256)`` of one turbine file.

    The curves are the WRF tables a WRF list reads (each type's number and
    table sha256, in type order, through :func:`wrf_table_paths`, the
    reader's own) or the turbine library a woof table resolves its curves
    through.  Read afresh every call -- the files are kilobytes, and a
    cache keyed on stat would miss an edit inside one timestamp tick.
    """
    source = Path(path).expanduser().resolve()
    wrf = turbine_file_kind(source) == "wrf"
    tables = {}
    if wrf:
        try:
            tables = wrf_table_paths(source)
        except WindFarmRefused:
            tables = {}
    if wrf:
        curves = {str(kind): _sha256(table) if table.is_file() else "missing"
                  for kind, table in tables.items()}
        curves_digest = hashlib.sha256(json.dumps(
            curves, sort_keys=True).encode()).hexdigest()
    else:
        from woof.energy.turbine_library import load_turbine_library
        curves_digest = load_turbine_library().sha256
    return (_sha256(source), curves_digest)


def bind_windfarm_identity(values) -> dict:
    """Apply the ``fitch_wrf`` identity rule to a run document, in place.

    Another scheme (or off): the two digests leave, so the document is the
    one upstream's rule (woof.wind_table.drop_inert_config) writes.
    ``fitch_wrf``: the turbine file's and its curves' SHA-256 bind beside
    the path (``wind_farm_table``).  A saved digest stays saved --
    digesting a stored checkpoint must not replace the bytes it ran with by
    the file now at that path, which is what lets a resume refuse a file
    changed under it.
    """
    if values.get("wind_farm_scheme", "off") != WRF_FITCH_SCHEME:
        for key in (TURBINE_FILE_IDENTITY_KEY, CURVES_IDENTITY_KEY):
            values.pop(key, None)
        return values
    path = (values.get("wind_farm_table", "") or values.get("wind_farms", "")
            or "")
    if path and TURBINE_FILE_IDENTITY_KEY not in values:
        try:
            # The bytes the running model was built from, when it was; the
            # file on disk otherwise (a fingerprint before any build).
            built = _BUILT.get(str(Path(path).expanduser().resolve()))
            values[TURBINE_FILE_IDENTITY_KEY], values[CURVES_IDENTITY_KEY] = \
                built if built is not None else windfarm_input_digests(path)
        except OSError:
            # Absent here is refused where the layout is built; a document
            # is still digestable so the refusal names the file, not this.
            values[TURBINE_FILE_IDENTITY_KEY] = "missing"
            values[CURVES_IDENTITY_KEY] = "missing"
    return values


# --------------------------------------------------------------------------
# the carrier


@dataclass
class WindFarm:
    """One domain's Fitch turbines on the device, and the call WRF makes.

    Built once by :func:`build_windfarm`; it holds only the uploaded layout
    and two settings, never a value a step writes (POWER lives in the
    driver's serialized ``fields``), so a resumed run rebuilds it exactly.
    """

    layout: object
    tke_factor: float
    #: An unstructured mesh's per-column ``wfdensity`` (1/area, float32, on
    #: the device; :func:`woof.core.windfarm_fitch.cell_wfdensity`), or
    #: ``None`` on a WRF grid, whose cells are all ``dx*dx``.
    wfdensity: object | None = None

    def apply(self, atmosphere: Mapping[str, object], du, dv, *, qke, power,
              dx: float, dt: float) -> None:
        """``dragforce``: add the rotors' drag to ``du``/``dv`` and their
        TKE to ``qke`` in place, and write ``power`` (W per cell).

        On an unstructured mesh ``dx`` is not read: each column's own area
        takes WRF's ``dx*dx`` (``fitch_column_cells``)."""
        from woof.core.windfarm_fitch import launch_fitch

        if self.wfdensity is not None:
            launch_fitch(atmosphere["u"], atmosphere["v"],
                         atmosphere["z_interface"], self.layout, du, dv,
                         wfdensity=self.wfdensity, qke=qke, dt=dt,
                         tke_factor=self.tke_factor, power=power)
            return
        launch_fitch(atmosphere["u"], atmosphere["v"],
                     atmosphere["z_interface"], self.layout, du, dv,
                     dx=dx, qke=qke, dt=dt, tke_factor=self.tke_factor,
                     power=power)



def wrf_fitch_selected(cfg) -> bool:
    """Whether ``cfg`` (a RunConfig or a run mapping) runs ``fitch_wrf``."""
    get = cfg.get if isinstance(cfg, Mapping) else (
        lambda key, default=None: getattr(cfg, key, default))
    return get("wind_farm_scheme", "off") == WRF_FITCH_SCHEME


def turbine_file_of(cfg) -> str:
    """The ``fitch_wrf`` turbine file: ``wind_farm_table`` (which RunConfig
    sets from ``wind_farms``), else ``wind_farms``."""
    get = cfg.get if isinstance(cfg, Mapping) else (
        lambda key, default=None: getattr(cfg, key, default))
    return str(get("wind_farm_table", "") or get("wind_farms", "") or "")


def build_windfarm(cfg, *, shape, grid=None) -> WindFarm:
    """The domain's :class:`WindFarm`, its layout uploaded once.

    The file's and its curves' digests are taken as the layout is built and
    kept by path, so a checkpoint names the bytes the model runs, not a
    file edited under it later.
    """
    turbine_file = turbine_file_of(cfg)
    path = Path(turbine_file).expanduser().resolve()
    before = windfarm_input_digests(path) if path.is_file() else None
    layout = fitch_layout_for_file(
        turbine_file, ij=wrf_turbine_list_ij(turbine_file), shape=shape,
        grid=grid, grid_id=int(getattr(cfg, "grid_id", 1) or 1))
    after = windfarm_input_digests(path)
    if before is not None and before != after:
        raise WindFarmRefused(
            f"{path} or its tables changed while the layout was read; "
            "build again from settled files")
    _BUILT[str(path)] = after
    layout.device_arrays()
    wfdensity = None
    if grid is not None and _is_cells(grid):
        import cupy as cp
        from woof.core.windfarm_fitch import cell_wfdensity
        wfdensity = cp.asarray(cell_wfdensity(grid.area_m2))
    return WindFarm(layout=layout, tke_factor=float(cfg.windfarm_tke_factor),
                    wfdensity=wfdensity)


# --------------------------------------------------------------------------
# upstream's seam: the carrier woof.core.wind_farm attaches


class WrfFitchFarm:
    """``fitch_wrf``: WRF v4.7.1's ``dragforce`` call behind upstream's seam.

    Holds the fork's :class:`woof.core.windfarm_coupling.WindFarm` (the
    uploaded layout, the TKE factor and an unstructured mesh's per-column
    density) and WRF's POWER plane, which lives in the driver's serialized
    ``fields`` so a resumed run carries it.  :meth:`apply_mynn` is the call
    module_pbl_driver.F:2045-2125 makes right after MYNN and before the
    orographic drag: the turbines' drag into the slot's A-grid ``du``/``dv``,
    their TKE into MYNN's QKE as MYNN left it (WRF adds to QKE alone), and
    POWER overwritten.  ``DT=dt`` is the MODEL step (module_pbl_driver.F:
    2056), not the PBL interval: with ``bldt > 0`` WRF adds one model
    step's TKE per PBL call, so the live ``cfg.dt`` (an adaptive clock
    rewrites it per step) is read at each call.
    """

    scheme = "fitch_wrf"

    def __init__(self, farm, power, cfg, *, turbine_file):
        self.farm = farm
        self.power = power
        self.dx = float(cfg.dx)
        self.dt = float(cfg.dt)
        self.turbine_file = str(turbine_file)

    def sample(self, atmosphere, fields):
        """WRF writes POWER only inside dragforce; the initial frame keeps
        the zero plane it was allocated with."""
        return None

    def apply_mynn(self, atmosphere, fields, rates, *, step_seconds=None, cfg=None):
        from woof.core.physics import _pbl_momentum_targets

        du, dv = _pbl_momentum_targets(rates, "wind_farm_scheme = fitch_wrf")
        dt = float(cfg.dt) if cfg is not None else self.dt
        self.farm.apply(atmosphere, du, dv, qke=fields["qke"], power=self.power,
                        dx=self.dx, dt=dt)

    def history_fields(self):
        """WRF's POWER (Registry.EM_COMMON:413, package fitchscheme)."""
        return {"POWER": self.power}

    def write_replay(self, path):
        return None

    def receipt(self):
        layout = self.farm.layout
        try:
            turbine_sha256, curves_sha256 = windfarm_input_digests(self.turbine_file)
        except OSError:
            turbine_sha256 = curves_sha256 = "missing"
        return dict(schema="gpuwm-wind-receipt.v1", scheme=self.scheme,
                    implementation="WRF v4.7.1 module_wind_fitch.F word for word "
                                   "(woof.core.windfarm_fitch)",
                    turbine_file=self.turbine_file,
                    turbine_count=int(np.asarray(layout.count).sum()),
                    tke_factor=float(self.farm.tke_factor),
                    unequal_area_columns=self.farm.wfdensity is not None,
                    windfarm_turbine_sha256=turbine_sha256,
                    windfarm_curves_sha256=curves_sha256)


def attach_wrf_fitch(state, cfg, fields, grid):
    """``fitch_wrf`` on one domain: the layout placed and uploaded once.

    ``grid`` is the domain's projected grid (lat/lon placement needs it) or
    an unstructured mesh's columns (:class:`woof.windfarm.cells.
    CellPlacement`).  Refused by name when the route hands a grid for other
    columns than this state's (a streamed slab, a tile, a rank's window):
    the layout's cell indices are whole-domain columns.
    """
    import cupy as cp
    from woof.core.state import DTYPE

    shape = tuple(int(n) for n in state.mup.shape)
    turbine_file = turbine_file_of(cfg)
    where = (f'wind_farm_scheme = "fitch_wrf" on grid_id = {cfg.grid_id} '
             f"(wind_farms = {turbine_file!r})")
    if grid is None:
        raise ValueError(
            f"{where}: this initialization route hands the physics no "
            "projected grid, so the turbines cannot be placed and nothing "
            "says this state holds the whole domain.  [tiles] / streamed "
            "tile buffers, the offline child, the WRF-input route and the "
            "idealized harnesses are such routes; the wind farm runs on a "
            "resident real-case domain (woof go, the prepared single-domain "
            "and domain-tree forecasts without [tiles]), which passes the "
            "domain's grid.")
    elif _is_cells(grid):
        # An unstructured mesh's columns (woof.windfarm.cells.CellPlacement,
        # hex's generic column seam): one row of columns, each its own area.
        if tuple(grid.shape) != shape:
            raise ValueError(
                f"{where}: this state's columns are {shape} and the mesh "
                f"placement holds {tuple(grid.shape)}; the wind farm's cells "
                "are whole-mesh columns.")
    else:
        grid_shape = (int(grid.e_sn) - 1, int(grid.e_we) - 1)
        if grid_shape != shape:
            raise ValueError(
                f"{where}: this state's columns are {shape} and the domain's "
                f"mass grid is {grid_shape} (a streamed slab, a tile or a "
                "rank's window).  The wind farm's cells are whole-domain "
                "columns, so it runs on a resident single-device domain "
                "only; run without [tiles] / streaming, on one device.")
    try:
        farm = build_windfarm(cfg, shape=shape, grid=grid)
    except WindFarmRefused as error:
        raise ValueError(f"{where}: {error}") from error
    fields["power"] = cp.zeros(shape, dtype=DTYPE)
    wrapped = WrfFitchFarm(farm, fields["power"], cfg, turbine_file=turbine_file)
    detection = dict(requested="fitch_wrf", enabled=True,
                     reason="WRF v4.7.1 Fitch turbines placed on this domain",
                     turbine_count=int(np.asarray(farm.layout.count).sum()))
    return wrapped, detection


# --------------------------------------------------------------------------
# pricing


def windfarm_array_shapes(cfg) -> dict[str, tuple[int, ...]]:
    """The device arrays ``fitch_wrf`` holds for the run.

    ``fields/power`` (WRF's POWER, serialized) and the layout's
    ``cell_offsets`` (ny*nx + 1 int32) are exact.  The group and curve
    arrays depend on the file; they are priced from it when it can be read
    (five 4-byte words per turbine bounds five per group, and a curve is
    three MAXVALS2-row tables plus two words), and at one group per column
    and one curve otherwise.
    """
    if not wrf_fitch_selected(cfg):
        return {}
    ny, nx = int(cfg.ny), int(cfg.nx)
    turbines, curves = ny * nx, 1
    path = turbine_file_of(cfg)
    try:
        source = Path(path)
        if turbine_file_kind(source) == "wrf":
            rows = [r for r in _records(source) if r]
            turbines = len(rows)
            curves = len({r[-1] for r in rows})
        else:
            from woof.windfarm.layout import read_turbines
            found = read_turbines(source)
            turbines = len(found)
            curves = len({t.curve for t in found}) or 1
    except Exception:  # noqa: BLE001 - priced at the bound instead
        pass
    return {
        "fields/power": (ny, nx),
        "windfarm/cell_offsets": (ny * nx + 1,),
        "windfarm/groups": (5, max(int(turbines), 1)),
        "windfarm/curves": (curves, 3 * MAXVALS2 + 2),
    }


__all__ = [
    "CURVES_IDENTITY_KEY", "TURBINE_FILE_IDENTITY_KEY",
    "WindFarm", "WindFarmRefused", "WrfFitchFarm", "attach_wrf_fitch",
    "WrfTurbineTable", "bind_windfarm_identity", "build_windfarm",
    "fitch_layout_for_file", "fitch_layout_from_layout", "read_wrf_table",
    "turbine_file_kind", "turbine_file_of", "windfarm_array_shapes",
    "windfarm_input_digests", "wrf_fitch_layout", "wrf_fitch_selected",
    "wrf_table_paths",
]
