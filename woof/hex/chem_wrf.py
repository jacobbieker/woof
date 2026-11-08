"""WRF-Chem column processes on the hex WOOF route: a chemistry config's ``wrf_chem`` block.

``--chemistry-config FILE`` whose ``gpuwm-hex.chemistry/v1`` document carries a ``wrf_chem`` block
(instead of ``species``) runs the WRF path's own chemistry on the hex mesh:

* the species set is WRF-Chem's, chosen as on the WRF namelist door: ``chem_opt`` through the set files'
  ``wrf_chem_opt`` (woof/data/chem/sets/*.json; 401 -> ``dust``), or ``chem_sets`` by name;
* the carried ``&chem`` knobs keep their WRF names and meanings (``dust_opt``, ``seas_opt``, ``chemdt``,
  ``wetscav_onoff``, ``vertmix_onoff``, ...) and pass through :func:`woof.config.validate_chem_config` on
  the column batch's own RunConfig, so the chem law the WRF path enforces is enforced here too;
* the rows are appended hex scalars, carried in their WRF units (``ug kg-1``, ``ppmv``) by the hex
  monotonic scalar FCT -- the mesh-native horizontal transport -- and handed to the processes as
  zero-copy ``(nz, 1, ncells)`` views;
* after each step's microphysics (WRF's ``solve_em`` then ``chem_driver`` order) the WRF process modules
  run in :data:`woof.core.chem_context.CHEM_STEP_ORDER` on a WRF-shaped column state.  The hex physics is
  the WRF PhysicsDriver on ``ny = 1`` columns (woof.core.mpas_column_batch), so every process reads the
  same met names (``ust``, ``rmol``, ``exch_h``, ...) and launches the same kernel on the same column bytes
  as on a WRF grid.  ``chem_prep`` (module_chem_utilities.F) is the WRF kernel itself; the one hex
  difference is the wind: MPAS winds are already A-grid, so ``u_phy``/``v_phy`` are the reconstructed
  cell winds rather than a destaggering of C-grid faces.

The mass ledger is the WRF driver's (every process's change filed under its declared bucket, the rest
under ``transport``), with hex layer mass ``dryrho * dz8w * areaCell`` in place of ARW's ``mu * dnw``.

Processes that need a source the hex mesh cannot yet sample (GOCART sulfur's oxidant fields, inventory and
fire emission frames, SFIRE) and options whose coupling hex does not yet carry (aerosol-radiation feedback,
aerosol-microphysics coupling, subgrid convective transport, regional chemistry boundaries) are refused by
name rather than run without their input.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODE = "wrf_chem"
#: The process keys hex runs (woof.chem_table.CHEM_PROCESS_MODULES).
PORTED_PROCESSES = frozenset((
    "emission.dust", "emission.seasalt", "emission.inventory", "drydep.wesely", "drydep.gocart",
    "mixing.vertmx", "settling.gocart", "aging.gocart", "wetdep.ls", "optics.gocart",
    "coupling.thompson"))
#: Process keys the hex mesh cannot run yet, and why.
UNPORTED_PROCESSES = {
    "chem.sulfur": "GOCART sulfur reads background oxidant source fields that are not yet remapped onto "
                   "the hex mesh",
    "emission.fire": "fire emission frames are not yet remapped onto the hex mesh",
    "plumerise.freitas": "fire plume rise needs the fire emission frames, not yet remapped onto the hex mesh",
    "emission.sfire": "SFIRE needs the WRF fire grid, which a Voronoi mesh does not have",
}
#: ``&chem`` keys carried with their WRF meaning, and their Python types.
INT_KEYS = ("dust_opt", "seas_opt", "dmsemis_opt", "wetscav_onoff", "vertmix_onoff", "aer_op_opt",
            "kemit", "chem_conv_tr", "aer_drydep_opt", "aer_ra_feedback")
FLOAT_KEYS = ("chemdt", "dust_alpha", "dust_gamma", "dust_smtune", "dust_ustune")
#: Keys the hex route refuses at any value other than WRF's "off", with the reason.
OFF_ONLY_KEYS = {
    "biomass_burn_opt": "fire emission frames are not yet remapped onto the hex mesh",
    "mynn_chem_vertmx": "MYNN's chem mixing is not wired on the hex column route",
}
#: Hex keys that are not WRF ``&chem`` keys.
HEX_KEYS = ("chem_opt", "chem_sets", "geog_root")
#: Every key a wrf_chem block may carry.
KNOWN_KEYS = frozenset(INT_KEYS) | frozenset(FLOAT_KEYS) | frozenset(OFF_ONLY_KEYS) | frozenset(HEX_KEYS)
#: Product suffixes of every transported row.
ROW_SUFFIXES = ("_column_mass", "_surface_concentration")
#: units of every product configure() declared, by name (history_units)
_UNITS = {}
G = 9.81


def _int(block, key, default):
    value = block.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"wrf_chem {key} must be an integer (WRF declares it an integer namelist key), got {value!r}")
    return value


def chem_set(block):
    """The one WOOF chem set the block selects."""
    from woof.chem_table import catalog
    if ("chem_opt" in block) == ("chem_sets" in block):
        raise ValueError("wrf_chem needs exactly one of chem_opt (WRF's package number) or chem_sets "
                         "(a woof/data/chem/sets name)")
    if "chem_opt" in block:
        from woof.namelist_chem import carried_menu, chem_set_for, package_status
        option = _int(block, "chem_opt", 0)
        name = chem_set_for(option, fork=False)
        if name is None:
            raise ValueError(f"wrf_chem chem_opt={option} has no WOOF species set; carried: {carried_menu()}."
                             + package_status(option))
        return name
    name = block["chem_sets"]
    if not isinstance(name, str) or "," in name or name not in catalog().sets:
        raise ValueError(f"wrf_chem chem_sets must name one set of {sorted(catalog().sets)}, got {name!r}")
    return name


def knobs(block):
    """The RunConfig fields the block sets, in WRF's names, type-checked."""
    unknown = sorted(set(block) - KNOWN_KEYS)
    if unknown:
        raise ValueError(f"wrf_chem keys {unknown} are not carried on the hex route; carried: "
                         f"{sorted(KNOWN_KEYS)}")
    out = {}
    for key in INT_KEYS:
        if key in block:
            out[key] = _int(block, key, 0)
    for key in FLOAT_KEYS:
        if key in block:
            value = block[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"wrf_chem {key} must be a finite number, got {value!r}")
            out[key] = float(value)
    for key, reason in OFF_ONLY_KEYS.items():
        if key in block and block[key] not in (0, False):
            raise ValueError(f"wrf_chem {key}={block[key]!r}: {reason}; only WRF's off value is accepted")
    return out


def load_table(raw):
    """(set name, ChemTable, knobs) for a wrf_chem config, refusing what hex cannot run."""
    from woof.chem_table import load_sets
    block = raw["wrf_chem"]
    if not isinstance(block, dict):
        raise ValueError("wrf_chem must be an object")
    if raw.get("inputs"):
        raise ValueError("wrf_chem takes its emissions from the WRF processes; an inputs block (initial "
                         "composition and emission frames) is not remapped onto WRF-Chem rows yet")
    name = chem_set(block)
    values = knobs(block)
    table = load_sets([name])
    for key in table.processes:
        if key in UNPORTED_PROCESSES:
            raise ValueError(f"wrf_chem set {name!r} runs {key}: {UNPORTED_PROCESSES[key]}")
        if key not in PORTED_PROCESSES:
            raise ValueError(f"wrf_chem set {name!r} runs {key}, which the hex route does not carry")
    if table.prescribed:
        raise ValueError(f"wrf_chem set {name!r} has prescribed rows {[r.name for r in table.prescribed]} "
                         "that need source frames on the hex mesh")
    if not table.transported:
        raise ValueError(f"wrf_chem set {name!r} transports no rows")
    return name, table, values


def product_names(table):
    from woof.core.chem_context import chem_processes
    names = [row.name + suffix for row in table.transported for suffix in ROW_SUFFIXES]
    units = {row.name + "_column_mass": "kg m-2" for row in table.transported}
    units.update({row.name + "_surface_concentration": "ug m-3" for row in table.transported})
    for _key, module in chem_processes(table):
        rows = module.rows(table)
        for alloc in getattr(module, "ALLOCATES", ()):
            for name in _output_names(alloc, len(rows)):
                names.append(name)
                units[name] = alloc.units
    for row in table.transported:
        units[row.name] = row.units
    return tuple(names), units


def _output_names(alloc, nrows):
    if alloc.output_name is None or alloc.shape not in ("2d", "rows_2d"):
        return ()
    if alloc.shape == "2d":
        return (alloc.output_name.lower(),)
    if "{n}" not in alloc.output_name:
        return ()
    return tuple(alloc.output_name.format(n=i + 1).lower() for i in range(nrows))


def declare(raw):
    """Register the product units; returns the transported rows."""
    _name, table, _values = load_table(raw)
    _UNITS.update(product_names(table)[1])
    return table.transported


def history_units(name):
    return _UNITS.get(name)


def history_metadata(raw, digest):
    name, table, values = load_table(raw)
    return json.dumps({"schema": "gpuwm-hex.chemistry/v1", "mode": MODE, "config_sha256": digest,
                       "chem_set": name, "knobs": values,
                       "rows": [{"name": r.name, "units": r.units} for r in table.transported],
                       "transport": "hex monotonic scalar FCT (mesh-native)",
                       "processes": list(table.processes)}, sort_keys=True)


def product_slugs(receipt):
    return tuple("var:wrf_" + name for name in receipt["products"])


class HexChemState:
    """``state.chem`` for the column state: the WRF ChemState's surface, no arena.

    The rows live in the hex scalar array, so there is no ``(2, n, nz, ny, nx)`` arena and no time copy
    (no hex process reads one: pre-transport sources are refused).
    """

    def __init__(self, table):
        self.table = table
        self.transported = table.transported
        self.prescribed = table.prescribed
        self.processes = None
        self.initialized = False
        self.ledger_started = True
        #: A mesh has no WRF domain edge: GOCART deposition runs in every column
        #: (woof.core.chem_drydep_aerosol).
        self.no_domain_edge = True

    def fields(self, state):
        return tuple(getattr(state, row.state_attr) for row in self.transported)


def _layer_mass(dryrho, dz8w, area):
    """kg of dry air per layer and cell, float64 (nz, 1, nc)."""
    import cupy as cp
    return dryrho.astype(cp.float64) * dz8w.astype(cp.float64) * area


def _masses(fields, rows, layer):
    """Each row's mass in kg (float64 vector on the device)."""
    import cupy as cp
    from woof.core.chem_driver import _kg_per_unit
    if not fields:
        return cp.zeros(0, cp.float64)
    return cp.stack([cp.sum(f.astype(cp.float64) * layer) * _kg_per_unit(row)
                     for f, row in zip(fields, rows)])


class _RadiationPrep:
    """The chem_prep fields the step's radiation reads: already refreshed from the phase-one columns."""

    def __init__(self, runtime):
        self.runtime = runtime

    def get(self, name):
        return self.runtime.prep.get(name)

    def refresh(self, state):
        if self.runtime.radiation_time != float(state.elapsed_seconds):
            raise RuntimeError("chemistry aerosol optics were not prepared from this step's phase-one columns")


class _RadiationChem:
    """``state.chem`` on the column seam's own state, for woof.core.chem_radiation.prepare."""

    def __init__(self, runtime):
        chem = runtime.state.chem
        self.table, self.transported, self.prescribed = chem.table, chem.transported, chem.prescribed
        self.processes = runtime.processes
        self.start_time = runtime.state.start_time
        self.prep = _RadiationPrep(runtime)

    def fields(self, state):
        return tuple(getattr(state, row.state_attr) for row in self.transported)


class HexWrfChemistry:
    """The WRF chem step on hex columns (see module docstring)."""

    def __init__(self, path, raw, digest, *, cfg, area, nlev, ncells, offset, start_time,
                 fnm, fnp, terrain=None):
        import cupy as cp
        from woof.config import validate_chem_config
        from woof.core.chem_context import (LEDGER_BUCKETS, allocation_shape, chem_processes)
        from woof.core.chem_state import LEDGER_TOTALS, ledger_attr, process_attr
        self.path, self.raw, self.digest = Path(path), raw, digest
        self.set_name, self.table, self.values = load_table(raw)
        self.nlev, self.ncells, self.offset = int(nlev), int(ncells), int(offset)
        nx = self.ncells
        self.cfg = replace(cfg, chem_sets=self.set_name, chem_sources="", aerosol_mp_coupling="none",
                           **self.values)
        validate_chem_config(self.cfg)
        if self.cfg.nz != self.nlev or self.cfg.ny != 1 or self.cfg.nx != nx:
            raise ValueError("chemistry column config and hex column shape differ")
        self.area = cp.asarray(area, dtype=cp.float64).reshape(1, nx)
        self.rows = self.table.transported
        self.state = s = SimpleNamespace()
        s.chem = HexChemState(self.table)
        s.start_time = start_time
        s.chem.start_time = start_time
        s.elapsed_seconds = 0.0
        shape3 = (self.nlev, 1, nx)
        z = lambda *shape: cp.zeros(shape, dtype=cp.float32)
        s.p, s.thp, s.alt = z(*shape3), z(*shape3), z(*shape3)
        s.qv, s.qc, s.qr = z(*shape3), z(*shape3), z(*shape3)
        s.thb, s.phb = z(self.nlev), z(self.nlev + 1)
        s.php, s.w = z(self.nlev + 1, 1, nx), z(self.nlev + 1, 1, nx)
        # chem_prep destaggers C-grid faces; MPAS winds are cell winds, written over u_phy/v_phy.
        s.u, s.v = z(self.nlev, 1, nx + 1), z(self.nlev, 2, nx)
        s.fnm, s.fnp = cp.asarray(fnm, dtype=cp.float32), cp.asarray(fnp, dtype=cp.float32)
        s.use_theta_m = 0
        s.msft = cp.ones((1, nx), dtype=cp.float32)
        s.has_msf = False
        s.ht = z(1, nx) if terrain is None else cp.asarray(terrain, dtype=cp.float32).reshape(1, nx)
        s.nwfa = s.nifa = None
        s.physics = SimpleNamespace(fields={}, microphysics=None)
        self.winds = (z(*shape3), z(*shape3))
        n = len(self.rows)
        for name in (*LEDGER_TOTALS, *LEDGER_BUCKETS):
            setattr(s, ledger_attr(name), cp.zeros(n, dtype=cp.float64))
        setattr(s, ledger_attr("started"), cp.ones(1, dtype=cp.int32))
        self.serialized = [ledger_attr(name) for name in (*LEDGER_TOTALS, *LEDGER_BUCKETS)]
        seen = set()
        for key, module in chem_processes(self.table):
            if getattr(module, "TIMING", "post_step") != "post_step":
                raise ValueError(f"wrf_chem process {key} runs before transport, which hex does not schedule")
            nrows = len(module.rows(self.table))
            for alloc in getattr(module, "ALLOCATES", ()):
                if alloc.name in seen:
                    raise ValueError(f"chem process {key}: array {alloc.name!r} is declared twice")
                seen.add(alloc.name)
                attr = process_attr(alloc)
                setattr(s, attr, cp.zeros(allocation_shape(alloc, nrows, self.nlev, 1, nx), dtype=alloc.dtype))
                if alloc.restart == "serialize":
                    self.serialized.append(attr)
        self.processes = chem_processes(self.table)
        s.chem.processes = self.processes
        self.ddvel = cp.zeros((len(self.table.rows), 1, nx), dtype=cp.float32)
        self.prep = None
        self.pending = None
        self.steps = 0
        self.driver = None
        self.provider_receipt = None
        self.restart_provider_receipt = None
        self.initial_mass_kg = [0.0] * n
        self.radiation_time = None
        self.seam_state = None
        if self.feedback:
            from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
            if rrtmg_variant(self.cfg) != RRTMG_VARIANT_LEGACY:
                raise ValueError("wrf_chem aer_ra_feedback on the hex route feeds the legacy RRTMG the column "
                                 "physics runs; this configuration selects another radiation")

    @property
    def feedback(self):
        return bool(self.cfg.aer_ra_feedback)

    # -- aerosol-radiation feedback ---------------------------------------------------------------
    def bind_seam(self, seam):
        """Hand the column seam's radiation the chemistry (WRF aer_ra_feedback): its RunConfig gains the
        chem keys, its state the rows and optics arrays, exactly as a WRF domain's radiation sees them
        through woof.core.chem_radiation.prepare.  A no-op without feedback."""
        if not self.feedback:
            return
        from woof.core.chem_state import process_attr
        seam._cfg = replace(seam._cfg, chem_sets=self.cfg.chem_sets, chem_sources="",
                            aerosol_mp_coupling="none", aer_ra_feedback=self.cfg.aer_ra_feedback,
                            aer_op_opt=self.cfg.aer_op_opt)
        state = seam._state
        state.chem = _RadiationChem(self)
        for _key, module in self.processes:
            for alloc in getattr(module, "ALLOCATES", ()):
                attr = process_attr(alloc)
                setattr(state, attr, getattr(self.state, attr))
        self.seam_state = state

    def prepare_radiation(self, scalars, columns, *, elapsed_seconds):
        """Before phase one: the rows at the start of the step and chem_prep from the phase-one columns
        (WRF's radiation reads chem at the start of its step)."""
        if not self.feedback:
            return
        if self.seam_state is None:
            raise RuntimeError("aer_ra_feedback needs the column seam bound (bind_seam)")
        self.bind_rows(scalars)
        for row in self.rows:
            setattr(self.seam_state, row.state_attr, getattr(self.state, row.state_attr))
        self.state.elapsed_seconds = float(elapsed_seconds)
        self.refresh(columns)
        self.radiation_time = float(elapsed_seconds)

    # -- column binding ---------------------------------------------------------------------------
    def bind_rows(self, scalars):
        """Point every row's ``chem_<name>`` at its hex scalar row (zero copy)."""
        shape = (self.nlev, 1, self.ncells)
        for i, row in enumerate(self.rows):
            view = scalars[self.offset + i].reshape(shape)
            if view.data.ptr != scalars[self.offset + i].data.ptr:
                raise ValueError("hex chemistry row is not a contiguous scalar plane")
            setattr(self.state, row.state_attr, view)

    def refresh(self, columns, *, physics_fields=None, microphysics=None):
        """WRF chem_prep on hex columns.

        ``columns`` holds ``[level, column]`` float32 arrays: ``pressure``, ``theta`` (dry), ``rho_dry``,
        ``z_interface``, ``qv``, ``qc``, ``qr``, ``u``, ``v`` (cell winds) and ``w`` (interfaces).
        Thermodynamics enter exactly as the column-batch physics seam hands them to the WRF driver:
        ``alt = 1/rho_dry`` and ``php = z_interface * g`` with zero base state.
        """
        import cupy as cp
        from woof.core.chem_prep import ChemPrep
        s, nz, nc = self.state, self.nlev, self.ncells
        mass, iface = (nz, nc), (nz + 1, nc)
        for name, shape in (("pressure", mass), ("theta", mass), ("rho_dry", mass), ("qv", mass),
                            ("qc", mass), ("qr", mass), ("u", mass), ("v", mass),
                            ("z_interface", iface), ("w", iface)):
            value = columns[name]
            if tuple(value.shape) != shape or value.dtype != np.float32:
                raise ValueError(f"chemistry column {name} must be float32 {shape}")
        s.p[...] = columns["pressure"].reshape(nz, 1, nc)
        s.thp[...] = columns["theta"].reshape(nz, 1, nc)
        cp.divide(np.float32(1.0), columns["rho_dry"].reshape(nz, 1, nc), out=s.alt)
        cp.multiply(columns["z_interface"].reshape(nz + 1, 1, nc), np.float32(G), out=s.php)
        s.w[...] = columns["w"].reshape(nz + 1, 1, nc)
        for name in ("qv", "qc", "qr"):
            getattr(s, name)[...] = columns[name].reshape(nz, 1, nc)
        self.winds[0][...] = columns["u"].reshape(nz, 1, nc)
        self.winds[1][...] = columns["v"].reshape(nz, 1, nc)
        s.physics.fields = {} if physics_fields is None else physics_fields
        s.physics.microphysics = microphysics
        if self.prep is None:
            self.prep = ChemPrep(s)
        else:
            self.prep.refresh(s)
        self.prep._fields["u_phy"][...] = self.winds[0]
        self.prep._fields["v_phy"][...] = self.winds[1]
        return self.prep

    def context(self, ktau, dt):
        from woof.core.chem_context import ChemClock, ChemContext
        from woof.core.chem_state import process_attr
        s = self.state
        diag = {}
        for _key, module in self.processes:
            for alloc in getattr(module, "ALLOCATES", ()):
                diag[alloc.name] = getattr(s, process_attr(alloc))
        start = s.start_time
        gmt = start.hour + start.minute / 60.0 + start.second / 3600.0
        clock = ChemClock(gmt=gmt, julday=int(start.timetuple().tm_yday),
                          curr_secs=float(s.elapsed_seconds), ktau=int(ktau), dt=float(dt))
        return ChemContext(s, self.cfg, self.table, prep=self.prep, physics_fields=s.physics.fields,
                           diag=diag, ddvel=self.ddvel, clock=clock)

    # -- the step ---------------------------------------------------------------------------------
    def step(self, scalars, columns, *, dt, elapsed_seconds, physics_fields=None, microphysics=None,
             area_layer=None):
        """One chem step on the post-microphysics columns; ``commit``/``abort`` settle it."""
        import cupy as cp
        from woof.core.chem_state import ledger_attr
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("chemistry dt must be finite and positive")
        if float(dt) != float(self.cfg.dt):
            raise ValueError("chemistry dt differs from the column physics dt")
        s = self.state
        self.pending = {attr: getattr(s, attr).copy() for attr in self.serialized}
        self.bind_rows(scalars)
        # WRF curr_secs: the start of the step being closed.
        s.elapsed_seconds = float(elapsed_seconds) - float(dt)
        self.refresh(columns, physics_fields=physics_fields, microphysics=microphysics)
        ktau = self.steps + 1
        layer = _layer_mass(self.prep.get("dryrho"), self.prep.get("dz8w"), self.area)
        mass = _masses(s.chem.fields(s), self.rows, layer)
        current = getattr(s, ledger_attr("current"))
        getattr(s, ledger_attr("transport"))[...] += mass - current
        ctx = self.context(ktau, dt)
        self.ddvel[...] = 0
        if not s.chem.initialized:
            for key, module in self.processes:
                missing = [name for name in getattr(module, "REQUIRES", ()) if not ctx.has(name)]
                if missing:
                    raise RuntimeError(f"chem process {key} reads {missing}, which neither chem_prep nor "
                                       "the hex column physics publishes")
                module.init(ctx)
            s.chem.initialized = True
        index = {row.name: i for i, row in enumerate(self.rows)}
        for _key, module in self.processes:
            acted = [row for row in module.rows(self.table) if row.name in index]
            module.step(ctx, dt, ktau)
            if acted:
                idx = cp.asarray([index[row.name] for row in acted])
                now = _masses(tuple(getattr(s, row.state_attr) for row in acted), acted, layer)
                getattr(s, ledger_attr(module.LEDGER))[idx] += now - mass[idx]
                mass[idx] = now
        current[...] = mass
        for _key, module in self.processes:
            refresh = getattr(module, "refresh_output", None)
            if refresh is not None:
                refresh(s)

    def commit(self):
        self.pending = None
        self.steps += 1

    def abort(self):
        if self.pending is not None:
            for attr, value in self.pending.items():
                getattr(self.state, attr)[...] = value
        self.pending = None

    # -- products, receipts, restart --------------------------------------------------------------
    def products(self, scalars, columns):
        """2-D product planes ``(ncells,)`` by name (row column mass and surface concentration, and every
        2-D process output WRF writes)."""
        import cupy as cp
        from woof.core.chem_driver import _kg_per_unit
        from woof.core.chem_state import process_attr
        self.bind_rows(scalars)
        prep = self.refresh(columns)
        dryrho, dz8w = prep.get("dryrho"), prep.get("dz8w")
        layer = dryrho.astype(cp.float64) * dz8w.astype(cp.float64)
        out = {}
        for row in self.rows:
            q = getattr(self.state, row.state_attr).astype(cp.float64) * _kg_per_unit(row)
            out[row.name + "_column_mass"] = cp.sum(q * layer, axis=0)[0].astype(cp.float32)
            out[row.name + "_surface_concentration"] = (q[0, 0] * dryrho[0, 0] * 1.0e9).astype(cp.float32)
        for _key, module in self.processes:
            nrows = len(module.rows(self.table))
            for alloc in getattr(module, "ALLOCATES", ()):
                names = _output_names(alloc, nrows)
                if not names:
                    continue
                array = getattr(self.state, process_attr(alloc))
                planes = (array,) if alloc.shape == "2d" else tuple(array[i] for i in range(nrows))
                for name, plane in zip(names, planes):
                    out[name] = cp.ascontiguousarray(plane.reshape(self.ncells), dtype=cp.float32)
        return out

    def ledger(self):
        import cupy as cp
        from woof.core.chem_context import LEDGER_BUCKETS
        from woof.core.chem_state import ledger_attr
        host = {name: cp.asnumpy(getattr(self.state, ledger_attr(name)))
                for name in ("initial", "current", *LEDGER_BUCKETS)}
        rows = {}
        worst = 0.0
        for i, row in enumerate(self.rows):
            entry = {f"{name}_kg": float(host[name][i]) for name in LEDGER_BUCKETS}
            entry["initial_kg"] = float(host["initial"][i])
            entry["current_kg"] = float(host["current"][i])
            entry["closure_kg"] = (entry["initial_kg"] + sum(float(host[n][i]) for n in LEDGER_BUCKETS)
                                   - entry["current_kg"])
            scale = max(abs(entry["initial_kg"]), abs(entry["current_kg"]))
            if scale > 0:
                worst = max(worst, abs(entry["closure_kg"]) / scale)
            rows[row.name] = entry
        return {"rows": rows, "max_relative_closure": worst}

    def receipt(self):
        return {"schema": "gpuwm-hex.chemistry/v1", "mode": MODE, "config": str(self.path),
                "config_sha256": self.digest, "chem_set": self.set_name, "knobs": self.values,
                "rows": [{"name": r.name, "units": r.units} for r in self.rows],
                "products": list(product_names(self.table)[0]),
                "processes": [key for key, _ in self.processes],
                "transport": "hex monotonic scalar FCT (mesh-native)",
                "column_processes": "WRF-Chem process modules on ny=1 columns (woof.core.chem_*)",
                "committed_steps": self.steps, "provider": self.provider_receipt}

    def restart_payload(self):
        import cupy as cp
        return {"mode": MODE, "config_sha256": self.digest, "steps": self.steps,
                "provider": self.provider_receipt,
                "arrays": {attr: cp.asnumpy(getattr(self.state, attr)) for attr in self.serialized}}

    def restore_payload(self, payload):
        import cupy as cp
        if payload is None or payload.get("mode") != MODE or payload.get("config_sha256") != self.digest:
            raise ValueError("chemistry restart config identity differs")
        steps = payload["steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("invalid chemistry restart step count")
        arrays = payload["arrays"]
        if set(arrays) != set(self.serialized):
            raise ValueError("chemistry restart arrays differ from this configuration's")
        staged = {}
        for attr in self.serialized:
            target = getattr(self.state, attr)
            value = cp.asarray(arrays[attr])
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"invalid chemistry restart {attr}")
            staged[attr] = value
        for attr, value in staged.items():
            getattr(self.state, attr)[...] = value
        self.steps = steps
        self.restart_provider_receipt = payload.get("provider")
        # A restored run re-runs each process's init (statics, caches) before its first step.
        self.state.chem.initialized = False


def sample_dust_statics(geog_root, lat_deg, lon_deg, landmask):
    """EROD (3 classes), CLAYFRAC and SANDFRAC at the cell centres.

    Each variable takes its static source row's geogrid interpolation sequence (GEOGRID.TBL.ARW_CHEM, as
    the chem table records it), evaluated at the centre's fractional source index: ``four_pt`` (bilinear,
    all four points valid), ``average_4pt`` (mean of the valid points of those four), ``average_16pt``
    (mean of the valid points of the surrounding 4 x 4) and ``search`` (the nearest valid point), the first
    that finds data winning; a centre none of them reaches takes the row's ``fill_missing``.  Water cells
    take the fill, the mask being the run's own land mask (geogrid masks with the run's land use).  This
    is a point sampler written for the mesh, not a transcription of geogrid.
    """
    from woof.static.geog import GeogDataset
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    land = np.asarray(landmask) > 0.5
    out = {}
    for variable, (dataset, selector) in _static_rows().items():
        options = tuple(selector["interp_options"])
        unknown = sorted(set(options) - set(_INTERPOLATORS))
        if unknown:
            raise ValueError(f"{variable}: the hex point sampler has no {unknown} interpolation")
        out[variable] = _sample(GeogDataset(Path(geog_root) / dataset), lat, lon,
                                float(selector["fill_missing"]), land, options)
    out["EROD"] = out["EROD"].reshape(-1, lat.size)
    return out


def _static_rows():
    """variable -> (WPS GEOG dataset, selector) for the dust statics, from the chem table's static rows."""
    from woof.chem_table import catalog
    from woof.core.chem_statics import STATIC_VARIABLES
    rows = {}
    for source in catalog().sources.values():
        if source.kind != "static" or source.format != "wps_geog":
            continue
        for variable, spec in source.variables.items():
            if variable in STATIC_VARIABLES:
                rows[variable] = (source.grid["dataset"], spec["selector"])
    missing = [v for v in STATIC_VARIABLES if v not in rows]
    if missing:
        raise ValueError(f"no static chem source provides {missing}")
    return rows


def statics_receipt(geog_root):
    """What the dust statics were sampled from: each variable's dataset, interpolation sequence and the
    SHA-256 of the dataset's index and every file beside it, so a restart detects changed sources."""
    import hashlib
    from .file_digest import sha256_file
    out = {}
    for variable, (dataset, selector) in sorted(_static_rows().items()):
        directory = Path(geog_root) / dataset
        digest = hashlib.sha256()
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            digest.update(f"{path.name}\0{sha256_file(path)}\n".encode())
        out[variable] = {"dataset": dataset, "interp_options": list(selector["interp_options"]),
                         "files_sha256": digest.hexdigest()}
    return out


def _four_pt(values, fx, fy, i, j):
    p = [values[j + b, i + a] for b in (0, 1) for a in (0, 1)]
    ok = np.isfinite(p[0]) & np.isfinite(p[1]) & np.isfinite(p[2]) & np.isfinite(p[3])
    lower = p[0] * (1 - fx) + p[1] * fx
    upper = p[2] * (1 - fx) + p[3] * fx
    return np.where(ok, lower * (1 - fy) + upper * fy, np.nan)


def _mean(values, i, j, offsets):
    total = np.zeros(i.shape)
    count = np.zeros(i.shape)
    for a in offsets:
        for b in offsets:
            v = values[j + b, i + a]
            ok = np.isfinite(v)
            total += np.where(ok, v, 0.0)
            count += ok
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _search(values, fx, fy, i, j):
    good = np.argwhere(np.isfinite(values))
    if not good.size:
        return np.full(i.shape, np.nan)
    out = np.empty(i.shape)
    for n in range(i.size):
        d = (good[:, 1] - (i[n] + fx[n])) ** 2 + (good[:, 0] - (j[n] + fy[n])) ** 2
        jj, ii = good[int(np.argmin(d))]
        out[n] = values[jj, ii]
    return out


_INTERPOLATORS = {
    "four_pt": _four_pt,
    "average_4pt": lambda values, fx, fy, i, j: _mean(values, i, j, (0, 1)),
    "average_16pt": lambda values, fx, fy, i, j: _mean(values, i, j, (-1, 0, 1, 2)),
    "search": _search,
}
#: source points around the window kept for the 16-point average and the nearest-point search
_MARGIN = 8


def _sample(data, lat, lon, fill, land, options):
    x, y = data.latlon_to_xy(lat, lon)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    wx0, wx1 = int(x0.min()) - _MARGIN, int(x0.max()) + 1 + _MARGIN
    wy0, wy1 = int(y0.min()) - _MARGIN, int(y0.max()) + 1 + _MARGIN
    # A dataset that does not wrap is read inside its extent; the margin beyond it is missing.
    rx0, rx1 = (wx0, wx1) if data.wraps_x else (max(wx0, 1), min(wx1, data.nx_global))
    window = data.read_window(rx0, rx1, wy0, wy1)
    i, j = x0 - wx0, y0 - wy0
    fx, fy = x - x0, y - y0
    planes = []
    for k in range(data.index.nz):
        values = np.full((wy1 - wy0 + 1, wx1 - wx0 + 1), np.nan)
        # float64, the missing value masked as NaN
        covered = window.values(k)
        if window.coverage is not None:
            # Rows beyond the dataset and absent tiles are padding, not data (with no declared
            # missing value read_window pads them with zeros).
            covered = np.where(window.coverage, covered, np.nan)
        values[:, rx0 - wx0:rx1 - wx0 + 1] = covered
        result = np.full(lat.size, np.nan)
        for option in options:
            todo = land & ~np.isfinite(result)  # water takes the fill whatever the source says
            if not todo.any():
                break
            result[todo] = _INTERPOLATORS[option](values, fx[todo], fy[todo], i[todo], j[todo])
        planes.append(np.where(land & np.isfinite(result), result, fill))
    return planes[0] if data.index.nz == 1 else np.stack(planes)


def attach_statics(runtime, *, lat_deg, lon_deg, landmask=None, fields=None, geog_root=None):
    """Write the dust statics the dust process reads at init; returns the source receipt."""
    from woof.core.chem_statics import DUST_KEY, attach_dust_static_fields
    s = runtime.state
    if not runtime.table.rows_for(DUST_KEY):
        return None
    receipt = {"dust_statics": "supplied fields"}
    if fields is None:
        if geog_root is None:
            from woof.geog_assets import default_geog_root
            geog_root = default_geog_root()
        if landmask is None:
            raise ValueError("dust statics sampling needs the run's land mask")
        fields = sample_dust_statics(geog_root, lat_deg, lon_deg, landmask)
        receipt = {"dust_statics": "sampled at the cell centres", "geog_root": str(geog_root),
                   "sources": statics_receipt(geog_root)}
    shaped = {name: np.asarray(value).reshape(*np.shape(value)[:-1], 1, runtime.ncells)
              for name, value in fields.items()}
    attach_dust_static_fields(s, shaped)
    return receipt


def columns_from_prep(prepared, *, pressure="pres_p"):
    """The ``refresh`` column dict from a hex MPAS-to-physics preparation.  ``pressure`` names the carrier
    the physics phase reads: the EOS pressure after the RK step (phase two), the hydrostatic pressure at
    its start (phase one)."""
    return {"pressure": getattr(prepared, pressure), "theta": prepared.th_p, "rho_dry": prepared.rho_dry,
            "z_interface": prepared.z_p, "qv": prepared.qv_p, "qc": prepared.qc_p, "qr": prepared.qr_p,
            "u": prepared.u_p, "v": prepared.v_p, "w": prepared.w_p}


def build_column_backend(row, config, *, constructor, prep_geometry, kernel_cache, gwdo_static=None,
                         gwdo_kernel_cache=None, checkout=None, cell_area_m2=None, rollback_snapshot=True):
    from .cuda_arwen_physics_v841 import PersistentTwoPhaseCudaPhysicsBackendV841
    from .cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
    from .species_row import WSM6_SPECIES_ROW
    path, raw, digest, species_row = config

    class WrfChemistryBackend(PersistentTwoPhaseCudaPhysicsBackendV841):
        def _species_row(self, scalar_names):
            return WSM6_SPECIES_ROW

        def _new_seam(self):
            seam = super()._new_seam()
            runtime = getattr(self, "chemistry", None)
            if runtime is not None:
                runtime.bind_seam(seam)
            return seam

        def begin_step(self, *, atmosphere, scalar_names, dt, dynamics_tendencies=None):
            runtime = self.chemistry
            if runtime.feedback:
                # The columns phase one hands the seam (hydrostatic pressure, as the radiation reads it).
                prepared = prepare_mpas_to_phys_cuda_v841(
                    atmosphere, scalar_names=scalar_names, geometry=self._prep_geometry,
                    kernel_cache=self._kernel_cache, post_rk_wsm6=False)
                runtime.prepare_radiation(atmosphere.state.scalars,
                                          columns_from_prep(prepared, pressure="pres_hyd_p"),
                                          elapsed_seconds=float(atmosphere.state.time_seconds))
            return super().begin_step(atmosphere=atmosphere, scalar_names=scalar_names, dt=dt,
                                      dynamics_tendencies=dynamics_tendencies)

        def _columns(self, atmosphere, scalar_names, *, post_rk):
            prepared = prepare_mpas_to_phys_cuda_v841(
                atmosphere, scalar_names=scalar_names, geometry=self._prep_geometry,
                kernel_cache=self._kernel_cache, post_rk_wsm6=post_rk, phase2_only=post_rk,
                phase2_winds=post_rk)
            return columns_from_prep(prepared)

        def finish_step(self, *, atmosphere, scalar_names, dt, refl_10cm_due=False):
            update = super().finish_step(atmosphere=atmosphere, scalar_names=scalar_names,
                                        dt=dt, refl_10cm_due=refl_10cm_due)
            driver = self._seam._driver
            self.chemistry.step(atmosphere.state.scalars, self._columns(atmosphere, scalar_names, post_rk=True),
                                dt=float(dt), elapsed_seconds=float(atmosphere.state.time_seconds),
                                physics_fields=driver.fields,
                                microphysics=getattr(driver, "microphysics", None))
            return update

        def commit_step(self, **kwargs):
            result = super().commit_step(**kwargs)
            self.chemistry.commit()
            return result

        def abort_step(self, **kwargs):
            self.chemistry.abort()
            return super().abort_step(**kwargs)

        def diagnostic_snapshot(self, **kwargs):
            snapshot = super().diagnostic_snapshot(**kwargs)
            runtime = self.chemistry
            if runtime.driver is None:
                return snapshot
            atmosphere = runtime.driver.atmosphere
            extra = runtime.products(atmosphere.state.scalars,
                                     self._columns(atmosphere, runtime.species_row.names(), post_rk=False))
            return replace(snapshot, surface={**snapshot.surface, **extra},
                           metadata={**snapshot.metadata, "chemistry": runtime.receipt()})

        def restart_state(self):
            state = dict(super().restart_state())
            state["chemistry"] = self.chemistry.restart_payload()
            return state

        def restore_restart_state(self, payload):
            chem = payload.get("chemistry")
            self.chemistry.restore_payload(chem)
            super().restore_restart_state({k: v for k, v in payload.items() if k != "chemistry"})

        def write_ledger(self, output):
            runtime = self.chemistry
            Path(output, "chemistry-receipt.json").write_text(json.dumps(
                {**runtime.receipt(), "ledger": runtime.ledger()}, indent=2) + "\n")

    backend = WrfChemistryBackend(
        constructor=constructor, prep_geometry=prep_geometry, kernel_cache=kernel_cache,
        gwdo_static=gwdo_static, gwdo_kernel_cache=gwdo_kernel_cache,
        arwen_checkout=checkout, rollback_snapshot=rollback_snapshot)
    seam = backend._seam
    offset = len(species_row.species) - len(declare(raw))
    runtime = HexWrfChemistry(
        path, raw, digest, cfg=seam._cfg, area=cell_area_m2, nlev=constructor.n_levels,
        ncells=constructor.n_columns, offset=offset, start_time=seam._start_time,
        fnm=seam._state.fnm, fnp=seam._state.fnp, terrain=seam._state.ht)
    runtime.species_row = species_row
    backend.chemistry = runtime
    runtime.bind_seam(seam)
    return backend


def attach(backend, host, driver):
    """Bind the committed atmosphere, the dust statics and the zero initial rows."""
    import cupy as cp
    runtime = backend.chemistry
    runtime.driver = driver
    if host.get("regional") is not None:
        raise ValueError("wrf_chem on a regional mesh needs chemistry lateral boundaries over the specified "
                         "and relaxation zones, which the hex route does not drive yet")
    arrays = host["prepared"].mesh.arrays
    lat = np.degrees(np.asarray(arrays["latCell"], dtype=np.float64))[:runtime.ncells]
    lon = np.degrees(np.asarray(arrays["lonCell"], dtype=np.float64))[:runtime.ncells]
    landmask = cp.asnumpy(backend._seam._driver.fields["xland"]).reshape(-1) < 1.5
    block = runtime.raw["wrf_chem"]
    runtime.provider_receipt = attach_statics(runtime, lat_deg=lat, lon_deg=lon, landmask=landmask,
                                              geog_root=block.get("geog_root"))
    if (runtime.restart_provider_receipt is not None
            and runtime.restart_provider_receipt != runtime.provider_receipt):
        raise ValueError("chemistry restart source-file identity differs")
    if runtime.steps == 0:
        rows = driver.atmosphere.state.scalars[runtime.offset:runtime.offset + len(runtime.rows)]
        if bool(cp.any(rows != 0)):
            raise ValueError("wrf_chem starts from zero composition (WRF chem_in_opt = 0); the rows carry "
                             "mass before the first step")
