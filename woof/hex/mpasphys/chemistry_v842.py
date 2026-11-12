"""Aerosol mass transport on the MPAS-A v8.4.2 suite route: the WOOF route's chemistry, same operator.

``forecast --chemistry-config FILE`` appends one dry-air mass mixing ratio per configured species to the
atmosphere.  On the WOOF route (``woof.hex.chem_transport``) those rows ride the dycore's monotonic scalar
transport and, after each step, the conservative implicit column operator (PBL mixing with the active
scheme's diffusivity, downward settling, dry deposition at the ground, surface emissions).  This module
gives the suite route the same thing with the same code:

* the species rows are appended after the Registry's scalars in the dycore's ``state.scalars`` (the MPAS
  physics never sees them: every exchange with the physics pools takes the Registry rows only, and the
  physics scalar sources carry zero for the appended rows);
* after each completed step, ``woof.hex.chem_transport.ChemistryTransport.step`` runs on the appended rows
  with the step-end dry density and interface heights and MYNN's ``kzh`` (the suite's eddy diffusivity for
  heat, MPAS's ``exch_h``);
* initial composition, emissions and their coverage checks are ``woof.hex.chem_emissions.attach`` (the
  provider contract: dated frames bound to the grid's SHA-256, missing coverage refused);
* every history frame carries the mixing ratios and the per-species column mass, surface concentration,
  AOD proxy, dry deposition and settled mass, plus the ``CHEMISTRY`` attribute; ``chemistry-receipt.json``
  records the mass closure; a checkpoint carries the ledgers and a restart refuses another configuration.

THE BREAKAGE THIS FIXES: ``--chemistry-config`` with an MPAS suite was accepted and dropped without a word
(no tracer, no attribute, no receipt, output bitwise the plain run's; HEX-PHYSICS-PARITY run group 6 #24).

A ``wrf_chem`` or ``micm`` block runs the WOOF route's column runtime itself (``woof.hex.chem_wrf.HexWrfChemistry``,
``woof.hex.chem_micm_hex.HexMicmChemistry``) on the same appended rows.  The prep adapter here is the only new
part: the post-step columns come from the same MPAS-to-physics preparation the WOOF route reads
(``columns_from_prep`` / ``HexMicmChemistry.columns``), and the WRF met names the processes read are the MPAS
v8.4.2 physics pools' own words under the names MPAS's drivers hand WRF's schemes (``SUITE_MET_FIELDS``:
``kzh`` -> ``exch_h``, ``hpbl`` -> ``pblh``, ``skintemp`` -> ``tsk``, ``swdnb`` -> ``swdown``, the rest by the same
name), reshaped to the column runtime's ``(1, ncells)`` / ``(nz, 1, ncells)`` layout without a change of value.
The column config is a ``RunConfig`` on ``ny = 1`` columns whose physics selectors are the suite's
(``suite_column_config``).  Couplings the suite's physics does not share with the WRF column driver are refused
by name: aerosol-radiation feedback (the suite's radiation is MPAS's RRTMG, not the WRF driver's) and the
aerosol-aware Thompson diagnose coupling; a limited-area suite run (chemistry lateral boundaries) is refused too.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _Row:
    """The species-row surface ChemistryTransport and chem_emissions read: Registry scalars, then species."""

    def __init__(self, registry, species):
        self.registry = tuple(registry)
        self.species = self.registry + tuple(species)

    def names(self):
        return self.species

    def index_of(self, name):
        return self.species.index(name)


#: WRF met name -> (MPAS v8.4.2 pool, field): the words MPAS's physics drivers hand the WRF schemes under
#: these names (mpas_atmphys_driver_pbl.F:602 kzh <- exch_h, :587 hpbl <- pblh; mpas_atmphys_driver_lsm.F:348
#: skintemp -> tsk, :422 vegfra -> vegfra_p), the rest by the same name.  A field the active suite does not
#: allocate is left out, and a process that reads it refuses by name at its first step (HexWrfChemistry.step).
SUITE_MET_FIELDS = (
    ("exch_h", "diag_physics", "kzh"),
    ("ust", "diag_physics", "ust"),
    ("rmol", "diag_physics", "rmol"),
    ("znt", "diag_physics", "znt"),
    ("pblh", "diag_physics", "hpbl"),
    ("u10", "diag_physics", "u10"),
    ("v10", "diag_physics", "v10"),
    ("gsw", "diag_physics", "gsw"),
    ("raincv", "diag_physics", "raincv"),
    ("hfx", "diag_physics", "hfx"),
    ("qfx", "diag_physics", "qfx"),
    ("tsk", "sfc_input", "skintemp"),
    ("xland", "sfc_input", "xland"),
    ("smois", "sfc_input", "smois"),
    ("isltyp", "sfc_input", "isltyp"),
    ("ivgtyp", "sfc_input", "ivgtyp"),
    ("vegfra", "sfc_input", "vegfra"),
    ("snowh", "sfc_input", "snowh"),
    ("snow", "sfc_input", "snow"),
)
#: MPAS-A v8.4.2 mpas_atmphys_control.F:131-163: what ``suite`` selects under each physics suite.
SUITE_DEFAULTS = {
    "mesoscale_reference": {"config_microp_scheme": "mp_wsm6", "config_convection_scheme": "cu_ntiedtke",
                            "config_pbl_scheme": "bl_ysu", "config_lsm_scheme": "sf_noah"},
    "convection_permitting": {"config_microp_scheme": "mp_thompson",
                              "config_convection_scheme": "cu_grell_freitas", "config_pbl_scheme": "bl_mynn",
                              "config_lsm_scheme": "sf_noah"},
    "none": {"config_microp_scheme": "off", "config_convection_scheme": "off", "config_pbl_scheme": "off",
             "config_lsm_scheme": "off"},
}
#: MPAS scheme name -> the WRF option number the chem processes and validate_chem_config read, per selector.
WRF_OPTIONS = {
    "config_microp_scheme": ("mp_physics", {"off": 0, "mp_kessler": 1, "mp_wsm6": 6, "mp_thompson": 8,
                                            "mp_thompson_aerosols": 28}),
    "config_convection_scheme": ("cu_physics", {"off": 0, "cu_kain_fritsch": 1, "cu_grell_freitas": 3,
                                                "cu_tiedtke": 6, "cu_ntiedtke": 16}),
    "config_pbl_scheme": ("bl_pbl_physics", {"off": 0, "bl_ysu": 1, "bl_mynn": 5}),
    "config_lsm_scheme": ("sf_surface_physics", {"off": 0, "sf_noah": 2, "sf_noahmp": 4}),
}


def _suite_field(session, pool, name):
    try:
        return session.field(pool, name)
    except (KeyError, RuntimeError):
        return None


def _column_layout(value, ncells, what):
    """An MPAS ``(..., nCells)`` field as the column runtime's ``(1, ncells)`` or ``(n, 1, ncells)``, the
    words unchanged (float32 / int32, as the WRF driver's fields)."""
    import cupy as cp
    value = cp.asarray(value)
    if value.ndim not in (1, 2) or value.shape[-1] < ncells:
        raise ValueError(f"suite field {what} has shape {tuple(value.shape)} for {ncells} columns")
    value = value[..., :ncells]
    value = value.reshape(1, ncells) if value.ndim == 1 else value.reshape(value.shape[0], 1, ncells)
    kind = cp.float32 if value.dtype.kind == "f" else cp.int32
    return cp.ascontiguousarray(value, dtype=kind)


def suite_physics_fields(session, *, ncells):
    """The WRF-named physics fields (``SUITE_MET_FIELDS``) from the suite's pools, in the column layout.

    ``swdown`` is the word MPAS's land-surface driver hands Noah under that name,
    ``gsw / (1 - sfc_albedo)`` in the physics' single precision (mpas_atmphys_driver_lsm.F:472), WRF's own
    SWDOWN (module_radiation_driver)."""
    import cupy as cp
    out = {}
    for wrf_name, pool, name in SUITE_MET_FIELDS:
        value = _suite_field(session, pool, name)
        if value is not None:
            out[wrf_name] = _column_layout(value, ncells, f"{pool}.{name}")
    albedo = _suite_field(session, "diag_physics", "sfc_albedo")
    if "gsw" in out and albedo is not None:
        albedo = _column_layout(albedo, ncells, "diag_physics.sfc_albedo")
        out["swdown"] = cp.ascontiguousarray(out["gsw"] / (np.float32(1.0) - albedo))
    return out


def suite_microphysics(session, *, ncells):
    """The step's grid-scale rain (``rainncv``, mm per step) as the WRF driver's microphysics view, or None
    when the suite runs no microphysics (WRF's mp_physics = 0, where the driver publishes none)."""
    value = _suite_field(session, "diag_physics", "rainncv")
    if value is None:
        return None
    return SimpleNamespace(rainncv=_column_layout(value, ncells, "diag_physics.rainncv"))


def suite_selections(configs):
    """The suite's four scheme selections with ``suite`` resolved as MPAS resolves it."""
    suite = str(configs.get("config_physics_suite", "mesoscale_reference")).strip()
    if suite not in SUITE_DEFAULTS:
        raise ValueError(f"config_physics_suite = {suite!r} is not an MPAS-A v8.4.2 physics suite")
    out = {}
    for key in WRF_OPTIONS:
        value = str(configs.get(key, "suite")).strip()
        out[key] = SUITE_DEFAULTS[suite][key] if value == "suite" else value
    return out


def suite_column_config(*, ncells, nlev, dt, dx_m, ztop, configs):
    """The ``ny = 1`` column RunConfig the WRF chem processes read: shape, dt, dx, and the WRF option numbers
    of the suite's microphysics, convection, PBL and land surface (the selectors the processes and
    validate_chem_config branch on)."""
    from woof.config import RunConfig
    options = {}
    for key, value in suite_selections(configs).items():
        name, table = WRF_OPTIONS[key]
        if value not in table:
            raise ValueError(f"chemistry on the MPAS suite route: {key} = {value!r} has no WRF option the chem "
                             f"processes know ({sorted(table)})")
        options[name] = table[value]
    return RunConfig(nx=int(ncells), ny=1, nz=int(nlev), dx=float(dx_m), dy=float(dx_m), ztop=float(ztop),
                     dt=float(dt), run_seconds=0.0, moist=True, **options)


def column_mode(raw):
    return "wrf_chem" if "wrf_chem" in raw else "micm" if "micm" in raw else None


class SuiteChemistry:
    """Owner of the appended species rows on the suite route."""

    def __init__(self, path, *, registry_names, mesh, nlev, grid_path, start_text, vertical=None,
                 configs=None, dt=None):
        from ..chem_transport import RESERVED_FIELDS, load_config
        path, raw, species = load_config(path)
        self.mode = column_mode(raw)
        # The column modes read the vertical grid, the namelist (scheme selections, read at bind, after the
        # physics initialization) and the step.
        self.vertical, self.configs, self.dt = vertical, configs, dt
        if self.mode is not None and (configs is None or dt is None or (self.mode == "wrf_chem" and vertical is None)):
            raise ValueError(f"{self.mode} on the MPAS suite route needs the vertical grid, configs and dt")
        self.runtime = None
        if self.mode is not None:
            names = self._column_rows(path, raw, mesh)
        else:
            names = tuple(s.name for s in species)
        overlap = set(names) & (set(registry_names) | RESERVED_FIELDS)
        if overlap:
            raise ValueError(f'chemistry species overlap the MPAS suite scalars or fields: {sorted(overlap)}')
        self.path, self.raw, self.species = path, raw, species
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.registry_names = tuple(registry_names)
        self.names = names
        self.row = _Row(registry_names, names)
        self.n_registry = len(self.registry_names)
        self.nlev, self.ncells = int(nlev), int(mesh.n_cells)
        area = np.asarray(mesh.arrays['areaCell'], dtype=np.float64)[:self.ncells]
        if float(area.sum()) < 4 * math.pi * 1.001:
            # A unit-sphere mesh (sphere_radius = 1) stores areas in radians squared.
            area = area * float(mesh.attrs.get('sphere_radius_m', 6371229.0)) ** 2
        self.area = area
        self.grid_path = Path(grid_path)
        self.mesh = mesh
        self.start_text = start_text
        self.case_start = start_text
        self.transport = None
        self.restart = None

    def _column_rows(self, path, raw, mesh):
        """The column runtime's rows, refusing what the suite's physics does not share with the WRF driver."""
        from .. import chem_micm_hex, chem_wrf
        if self.mode == "wrf_chem":
            block = raw["wrf_chem"]
            if int(block.get("aer_ra_feedback", 0)):
                raise ValueError("wrf_chem aer_ra_feedback on the MPAS suite route: the suite's radiation is "
                                 "MPAS's RRTMG, which reads no WRF-Chem optics; run it on the hex WOOF route")
            if chem_wrf.mp_coupling(block) != "none":
                raise ValueError("wrf_chem aerosol_mp_coupling on the MPAS suite route: the suite's Thompson "
                                 "carries its own nwfa/nifa; the diagnose coupling runs on the hex WOOF route")
            rows = chem_wrf.declare(raw)
        else:
            rows = chem_micm_hex.declare(path, raw)
        mask = getattr(mesh, "arrays", {}).get("bdyMaskCell")
        if mask is not None and bool(np.any(np.asarray(mask) != 0)):
            raise ValueError(f"{self.mode} on a limited-area MPAS suite run needs chemistry lateral boundaries, "
                             "which the suite route does not drive; run it on the hex WOOF route")
        return tuple(r.name for r in rows)

    # ------------------------------------------------------------------ before the dycore exists
    def extend_state(self, state):
        """Append zero rows for the species to the host scalar block (the provider binds the start)."""
        if state.scalars.shape[0] == self.n_registry + len(self.names):
            return state      # a checkpoint's state already holds them
        if state.scalars.shape[0] != self.n_registry:
            raise ValueError('the start state scalar block differs from the MPAS Registry declaration')
        extra = np.zeros((len(self.names),) + tuple(state.scalars.shape[1:]), dtype=state.scalars.dtype)
        state.scalars = np.ascontiguousarray(np.concatenate([state.scalars, extra], axis=0))
        return state

    # ------------------------------------------------------------------ once the dycore exists
    def bind(self, *, dycore, cache, coupling):
        if self.mode is not None:
            return self._bind_column(dycore=dycore, coupling=coupling)
        from ..chem_transport import ChemistryTransport
        config = (self.path, self.raw, self.species, self.row, self.digest)
        self.transport = ChemistryTransport(config, cache=cache, area=self.area,
                                            nlev=self.nlev, ncells=self.ncells)
        self.coupling, self.dycore = coupling, dycore
        self.transport.driver = SimpleNamespace(atmosphere=None)
        self._refresh_driver()
        if self.restart is not None:
            self._restore(self.restart)
        self._attach_inputs()

    def _refresh_driver(self):
        if self.mode is not None:
            self.runtime.driver.atmosphere = self.dycore.atmosphere
            return
        self.transport.driver.atmosphere = self.dycore.atmosphere

    # ------------------------------------------------------------------ wrf_chem / micm column runtime
    def _bind_column(self, *, dycore, coupling):
        """The WOOF route's column runtime on the appended rows, attached through its own ``attach``.

        The chemistry clock is the case's: a checkpoint carries the case start (``config_start_time`` is the
        restart time on a restart), and the step's end is the committed step count times dt."""
        from .. import chem_micm_hex, chem_wrf
        from datetime import datetime
        self.coupling, self.dycore = coupling, dycore
        if self.restart is not None:
            self.case_start = self.restart.get('case_start')
            if not isinstance(self.case_start, str):
                raise ValueError('this chemistry checkpoint carries no case start time')
        start = datetime.strptime(self.case_start, '%Y-%m-%d_%H:%M:%S')
        if self.mode == "wrf_chem":
            from woof.core.mpas_column_batch import _interface_weights
            vertical = self.vertical
            # The column physics' nominal interfaces and terrain, as the WOOF route's constructor takes them
            # (run_cuda_v841_forecast: vertical.zw, zgrid[:, 0]).
            zw = np.asarray(vertical.zw)
            fnm, fnp = _interface_weights(zw)
            cfg = suite_column_config(ncells=self.ncells, nlev=self.nlev, dt=self.dt,
                                      dx_m=float(self.configs['config_len_disp']), ztop=float(zw[-1]),
                                      configs=self.configs)
            terrain = np.ascontiguousarray(np.asarray(vertical.zgrid)[:self.ncells, 0], dtype=np.float32)
            runtime = chem_wrf.HexWrfChemistry(self.path, self.raw, self.digest, cfg=cfg, area=self.area,
                                               nlev=self.nlev, ncells=self.ncells, offset=self.n_registry,
                                               start_time=start, fnm=fnm, fnp=fnp, terrain=terrain)
            module = chem_wrf
        else:
            runtime = chem_micm_hex.HexMicmChemistry(self.path, self.raw, self.digest, nlev=self.nlev,
                                                     ncells=self.ncells, offset=self.n_registry,
                                                     start_time=start, dt=self.dt)
            module = chem_micm_hex
        runtime.species_row = self.row
        if self.mode == "wrf_chem":
            # Wesely deposition reads the run's land-use dataset (WRF MMINLU) from the surface scheme's tables,
            # as on the WOOF route (HexWrfChemistry.bind_seam); here it is MPAS's own sfc_input.mminlu, the
            # classification MPAS's Noah reads its tables under (mpas_atmphys_landuse.F:143-188).
            native = getattr(coupling.session, 'native', None)
            mminlu = None if native is None else native.get_text_field('sfc_input', 'mminlu').strip()
            if mminlu:
                runtime.state.physics.noah_params = SimpleNamespace(lutype=mminlu)
        self.runtime = runtime
        if self.restart is not None:
            runtime.restore_payload(self.restart)
        # chem_wrf.attach / chem_micm_hex.attach read the column seam's land mask, the start preparation and
        # the host's mesh: here the suite's sfc_input.xland and MPAS-to-physics preparation.
        session = coupling.session
        fields = {}
        xland = _suite_field(session, 'sfc_input', 'xland')
        if xland is not None:
            fields['xland'] = _column_layout(xland, self.ncells, 'sfc_input.xland')
        backend = SimpleNamespace(chemistry=runtime, _seam=SimpleNamespace(_driver=SimpleNamespace(fields=fields)),
                                  _prepare=lambda atmosphere, names, post_rk: self._columns(atmosphere))
        host = {'regional': None, 'prepared': SimpleNamespace(mesh=self.mesh), 'start_time_text': self.start_text}
        module.attach(backend, host, SimpleNamespace(atmosphere=dycore.atmosphere))

    def _attach_inputs(self):
        """The WOOF route's attach (chem_transport.attach), on this route's grid authority."""
        from .. import chem_emissions
        from datetime import datetime, timezone
        runtime = self.transport
        if not runtime.raw.get('inputs'):
            raise ValueError('chemistry forecast requires dated initial composition and emission coverage in its '
                             'inputs block')
        source_start = chem_emissions.instant(runtime.raw['inputs']['valid_time'])
        weather_start = int(datetime.strptime(self.start_text, '%Y-%m-%d_%H:%M:%S')
                            .replace(tzinfo=timezone.utc).timestamp())
        grid_hash = chem_emissions.digest(self.grid_path)
        host = dict(prepared=SimpleNamespace(input_bytes={'grid': {'sha256': grid_hash}}), regional=None)
        if self.restart is None:
            if source_start != weather_start:
                raise ValueError('chemistry inputs and weather initialization valid times differ')
        if runtime.raw['inputs']['mesh_sha256'] != grid_hash:
            raise ValueError('chemistry inputs and weather grid SHA-256 differ')
        # attach binds the emission/boundary update to model time since the provider's start; a restart's
        # dycore clock counts from the case start too (the carried time_seconds), so the update is the same.
        runtime.provider_receipt = chem_emissions.attach(SimpleNamespace(chemistry=runtime), host)
        if (runtime.restart_provider_receipt is not None
                and runtime.restart_provider_receipt != runtime.provider_receipt):
            raise ValueError('chemistry restart source-file identity differs')
        import cupy as cp
        scalars = self.dycore.atmosphere.state.scalars
        if runtime.steps == 0:
            for s in runtime.species:
                if s.name in runtime.initial:
                    values = cp.asarray(runtime.initial[s.name], dtype=cp.float32)
                    target = scalars[self.row.index_of(s.name)]
                    if values.shape != target.shape or not bool(cp.all(cp.isfinite(values) & (values >= 0))):
                        raise ValueError(f'chemistry IC for {s.name} has invalid shape or mass')
                    target[...] = values
        if runtime.initial_mass_kg is None:
            products = self.products()
            runtime.initial_mass_kg = [float(cp.sum(products[s.name + '_column_mass'] * runtime.area))
                                       for s in runtime.species]

    # ------------------------------------------------------------------ registry views
    def registry_view(self, atmosphere):
        """The atmosphere with the Registry's scalar rows only (what the MPAS physics preparation reads)."""
        return replace(atmosphere, state=replace(atmosphere.state,
                                                 scalars=atmosphere.state.scalars[:self.n_registry]))

    def pad_sources(self, cp, scalars):
        """Physics scalar sources with zero rows for the species (they have no MPAS physics source)."""
        extra = cp.zeros((len(self.names),) + tuple(scalars.shape[1:]), dtype=scalars.dtype)
        return cp.ascontiguousarray(cp.concatenate([scalars, extra], axis=0))

    def _columns(self, atmosphere):
        from ..cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
        coupling = self.coupling
        return prepare_mpas_to_phys_cuda_v841(self.registry_view(atmosphere), scalar_names=coupling.names,
                                              geometry=coupling.geometry, kernel_cache=coupling.physics_cache,
                                              registry_scalar_names=coupling.names)

    # ------------------------------------------------------------------ per step
    def step(self, *, session, dt):
        """After the completed step: mixing, settling, deposition and emissions on the species rows."""
        if self.mode is not None:
            return self._step_column(session=session, dt=dt)
        self._refresh_driver()
        atmosphere = self.dycore.atmosphere
        columns = self._columns(atmosphere)
        if ('diag_physics', 'kzh') not in getattr(session.native, '_fields', {}) and not self._has_kzh(session):
            raise ValueError('the active PBL publishes no kzh (exch_h) for chemistry mixing')
        kh = session.field('diag_physics', 'kzh')[:self.nlev, :self.ncells]
        self.transport.step(atmosphere, rho=columns.rho_dry, z=columns.z_p, kh=kh, dt=float(dt))
        self.transport.commit()

    def column_inputs(self, session, atmosphere, phase="post_rk"):
        """(columns, physics_fields, microphysics) the column runtime reads after a suite step."""
        runtime = self.runtime
        columns = runtime.columns(self._columns(atmosphere), phase)
        if self.mode == "micm":
            return columns, None, None
        return (columns, suite_physics_fields(session, ncells=self.ncells),
                suite_microphysics(session, ncells=self.ncells))

    def _step_column(self, *, session, dt):
        """The column runtime's step and commit; an exception aborts it (its process arrays and ledger
        return to the step's start) before it propagates.  The step's end on the chemistry clock is the
        committed step count times dt, so a restart continues on the same clock."""
        self._refresh_driver()
        runtime = self.runtime
        atmosphere = self.dycore.atmosphere
        columns, fields, micro = self.column_inputs(session, atmosphere)
        try:
            runtime.step(atmosphere.state.scalars, columns, dt=float(dt),
                         elapsed_seconds=(runtime.steps + 1) * float(dt), physics_fields=fields,
                         microphysics=micro)
        except BaseException:
            runtime.abort()
            raise
        runtime.commit()

    @staticmethod
    def _has_kzh(session):
        try:
            session.field('diag_physics', 'kzh')
            return True
        except (KeyError, RuntimeError):
            return False

    def products(self):
        self._refresh_driver()
        atmosphere = self.dycore.atmosphere
        columns = self._columns(atmosphere)
        if self.mode is not None:
            return self.runtime.products(atmosphere.state.scalars, self.runtime.columns(columns, "snapshot"))
        return self.transport.products(atmosphere, columns.rho_dry, columns.z_p)

    # ------------------------------------------------------------------ history
    def history_fields(self):
        """Extra history variables for one frame: name -> (array, dims, units), on the host."""
        from ..chem_transport import history_units
        import cupy as cp
        scalars = self.dycore.atmosphere.state.scalars
        out = {}
        units = ({r.name: r.units for r in self.runtime.rows} if self.mode is not None
                 else dict.fromkeys(self.names, 'kg kg-1'))
        for name in self.names:
            value = cp.asnumpy(scalars[self.row.index_of(name)][:, :self.ncells])
            out[name] = (np.ascontiguousarray(value), ('nVertLevels', 'nCells'), units[name])
        for name, value in self.products().items():
            out[name] = (np.ascontiguousarray(cp.asnumpy(value)[:self.ncells]), ('nCells',),
                         history_units(name) or '')
        return out

    def history_attribute(self):
        from ..chem_transport import history_metadata
        return history_metadata(self.path, self.digest)

    # ------------------------------------------------------------------ checkpoint
    def restart_state(self):
        if self.mode is not None:
            return {**self.runtime.restart_payload(), 'case_start': self.case_start}
        t = self.transport
        return dict(config_sha256=self.digest, names=list(self.names), dry=t.dry.get(),
                    settled=t.settled.get(), emitted=t.emitted.get(), steps=t.steps,
                    initial_mass_kg=t.initial_mass_kg, provider=t.provider_receipt)

    def _restore(self, chem):
        import cupy as cp
        t = self.transport
        if chem is None or chem.get('config_sha256') != self.digest or list(chem.get('names', [])) != list(self.names):
            raise ValueError('chemistry restart config identity differs')
        for name in ('dry', 'settled', 'emitted'):
            array = cp.asarray(chem[name])
            target = getattr(t, name)
            if (array.shape != target.shape or array.dtype != target.dtype
                    or not bool(cp.all(cp.isfinite(array) & (array >= 0)))):
                raise ValueError(f'invalid chemistry restart {name}')
            target[...] = array
        steps = chem['steps']
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('invalid chemistry restart step count')
        t.steps = steps
        t.initial_mass_kg = list(chem['initial_mass_kg'])
        t.restart_provider_receipt = chem.get('provider')

    # ------------------------------------------------------------------ receipt
    def write_ledger(self, output):
        import cupy as cp
        if self.mode is not None:
            return self._write_column_ledger(output)
        t = self.transport
        totals = {key: cp.sum(getattr(t, name) * t.area, axis=1).get().tolist()
                  for key, name in (('dry_deposition_kg', 'dry'), ('settled_kg', 'settled'),
                                    ('emitted_kg', 'emitted'))}
        fields = self.products()
        totals['final_mass_kg'] = [float(cp.sum(fields[s.name + '_column_mass'] * t.area)) for s in t.species]
        totals['initial_mass_kg'] = t.initial_mass_kg
        totals['mass_closure_kg'] = [final + dry + settled - initial - emitted
                                     for final, dry, settled, initial, emitted in zip(
                                         totals['final_mass_kg'], totals['dry_deposition_kg'],
                                         totals['settled_kg'], t.initial_mass_kg, totals['emitted_kg'])]
        totals['surface_max_ug_m3'] = [float(cp.max(fields[s.name + '_surface_concentration']))
                                       for s in t.species]
        receipt = {**t.receipt(), **totals,
                   'route': 'mpas-suite (species rows after the Registry scalars; MYNN kzh mixing)'}
        Path(output, 'chemistry-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt

    def _write_column_ledger(self, output):
        """The column runtime's receipt (chem_backend's write_ledger), with the suite route named and, for
        wrf_chem, the per-row ledger's final mass, emissions and closure under the receipt keys the run
        receipt carries."""
        runtime = self.runtime
        receipt = {**runtime.receipt(),
                   'route': f'mpas-suite ({self.mode} column runtime on the rows after the Registry scalars)'}
        if hasattr(runtime, 'ledger'):
            ledger = runtime.ledger()
            rows = [ledger['rows'][r.name] for r in runtime.rows]
            receipt.update(ledger=ledger, final_mass_kg=[r['current_kg'] for r in rows],
                           emitted_kg=[r['emitted_kg'] for r in rows],
                           mass_closure_kg=[r['closure_kg'] for r in rows])
        Path(output, 'chemistry-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt
