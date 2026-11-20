"""E3SM v3.2.0's MPAS-Ocean coupler import and export on the GPU, opt-in.

E3SM's ocean (``components/mpas-ocean/driver/ocn_comp_mct.F``) reads what
the coupler sends it (``x2o``) into its forcing pool with ``ocn_import_mct``
and fills what it sends back (``o2x``) with ``ocn_export_mct``.
:func:`indices` is ``mpaso_cpl_indices_set`` (the columns, with the stops
E3SM makes on a list that lacks a required field); :func:`import_x2o` and
:func:`export_o2x` are the two routines, word for word:

- the import copies the fluxes and states into the forcing arrays a package
  has on (an absent array is left alone), takes ``max(Foxx_swnet, 0)``, the
  rain times ``precip_fact`` (under ``config_use_precip_scaling``), the
  sea-ice pressure capped at ``config_density0 * gravity * 5`` (gravity is
  ``mpas_constants``' default-real 9.80616, float32-rounded), the wind stress
  less the waves' share (under ``config_momentum_use_active_wave``), the
  waves coupled two ways, ``sqrt(So_duu10n)`` under the KPP theory-wave or
  EKE options, and moves the runoff south of 60S (liquid) and 57S (solid)
  into the removed-runoff arrays when ``config_remove_ais_*_runoff`` is on;
- the export writes the surface state, ``Faoo_h2otemp`` (the freshwater
  temperature flux times ``rho_sw * cp_sw``), the land-ice and removed-runoff
  fluxes, the frazil's heat and mass (the accumulated mass's latent heat, or
  the energy to the freezing point, or the land-ice frazil where the mask is
  1) and resets the frazil arrays, the land-ice boundary layer (standalone or
  coupled) and the 3-D thermal forcing.

This is not called by any default path. ``real(r8)`` only. The halo
exchanges are the identity on one block. Refused by name: the ecosystem,
DMS, MacroMolecules and CFC tracers (``config_use_*Tracers``), whose
branches are not ported; and what E3SM gets wrong past bounds (see QUIRKS in
the module's tests): the active-wave stress without the waves coupled two
ways (index 0 of ``x2o``), an ``o2x`` field the export writes that the
list lacks (``Faoo_h2otemp``, ``Fioo_q``, ``Foxo_*``: looked up quietly,
written to column 0), more ``nGlcZLevels`` than ``glc_nzoc``, and the
removed runoff or the 10 m wind switched on without their arrays.

NaN inputs are outside the word-for-word claim. gfortran's MAX and MIN with
a NaN argument depend on the code's shape (on the oracle compiler,
``max(nan, z)`` with a variable ``z`` returns ``z`` but ``max(nan, 0.0)`` with
the literal returns the NaN), and GPU arithmetic returns a canonical NaN
where x86 keeps the operand's payload.

The kernels (``ocn_coupling_e3sm.cu``) are held word for word to E3SM's
routines cut verbatim out of ``ocn_comp_mct.F`` and run on the pinned
framework stubs, built with gfortran ``-O0 -ffp-contract=off``
(``tools/e3sm_coupler_oracle``, ``tests/test_e3sm_ocean_coupling_parity.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np

E3SM_TAG = "v3.2.0"
E3SM_COMMIT = "648b167b7f95267a159a9fdee25faf604c2bfc4f"
SOURCE_FILES = ("components/mpas-ocean/driver/ocn_comp_mct.F:ocn_import_mct,ocn_export_mct",
                "components/mpas-ocean/driver/mpaso_cpl_indices.F:mpaso_cpl_indices_set",
                "components/mpas-ocean/src/shared/mpas_ocn_equation_of_state.F:ocn_freezing_temperature")
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128

#: mpas_constants' ``gravity = 9.80616``: a default-real literal, so float32-rounded
GRAVITY = float(np.float32(9.80616))
#: shr_const_mod's, which ocn_constants_init takes under MPAS_ESM_SHR_CONST (E3SM's build)
RHO_SW = 1.026e3
CP_SW = 3.996e3
T0_KELVIN = 273.15
GLC_NZOC = (0, 4, 30)
NUTRIENTS = ("DIN", "DIP", "DON", "DOP", "DOC", "PP", "DSi", "POC", "PN", "DIC", "Fe")


class CouplingAbort(RuntimeError):
    """What E3SM stops on (MCT's ``die`` on a field ``mct_aVect_indexRA``
    must find, or ``shr_sys_abort``); the message is the Fortran's."""


_O2X_REQUIRED = ("So_t", "So_u", "So_v", "So_s", "So_dhdx", "So_dhdy")
_O2X_QUIET = ("Fioo_q", "Foxo_q_li", "Fioo_frazil", "Foxo_frazil_li", "Faoo_h2otemp", "Faoo_fco2_ocn",
              "Faoo_fdms_ocn")
_O2X_QUIET_2 = ("Foxo_ismw", "Foxo_rrofl", "Foxo_rrofi", "Foxo_ismh", "Foxo_rrofih")
_O2X_LAND_ICE = ("So_blt", "So_bls", "So_htv", "So_stv", "So_rhoeff")
_X2O_REQUIRED = ("Si_ifrac Si_bpress Sa_pslv So_duu10n Foxx_tauy Foxx_taux Foxx_swnet Foxx_lat Foxx_sen Foxx_lwup "
                 "Faxa_lwdn Fioi_melth Fioi_meltw PFioi_bergh PFioi_bergw Fioi_salt Faxa_prec Faxa_snow Faxa_rain "
                 "Foxx_evap Foxx_rofl Foxx_rofi").split()
_X2O_AEROSOLS = ("Faxa_bcphidry Faxa_bcphodry Faxa_bcphiwet Faxa_ocphidry Faxa_ocphodry Faxa_ocphiwet "
                 "Faxa_dstdry1 Faxa_dstdry2 Faxa_dstdry3 Faxa_dstdry4 Faxa_dstwet1 Faxa_dstwet2 Faxa_dstwet3 "
                 "Faxa_dstwet4").split()
_X2O_GLC = "Fogx_qicelo Fogx_qiceho Sg_blit Sg_blis Sg_lithop Sg_icemask Sg_icemask_grounded Sg_icemask_floating"
_X2O_WAVES = ([n for m in range(1, 7) for n in (f"Sw_ustokes_wavenumber_{m}", f"Sw_vstokes_wavenumber_{m}")]
              + "Sw_Hs Sw_Fp Sw_Dp Sw_Z0 Sw_Ustar Sw_Charn Faww_Tawx Faww_Tawy Fwow_Twox Fwow_Twoy Faow_Tocx "
                "Faow_Tocy".split())


@dataclass(frozen=True)
class Indices:
    """mpaso_cpl_indices_set's columns, 0-based (-1: not in the list)."""

    x2o: dict
    o2x: dict
    nx2o: int
    no2x: int
    wav_ocn_coup: str
    glc_nzoc: int
    tf3d: tuple = ()
    tf3d_mask: tuple = ()

    def x(self, name: str) -> int:
        return self.x2o.get(name, -1)

    def o(self, name: str) -> int:
        return self.o2x.get(name, -1)


def indices(x2o_fields, o2x_fields, *, rof2ocn_nutrients: bool = False, wav_ocn_coup: str = "none",
            ocn_lnd_one_way: bool = False, glc_nzoc: int = 0) -> Indices:
    """``mpaso_cpl_indices_set`` for these lists (MCT's ``rList``), after
    ``glc_zocnclass_init(glc_nzoc)``; the lookups in the Fortran's order, so a
    missing field stops where E3SM's does. ``ocn_lnd_one_way`` only looks
    So_ssh up again (as ``So_frac_h2oocn``, a quirk)."""
    x2o_fields, o2x_fields = tuple(x2o_fields), tuple(o2x_fields)
    for label, names in (("x2o", x2o_fields), ("o2x", o2x_fields)):
        if len(set(names)) != len(names):
            raise ValueError(f"{label} repeats a field name; MCT's lists hold each once")
    if glc_nzoc not in GLC_NZOC:
        raise CouplingAbort("glc_zocnclass_init ERROR: unknown glc_nzoc")
    xpos = {n: k for k, n in enumerate(x2o_fields)}
    opos = {n: k for k, n in enumerate(o2x_fields)}

    def need(pos, name):
        if name not in pos:
            raise CouplingAbort(f'm_AttrVect::indexRA_:: FATAL--attribute not found: "{name}"')
        return pos[name]

    for name in _O2X_REQUIRED:
        need(opos, name)
    need(opos, "So_ssh")
    for name in _O2X_LAND_ICE:
        need(opos, name)
    tf, tfm = [], []
    for k in range(1, glc_nzoc + 1):
        tf.append(need(opos, f"So_tf3d{k:02d}"))
        tfm.append(need(opos, f"So_tf3d_mask{k:02d}"))
    for name in _X2O_REQUIRED:
        need(xpos, name)
    if rof2ocn_nutrients:
        for t in NUTRIENTS:
            need(xpos, f"Foxx_rof{t}")
    for name in _X2O_AEROSOLS + _X2O_GLC.split():
        need(xpos, name)
    if wav_ocn_coup.rstrip() == "twoway":  # Fortran's == ignores trailing blanks
        for name in _X2O_WAVES:
            need(xpos, name)
    del ocn_lnd_one_way  # its only lookup is So_ssh's, again
    return Indices(x2o=xpos, o2x=opos, nx2o=len(x2o_fields), no2x=len(o2x_fields),
                   wav_ocn_coup=wav_ocn_coup.rstrip(), glc_nzoc=glc_nzoc, tf3d=tuple(tf), tf3d_mask=tuple(tfm))


@dataclass(frozen=True)
class Options:
    """The namelist options, infodata and package the two routines read."""

    ocn_cpl_dt: int
    config_density0: float = 1026.0
    config_use_precip_scaling: bool = False
    precip_fact: float = 1.0
    config_remove_ais_river_runoff: bool = False
    config_remove_ais_ice_runoff: bool = False
    config_cvmix_kpp_use_theory_wave: bool = False
    config_eke_correction_enable: bool = False
    config_momentum_use_active_wave: bool = False
    frazil_ice_active: bool = True
    config_land_ice_flux_mode: str = "off"
    config_glc_thermal_forcing_coupling_mode: str = "off"
    config_frazil_heat_of_fusion: float = 3.34e5
    #: config_open_ocean_freezing_temperature_coeff_0, _S, _p, _pS, _mushy_az1_liq
    open_ocean_freezing: tuple = (0.0, 0.0, 0.0, 0.0, -18.48)
    config_use_ecosysTracers: bool = False
    config_use_DMSTracers: bool = False
    config_use_MacroMoleculesTracers: bool = False
    config_use_CFCTracers: bool = False

    def check(self) -> None:
        refused = [n for n in ("config_use_ecosysTracers", "config_use_DMSTracers",
                               "config_use_MacroMoleculesTracers", "config_use_CFCTracers") if getattr(self, n)]
        if refused:
            raise NotImplementedError(f"E3SM's {', '.join(refused)} coupling is not ported")
        if int(self.ocn_cpl_dt) != self.ocn_cpl_dt or self.ocn_cpl_dt == 0:
            raise ValueError("ocn_cpl_dt is the coupling interval in whole seconds, nonzero")
        _land_ice_mode(self.config_land_ice_flux_mode)
        _glc_3d(self.config_glc_thermal_forcing_coupling_mode)


_IMPORT_COPIES = (
    ("latentHeatFlux", "Foxx_lat"), ("sensibleHeatFlux", "Foxx_sen"), ("longWaveHeatFluxUp", "Foxx_lwup"),
    ("longWaveHeatFluxDown", "Faxa_lwdn"), ("evaporationFlux", "Foxx_evap"), ("seaIceHeatFlux", "Fioi_melth"),
    ("icebergHeatFlux", "PFioi_bergh"), ("snowFlux", "Faxa_snow"), ("seaIceFreshWaterFlux", "Fioi_meltw"),
    ("icebergFreshWaterFlux", "PFioi_bergw"), ("seaIceSalinityFlux", "Fioi_salt"),
)
_WAVE_COPIES = (
    ("significantWaveHeight", "Sw_Hs"), ("peakWaveFrequency", "Sw_Fp"), ("peakWaveDirection", "Sw_Dp"),
    ("z0wav", "Sw_Z0"), ("ustarwav", "Sw_Ustar"), ("charnockSeaState", "Sw_Charn"),
    ("waveSupportedZonalWindStress", "Faww_Tawx"), ("waveSupportedMeridionalWindStress", "Faww_Tawy"),
    ("waveToOceanZonalWindStress", "Fwow_Twox"), ("waveToOceanMeridionalWindStress", "Fwow_Twoy"),
    ("waveNetOceanZonalWindStress", "Faow_Tocx"), ("waveNetOceanMeridionalWindStress", "Faow_Tocy"),
)
R_NONE, R_COPY, R_TAU_WAVE, R_ROFL, R_ROFI, R_SWNET, R_RAIN, R_BPRESS, R_SQRT = range(-1, 8)


def import_rows(idx: Indices, opts: Options, present) -> list[tuple]:
    """The import as rows ``(array, component or None, code, c1, c2, c3,
    zeroFirst)`` in the Fortran's order; ``present`` is the forcing arrays a
    package has on. Row c2 of R_ROFL/R_ROFI is the removed row (-1: off)."""
    present = set(present)
    rows: list[list] = []
    where = {}

    def row(name, comp=None, code=R_COPY, c1=-1, c2=-1, c3=-1, zero=0):
        where[name, comp] = len(rows)
        rows.append([name, comp, code, c1, c2, c3, zero])

    x = idx.x
    twoway = idx.wav_ocn_coup == "twoway"
    if opts.config_momentum_use_active_wave and not twoway:
        raise ValueError("config_momentum_use_active_wave without wav_ocn_coup = 'twoway': E3SM reads x2o's "
                         "column 0 (out of bounds) for the waves' stress")
    # zeroed before the loop
    for config, name in ((opts.config_remove_ais_river_runoff, "removedRiverRunoffFlux"),
                         (opts.config_remove_ais_ice_runoff, "removedIceRunoffFlux")):
        if config:
            if name not in present:
                raise ValueError(f"{name} must be present when its removal is on (E3SM zeroes it)")
            row(name, code=R_NONE, zero=1)
    wind10 = opts.config_cvmix_kpp_use_theory_wave or opts.config_eke_correction_enable
    if wind10 and "windSpeed10m" not in present:
        raise ValueError("windSpeed10m must be present under the KPP theory-wave or EKE options (E3SM zeroes it)")
    for name, (main, wave_a, wave_b) in (("windStressZonal", ("Foxx_taux", "Faww_Tawx", "Fwow_Twox")),
                                         ("windStressMeridional", ("Foxx_tauy", "Faww_Tawy", "Fwow_Twoy"))):
        if name in present:
            if opts.config_momentum_use_active_wave:
                row(name, code=R_TAU_WAVE, c1=x(main), c2=x(wave_a), c3=x(wave_b))
            else:
                row(name, c1=x(main))
    for name, source in _IMPORT_COPIES:
        if name in present:
            row(name, c1=x(source))
    for name, source, code, config, removed in (
            ("riverRunoffFlux", "Foxx_rofl", R_ROFL, opts.config_remove_ais_river_runoff, "removedRiverRunoffFlux"),
            ("iceRunoffFlux", "Foxx_rofi", R_ROFI, opts.config_remove_ais_ice_runoff, "removedIceRunoffFlux")):
        if name in present:
            row(name, code=code, c1=x(source), c2=where[removed, None] if config else -1)
    for name, source, code in (("shortWaveHeatFlux", "Foxx_swnet", R_SWNET), ("rainFlux", "Faxa_rain", R_RAIN),
                               ("atmosphericPressure", "Sa_pslv", R_COPY),
                               ("seaIcePressure", "Si_bpress", R_BPRESS), ("iceFraction", "Si_ifrac", R_COPY)):
        if name in present:
            row(name, code=code, c1=x(source))
    if twoway:
        for name, letter in (("stokesDriftZonalWavenumber", "u"), ("stokesDriftMeridionalWavenumber", "v")):
            if name in present:
                for m in range(6):
                    row(name, m, c1=x(f"Sw_{letter}stokes_wavenumber_{m + 1}"))
        for name, source in _WAVE_COPIES:
            if name in present:
                row(name, c1=x(source))
    if wind10:
        row("windSpeed10m", code=R_SQRT, c1=x("So_duu10n"), zero=1)
    return [tuple(r) for r in rows]


def module_source() -> str:
    return files("woof.hex.coupled").joinpath("ocn_coupling_e3sm.cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "ocn_coupling_e3sm.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _options() -> tuple[str, ...]:
    from woof.hex import numerics as _hex_numerics
    return _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True, numerics=_hex_numerics._bind())


def _ints(values):
    import cupy as cp
    return cp.asarray(np.array(list(values) or [0], dtype=np.int32))


def _f64(name, array, shape=None):
    if array.dtype != np.float64 or not array.flags.c_contiguous or (shape is not None and array.shape != shape):
        raise ValueError(f"{name} must be C-contiguous float64" + (f" of shape {shape}" if shape else ""))
    return array


def _launch(kernel: str, n: int, args) -> None:
    import cupy as cp
    if n > 0:
        _module_on(cp.cuda.Device().id, _options()).get_function(kernel)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), tuple(args))


def import_x2o(idx: Indices, opts: Options, x2o, *, n_cells_solve: int, lat_cell, forcing: dict) -> None:
    """``ocn_import_mct``: ``x2o`` ``(lsize, nx2o)`` into ``forcing``'s
    arrays in place. ``forcing`` maps MPAS names to the device arrays a
    package has on: 1-D ``(nCells,)``, the Stokes drift ``(nCells, 6)``
    (MPAS's ``(6, nCells)`` in memory order); ``lat_cell`` is ``(nCells,)``."""
    import cupy as cp
    opts.check()
    if int(x2o.shape[0]) != n_cells_solve:
        raise ValueError("x2o's lsize must be nCellsSolve (one block)")
    _f64("x2o", x2o, (n_cells_solve, idx.nx2o))
    rows = import_rows(idx, opts, forcing)
    if not rows:
        return
    n_cells = int(lat_cell.shape[0])
    _f64("lat_cell", lat_cell, (n_cells,))
    for name, comp, *_ in rows:
        _f64(name, forcing[name], (n_cells,) if comp is None else (n_cells, 6))
    buf = cp.stack([forcing[name] if comp is None else forcing[name][:, comp] for name, comp, *_ in rows])
    buf = cp.ascontiguousarray(buf)
    prec = opts.precip_fact if opts.config_use_precip_scaling else 1.0
    _launch("ocn_import_e3sm", n_cells, (
        buf, np.int32(n_cells), np.int32(len(rows)), *(_ints(r[k] for r in rows) for k in (2, 3, 4, 5, 6)),
        x2o, np.int32(idx.nx2o), np.int32(n_cells_solve), lat_cell, np.float64(prec),
        np.float64(opts.config_density0), np.float64(GRAVITY)))
    for j, (name, comp, *_) in enumerate(rows):
        if comp is None:
            forcing[name][...] = buf[j]
        else:
            forcing[name][:, comp] = buf[j]


_O_ORDER = ("So_t", "So_s", "So_u", "So_v", "So_ssh", "So_dhdx", "So_dhdy", "Faoo_h2otemp", "Foxo_ismw",
            "Foxo_ismh", "Foxo_rrofl", "Foxo_rrofi", "Foxo_rrofih", "Fioo_q", "Fioo_frazil", "Foxo_q_li",
            "Foxo_frazil_li", "So_blt", "So_bls", "So_htv", "So_stv", "So_rhoeff")


#: config_land_ice_flux_mode's possible values (Registry.xml); E3SM stops on any other at init
_LAND_ICE_MODES = {"off": 0, "pressure_only": 0, "standalone": 1, "data": 2, "coupled": 3}
#: config_glc_thermal_forcing_coupling_mode's possible values
_GLC_TF_MODES = ("off", "3d")


def _land_ice_mode(mode: str) -> int:
    key = mode.rstrip()  # Fortran's trim drops trailing blanks only
    if key not in _LAND_ICE_MODES:
        raise ValueError(f"config_land_ice_flux_mode {mode!r} is not one of {', '.join(_LAND_ICE_MODES)} "
                         "(E3SM stops on it: 'unknown land_ice_flux_mode')")
    return _LAND_ICE_MODES[key]


def _glc_3d(mode: str) -> bool:
    if mode.rstrip() not in _GLC_TF_MODES:
        raise ValueError(f"config_glc_thermal_forcing_coupling_mode {mode!r} is not one of "
                         f"{', '.join(_GLC_TF_MODES)}")
    return mode.rstrip() == "3d"


def export_columns(idx: Indices, opts: Options) -> list[str]:
    """The o2x fields the export writes under these options."""
    li = _land_ice_mode(opts.config_land_ice_flux_mode)
    out = ["So_t", "So_s", "So_u", "So_v", "So_ssh", "So_dhdx", "So_dhdy", "Faoo_h2otemp"]
    out += ["Foxo_ismw", "Foxo_ismh"] if li in (1, 2) else []
    out += ["Foxo_rrofl"] if opts.config_remove_ais_river_runoff else []
    out += ["Foxo_rrofi", "Foxo_rrofih"] if opts.config_remove_ais_ice_runoff else []
    out += ["Fioo_q", "Fioo_frazil"] + (["Foxo_q_li", "Foxo_frazil_li"] if li in (1, 2) else []) \
        if opts.frazil_ice_active else []
    out += list(_O2X_LAND_ICE) if li in (1, 3) else []
    return out


def export_o2x(idx: Indices, opts: Options, o2x, *, n_cells_solve: int, forcing: dict, state: dict, dims: dict,
               diagnostics: dict) -> None:
    """``ocn_export_mct`` into ``o2x`` ``(lsize, no2x)`` in place; resets the
    frazil arrays (``seaIceEnergy``, ``frazilSurfacePressure``,
    ``accumulatedFrazilIceMass``) as E3SM does.

    ``forcing``/``state`` map MPAS names to device arrays (a ``(k, nCells)``
    array as ``(nCells, k)``; ``landIceMask`` int32, absent when its package
    is off); ``dims`` holds the forcing pool's 1-based tracer indices
    (``index_avgTemperatureSurfaceValue``, ...) and ``nGlcZLevels``;
    ``diagnostics`` holds ``landIceBoundaryLayerTracers``,
    ``landIceTracerTransferVelocities`` and their 1-based ``indexBLT``,
    ``indexBLS``, ``indexHeatTrans``, ``indexSaltTrans``."""
    opts.check()
    _f64("o2x", o2x, (n_cells_solve, idx.no2x))
    written = export_columns(idx, opts)
    missing = [n for n in written if idx.o(n) < 0]
    if missing:
        raise ValueError(f"o2x lacks {', '.join(missing)}, which the export writes: E3SM looks them up quietly "
                         "and writes column 0 (out of bounds)")
    li = _land_ice_mode(opts.config_land_ice_flux_mode)
    glc3d = _glc_3d(opts.config_glc_thermal_forcing_coupling_mode)
    n_glc_z = int(dims.get("nGlcZLevels", 0)) if glc3d else 0
    if n_glc_z > idx.glc_nzoc:
        raise ValueError(f"nGlcZLevels ({n_glc_z}) exceeds glc_nzoc ({idx.glc_nzoc}): E3SM writes column 0")
    import cupy as cp
    col = [idx.o(n) for n in _O_ORDER]
    dummy = cp.zeros(1, dtype=np.float64)
    idummy = cp.zeros(1, dtype=np.int32)

    def arr(pool, name, on=True, columns=None):
        """The array, its width; refuses what the kernel would read past:
        fewer than nCellsSolve cells, or (``columns``: the 0-based
        components it reads) a component outside a 2-D array."""
        if not on:
            return dummy, 1
        a = _f64(name, pool[name])
        if a.ndim != (1 if columns is None else 2) or int(a.shape[0]) < n_cells_solve:
            raise ValueError(f"{name} must be {'1' if columns is None else '2'}-D with at least nCellsSolve "
                             f"({n_cells_solve}) cells; it is {a.shape}")
        width = int(a.shape[1]) if a.ndim == 2 else 1
        bad = [k + 1 for k in (columns or ()) if not 0 <= k < width]
        if bad:
            raise ValueError(f"{name} has {width} components; 1-based index {bad[0]} is outside it")
        return a, width

    def index(pool, name):
        return int(pool.get(name, 1)) - 1

    i_t, i_s = dims["index_avgTemperatureSurfaceValue"] - 1, dims["index_avgSalinitySurfaceValue"] - 1
    i_u, i_v = dims["index_avgSurfaceVelocityZonal"] - 1, dims["index_avgSurfaceVelocityMeridional"] - 1
    i_dx, i_dy = dims["index_avgSSHGradientZonal"] - 1, dims["index_avgSSHGradientMeridional"] - 1
    i_blt, i_bls = index(diagnostics, "indexBLT"), index(diagnostics, "indexBLS")
    i_ht, i_st = index(diagnostics, "indexHeatTrans"), index(diagnostics, "indexSaltTrans")
    tsv, n_tsv = arr(forcing, "avgTracersSurfaceValue", columns=(i_t, i_s))
    vel, n_vel = arr(forcing, "avgSurfaceVelocity", columns=(i_u, i_v))
    grad, n_grad = arr(forcing, "avgSSHGradient", columns=(i_dx, i_dy))
    sa = li in (1, 2)
    frz = opts.frazil_ice_active
    mask = forcing.get("landIceMask") if frz else None
    if mask is not None and (mask.dtype != np.int32 or not mask.flags.c_contiguous or mask.ndim != 1
                             or int(mask.shape[0]) < n_cells_solve):
        raise ValueError("landIceMask must be C-contiguous 1-D int32 with at least nCellsSolve cells")
    lt, n_vert = arr(state, "layerThickness", frz, columns=(0,))
    blt, n_blt = arr(diagnostics, "landIceBoundaryLayerTracers", li in (1, 3), columns=(i_blt, i_bls))
    tv, n_tv = arr(diagnostics, "landIceTracerTransferVelocities", li in (1, 3), columns=(i_ht, i_st))
    levels = tuple(range(n_glc_z))
    tf, n_z = arr(forcing, "avgThermalForcingAtZLevels", glc3d, columns=levels)
    tfm, _ = arr(forcing, "avgThermalForcingAtZLevelsMask", glc3d, columns=levels)
    if glc3d and tfm.shape != tf.shape:
        raise ValueError("avgThermalForcingAtZLevelsMask must have avgThermalForcingAtZLevels' shape")
    _launch("ocn_export_e3sm", n_cells_solve, (
        o2x, np.int32(idx.no2x), _ints(col), np.int32(n_cells_solve),
        tsv, np.int32(n_tsv), np.int32(i_t), np.int32(i_s),
        vel, np.int32(n_vel), np.int32(i_u), np.int32(i_v),
        arr(state, "ssh")[0], grad, np.int32(n_grad), np.int32(i_dx), np.int32(i_dy),
        arr(forcing, "avgTotalFreshWaterTemperatureFlux")[0], np.float64(RHO_SW), np.float64(CP_SW),
        np.float64(T0_KELVIN),
        np.int32(li), arr(forcing, "avgLandIceFreshwaterFlux", sa)[0], arr(forcing, "avgLandIceHeatFlux", sa)[0],
        np.int32(opts.config_remove_ais_river_runoff),
        arr(forcing, "avgRemovedRiverRunoffFlux", opts.config_remove_ais_river_runoff)[0],
        np.int32(opts.config_remove_ais_ice_runoff),
        arr(forcing, "avgRemovedIceRunoffFlux", opts.config_remove_ais_ice_runoff)[0],
        arr(forcing, "avgRemovedIceRunoffHeatFlux", opts.config_remove_ais_ice_runoff)[0],
        np.int32(frz), np.int32(mask is not None), idummy if mask is None else mask,
        arr(forcing, "seaIceEnergy", frz)[0], arr(forcing, "frazilSurfacePressure", frz)[0],
        arr(state, "accumulatedFrazilIceMass", frz)[0], lt, np.int32(n_vert),
        np.float64(opts.config_frazil_heat_of_fusion), np.float64(int(opts.ocn_cpl_dt)),
        cp.asarray(np.array(opts.open_ocean_freezing, dtype=np.float64)),
        blt, np.int32(n_blt), np.int32(i_blt), np.int32(i_bls),
        tv, np.int32(n_tv), np.int32(i_ht), np.int32(i_st),
        np.int32(glc3d), np.int32(n_glc_z), _ints(idx.tf3d), _ints(idx.tf3d_mask), tf, tfm, np.int32(n_z)))
