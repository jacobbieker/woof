"""E3SM v3.2.0's MPAS-Seaice coupler import and export on the GPU, opt-in.

E3SM's sea ice (``components/mpas-seaice/driver/ice_comp_mct.F``) reads what
the coupler sends it (``x2i``) into its pools with ``ice_import_mct`` and
fills what it sends back (``i2x``) with ``ice_export_mct``. :func:`indices`
is ``mpassi_cpl_indices_set`` (the columns, with the stops MCT makes on a
list that lacks a required field); :func:`import_x2i` and :func:`export_i2x`
are the two routines, word for word:

- the import copies the ocean and atmosphere states and fluxes into the
  ``ocean_coupling`` and ``atmos_coupling`` arrays, sets the sea freezing
  temperature with Icepack's ``icepack_sea_freezing_temperature`` (every
  ``tfrz_option``: ``mushy``, which is the mush liquidus under mushy
  thermodynamics and the linear one otherwise, ``linear_salt``, ``constant``
  and ``minus1p8``), takes the frazil mass the ocean made less the mass
  ``frazil_mass`` makes of the freezing potential (free ocean surface only),
  converts the sea surface temperature to Celsius, sums the aerosol and
  z-aerosol deposition (bulk or modal) and zeroes the z-aerosol ocean
  concentrations, and reads the waves coupled two ways (the column waves);
- the export zeroes ``i2x`` and writes the ice fraction (``min(area, 1)``),
  the icebergs' fluxes, the floe size (waves two ways with the floe size
  distribution) and, where the fraction is positive, the surface
  temperature in Kelvin, the basal pressure (with the topographic ponds),
  the 10 m and 2 m references, the snow and ice thicknesses, the
  atmosphere and ocean stresses rotated back to the geographic grid, the
  atmosphere fluxes, the albedos (column shortwave) and the ocean fluxes
  with the frazil mass adjustment.

This is not called by any default path. ``real(r8)`` only. The halo
exchanges are the identity on one block. The stress rotation's sine,
cosine, arcsine and two-argument arctangent are glibc's in E3SM; they come
in as ``trig`` (per cell, sin and cos of the latitude, the longitude, and
the rotated grid's latitude and longitude), so no libm runs on the device.
:func:`rotation_trig` makes them with Python's ``math`` module, which is the
platform libm: under the glibc the oracle was recorded with
(:data:`TRIG_GLIBC`, 2.44) its words are the oracle's; under another glibc
release or another libm (macOS's) they may differ in the last place, and it
warns. The parity tests feed the oracle's recorded words.

Refused by name (see QUIRKS in the module's tests): the column
biogeochemistry coupled to the ocean (``config_use_column_biogeochemistry``
with ``config_couple_biogeochemistry_fields``; its ``i2x``/``x2i`` columns
are looked up quietly, so an absent field is written at column 0, and its
branches are not ported); the z-aerosols coupled without it (the export
reads ``oceanDustIronFlux`` through a pool pointer only the column
biogeochemistry sets: E3SM faults; so ``Fioi_dust1`` is never written); more ``nFrequencies`` than ``wav_nfreq`` under the
waves coupled two ways with the column waves (E3SM reads past
``index_x2i_Sw_wavespec``); an unknown thermodynamics or freezing type.
Kept as E3SM has them: ``Si_u10withgusts`` is ``Si_u10``; under an ocean
surface that is not ``free`` the import leaves ``frazilMassAdjust`` alone
and the export still reads it; ``Si_ifloe`` must be in the list under the
waves two ways but is written only with the floe size distribution; a
wave frequency past 99 is named ``Sw_wavespec**`` (``i2.2``).

NaN inputs are outside the word-for-word claim (gfortran's MIN and MAX with
a NaN depend on the code's shape; GPU arithmetic returns a canonical NaN).

The kernels (``ice_coupling_e3sm.cu``) are held word for word to E3SM's
routines cut verbatim out of ``ice_comp_mct.F`` over ``mpassi_cpl_indices``,
``seaice_constants`` (``CCSMCOUPLED``: ``shr_const_mod``'s values) and the
pinned Icepack's column physics, built with gfortran ``-O0
-ffp-contract=off`` (``tools/e3sm_coupler_oracle``,
``tests/test_e3sm_seaice_coupling_parity.py``).
"""
from __future__ import annotations

import math
import platform
import warnings
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np

E3SM_TAG = "v3.2.0"
E3SM_COMMIT = "648b167b7f95267a159a9fdee25faf604c2bfc4f"
ICEPACK_COMMIT = "cf1d460fbb6a13d28988f798b33734f46d70a9dc"
SOURCE_FILES = ("components/mpas-seaice/driver/ice_comp_mct.F:ice_import_mct,ice_export_mct,basal_pressure,"
                "frazil_mass",
                "components/mpas-seaice/driver/mpassi_cpl_indices.F:mpassi_cpl_indices_set",
                "components/mpas-seaice/src/shared/mpas_seaice_mesh.F:seaice_latlon_vector_rotation_backward",
                "components/mpas-seaice/src/shared/mpas_seaice_constants.F",
                "icepack/columnphysics/icepack_therm_shared.F90:icepack_sea_freezing_temperature,"
                "icepack_liquidus_temperature",
                "icepack/columnphysics/icepack_mushy_physics.F90:liquidus_temperature_mush,"
                "liquidus_brine_salinity_mush,icepack_enthalpy_mush")
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128

# seaice_constants under CCSMCOUPLED (shr_const_mod's) and its own
TFFRESH = 273.15  # seaiceFreshWaterFreezingPoint
RHOI = 0.917e3  # seaiceDensityIce
RHOW = 1.026e3  # seaiceDensitySeaWater
RHOS = 330.0  # seaiceDensitySnow
GRAVITY = 9.80616  # seaiceGravity
CP_ICE = 2.11727e3  # seaiceFreshIceSpecificHeat
CP_OCN = 3.996e3  # seaiceSeaWaterSpecificHeat
LATICE = 3.337e5  # seaiceLatentHeatMelting
LATVAP = 2.501e6  # seaiceLatentHeatVaporization
LATSUB = LATICE + LATVAP  # SHR_CONST_LATSUB
LFRESH = LATSUB - LATVAP  # Icepack's Lfresh = Lsub - Lvap
REF_SALINITY = 4.0  # seaiceReferenceSalinity
PUNY = 1.0e-11
DEPRESS_T = 0.054  # seaiceMeltingTemperatureDepression
TOCNFRZ = -1.8  # seaiceFreezingTemperatureConstant
FRAZIL_SALINITY_REDUCTION = 3.0
FRAZIL_ICE_POROSITY = 0.75

#: the glibc release the oracle's trigonometric words were recorded with
TRIG_GLIBC = "2.44"
#: mpas_seaice_initialize's ocean surfaces ("free"; the rest is the slab ocean) and seq_flds_mod's wave couplings
OCEAN_SURFACES = ("free", "mixed_layer")
WAV_ICE_COUPLINGS = ("none", "oneway", "twoway")
KTHERM = {"zero layer": 0, "BL99": 1, "mushy": 2}
TFRZ = {"mushy": 0, "linear_salt": 1, "constant": 2, "minus1p8": 3}
R_COPY, R_SUM, R_ZERO, R_SST, R_TF, R_FRAZIL = range(6)
_NC = 8


class CouplingAbort(RuntimeError):
    """What E3SM stops on (MCT's ``die`` on a field ``mct_aVect_indexRA``
    must find); the message is MCT's."""


_I2X_REQUIRED = ("Si_t Si_bpress Si_tref Si_qref Si_ifrac Si_ithick Si_avsdr Si_anidr Si_avsdf Si_anidf Si_snowh "
                 "Si_u10 Si_u10withgusts Faii_taux Faii_tauy Faii_lat Faii_sen Faii_lwup Faii_evap Faii_swnet "
                 "Fioi_swpen Fioi_melth Fioi_meltw Fioi_salt Fioi_taux Fioi_tauy PFioi_bergh PFioi_bergw").split()
_X2I_REQUIRED = ("So_t So_s So_u So_v Sa_z Sa_u Sa_v Sa_tbot Sa_ptem Sa_shum Sa_dens So_dhdx So_dhdy Faxa_lwdn "
                 "Faxa_rain Faxa_snow Faxa_swndr Faxa_swvdr Faxa_swndf Faxa_swvdf Fioo_q Fioo_frazil "
                 "Faxa_bcphidry Faxa_bcphodry Faxa_bcphiwet Faxa_ocphidry Faxa_ocphodry Faxa_ocphiwet "
                 "Faxa_dstdry1 Faxa_dstdry2 Faxa_dstdry3 Faxa_dstdry4 Faxa_dstwet1 Faxa_dstwet2 Faxa_dstwet3 "
                 "Faxa_dstwet4").split()


def wavespec_name(k: int) -> str:
    """``'Sw_wavespec' // freqnum`` with ``write(freqnum, '(i2.2)') k``."""
    return "Sw_wavespec" + (f"{k:02d}" if 0 <= k <= 99 else "**")


@dataclass(frozen=True)
class Indices:
    """mpassi_cpl_indices_set's columns, 0-based (-1: not in the list)."""

    x2i: dict
    i2x: dict
    nx2i: int
    ni2x: int
    wav_ice_coup: str
    wav_nfreq: int
    wavespec: tuple = ()

    def x(self, name: str) -> int:
        return self.x2i.get(name, -1)

    def i(self, name: str) -> int:
        return self.i2x.get(name, -1)


def indices(x2i_fields, i2x_fields, *, wav_ice_coup: str = "none", wav_nfreq: int = 0) -> Indices:
    """``mpassi_cpl_indices_set`` for these lists (MCT's ``rList``); the
    lookups in the Fortran's order, so a missing field stops where E3SM's
    does. The biogeochemistry lookups are quiet (and unused: refused)."""
    x2i_fields, i2x_fields = tuple(x2i_fields), tuple(i2x_fields)
    for label, names in (("x2i", x2i_fields), ("i2x", i2x_fields)):
        if len(set(names)) != len(names):
            raise ValueError(f"{label} repeats a field name; MCT's lists hold each once")
    xpos = {n: k for k, n in enumerate(x2i_fields)}
    ipos = {n: k for k, n in enumerate(i2x_fields)}

    def need(pos, name):
        if name not in pos:
            raise CouplingAbort(f'm_AttrVect::indexRA_:: FATAL--attribute not found: "{name}"')
        return pos[name]

    if wav_ice_coup.rstrip() not in WAV_ICE_COUPLINGS:
        raise ValueError(f"unknown wav_ice_coup {wav_ice_coup!r}: one of {WAV_ICE_COUPLINGS}")
    if int(wav_nfreq) < 0:
        raise ValueError(f"wav_nfreq must not be negative; it is {wav_nfreq}")
    twoway = wav_ice_coup.rstrip() == "twoway"  # Fortran's .eq. ignores trailing blanks
    for name in _I2X_REQUIRED:
        need(ipos, name)
    if twoway:
        need(ipos, "Si_ifloe")
    for name in _X2I_REQUIRED:
        need(xpos, name)
    spec = ()
    if twoway:
        need(xpos, "Sw_Hs")
        spec = tuple(need(xpos, wavespec_name(k)) for k in range(1, wav_nfreq + 1))
    return Indices(x2i=xpos, i2x=ipos, nx2i=len(x2i_fields), ni2x=len(i2x_fields), wav_ice_coup=wav_ice_coup.rstrip(),
                   wav_nfreq=int(wav_nfreq), wavespec=spec)


@dataclass(frozen=True)
class Options:
    """The namelist switches the two routines read (E3SM's defaults)."""

    config_thermodynamics_type: str = "mushy"
    config_sea_freezing_temperature_type: str = "mushy"  # Icepack's tfrz_option
    config_ocean_surface_type: str = "free"
    config_rotate_cartesian_grid: bool = True
    config_use_topo_meltponds: bool = False
    config_use_column_shortwave: bool = True
    config_use_data_icebergs: bool = False
    config_use_aerosols: bool = False
    config_use_modal_aerosols: bool = False
    config_use_zaerosols: bool = False
    config_use_column_biogeochemistry: bool = False
    config_couple_biogeochemistry_fields: bool = False
    config_use_column_waves: bool = False
    config_use_floe_size_distribution: bool = False

    def check(self) -> None:
        if self.config_thermodynamics_type.rstrip() not in KTHERM:
            raise ValueError(f"unknown config_thermodynamics_type {self.config_thermodynamics_type!r}")
        if self.config_sea_freezing_temperature_type.rstrip() not in TFRZ:
            raise ValueError(f"unknown tfrz_option {self.config_sea_freezing_temperature_type!r}: "
                             "Icepack's icepack_sea_freezing_temperature aborts")
        if self.config_ocean_surface_type.rstrip() not in OCEAN_SURFACES:
            raise ValueError(f"unknown config_ocean_surface_type {self.config_ocean_surface_type!r}: one of "
                             f"{OCEAN_SURFACES} (E3SM takes anything but 'free' as the slab ocean)")
        if self.config_use_column_biogeochemistry and self.config_couple_biogeochemistry_fields:
            raise ValueError("the column biogeochemistry coupled to the ocean is not ported (its x2i and i2x "
                             "columns are looked up quietly, so an absent field is written at column 0)")
        if self.config_use_zaerosols and self.config_couple_biogeochemistry_fields:
            raise ValueError("the coupled z-aerosols without the column biogeochemistry: E3SM's export reads "
                             "oceanDustIronFlux through a biogeochemistry pool pointer it never sets")

    @property
    def ktherm(self) -> int:
        return KTHERM[self.config_thermodynamics_type.rstrip()]

    @property
    def tfrz(self) -> int:
        return TFRZ[self.config_sea_freezing_temperature_type.rstrip()]

    @property
    def mushy(self) -> bool:
        return self.config_thermodynamics_type.rstrip() == "mushy"

    @property
    def free_surface(self) -> bool:
        return self.config_ocean_surface_type.rstrip() == "free"


_DUST_WET = tuple(f"Faxa_dstwet{k}" for k in range(1, 5))
_DUST_DRY = tuple(f"Faxa_dstdry{k}" for k in range(1, 5))


def import_rows(idx: Indices, opts: Options, n_frequencies: int) -> list[tuple]:
    """``(pool, name, component or None, code, x2i columns)`` for each row
    the import writes, in the Fortran's order."""
    x = idx.x
    rows = []

    def row(pool, name, code, *fields, comp=None):
        rows.append((pool, name, comp, code, tuple(x(f) for f in fields)))

    oc, ac = "ocean_coupling", "atmos_coupling"
    row(oc, "seaSurfaceTemperature", R_SST, "So_t")
    row(oc, "seaSurfaceSalinity", R_COPY, "So_s")
    row(oc, "seaFreezingTemperature", R_TF, "So_s")
    for name, f in (("uOceanVelocity", "So_u"), ("vOceanVelocity", "So_v"), ("seaSurfaceTiltU", "So_dhdx"),
                    ("seaSurfaceTiltV", "So_dhdy")):
        row(oc, name, R_COPY, f)
    if idx.wav_ice_coup == "twoway" and opts.config_use_column_waves:
        row("wave_coupling", "significantWaveHeight", R_COPY, "Sw_Hs")
        for k in range(n_frequencies):
            rows.append(("wave_coupling", "waveSpectra", k, R_COPY, (idx.wavespec[k],)))
    row(oc, "freezingMeltingPotential", R_COPY, "Fioo_q")
    if opts.free_surface:
        row(oc, "frazilMassAdjust", R_FRAZIL, "Fioo_frazil", "Fioo_q", "So_s")
    for name, f in (("airLevelHeight", "Sa_z"), ("airPotentialTemperature", "Sa_ptem"), ("airTemperature", "Sa_tbot"),
                    ("airSpecificHumidity", "Sa_shum"), ("airDensity", "Sa_dens"),
                    ("shortwaveVisibleDirectDown", "Faxa_swvdr"), ("shortwaveVisibleDiffuseDown", "Faxa_swvdf"),
                    ("shortwaveIRDirectDown", "Faxa_swndr"), ("shortwaveIRDiffuseDown", "Faxa_swndf"),
                    ("longwaveDown", "Faxa_lwdn"), ("rainfallRate", "Faxa_rain"), ("snowfallRate", "Faxa_snow"),
                    ("uAirVelocity", "Sa_u"), ("vAirVelocity", "Sa_v")):
        row(ac, name, R_COPY, f)
    modal = opts.config_use_modal_aerosols

    def black_carbon(pool, name):
        if modal:
            row(pool, name, R_SUM, "Faxa_bcphodry", "Faxa_bcphidry", comp=0)
            row(pool, name, R_COPY, "Faxa_bcphiwet", comp=1)
        else:
            row(pool, name, R_COPY, "Faxa_bcphodry", comp=0)
            row(pool, name, R_SUM, "Faxa_bcphidry", "Faxa_bcphiwet", comp=1)

    if opts.config_use_aerosols:
        black_carbon("aerosols", "atmosAerosolFlux")
        row("aerosols", "atmosAerosolFlux", R_SUM, *_DUST_WET, *_DUST_DRY, comp=2)
    bgc = "biogeochemistry"
    if opts.config_use_zaerosols or opts.config_use_column_biogeochemistry:
        for k in range(6):
            row(bgc, "oceanZAerosolConc", R_ZERO, comp=k)
    if opts.config_use_zaerosols:
        black_carbon(bgc, "atmosBlackCarbonFlux")
        for k in range(4):
            row(bgc, "atmosDustFlux", R_SUM, _DUST_WET[k], _DUST_DRY[k], comp=k)
        # the modal branch writes the wet dust first, the bulk the dry; the words are the same
        for k in range(4):
            row(bgc, "atmosWetDustFlux", R_COPY, _DUST_WET[k], comp=k)
        for k in range(4):
            row(bgc, "atmosDryDustFlux", R_COPY, _DUST_DRY[k], comp=k)
    return rows


def module_source() -> str:
    return files("woof.hex.coupled").joinpath("ice_coupling_e3sm.cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "ice_coupling_e3sm.cu")
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
    import cupy as cp
    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a cupy (device) array; it is {type(array).__name__}")
    if array.dtype != np.float64 or not array.flags.c_contiguous or (shape is not None and array.shape != shape):
        raise ValueError(f"{name} must be C-contiguous float64" + (f" of shape {shape}" if shape else ""))
    return array


def _launch(kernel: str, n: int, args) -> None:
    import cupy as cp
    if n > 0:
        _module_on(cp.cuda.Device().id, _options()).get_function(kernel)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), tuple(args))


def _pool_array(pools: dict, pool: str, name: str):
    try:
        return pools[pool][name]
    except KeyError:
        raise ValueError(f"{pool}/{name} is needed and missing") from None


def import_x2i(idx: Indices, opts: Options, x2i, *, n_cells_solve: int, pools: dict, n_frequencies: int = 0) -> None:
    """``ice_import_mct``: ``x2i`` ``(lsize, nx2i)`` into the pools' arrays in
    place. ``pools`` maps a pool name to ``{array name: device array}``: 1-D
    ``(nCells,)``, 2-D ``(nCells, k)`` (MPAS's ``(k, nCells)`` in memory
    order). ``n_frequencies`` is the ``nFrequencies`` dimension."""
    import cupy as cp
    opts.check()
    if n_cells_solve < 0 or n_frequencies < 0:
        raise ValueError(f"nCellsSolve ({n_cells_solve}) and nFrequencies ({n_frequencies}) must not be negative")
    _f64("x2i", x2i)
    if int(x2i.shape[0]) != n_cells_solve:
        raise ValueError("x2i's lsize must be nCellsSolve (one block)")
    _f64("x2i", x2i, (n_cells_solve, idx.nx2i))
    waves = idx.wav_ice_coup == "twoway" and opts.config_use_column_waves
    if waves and n_frequencies > idx.wav_nfreq:
        raise ValueError(f"nFrequencies ({n_frequencies}) exceeds wav_nfreq ({idx.wav_nfreq}): E3SM reads past "
                         "index_x2i_Sw_wavespec")
    rows = import_rows(idx, opts, n_frequencies if waves else 0)
    n_cells = int(_pool_array(pools, *rows[0][:2]).shape[0])
    for pool, name, comp, _, cols in rows:
        a = _f64(f"{pool}/{name}", _pool_array(pools, pool, name))
        if a.ndim != (1 if comp is None else 2) or int(a.shape[0]) != n_cells:
            raise ValueError(f"{pool}/{name} must be {'1' if comp is None else '2'}-D over the {n_cells} cells; "
                             f"it is {a.shape}")
        if comp is not None and not comp < int(a.shape[1]):
            raise ValueError(f"{pool}/{name} has {a.shape[1]} components; 1-based index {comp + 1} is outside it")
        if any(c < 0 for c in cols):
            raise ValueError(f"{pool}/{name} reads an x2i field the list lacks")
    if n_cells < n_cells_solve:
        raise ValueError(f"the arrays have {n_cells} cells, fewer than nCellsSolve ({n_cells_solve})")
    buf = cp.ascontiguousarray(cp.stack([
        _pool_array(pools, p, n) if c is None else _pool_array(pools, p, n)[:, c] for p, n, c, *_ in rows]))
    cols = np.full((len(rows), _NC), -1, dtype=np.int32)
    for j, r in enumerate(rows):
        cols[j, :len(r[4])] = r[4]
    _launch("ice_import_e3sm", n_cells_solve, (
        buf, np.int32(n_cells), np.int32(len(rows)), _ints(r[3] for r in rows), _ints(len(r[4]) for r in rows),
        cp.asarray(cols), x2i, np.int32(idx.nx2i), np.int32(n_cells_solve),
        np.float64(TFFRESH), np.int32(opts.tfrz), np.int32(opts.ktherm), np.int32(opts.mushy),
        np.float64(RHOI), np.float64(LATICE), np.float64(FRAZIL_SALINITY_REDUCTION), np.float64(FRAZIL_ICE_POROSITY),
        np.float64(RHOI), np.float64(RHOW), np.float64(CP_ICE), np.float64(CP_OCN), np.float64(LFRESH),
        np.float64(DEPRESS_T), np.float64(TOCNFRZ), np.float64(PUNY)))
    # the solve cells only: the halo is E3SM's to leave alone
    for j, (pool, name, comp, *_) in enumerate(rows):
        if comp is None:
            pools[pool][name][:n_cells_solve] = buf[j, :n_cells_solve]
        else:
            pools[pool][name][:n_cells_solve, comp] = buf[j, :n_cells_solve]


#: the export's i2x columns, in the kernel's I_* order
_I_ORDER = ("Si_ifrac", "PFioi_bergw", "PFioi_bergh", "Si_ifloe", "Si_t", "Si_bpress", "Si_u10", "Si_u10withgusts",
            "Si_tref", "Si_qref", "Si_snowh", "Si_ithick", "Faii_taux", "Faii_tauy", "Faii_lat", "Faii_sen",
            "Faii_lwup", "Faii_evap", "Faii_swnet", "Si_avsdr", "Si_anidr", "Si_avsdf", "Si_anidf", "Fioi_melth",
            "Fioi_swpen", "Fioi_meltw", "Fioi_salt", "Fioi_taux", "Fioi_tauy")
#: the export's arrays, in the kernel's A_* order: (pool, name)
_A_ORDER = tuple((p, n) for p, names in (
    ("tracers_aggregate", "iceAreaCell iceVolumeCell snowVolumeCell pondDepthCell pondLidThicknessCell pondAreaCell "
                          "surfaceTemperatureCell"),
    ("velocity_solver", "airStressCellU airStressCellV oceanStressCellU oceanStressCellV"),
    ("shortwave", "albedoVisibleDirectCell albedoIRDirectCell albedoVisibleDiffuseCell albedoIRDiffuseCell "
                  "absorbedShortwaveFlux"),
    ("atmos_coupling", "atmosReferenceSpeed10m atmosReferenceTemperature2m atmosReferenceHumidity2m"),
    ("ocean_coupling", "frazilMassAdjust"),
    ("atmos_fluxes", "latentHeatFlux sensibleHeatFlux longwaveUp evaporativeWaterFlux"),
    ("ocean_fluxes", "oceanHeatFlux oceanShortwaveFlux oceanFreshWaterFlux oceanSaltFlux"),
    ("berg_fluxes", "bergFreshwaterFlux bergLatentHeatFlux"),
    ("floe_size_distribution", "floeSizeDiameter")) for n in names.split())


def export_i2x(idx: Indices, opts: Options, i2x, *, n_cells_solve: int, pools: dict, trig) -> None:
    """``ice_export_mct``: fills ``i2x`` ``(lsize, ni2x)`` from the pools'
    arrays. ``trig`` is ``(nCellsSolve, 8)``: per cell sin and cos of
    ``latCell``, ``lonCell`` and the rotated grid's latitude and longitude
    (:func:`rotation_trig`, or the words E3SM's libm makes)."""
    import cupy as cp
    opts.check()
    if n_cells_solve < 0:
        raise ValueError(f"nCellsSolve ({n_cells_solve}) must not be negative")
    _f64("i2x", i2x)
    if int(i2x.shape[0]) != n_cells_solve:
        raise ValueError("i2x's lsize must be nCellsSolve (one block)")
    _f64("i2x", i2x, (n_cells_solve, idx.ni2x))
    _f64("trig", trig, (n_cells_solve, 8))
    floe = idx.wav_ice_coup == "twoway" and opts.config_use_floe_size_distribution
    on = {"berg_fluxes": opts.config_use_data_icebergs, "floe_size_distribution": opts.config_use_floe_size_distribution}
    stacked = []
    for pool, name in _A_ORDER:
        if not on.get(pool, True):
            stacked.append(cp.zeros(n_cells_solve, dtype=np.float64))
            continue
        a = _f64(f"{pool}/{name}", _pool_array(pools, pool, name))
        if a.ndim != 1 or int(a.shape[0]) < n_cells_solve:
            raise ValueError(f"{pool}/{name} must be 1-D with at least nCellsSolve ({n_cells_solve}) cells; "
                             f"it is {a.shape}")
        stacked.append(a[:n_cells_solve])
    arr = cp.ascontiguousarray(cp.stack(stacked))
    _launch("ice_export_e3sm", n_cells_solve, (
        i2x, np.int32(idx.ni2x), _ints(idx.i(n) for n in _I_ORDER), np.int32(n_cells_solve), arr, trig,
        np.int32(opts.config_rotate_cartesian_grid), np.int32(opts.config_use_topo_meltponds),
        np.int32(opts.config_use_column_shortwave), np.int32(opts.config_use_data_icebergs), np.int32(floe),
        np.float64(TFFRESH), np.float64(RHOI), np.float64(RHOS), np.float64(RHOW),
        np.float64(GRAVITY), np.float64(REF_SALINITY)))


def _asin(v: float) -> float:
    try:
        return math.asin(v)
    except ValueError:  # |v| > 1: glibc returns a NaN
        return math.nan


def rotation_trig(lat_cell, lon_cell, x_cell, y_cell, z_cell, sphere_radius: float, rotate: bool,
                  n_cells_solve: int) -> np.ndarray:
    """``trig`` for :func:`export_i2x`, as ``seaice_latlon_vector_rotation_backward``
    makes it: ``seaice_grid_rotation_forward``, ``latp = asin(zp / r)``,
    ``lonp = atan2(yp, xp)``, then sin and cos of ``lat``, ``lon``, ``latp``
    and ``lonp``. Host arrays; Python's ``math`` is the platform libm: under
    glibc :data:`TRIG_GLIBC` the words are the oracle's; under another glibc
    or another libm (macOS's) they may differ in the last place, and a
    ``RuntimeWarning`` says so."""
    r = float(sphere_radius)
    if not r > 0.0:
        raise ValueError(f"sphere_radius must be positive; it is {sphere_radius}")
    if n_cells_solve < 0:
        raise ValueError(f"nCellsSolve ({n_cells_solve}) must not be negative")
    lat, lon = np.asarray(lat_cell, dtype=np.float64), np.asarray(lon_cell, dtype=np.float64)
    xs, ys, zs = (np.asarray(a, dtype=np.float64) for a in (x_cell, y_cell, z_cell))
    for name, a in (("lat_cell", lat), ("lon_cell", lon), ("x_cell", xs), ("y_cell", ys), ("z_cell", zs)):
        if a.ndim != 1 or a.shape[0] < n_cells_solve:
            raise ValueError(f"{name} must be 1-D with at least nCellsSolve ({n_cells_solve}) cells; it is {a.shape}")
    if platform.libc_ver() != ("glibc", TRIG_GLIBC):
        warnings.warn(f"rotation_trig uses this host's libm ({platform.libc_ver()}), not glibc {TRIG_GLIBC}: its "
                      "words may differ from E3SM's in the last place", RuntimeWarning, stacklevel=2)
    out = np.empty((n_cells_solve, 8), dtype=np.float64)
    for i in range(n_cells_solve):
        x, y, z = float(xs[i]), float(ys[i]), float(zs[i])
        xp, yp, zp = (-z, y, x) if rotate else (x, y, z)
        lonp = math.atan2(yp, xp)
        latp = _asin(zp / r)
        la, lo = float(lat[i]), float(lon[i])
        out[i] = (math.sin(la), math.cos(la), math.sin(lo), math.cos(lo),
                  math.sin(latp), math.cos(latp), math.sin(lonp), math.cos(lonp))
    return out
