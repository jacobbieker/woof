# gpuwm/data/vtables

## Vtable.ERA5_CDO

- sha256: `64282b5b35ac7302e274f764327923080883f164f4e605ef06529d1baef6620e`
- WPS-format Vtable describing ERA5 GRIB1 parameters (pressure-level +
  single-level, ECMWF table-128 codes) for the native GRIB1 ingest route.
  Derived from the WPS v4.6.1 `ungrib/Variable_Tables/Vtable.ERA-interim.pl`
  family, adjusted for CDS ERA5 retrievals (level-type 112 soil layers are
  additionally aliased by `gpuwm/ingest/grib.py` for native CDS files).
- Byte-identical to the reference bundle's `era5_grib/Vtable.ERA5_CDO`,
  the exact schema the certified real-data lineage decoded with; the
  canonical parameter/spelling mapping is pinned by
  `gpuwm/ingest/grib.py::_CANONICAL_SPECS`, which fails loudly on drift.
- Packaged so `gpuwm domain` can emit relocatable configs that declare a
  vtable without referencing a machine-local bundle path.  WPS Vtables
  carry the WRF public-domain license.

## Vtable.GFS.rw, Vtable.ECMWF-OD.rw, Vtable.ERA5.rw

The tables `woof hex intermediate` hands the Rust `met_intermediate` writer
(`woof/hex/met_intermediate_door.py`) for the global lat-lon sources.  The
digests below are informational and not pinned by a test.  Each file's
trailing comment block says which fields are derived, which are dropped and why.

- `Vtable.GFS.rw` (sha256 `a0a11e30b93c3a20481aef2be2b30781ae755f18f3bc9d3003f44d7283e2719a`):
  GFS and GDAS pgrb2.0p25, map source `ncep-gfs`.  Names, units and soil keys
  follow WPS's `Vtable.GFS`.  The GRIB2 soil identifiers (2-0-2, 2-0-192) are
  the ones `woof/authorities/Vtable.GFS.rw-wps` validates.  Pressure-level
  SPECHUMD rows were added.
- `Vtable.ECMWF-OD.rw` (sha256 `8959329ddae3fcea8d1fc540746a132df8c1d51da4854e0c9ee62dc489778600`):
  ECMWF IFS open data and AIFS single, map source `ecmwf`.  This file
  recreates the table `components/hex/docs/source-matrix.md` names; the
  original was not in this checkout or its history.  The GRIB2 identifiers
  were read off a 2026-09-24 12Z IFS oper file: ordinal type-151 soil, surface
  geopotential 0-3-4, snow depth (m of water equivalent) 0-1-254, snow density
  0-1-61.
- `Vtable.ERA5.rw` (sha256 `9d1a6514ed5af72a17e7cac7875ce1776e901dcd42dbe3588f1546bef5a416cc`):
  CDS ERA5 GRIB1 (table 128), map source `ecmwf`.  It follows WPS's
  `Vtable.ECMWF`, with native level-type 112 soil rows plus the CDO-flattened
  level-type 1 aliases that `Vtable.ERA5_CDO` uses.

## Vtable.ICON-GLOBAL.rw, Vtable.ICON-GLOBAL-ML.rw

DWD ICON global for `met_intermediate --map-source dwd`: pressure levels and
model levels (fixed-surface type 150).  Both read the regional regular lat-lon
GRIB2 that `gdt101_remap` writes from ICON's native GDT 3.101 objects, which
keeps each object's product section.  The digests below are informational and
not pinned by a test.  The Rust tests in `met_intermediate` parse both files
and check their soil rows.

Source: Deutscher Wetterdienst (DWD), CC BY 4.0.

- `Vtable.ICON-GLOBAL.rw` (sha256 `4c1b53d7d21ed2d6cee3d809ba88ea7609b4208bd4bed6321f3c29f63d29627b`):
  T, RELHUM, U, V and FI on the eighteen 1000-30 hPa levels.  GHT is
  FI / 9.81.
- `Vtable.ICON-GLOBAL-ML.rw` (sha256 `f796d5fb9ba133a736577e0e599500cee43917770ea00e0c17a142b1b46247a0`):
  T, QV (SPECHUMD), U, V and P (PRES) on the full model levels.  GHT is the
  mean of the two bounding HHL half-level heights.
- Both tables carry the same surface and soil rows.  T_SO is read at nine
  single depths, matched by depth.  W_SO is read per TERRA layer by both
  bounds.  They become SOILT/SOILM 001, 002, 006, 018, 054, 162, 486 and 999,
  the eight layers `rw_mpas_init` reads (1-1458 cm thick).  SOILM is
  W_SO / (1000 x layer thickness).  SOILT is T_SO at the layer midpoint,
  which is exactly one TERRA node per layer.
- The GRIB2 identifiers were read off real DWD v1 objects from run
  2026-10-11T00:00 (opendata.dwd.de/weather/nwp/v1/m/icon/p/).  W_SO carries
  both depth surfaces (106/106), T_SO carries one (106/255), full model levels
  are 150/150 between half levels k and k+1, and HHL is 150 with a second
  surface of 101.  HSURF and HHL share parameter 0-3-6 and differ only by level
  type.  PMSL is 0-3-1.

WPS Vtables carry the WRF public-domain license.
