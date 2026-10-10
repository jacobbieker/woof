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

WPS Vtables carry the WRF public-domain license.
