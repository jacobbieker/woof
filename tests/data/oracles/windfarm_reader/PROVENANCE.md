# WRF v4.7.1 wind-farm input reads: oracle provenance

These fixtures are WRF v4.7.1's own reads of a wind-farm run directory:
`init_module_wind_fitch`'s list-directed READs of `windturbines-ij.txt`
(`windfarm_ij = 1`), `windturbines.txt` (`windfarm_ij = 0`) and
`wind-turbine-N.tbl`. They hold `woof.core.windfarm_coupling`'s readers
(`_wrf_rows`, `read_wrf_table`, and the table padding `wrf_fitch_layout`
hands the Fitch kernel) to WRF at max ULP 0, raw-word equality, in
`tests/test_windfarm_reader_wrf471_parity.py`. No output is held out.

Upstream tag: `v4.7.1`.
Upstream commit: `f52c197ed39d12e087d02c50f412d90d418f6186`, public domain.
Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, through
`pixi exec -s gfortran -s binutils` (not a project dependency); see
`TOOLCHAIN.txt`.

## What is compiled

`tools/windfarm_reader_wrf471_oracle/extract.py` checks the sha256 of
`phys/module_wind_fitch.F` and of each block, then writes four INCLUDE
files byte for byte:

| lines | block |
|---|---|
| 369-375 | the record count of the turbine list (`nt`) |
| 393 | the `windfarm_ij = 1` row read, `ival, jval, nkind` |
| 399 | the `windfarm_ij = 0` row read, `lat, lon, nkind` |
| 491-510 | the table reads into `turbws/turbtc/turbpw`, `hubheight`, `diameter`, `stc`, `npower`, `cutin`, `cutout`, `stc2` |

`run_reader.F90` places them where WRF has them (the open, the allocation
and the per-turbine loop restated around them) and writes every array with
`../urban_wrf471_oracle/oracle_io.F90`. The rest of the routine projects
lat/lon turbines (held to WRF by `tools/windfarm_layout_wrf471_oracle`) and
broadcasts over MPI; neither reads a file. `-O0 -ffree-form
-ffree-line-length-none`, REAL is real(4). The reads call no libm (the
`libmvec-report.txt` records that and the positive control).

## Decks

`make_deck.py` writes two synthetic run directories into `decks/`:

* `ij`: five turbines of two types, one row comma-separated, one carrying
  values past the three it is read for; two tables whose READs continue
  across records, skip the rest of a record, read integers as REALs and
  take `d`/`D` exponents;
* `latlon`: four turbines of one type, one comma-separated, one with a `d`
  exponent and a surplus value.

Several decimals sit a hair above a binary32 rounding midpoint. Rounding
them through binary64 first lands exactly on the midpoint, and the
ties-to-even step can then pick the other word. libgfortran converts a
REAL item with `strtof`, which rounds the decimal itself. The parity test
checks that the decks exercise such a word, so a double-rounding reader
fails it.

## Regenerating

```sh
bash tools/windfarm_reader_wrf471_oracle/fetch_sources.sh /tmp/wrf471
pixi exec -s gfortran -s binutils -- \
    bash tools/windfarm_reader_wrf471_oracle/build.sh /tmp/wrf471 /tmp/reader-build
rm -rf tests/data/oracles/windfarm_reader/{ij,latlon,decks}
cp -r /tmp/reader-build/fixtures/. tests/data/oracles/windfarm_reader/
```

`FIXTURES.sha256` (every word and every deck file), `HARNESS.sha256` and
`SOURCES.sha256` are written by `build.sh` and checked by the parity test.
