# Recast WOOF

**Weather Observations and Open Forecasting: a GPU-native weather model you run on your own card.**

WOOF is a regional weather model with WRF-class physics that runs on one
NVIDIA GPU, with its own preprocessing, data assimilation and plots. One
install carries three models:

| Model | Command | Status |
| --- | --- | --- |
| Regional (WRF-ARW class, nests, DA) | `woof` | Production |
| Hex (MPAS-style unstructured mesh) | `woof hex` | Preview |
| Global (spectral, with ensemble DA) | `woof global` | Preview |

Preview means the model is in the package, documented and tested on CPU, and
its command says preview each time it runs; its config and checkpoint formats
may change in a minor release. The major-version promise covers the regional
model.

## Install

Linux on x86_64, Python 3.11 or newer, an NVIDIA GPU and driver. WOOF has
no Windows build of the native tools yet.

With Pixi (Python and the Rust/build toolchain managed in one env):

```bash
pixi install
pixi run setup
pixi run doctor
# or: pixi run doctor-explain
```

The default Pixi environment installs this checkout in editable mode with the
`all-cu12` extra. For a CUDA 13 setup use:

```bash
pixi install -e cuda13
```

On a source checkout, `pixi run setup` builds the native Rust tools locally.
`woof fetch-bridges` may report that no bundle pins are present; that is normal
outside a published wheel install.

```bash
python3 -m venv ~/woof-env
~/woof-env/bin/python -m pip install 'recast-woof[all-cu12]'
~/woof-env/bin/woof fetch-tables
~/woof-env/bin/woof fetch-geog --datasets wrf
~/woof-env/bin/woof doctor
```

For a CUDA 13 setup use `all-cu13`. The hex model runs only on CUDA 13, so
install `all-cu13` to use it. Install one CuPy/CUDA major per environment.
The Linux wheel carries the native tools (GRIB and NetCDF decoding, the
WPS-equivalent preprocessor, the renderer).

`fetch-geog` downloads the global geography every forecast builds its
terrain, land use and soil from: about 1.3 GB, 17 GB unpacked, once per
computer. `fetch-tables` adds two Thompson microphysics tables (about 314 MiB).
Reference tables and map assets arrive with the `recast-woof-data` package.

## Run a forecast

```bash
woof domain --point 35.3,-97.5 --source hrrr --cycle latest --hours 3 \
     --root-dx 3 --point-extent-km 300 --out forecast.toml
woof go forecast.toml --dry-run
woof go forecast.toml
```

`--dry-run` reviews the route, sizes it for your card and names anything
missing; the second line fetches, prepares, runs and draws. Pictures are drawn
as each history frame lands, beside the running forecast.

Compare two runs, any product, as run A minus run B:

```bash
woof render --diff runA/ runB/ --out diff/
```

The pair must share the grid and the valid times; a mismatch is refused by
name.

## Documentation

- `docs/public/`: the CLI manual, physics, verification record, hardware
  guidance and data sources.
- `components/hex/` and `components/globe/`: the hex and global models' manuals,
  tests and tools.
- `woof --help-all` lists every command; `woof hex --help` and
  `woof global --help` list the preview models' commands.

## Science and limits

WOOF implements a WRF-ARW-class regional model and WRF-derived physics on the
GPU. Its numerical comparisons apply to documented configurations and test
cases; they do not establish general equivalence with WRF or forecast skill for
every combination of physics, input data and hardware. The hex model is a port
of the MPAS-A v8.4.1 dynamical core with this engine's physics; it is not MPAS
and is not endorsed by UCAR. Use official meteorological services for
forecasts and warnings.

WOOF continues an engine released before under another name: WOOF 1.0 is
that engine's release 2.8.0 and the fixes made since, under the WOOF name,
with the hex and global models inside it. Settings spelled `GPUWM_*` keep
working beside the new `WOOF_*` spelling.

## Licence

Apache License 2.0. Third-party code, tables and datasets keep their own
terms; see [NOTICE](NOTICE) and [licenses/](licenses/).
