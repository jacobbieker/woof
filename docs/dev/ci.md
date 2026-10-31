# Continuous integration (`.github/workflows/tests.yml`)

Every push and pull request runs the jobs below. Every environment is a pixi
environment from the committed `pixi.lock`, installed with `locked: true`
through `prefix-dev/setup-pixi`, with the pixi cache on. The helper scripts
are in `.github/ci/`.

The file is named `tests.yml` and not `ci.yml` on purpose. Several tools and
tests in this tree describe the engine's own `ci.yml`
(`tools/promote_prepared_release.py`, `tools/workflow_rehearsals.py`,
`tests/test_publish_workflow_state_machine.py`). That workflow is left out of
this tree, and `tests/_woof_snapshot.py` skips those contracts while it is
absent. A `ci.yml` here would switch them on against a different workflow.
`publish.yml` and `publish-data.yml` are unchanged, because their file names
are bound to PyPI trusted publishing.

## Jobs

| job | runner | pixi env | what it runs | what a pass proves |
|---|---|---|---|---|
| `lint` | ubuntu-24.04 | `cpu` | `actionlint`; `tests/test_no_undefined_names.py` (ruff F821/F822/F823 over `woof`, `tools`, `tilestream`); `compileall` over the owned Python trees; `python -m tools.import_sweep`; a check that the test legs partition `tests/` | No undefined names, no syntax errors, and every shipped module imports (or names a declared optional extra) |
| `native (ubuntu-24.04)` | ubuntu-24.04 | `cpu` | `cargo build --release --locked --offline` in every workspace that the `build-native` pixi task names, read from `pyproject.toml` | The vendored Rust bridges build offline. Their executables and shared libraries are packed as the `native-linux` artifact |
| `cpu (ubuntu-24.04, N)` | ubuntu-24.04 | `cpu` | Legs `0`–`7`: `tests/test_*.py` split into 8 legs by a hash of each path. `hex`: `components/hex/tools/battery/cpu_files.txt`, run from `components/hex`. `stage1`: `tools/battery/run_stage1.py`, the maintained Stage 1 list. Every leg uses `-m "not gpu and not bigcard and not assets"` and `pytest-xdist -n auto --dist loadfile` | The CPU tier passes on Linux, with the bridges present |
| `native (macos-15)` | macos-15 (arm64) | `mlx` | The same Rust build on macOS | The bridges build on macOS |
| `macos (macos-15, mlx env, N)` | macos-15 | `mlx` | The same legs as `cpu` | The CPU tier passes on macOS in the environment Mac users run (`WOOF_BACKEND=mlx` is set by the env) |
| `metal probe + device-free translator checks` | macos-15 | `mlx` | Checks `mx.metal.is_available()`. Without a device it still runs `tools/metal_translation_census.py --translate-only --gate` (both rounding modes) and `tools/f64_soft_grade.py host` | Every CUDA kernel unit translates to MSL, and the soft-binary64 header is word-exact on the host. Whether Metal exists is reported |
| `metal gates (GPUWM_WRF_EXACT=0/1)` | macos-15 | `mlx` | Runs only if the probe found a device: the Kessler, WSM6 and YSU WRF 4.6.1 parity files, `test_metal_launch_ordering`, `test_metal_memory_order`, `test_metal_mapped_host_memory`, and `tools/f64_soft_grade.py gpu` | The Metal backend reproduces the WRF oracles word for word |
| `cuda (self-hosted, GPUWM_WRF_EXACT=0/1)` | `[self-hosted, linux, cuda]` | `default` | Opt-in. The three WRF 4.6.1 parity files in both modes. With `EXACT=0`, also the pytest members of `components/hex/tools/battery/gpu_gates.txt` (`-m "not assets"`; their `bigcard` members fit a 48 GB card) | The CUDA backend passes the parity gates and the hex GPU gates |

Each test leg uploads its junit XML (`junit-*` artifacts) and writes its
passed, failed, errors and skipped counts to the job summary. That way a leg
where every test skipped does not read as green.

### When Metal is missing

GitHub's macOS runners are virtual machines and may have no Metal device. If
the probe finds none, three things happen. The `metal gates` jobs are
**skipped**: they show grey, never green. The probe job prints a
`Metal gates NOT RUN` warning annotation and a summary line. Only the
translator checks run, because they need no device. A skipped Metal gate is
not a pass.

## Not covered

- **The big-card tier (Tier 3).** This is the x4.163842 full-physics harnesses
  that `gpu_gates.txt` names as commands, and `-m bigcard` on cards smaller
  than its admission floor. No hosted runner has a CUDA device. The opt-in
  `cuda` job runs only the pytest members.
- **Assets (Tier 2, `asset_gates.txt`, `-m assets`).** About 6.9 GiB of
  byte-pinned files have no fetch path, so no job runs them.
- **The full CUDA pytest shard** (`tools/battery/gpu_shard_files.txt`, 516
  files). It is not in the `cuda` job. Run it by hand on the CUDA host.
- **The full Metal compile census** (`tools/metal_translation_census.py`
  without `--translate-only`). Several units take more than 10 minutes each
  to compile.
- **Rust formatting.** `cargo fmt --check` is not clean across the path
  dependencies into vendored crates, so it is not enforced.
- **Windows.** No job runs on Windows.

## The opt-in CUDA runner

The `cuda` job runs only when the repository variable `WOOF_CUDA_RUNNER` is
`true` (Settings → Secrets and variables → Actions → Variables). Without the
variable it is skipped and never waits for a runner that does not exist. Pull
requests from forks never reach it, because their code would otherwise run on
the card.

To register a runner:

1. Use a Linux x86-64 host with an NVIDIA driver for CUDA 12, a card with at
   least 24 GB (48 GB for the `bigcard` members), `gcc`, and `pixi` on the
   PATH of the runner's user. pixi supplies Python, CuPy and Rust from the
   lock.
2. Go to Settings → Actions → Runners → New self-hosted runner → Linux x64,
   and follow the download and `./config.sh` steps. When asked for labels,
   add `cuda`. The runner then carries `self-hosted`, `linux`, `x64` and
   `cuda`.
3. Install the runner as a service with `sudo ./svc.sh install && sudo ./svc.sh start`.
4. Set the repository variable `WOOF_CUDA_RUNNER=true`.

The job runs its two legs one after another (`max-parallel: 1`) so they do
not share the card. On a host that also does timed work, reserve the card
with the host's own mutex before you give the runner the `cuda` label. A
timed card must be reserved, not just found idle (`gpu_gates.txt`).

## Running the jobs locally

```sh
# lint
pixi run -e cpu python -m pytest -q tests/test_no_undefined_names.py
pixi run -e cpu python -m tools.import_sweep
pixi exec --spec actionlint==1.7.12 actionlint

# natives, then one CPU leg (0..7), the hex tier, Stage 1
pixi run -e cpu python .github/ci/build_native.py build
GPUWM_NO_LOCAL_GPU=1 pixi run -e cpu python -m pytest -q -n auto --dist loadfile \
    -m "not gpu and not bigcard and not assets" $(python3 .github/ci/select_tests.py shard 0 8)
(cd components/hex && GPUWM_NO_LOCAL_GPU=1 pixi run -e cpu python -m pytest -q -n auto \
    -m "not gpu and not bigcard and not assets" $(python3 ../../.github/ci/select_tests.py hex))
GPUWM_NO_LOCAL_GPU=1 pixi run -e cpu python tools/battery/run_stage1.py -- -q -n auto \
    -m "not gpu and not bigcard and not assets"

# Metal (on a Mac)
pixi run -e mlx python tools/metal_translation_census.py --translate-only --gate --fresh --out /tmp/c.jsonl
GPUWM_WRF_EXACT=1 pixi run -e mlx python -m pytest -q tests/test_kessler_wrf461_parity.py
```
