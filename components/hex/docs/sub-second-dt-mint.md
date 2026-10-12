# Minting a sub-second timestep anchor on a regional cull

This page is for the coordinator. It says how to run a real anchor mint at
0.5 s or 0.25 s on a regional (limited-area) corridor cull, once the user has
supplied the candidate-mint authorization sentence, and which inputs a 100 m
corridor mint needs.

Nothing on this page registers an anchor. `ADMITTED_TIMESTEPS` in
`woof/hex/dt_admission.py` and the regional class table in
`woof/hex/cuda_backend/regional_admission.py` are never edited by a mint. A
passing mint goes back to the user, who rules on it.

## Why sub-second, and why regional

A 100 m corridor cull has a finest edge of about 100 m. Under the outer-step
Courant policy every gate uses (125 m/s, safety 0.9), its limit is
0.9 x 100 / 125 = 0.72 s. A 50 m cull's limit is 0.36 s. The smallest
anchored timestep is 5 s, so no anchored timestep fits either mesh. Two
consequences follow:

- The mint runs at a timestep that divides the 600 s radiation cadence and
  closes its clock exactly in binary64: 0.5 s (1,200 steps per radiation
  call) or 0.25 s (2,400). 0.1 s and 0.3 s are refused, because their
  multiples are not exact.
- The physics-health control cannot run at an anchored timestep on the same
  mesh. It runs at a finer *unanchored* step (by default dt/2) through the
  same candidate path, and the anchor JSON labels it
  `step-halving-unanchored`. Agreement with that control shows the solution
  is converged in dt. It does not show agreement with an anchored run.

Convection is off for these mints, because convection is off below 3 km
(`woof.hex.convection_admission`).

## What a mint is

A mint has four parts:

| part | what it is | card | authorization |
|---|---|---|---|
| schedule receipt | host arithmetic: cadence step counts, RK schedule shape, WSM6 split, binary64 clock closure | no | no |
| integration anchor | two candidate arms through the forecast door on the cull, finite at every step, history byte-identical under masked digests | yes | yes |
| physics_health | the candidate's band against the control's band on the same card, mesh and init | yes | yes (step-halving control) |
| regional forecast mint pair | the dry regional dycore twice (`run_cuda_regional_forecast.py --mint-anchor`), plus the cull's geometry contract deck | yes | no |

## Step 0: plan it (no card, no authorization)

```bash
python components/hex/tools/mint_dt_anchor.py --regional-plan \
    --dt 0.5 --mesh <corridor-row> \
    --grid <cull>/corridor.grid.nc --init <cull>/corridor.init.nc \
    --static <cull>/corridor.static.nc --lbc-dir <lbc-dir> \
    --start-time <YYYY-MM-DD_HH:MM:SS> --init-source '<what produced the init>' \
    --gpuwm-checkout <arwen-checkout> --card '<card>' --work <work>
```

Run it again with `--dt 0.25`. The plan reads the cull's own files and checks
every host-checkable rule. A failed rule is listed by name under `blockers`,
and `runnable` is false:

- Courant from the cull's `min(dcEdge)`
- radiation divisibility
- binary64 clock closure
- a dt that is a whole number of milliseconds (the regional class key
  quantises dt to milliseconds)
- a real boundary zone (`bdyMaskCell`)
- the configuration is not already anchored

The level count is checked as well. The full-physics lane binds any column
up to its physics ceiling (`woof.hex.vertical_spec.hex_column_level_ceiling`,
256 levels on the frozen WSM6 + YSU lane), so only a deeper cull is listed
under `integration_blockers`. Every existing anchor was measured at 55
levels, so a cull at another count mints a new `-l<levels>` class rather
than reusing one.

The plan prints, for each part, the exact command, the inputs it reads and
the outputs it writes. The authorization appears only as the name
`<CANDIDATE_MINT_AUTHORIZATION>`. The campaign's own dry run prints the same
card legs:

```bash
python components/hex/tools/run_dt_anchor_campaign.py --dry-run --dt 0.5 \
    --control-dt 0.25 --convection off ... (the plan's dry_run_command)
```

## Step 1: the schedule receipt (host only)

```bash
python components/hex/tools/mint_dt_anchor.py --dt 0.5 --cumulus-scheme off \
    --run-steps 7200 --out <work>/schedule-receipt-dt0p5.json
```

## Step 2: the regional forecast mint pair (card, no authorization)

1. Run the contract deck once per cull. It depends on the cull's geometry,
   not on the timestep:

   ```bash
   python -m woof.hex.drivers.run_cuda_regional_contract --grid <grid> \
       --init <init> --lbc-dir <lbc-dir> --start-time <T> \
       --mesh-row <row> --out <work>/contract
   ```

2. Run the two mint arms, in separate processes, with the commands from the
   plan's `regional_forecast_mint_pair.arm_commands`. Each arm runs 1,080
   steps with `--len-disp` set to the cull's finest edge. `--history-every`
   is chosen so that every frame lands on a whole second. The instrument
   refuses any cadence that would put two frames in the same second,
   because frames are named and stamped to the second and the later one
   would overwrite the earlier.
3. Compare the arms with `compare_command`. The pair passes when every
   frame's `masked_sha256` matches between `run1/forecast.json` and
   `run2/forecast.json`, and every field is finite.

## Step 3: the integration pair and physics_health (card, authorization)

Run the plan's `integration_anchor.command`, supplying the sentence the user
gave in place of `<CANDIDATE_MINT_AUTHORIZATION>`. The safest way is to
leave `--authorization` off and type the sentence into the environment
variable the campaign reads, so it stays out of shell history and the
process list. Do not put it in a script, a file or a commit.

```bash
export WOOF_HEX_MESH_ROWS=<rows.json>     # the corridor row, see "Inputs"
read -rs WOOF_HEX_CANDIDATE_MINT_AUTHORIZATION   # type the sentence
export WOOF_HEX_CANDIDATE_MINT_AUTHORIZATION
python components/hex/tools/run_dt_anchor_campaign.py --dt 0.5 \
    --control-dt 0.25 --convection off --card '<card>' --mesh <row> \
    --grid <grid> --static <static> --init <init> --init-source '<...>' \
    --lbc-dir <lbc-dir> --start-time <T> --hours 1 \
    --history-every-minutes 10 --gpuwm-checkout <arwen> \
    --work <work>/integration-dt0p5
unset WOOF_HEX_CANDIDATE_MINT_AUTHORIZATION
```

The campaign checks the authorization before the first arm starts, by
constructing the candidate admission without entering it. It hands the
sentence to each arm through the same variable (`--authorization-env`),
never through argv, and redacts it from its logs, its anchor JSON and its
markers. One gap remains: the timestep gate's candidate row quotes the
sentence in its `basis`, and the forecast door's own receipt
(`arm-*/out/forecast-receipt.json`) records that row as it was given. Treat
the arm output folders as private until a person has checked them. It runs the
step-halving control first, then arms A and B. It writes
`anchor-dt0p5.json` and either `DONE` or `FAILED`. The integration anchor
is earned when `integration_anchor_earned` is true. The `physics_health`
verdict is a person reading `health_against_control` and writing the
leading word (TRACKS, DIVERGES, ...).

## Inputs a 100 m corridor mint needs

- **A registered corridor cull row.** Register it with
  `woof hex register --grid G --static S --parent-row NAME --cull-receipt R
  --name N --dt-seconds DT --rows FILE`, then name the file to the door with
  `WOOF_HEX_MESH_ROWS=FILE`. The candidate override replaces the row's
  timestep for the length of one arm. It is applied after runtime rows are
  added, and the harness refuses to start if the override did not reach the
  registry the door loads.
- **The cull's grid, static and init.** The init comes from `init_door`. The
  static comes from `rw_mpas_static`, which reads WPS_GEOG 30" only. A
  100 m corridor needs the high-resolution statics as well.
- **A lateral-boundary series** (`lbc.*.nc` from `rw_mpas_lbc`) covering the
  arm length, and the start time it begins at.
- **55 vertical levels.** That is the count the full-physics lane binds
  today.
- **The Arwen checkout** (`--gpuwm-checkout`) and the card's name (`--card`).
- **Card memory.** The device model admits about 460k regional cells at 55
  levels on the RTX A6000. Check `nvidia-smi` first, because the card is
  shared.
- **Arm length.** At 0.5 s, one hour is 7,200 steps per arm. At 0.25 s it is
  14,400. The control at dt/2 doubles that again. Size `--hours` to the
  card time available. Every arm must be a whole number of steps, and the
  history stride must divide the arm.

## After a mint passes

Bring these to the user:

- the schedule receipt
- the anchor JSON
- both regional `forecast.json` files and the pair comparison
- the contract deck receipt

Registering the rows is the user's ruling. It needs the user to supply the
sentence in person, and nothing in this procedure does it.
