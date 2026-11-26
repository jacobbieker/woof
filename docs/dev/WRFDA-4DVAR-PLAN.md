# WRFDA 4D-Var (`var4d`) on woof: plan

Status (2026-11-26): 4D.1 (`transfer4d.py`, #737), 4D.2
(`FourDObsTerm`, `IdentityModel`, `analyse(var4d=)`, #741) and 4D.3
(`WindowModel`, #742) are merged. 4D.3b makes `WindowModel` advect `g_qv`
on a moist window (`moisture="advect"`). 4D.0 has run: the lanes L-id, L1
and L0 converge on pb with per-slot TL, adjoint-forcing and gradient files
(`tools/wrfda_oracle/run_4dvar.sh`). Not done: 4D.4 (outer loops and
`woof var --4dvar`, #867), the checkpoint
policy and its G3 receipt, and 4D.5 (L0, after P3a item 6). It is the
second half of V5 in
[WRFDA-3DVAR-PLAN.md](WRFDA-3DVAR-PLAN.md) section 7 (V5.2 and the
differentiable plan's P4.2). The model TL/AD is P3a's
(`woof/grad/wrfplus/window.py`, #718; see
[P3A-WRF-DRYCORE-ADJOINT.md](P3A-WRF-DRYCORE-ADJOINT.md)). Interface
requests to P3a go through the coordinator. This lane does not edit
`woof/grad/wrfplus`.

4D-Var is opt-in, like FGAT, and the 3D-Var defaults do not change. The
first deliverable is the tutorial case agreeing with WRFDA 4.7.1 plus
WRFPLUS 4.7.1.

## 1. What WRFDA 4.7.1 does with `var4d = .true.`

Sources: `var/da/da_main/da_solve.inc`, `da_minimisation/da_transform_vtoy.inc`
(and `_adj`), `da_transfer_model/da_transfer_xatowrftl{,_adj,_lbc,_adj_lbc}.inc`,
`da_transfer_model/da_transfer_wrftltoxa{,_adj}.inc`, `da_4dvar/da_{nl,tl,ad}_model.inc`,
`da_4dvar/da_4dvar_io.inc`.

- **Window and bins.** `var4d_bin` must divide `time_window_max -
  time_window_min`, and `time_step` must divide `var4d_bin`.
  `num_fgat_time = window / var4d_bin + 1`. The tutorial has a 6-h window
  and 3600-s bins, which gives 7 slots. The slot edges come from
  `da_get_time_slots` exactly as for FGAT: a half-width first and last slot,
  with slot `n` centred on `tmin + (n-1) var4d_bin`. The analysis time is
  the window start. The tutorial's `analysis_date` is `time_window_min`, so
  the analysis is a wrfinput valid at the start of the window. This avoids
  FGAT's `START_DATE` limitation for the bundle.
- **Observation binning** is FGAT's: one `obNN.ascii` per slot, `plocal`
  per type, and `ntotal` carried across slots. woof already has this:
  `woof.da.var.fgat.read_obs_ascii_fgat` and `slot`/`concat`.
- **Nonlinear trajectory.** Each outer loop, `da_nl_model(it)` runs WRF
  over the window from the current state with the namelist's full physics.
  For `it > 1` the state is the updated analysis: `da_solve` copies
  `grid` into `model_grid`, the lateral boundaries included. With
  `trajectory_io = .true.` the trajectory is kept in memory, every time
  step. The slot states (`nl_d01_<date>`) are the per-slot first guesses
  that `da_get_innov_vector` reads, through the FGAT path.
- **Inner loop, `y = H' M' U v`** (`da_transform_vtoy`, var4d branch):
  1. `da_transform_vtox` gives `xa` at the window start. The hybrid's
     `vpatox` is added when `be%ne > 0`.
  2. `da_transfer_xatowrftl` maps `xa` (A-grid, WRFDA variables) to the WRF
     TL state (C-grid):
     - `g_qv = q'/(1-q_b)^2`;
     - `g_mu` from `psfc'` and the column `qv'`;
     - `g_p` by hydrostatic integration down from the top;
     - `g_t = (t0+t_2)(t'/t_b - kappa p'/p_b)`;
     - `u, v` averaged A to C, with `(3a-b)/2` at the domain edges;
     - `w` copied;
     - hydrometeors per `cloud_cv_options`;
     - `g_ph_2 = 0`.

     It then calls `da_transfer_wrftl_lbc_t0`/`da_transfer_xatowrftl_lbc`
     for the boundary perturbation.
  3. `da_tl_model` runs the WRFPLUS TL over the window. It uses simplified
     physics, set from the namelist:
     - `mp_physics_ad` (Registry default 99, warm rain);
     - `bl_pbl_physics = 98` and `cu_physics = 98` if the original physics
       was non-zero;
     - radiation and surface layer off;
     - `ifsnow = 1`, `icloud = 0`.

     It writes `tl_d01_<date>` every `var4d_bin`.
  4. For each slot `n`, last to first:
     - `da_read_basicstates` makes `xb` the slot's NL state;
     - `da_transfer_wrftltoxa` (the inverse map, linearised about *that*
       slot's `xb`);
     - `da_transform_xtoxa`;
     - `da_transform_xtoy` on the slot's reports `plocal(n-1)+1 .. plocal(n)`.

  So both `H'` and the transfer are linearised about the slot's trajectory
  state. In FGAT they are linearised about the analysis time's state. This
  is the one difference in the observation side.
- **Adjoint** (`da_transform_vtoy_adj`) is the exact reverse:
  1. for each slot, `xtoy_adj`, `xtoxa_adj` and `wrftltoxa_adj`, which
     give the adjoint forcing at that bin;
  2. `da_ad_model`, with the same simplified physics, sweeps back
     through the window, adding each bin's forcing;
  3. `xatowrftl_adj`;
  4. `vtox_adj`.
- **Outer loops.** Each outer loop re-runs the NL model from the updated
  state and recomputes the innovations, so `H` and `M` are both
  relinearised. The increments are kept: the grid carries them, unlike
  FGAT's re-read of `fgNN`. The minimiser (`da_minimise_cg`/`_lz`) and
  `J = Jb + Jo` are unchanged. `Jo` sums over all slots.
- **Not in scope for the first port** (refused by name):
  - `var4d_lbc` (the boundary control at the window end, `x6a`);
  - `jcdfi_use` (the digital-filter constraint);
  - `use_rainobs`/`var4d_bin_rain`;
  - radiances under var4d;
  - `multi_inc`;
  - `var4d_detail_out`.

**Quirk to keep word-exact, with an opt-in correction:**
`da_transfer_xatowrftl` sets `g_ph_2 = 0` while perturbing `g_mu` and `g_p`.
The initial TL state is therefore not in hydrostatic balance, and the TL
model's first steps adjust it. The default reproduces this. The opt-in
`transfer="balanced"` would integrate the TL hypsometric relation for
`g_ph`.

**The mu halo at a specified window's start (`WindowModel(halo_start=)`):**
WRFPLUS's `start_domain` resets the primal `mu_2` halo to the edge row at
each window's start. Its TL `g_start_domain_em` calls `g_set_physical_bc2d`
(`wrftladj/start_em_tl.F:706`, `module_bc_tl.F`), which copies `g_mu`'s
edge row into the halo. woof's window carries the halo of the forecast it
starts from. On a warm specified window the two bases therefore differ.
The default, `halo_start="wrfplus"`, matches WRFPLUS: the primal halo
(`model.HALO`) is reset to `mup`, and the TL halo is seeded from `g_mu`
(`wrfda_tl_start(..., specified=True)`). `halo_start="carried"` keeps woof's
carried halo and starts the TL halo at zero. A cold start, or a periodic
domain, is the same under both.

## 2. How it maps onto woof

| WRFDA | woof |
|---|---|
| `da_get_time_slots`, `plocal`, ob per slot | `woof.da.var.fgat` (exists) |
| per-slot innovations against `nl_d01_<date>` | `fgat.innovations_fgat` against the trajectory's slot states (exists) |
| `da_transfer_xatowrftl` (+ `_adj`) | new `woof/da/var/transfer4d.py`: `xa_to_tl(xa, xb0, base0)` / `_adj` |
| `da_transfer_wrftltoxa` (+ `_adj`) | same module: `tl_to_xa(dx_n, xb_n, base_n)` / `_adj` |
| `da_tl_model` / `da_ad_model` | P3a `Window.tl(dx0, save_at=bins)` / `Window.ad(forcing={bin: ...})` |
| per-slot `H'` about the slot's `xb` | `obsop_wrfda` terms built per slot from that slot's `xb` (the FGAT path builds them all from `xb` at the analysis time) |
| `J`, CG, Lanczos | `VarCost` and `minimise` unchanged |

**The seam** is one `ObsTerm`, `FourDObsTerm` (d, err, mask, tl, ad,
jo_split), so `VarCost` itself does not change:

- `tl(xa)`:
  1. `dx0 = xa_to_tl(xa)`;
  2. `Window.tl(dx0, save_at=bins)`;
  3. for each slot, `tl_to_xa`, then that slot's `H'`;
  4. concatenate in slot order.
- `ad(y)`: slot by slot, `H'^T` and `tl_to_xa_adj` give `forcing[bin]`.
  Then `Window.ad(forcing=...)` and `xa_to_tl_adj`.
- The `Window` holds the trajectory. Its `forward` is run once per outer
  loop. Bin `n` is state `k_n = (n-1) var4d_bin / time_step`.
- `analyse(..., var4d=Var4dConfig(bin_s, window))` is the Python door.
  `woof var --4dvar --var4d-bin` follows only after the oracle gates pass.

## 3. What P3a must still provide

Via the coordinator. They are listed in the order the oracle lanes
(section 4) need them. Where they stand (2026-11-26): 1 is
`Window.forward(trajectory=)`; 2 is `wrfda_tl_start` and
`Window.tl(partial=True)`; 3 is the specified window (#743); 4 is the moist
window (qv, qc, qr, mp_physics 0), which `WindowModel(moisture="advect")`
uses; 5 is the diff_opt 1 / km_opt 4 binding (dry merged, the moist rows in
#825); 6 is not started; 7 holds on CUDA, and a Metal 4D-Var analysis is
refused (no binary64 there; PERF-QUEUE has the follow-up).

1. **The trajectory comes from outside the window.** WRFPLUS linearises
   about the NL run's states, read every time step (`auxinput6`). That run
   has full physics. P3a's `Window` linearises about its own dry rerun. For
   comparison with WRFDA, `Window` needs a mode that takes the base state
   per step from a supplied trajectory: a woof forecast with physics, or
   for isolation, WRF's own states. Without that mode only the
   identity-model and adiabatic lanes can agree with WRFDA.
2. **A cold TL start.** WRFDA gives the TL its prognostics (`g_u, g_v, g_w,
   g_t, g_mu, g_moist`) plus `g_p`, with `g_ph = 0`, and zeroes `g_al`/`g_alt`.
   `Window.tl` takes all of `STEP_INPUTS` (`p, al, alt, p_pert, muts`
   included). Needed: either the meaning of a dx0 whose diagnosis entries
   are zero or given, or a `diagnose_tl` helper. Also needed: which arrays
   carry `g_p` and `g_mu`, `mup` against `muts`.
3. **Specified lateral boundaries** (`spec_bdy_width = 5`, relaxation zone)
   in TL/AD, with the boundary perturbation that `da_transfer_wrftl_lbc_t0`
   sets and zero tendency (`var4d_lbc = .false.`). The tutorial is a
   regional domain. Map factors are listed as covered in `model.py`'s
   docstring; please confirm.
4. **Moisture.** Needed: `qv` (and `qc`/`qr` when `cloud_cv_options >= 1`)
   as TL state, the moist coupling in the dynamics, and scalar advection of
   the moist arrays. This gives an *adiabatic moist* TL/AD: physics-free,
   but moist. It is enough for oracle lane L1.
5. **Mixing:** `diff_opt = 1`, `km_opt = 4` TL/AD. The tutorial uses it.
   Lane L1 can switch it off, but L0 cannot.
6. **WRFPLUS's simplified physics** TL/AD: `mp_physics_ad = 99` (warm
   rain), `bl_pbl_physics = 98` (surface drag) and `cu_physics = 98`. This
   comes last. Only the full-tutorial lane L0 needs it.
7. **binary64** (`Window(dtype=np.float64)`, as now) on CUDA. A Metal port
   of the window comes later.

## 4. The oracle case

The **WRFDA 4D-Var tutorial case**:

- testdata `WRFDAV3.9-testdata`, 2008-02-05;
- window 12-18Z, `var4d_bin = 3600`, 7 slots;
- `ob01..ob07.ascii` hourly;
- the tutorial's cv5 `be.dat`;
- `rc/2008020512` wrfinput/wrfbdy;
- `namelist.input.4dvar`.

Build on pb, in `wrf-oracle:u2404`:

1. WRFPLUS 4.7.1 (`./configure wrfplus`, `./compile wrfplus`, r8).
2. WRFDA 4.7.1 with `./configure 4dvar` and `WRFPLUS_DIR`, plus the dump
   hooks of `WRFV4.7.1-dump` (`woof_dump_iv`, `woof_dump_u`).
3. New dumps: `xatowrftl` in and out, `tl_d01_<date>` per bin, and the
   adjoint forcing per bin.

The script is `tools/wrfda_oracle/run_4dvar.sh`, written as
`run_fgat.sh` was.

The lanes:

| lane | WRFDA namelist | needs from P3a | checks |
|---|---|---|---|
| L-id | `enable_identity = .true.` (WRFPLUS `&perturbation`: TL/AD are the identity) | nothing | binning, per-slot `H'` about the slot's `xb`, transfer and its adjoint, J0, iterations, J final, increments |
| L2 | none (woof-only OSSE) | nothing new (dry, periodic) | dot product and Taylor tests through `Window`; 4D-Var beats FGAT/3D-Var on a dry periodic case with synthetic obs |
| L1 | adiabatic: `mp_physics = 0`, `cu/bl/ra/sf_sfclay = 0`, `diff_opt = 0` | 1-4 | TL/AD per bin against `tl_d01_<date>`; full minimisation |
| L0 | the tutorial as shipped | 1-6 | J0 / iterations / J final; increments (O7 criteria) |

The gates (GV6) follow GV5:

- QC identical;
- innovations to 1e-10;
- J0 to 1e-12;
- the same iteration count;
- final J to 1e-12;
- increments to max|d|/rms 1e-4 with correlation >= 0.999999.

The trajectory comes from WRF's own NL states until woof's forecast is
shown to agree with them. The innovation gate then isolates the DA side.
It does not depend on how closely woof's forecast reproduces WRF.

## 5. PRs

1. **4D.0**: the oracle build and `run_4dvar.sh`, lanes L-id and L1 (and L0
   once built). No woof code.
2. **4D.1**: `transfer4d.py` (`xatowrftl`, `wrftltoxa` and their adjoints,
   with the `g_ph = 0` default plus the opt-in balanced `g_ph`).
   Dot-product tests, and word agreement with the L-id dumps.
3. **4D.2**: `FourDObsTerm` with a pluggable model (the identity first),
   per-slot `H'`, and the `analyse(var4d=...)` door. Gate: L-id.
4. **4D.3**: coupling with P3a's `Window`. The L2 OSSE first; then L1 once
   P3a items 1-4 land.
5. **4D.4**: outer loops (NL rerun from the updated analysis), `woof var
   --4dvar`, docs/var.md, README. Gate: L1 with `max_ext_its = 2`.
6. **4D.5**: L0, once P3a items 5-6 land.

**Refused throughout:**

- `var4d_lbc`, `jcdfi_use`, rain and radiance obs under var4d;
- `multi_inc`;
- a first-guess time other than the window start;
- a `var4d_bin` that does not divide the window, or that is not divisible
  by the time step.

The bin rules are WRFDA's own errors, refused with the same remedy.
