# Resume a prepared forecast

Set a nonzero `restart_interval_s` in the experiment before preparing it. Both single-domain and hierarchy forecasts write canonical `gpuwmrst_d01_*.npz` checkpoints at that cadence.

Use the existing prepared bundle and its exact configuration for continuation, with a new output directory:

```console
woof sim prepared --experiment-config prepared/experiment.toml --wps-namelist prepared/namelist.wps --restart previous/gpuwmrst_d01_<instant>.npz --outdir continued
```

For `go`, name the existing bundle explicitly:

```console
woof go prepared/experiment.toml --prepared-root prepared --wps-namelist prepared/namelist.wps --restart previous/gpuwmrst_d01_<instant>.npz --outdir continued --products none
```

The equivalent run-plan uses `route: "prepared"` and `run_options.prepared_root`, `wps_namelist`, and `restart`. These commands reuse the prepared input; they do not fetch or prepare it again. A single bundle binds the complete configuration, including the stop time. Its checkpoint continues the remaining interval and does not extend that sealed forecast. Sealed forcing extension remains a hierarchy operation.

A forecast started on a chained preparation's head (see PIPELINE-STAGES.md) can write a checkpoint before the preparation is sealed. That checkpoint is bound to the head and records only the boundary intervals that were prepared when it was written, the same preserved-prefix record a sealed forcing extension uses. It resumes only on the preparation with that head; any other is refused by name. A forecast that dies before the seal (killed, out of memory, a driver fault) leaves its preparation running to the seal, so `woof sim` with `--restart` resumes the checkpoint beside the preparation still being produced, or on the sealed tree afterwards, and waits only at boundary intervals that are not prepared yet. A checkpoint written after the seal is an ordinary checkpoint.

A checkpoint resumes only on the build that wrote it: 2.7.0 refuses every 2.6.5 checkpoint by name (see the CHANGELOG's "Restart and import compatibility" section, or run `woof doctor --since 2.6.5`).

A resumed or repeated run checks the bundle against its digests at the preflight, as the first run did. It does not re-read files and cache arrays that are stat-identical to the full read that last proved them (device, inode, size, mtime and ctime unchanged). Anything rewritten since is read and checked again. `report.json` records which in `input_verification`. To make every check read every byte, run with `WOOF_VERIFY=full` (`woof go --verify-inputs full`); PIPELINE-STAGES.md explains what a stat check cannot see.

History already committed at the checkpoint is retained in the previous output directory. The new directory contains only later frames. Restoring a checkpoint already at the configured stop performs no steps and writes no duplicate history. `--health-debug` on `sim` enables the shared per-step whole-domain validator, including the canonical store when streaming.

## How a checkpoint is written

A checkpoint no longer stops the forecast while it is written. At each
`restart_interval_s` the model thread checks and inventories the state as
before, then copies the device state into page-locked host memory on a
side stream and goes on stepping. The model's own stream waits for that
copy before its next kernel, so the checkpoint holds the state at its
instant. A background thread writes each member (`np.savez`, `fsync`,
rename, directory `fsync`), children first and the d01 root, the set's
commit marker, last.

A checkpoint becomes the resume point only once it is durable. That
means `last_checkpoint` in the progress file, the step log's
`restart_written`, and removal of the sets it replaces. Rules:

- At most one checkpoint is written at a time. If the next one comes due
  while the previous is still being written, the model waits for it and
  says so (`checkpoint: ... is still being written; ... waits for it`).
- Every exit waits for the checkpoint in flight: success, failure or
  signal. Its snapshot was taken before whatever stopped the run, so it
  is published, or its partial members are removed. A failed write
  publishes nothing. The previous checkpoint stays the resume point, and
  the run fails as a failed synchronous write did.
- History frames are not drained before a checkpoint any more. The
  checkpoint's root is renamed into place only after every wrfout frame
  at or before its instant is durable. A resume therefore never skips or
  repeats a frame. With a simulated radar attached, the writers are
  drained before the checkpoint, as before.
- A checkpoint requested through `WOOF_RESTART_REQUEST_FILE` is
  acknowledged only once it is durable.
- Numerics are unchanged. The snapshot is a copy, and the archive has
  the same members, member order and header as a synchronous one; only
  `created` differs.

The snapshot memory is allocated once per run, sized from the state, and
freed when the run ends. A checkpoint is written synchronously instead,
with the reason printed and recorded, when:

| reason | when |
|---|---|
| `disabled` | `[output] async_checkpoints = false`, or `WOOF_ASYNC_CHECKPOINTS=0` |
| `streamed-domain` | a domain streams from a host store (`[tiles]`); its store writer already copies nothing |
| `sealed-forcing-extension` | the sealed forcing extension is in force |
| `multi-device-state` | the state spans more than one GPU |
| `insufficient-host-ram` | the snapshot would leave less than max(2 GiB, 5% of RAM) free |
| `pinned-allocation-failed` | page-locked memory could not be allocated |

`WOOF_ASYNC_CHECKPOINTS` (`1`/`0`, `on`/`off`, `true`/`false`) overrides
the config key. The run receipt's `checkpoint_writes` block records the
mode requested and its source, the mode each checkpoint ran in, each
fallback's reason, the waits for a checkpoint in flight, the snapshot
memory, and each checkpoint's model-loop stall (`loop_stall_seconds`)
and background write time.

The synchronous writer also changed. It used to copy the whole state to
the host before writing. It now copies one member at a time through one
reusable page-locked buffer, so its peak host memory is the largest
member, not the whole state. The files are byte-identical. The setup
fingerprint and the forcing-prefix rows in every header are cached per
boundary inventory. They are recomputed only when an interval is added,
replaced or removed, or the setup arrays change.

## Checkpoint format compatibility

2.7.0 writes restart format 6. A checkpoint written by 2.6.5 (format 5) is refused by name: 2.7.0 adds the adaptive-timestep and `eta_levels` configuration echo, the `fields/ustm` surface-layer member and the `held/gf_*` cumulus forcing members, none of which a format-5 file carries. Restart such a run from its initial conditions, or complete it on 2.6.5. A format-6 checkpoint written with the adaptive clock off does not need the adaptive fields in its configuration echo, and an omitted `eta_levels` matches the default (inherit the source's ladder).
