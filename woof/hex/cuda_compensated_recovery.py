"""Compensated large-step recovery on the GPU, for timesteps below every anchor.

The device half of :func:`woof.hex.integration.recover_large_step_variables_compensated`.

Why it exists
-------------
At sub-second timesteps one timestep's increment of the carried binary32
large-step state (``rho_p``, ``rtheta_p``, ``ru``, ``rw``) is only a few units
in the last place of the field it is added to, so ``saved + perturbation``
rounds a measurable part of it away (``woof.hex.precision_probe.
large_step_increment_loss``: up to 36 % of the theta increment at dt 0.25 s).
TwoSum compensation carries the exact rounding error of that addition into
the next timestep, so the carried state is ``recovered + residual`` to about
twice working precision while the model keeps reading ``recovered``.

Why attachment rather than an edit of the dycore
------------------------------------------------
The stock recovery kernels (``recover_cells_f32`` / ``recover_edges_f32`` /
``recover_interfaces_f32``) live in ``cuda_driver.py``, an exact-byte pinned
execution source in the regional kernel set
(``cuda_backend.regional_admission.REGIONAL_KERNEL_SOURCES``); editing it moves
``kernel_set_sha256`` and lapses every minted regional class.  This module
therefore enters the way :mod:`woof.hex.cuda_les_v841` and
:mod:`woof.hex.cuda_driver_lts` do: :func:`attach_compensated_recovery` wraps
ONE call target on one built driver, ``driver._recover_candidate``, and leaves
every pinned byte where it is.

* Before the stock recovery runs, each of the four perturbations is replaced
  by ``perturbation + residual`` (``compensated_fold_f32``), and the rounding
  error of that fold is kept.  Every RK stage folds, as the CPU authority does.
* The stock recovery then runs unchanged on the folded perturbations: the same
  kernels, the same regional overwrites, the same endpoint enforcement.
* On RK stage 3 only -- the stage whose result the next dynamics subcycle
  copies as its saved state, the GPU form of ``final_stage`` -- the residual
  becomes the exact TwoSum error of ``saved + folded`` plus the fold error
  (``compensated_residual_f32``).

Two fail-closed guards keep the residual describing the value it belongs to.
At stage 3 a residual is kept only where the recovered value is bit-equal to
the ``saved + folded`` sum the residual was taken from (a regional
specified-zone overwrite, for one, leaves it zero there).  At every fold the
residual is applied only where the saved value is bit-equal to the value the
residual was recorded against; anything that rewrote a carried value between
two recoveries (a physics commit, a specified-zone reset) drops that element's
residual, so it recovers uncompensated for one subcycle rather than with a
correction for some other number.  A value guard cannot see a rollback at an
element whose whole increment rounded away (the restored value equals the
refused one), so the outer step is wrapped too: a step that starts at or
before the start time of the step before it (a retried or rewound step)
clears every residual first.

Stage 3 of the CUDA recovery carries no diabatic term (``recover_cells_f32``
forms ``rtheta_p = saved + rtheta_pp``), so the rtheta residual is the single
TwoSum of that addition: the CPU authority's two-rounding form with a zero
``rt_diabatic_tendency``.

Gate.  :func:`attach_compensated_recovery` returns ``None`` and rebinds nothing
unless the run's ``config_dt`` is below the smallest timestep the anchor table
admits (:func:`woof.hex.dt_admission.below_smallest_anchor`, the rule
:func:`woof.hex.integration.compensated_recovery_required` states for the CPU
authority), so every anchored run and every minted regional class executes
exactly the bytes it did before.  Only the experimental lane (``woof hex
forecast --experimental-dt``) reaches it, and its output is labelled
``large_step_recovery=compensated``.  It refuses the partitioned two-rank
driver (no halo rule for the residual is proved) and any schedule whose last
RK stage is not stage 3.

Detaching.  The forecast attaches local time stepping AFTER this module, so
LTS wraps this wrapper; :func:`detach_compensated_recovery` refuses unless it
is the outermost ``_recover_candidate`` (detach LTS first).

Memory.  Four persistent binary32 copies of each carried sum (residual,
anchor, folded perturbation, fold error): ``4 x (2 nCells + nEdges) x nlev``
plus ``4 x nCells x (nlev + 1)`` floats, not counted in the forecast's GPU
memory admission.

Non-claims.  The residual is not written into a restart checkpoint: a
restarted run starts the compensation again from zero, so it is not
bit-identical to an uninterrupted run.  No native MPAS build carries this
compensation; with it on, the sub-anchor lane is a declared divergence from
native arithmetic.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, NoReturn

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS
from .errors import ConfigurationRefusal

MODULE_KEY = "hexcore.cuda_compensated_recovery"
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false")

#: The label written to history files and run receipts when the lane is on.
RECOVERY_MODE_COMPENSATED = "compensated"
#: The attribute / receipt key that carries it.
RECOVERY_MODE_KEY = "large_step_recovery"

#: ``(residual name, saved owner, saved attribute, acoustic perturbation,
#: recovered owner, recovered attribute)``: the four carried sums, in the
#: order :data:`woof.hex.integration.COMPENSATED_RECOVERY_FIELDS` lists them.
#: ``saved``/``recovered`` owners are ``"diag"`` (the saved diagnostics) or
#: ``"state"`` (the prognostic state).
CARRIED_SUMS: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("rho_p", "diag", "density_perturbation", "rho_pp", "diag", "density_perturbation"),
    ("rtheta_p", "diag", "rho_theta_perturbation", "rtheta_pp", "diag", "rho_theta_perturbation"),
    ("ru", "state", "rho_u", "ru_p", "state", "rho_u"),
    ("rw", "state", "rho_w", "rw_p", "state", "rho_w"),
)

CUDA_COMPENSATED_RECOVERY_SOURCE = CUDA_FTZ_HELPERS + r"""
#define COMP_ELEMENT_LOOP(total) \
    for (int element = blockDim.x * blockIdx.x + threadIdx.x; \
         element < (total); element += gridDim.x * blockDim.x)

// Knuth TwoSum: the exact (a + b) - s for s = fl(a + b), branch-free and
// valid for any magnitudes.  Every operation goes through the subnormal-safe
// helpers, so the error is the IEEE one on any compile route.
__device__ __forceinline__ float comp_two_sum_error(
    const float a, const float b, const float s)
{
    const float bb = mpas_sub(s, a);
    return mpas_add(mpas_sub(a, mpas_sub(s, bb)), mpas_sub(b, bb));
}

__device__ __forceinline__ bool comp_same_bits(const float x, const float y)
{
    return __float_as_uint(x) == __float_as_uint(y);
}

// folded = pert + residual and, with has_error, fold_error = its exact
// rounding error.  With use_guard, the residual counts only where the saved
// value is bit-equal to the value it was recorded against (anchor);
// elsewhere it is dropped.  A zero residual leaves the perturbation's bits
// untouched.
extern "C" __global__ void compensated_fold_f32(
    const int total, const int use_guard, const int has_error,
    const float *pert, const float *residual,
    const float *saved, const float *anchor,
    float *folded, float *fold_error)
{
    COMP_ELEMENT_LOOP(total) {
        float carried = residual[element];
        if (use_guard && !comp_same_bits(saved[element], anchor[element])) {
            carried = 0.0f;
        }
        const float p = pert[element];
        if (carried == 0.0f) {
            folded[element] = p;
            if (has_error) fold_error[element] = 0.0f;
        } else {
            const float s = mpas_add(p, carried);
            folded[element] = s;
            if (has_error) fold_error[element] = comp_two_sum_error(p, carried, s);
        }
    }
}

// The final-stage residual of recovered = fl(saved + folded): its exact
// TwoSum error plus the fold error, kept only on rows [row_begin, row_end)
// of a row-major (rows, row_width) field and only where the recovered value
// is bit-equal to that sum.  anchor records the recovered value the residual
// belongs to.
extern "C" __global__ void compensated_residual_f32(
    const int total, const int row_width, const int row_begin,
    const int row_end, const int has_fold_error,
    const float *saved, const float *folded, const float *fold_error,
    const float *recovered, float *residual, float *anchor)
{
    COMP_ELEMENT_LOOP(total) {
        const int row = element / row_width;
        const float out = recovered[element];
        anchor[element] = out;
        float value = 0.0f;
        if (row >= row_begin && row < row_end) {
            const float a = saved[element];
            const float b = folded[element];
            const float s = mpas_add(a, b);
            if (comp_same_bits(s, out)) {
                value = comp_two_sum_error(a, b, s);
                if (has_fold_error) {
                    value = mpas_add(value, fold_error[element]);
                }
            }
        }
        residual[element] = value;
    }
}
"""

KERNEL_NAMES: tuple[str, ...] = ("compensated_fold_f32", "compensated_residual_f32")


def _cupy() -> Any:
    import cupy as cp

    return cp


class CompensatedRecoveryKernels:
    """The two kernels, compiled through the driver's cache when it has one."""

    def __init__(self, kernel_cache: Any | None = None) -> None:
        self.kernel_cache = kernel_cache
        self._kernels: dict[str, Any] = {}
        self._module: Any | None = None

    @property
    def cp(self) -> Any:
        # Imported on first use, so attaching (and the gate) needs no device.
        return _cupy()

    def _kernel(self, name: str) -> Any:
        kernel = self._kernels.get(name)
        if kernel is None:
            if self.kernel_cache is not None:
                kernel = self.kernel_cache.raw_kernel(
                    name, CUDA_COMPENSATED_RECOVERY_SOURCE, module_key=MODULE_KEY
                )
            else:
                if self._module is None:
                    self._module = self.cp.RawModule(
                        code=CUDA_COMPENSATED_RECOVERY_SOURCE, options=COMPILE_OPTIONS
                    )
                kernel = self._module.get_function(name)
            self._kernels[name] = kernel
        return kernel

    def _launch(self, name: str, total: int, args: tuple[Any, ...]) -> None:
        if total < 1:
            return
        threads = 128
        blocks = (int(total) + threads - 1) // threads
        self._kernel(name)((blocks,), (threads,), args)

    @staticmethod
    def _require(name: str, array: Any, like: Any) -> None:
        if array.shape != like.shape or array.dtype != np.float32:
            raise ValueError(
                f"compensated recovery: {name} must be float32 {like.shape}; "
                f"got {array.dtype} {array.shape}"
            )
        if not array.flags.c_contiguous:
            raise ValueError(f"compensated recovery: {name} must be C-contiguous")

    def fold(
        self,
        pert: Any,
        residual: Any,
        *,
        saved: Any | None = None,
        anchor: Any | None = None,
        want_error: bool = True,
        folded_out: Any | None = None,
        error_out: Any | None = None,
    ) -> tuple[Any, Any | None]:
        """``(pert + residual, rounding error)``, guarded when ``anchor`` is given.

        The error is ``None`` when ``want_error`` is false.  ``folded_out`` /
        ``error_out`` are reused buffers (allocated when absent).
        """

        for name, array in (
            ("residual", residual),
            ("saved", saved),
            ("anchor", anchor),
            ("folded_out", folded_out),
            ("error_out", error_out),
        ):
            if array is not None:
                self._require(name, array, pert)
        self._require("perturbation", pert, pert)
        if (saved is None) != (anchor is None):
            raise ValueError("the fold guard needs both saved and anchor")
        folded = self.cp.empty_like(pert) if folded_out is None else folded_out
        error = None
        if want_error:
            error = self.cp.empty_like(pert) if error_out is None else error_out
        use_guard = anchor is not None
        self._launch(
            "compensated_fold_f32",
            int(pert.size),
            (
                np.int32(pert.size),
                np.int32(1 if use_guard else 0),
                np.int32(1 if want_error else 0),
                pert,
                residual,
                saved if use_guard else pert,
                anchor if use_guard else pert,
                folded,
                folded if error is None else error,
            ),
        )
        return folded, error

    def residual(
        self,
        saved: Any,
        folded: Any,
        fold_error: Any | None,
        recovered: Any,
        *,
        rows: tuple[int, int] | None = None,
        residual_out: Any | None = None,
        anchor_out: Any | None = None,
    ) -> tuple[Any, Any]:
        """The final-stage residual of ``recovered = fl(saved + folded)``.

        ``rows`` limits it to a half-open range of leading-axis rows (the
        interior interfaces of ``rw``); elsewhere it is zero.
        """

        for name, array in (
            ("saved", saved),
            ("folded", folded),
            ("fold_error", fold_error),
            ("residual_out", residual_out),
            ("anchor_out", anchor_out),
        ):
            if array is not None:
                self._require(name, array, recovered)
        self._require("recovered", recovered, recovered)
        nrows = int(recovered.shape[0]) if recovered.ndim else 1
        width = int(recovered.size) // max(nrows, 1)
        begin, end = (0, nrows) if rows is None else (int(rows[0]), int(rows[1]))
        out = self.cp.empty_like(recovered) if residual_out is None else residual_out
        anchor = self.cp.empty_like(recovered) if anchor_out is None else anchor_out
        self._launch(
            "compensated_residual_f32",
            int(recovered.size),
            (
                np.int32(recovered.size),
                np.int32(max(width, 1)),
                np.int32(begin),
                np.int32(end),
                np.int32(0 if fold_error is None else 1),
                saved,
                folded,
                recovered if fold_error is None else fold_error,
                recovered,
                out,
                anchor,
            ),
        )
        return out, anchor


@dataclass
class CompensatedRecoveryAttachment:
    """What :func:`attach_compensated_recovery` wraps, and the residual it carries."""

    driver: Any
    kernels: CompensatedRecoveryKernels
    original_recover: Any
    recover_was_instance: bool
    dt_seconds: float
    smallest_anchored_dt_seconds: float
    #: ``driver._step_device_v841`` when the driver has one (the real CUDA
    #: driver); wrapped so a retried or rewound step clears the residual.
    original_step: Any | None = None
    step_was_instance: bool = False
    #: Residual per carried sum, absent until the first stage-3 recovery.
    residual: dict[str, Any] = field(default_factory=dict)
    #: The recovered value each residual element belongs to.
    anchor: dict[str, Any] = field(default_factory=dict)
    #: Reused fold buffers (folded perturbation, fold error) per carried sum.
    folded_buffers: dict[str, Any] = field(default_factory=dict)
    error_buffers: dict[str, Any] = field(default_factory=dict)
    stage_recoveries: int = 0
    residual_updates: int = 0
    residual_resets: int = 0
    last_step_start_seconds: float | None = None

    def clear_residual(self) -> None:
        """Restart the compensation from zero (the next fold is uncompensated)."""

        self.residual.clear()
        self.anchor.clear()

    def step(self, *args: Any, **kwargs: Any) -> Any:
        start = float(self.driver.atmosphere.state.time_seconds)
        previous = self.last_step_start_seconds
        if previous is not None and start <= previous and self.residual:
            # The same start time again (a refused step retried) or an
            # earlier one (a rewind): the residual describes a step that did
            # not happen, and the value guard cannot see that where a whole
            # increment rounded away.
            self.clear_residual()
            self.residual_resets += 1
        self.last_step_start_seconds = start
        assert self.original_step is not None
        return self.original_step(*args, **kwargs)

    def recover_candidate(
        self,
        saved_state: Any,
        saved_diag: Any,
        acoustic: Any,
        stage: int,
        acoustic_steps: int,
        scalars: Any,
    ) -> Any:
        final = int(stage) == 3
        owners_saved = {"diag": saved_diag, "state": saved_state}
        folded: dict[str, Any] = {}
        errors: dict[str, Any | None] = {}
        replacements: dict[str, Any] = {}
        for name, saved_owner, saved_attr, pert_attr, _r_owner, _r_attr in CARRIED_SUMS:
            pert = getattr(acoustic, pert_attr)
            carried = self.residual.get(name)
            if carried is None:
                folded[name], errors[name] = pert, None
                continue
            folded[name], errors[name] = self.kernels.fold(
                pert,
                carried,
                saved=getattr(owners_saved[saved_owner], saved_attr),
                anchor=self.anchor[name],
                want_error=final,
                folded_out=self.folded_buffers.get(name),
                error_out=self.error_buffers.get(name) if final else None,
            )
            self.folded_buffers[name] = folded[name]
            if errors[name] is not None:
                self.error_buffers[name] = errors[name]
            replacements[pert_attr] = folded[name]
        effective = dataclasses.replace(acoustic, **replacements) if replacements else acoustic
        result = self.original_recover(
            saved_state, saved_diag, effective, stage, acoustic_steps, scalars
        )
        self.stage_recoveries += 1
        if not final:
            return result
        state, diagnostics = result[0], result[1]
        owners_out = {"diag": diagnostics, "state": state}
        for name, saved_owner, saved_attr, _pert, r_owner, r_attr in CARRIED_SUMS:
            recovered = getattr(owners_out[r_owner], r_attr)
            rows = None
            if name == "rw":
                # rw is the interior-interface sum rw_save + rw_p; its bottom
                # and top interfaces are zero by construction.
                rows = (1, int(recovered.shape[0]) - 1)
            self.residual[name], self.anchor[name] = self.kernels.residual(
                getattr(owners_saved[saved_owner], saved_attr),
                folded[name],
                errors[name],
                recovered,
                rows=rows,
                residual_out=self.residual.get(name),
                anchor_out=self.anchor.get(name),
            )
        self.residual_updates += 1
        return result

    def summary(self) -> dict[str, Any]:
        """The receipt block.  Reads the residual magnitudes back (one D2H each)."""

        residual_max = {
            name: (
                None
                if self.residual.get(name) is None
                else float(self.kernels.cp.abs(self.residual[name]).max().get())
            )
            for name, *_ in CARRIED_SUMS
        }
        return {
            RECOVERY_MODE_KEY: RECOVERY_MODE_COMPENSATED,
            "dt_seconds": float(self.dt_seconds),
            "smallest_anchored_dt_seconds": float(self.smallest_anchored_dt_seconds),
            "carried_sums": [name for name, *_ in CARRIED_SUMS],
            "stage_recoveries": int(self.stage_recoveries),
            "residual_updates": int(self.residual_updates),
            "residual_resets": int(self.residual_resets),
            "residual_max_abs": residual_max,
            "attachment": (
                "woof.hex.cuda_compensated_recovery.attach_compensated_recovery "
                "wraps driver._recover_candidate (and _step_device_v841, to "
                "clear the residual on a retried step); no pinned source byte "
                "and no regional kernel-set byte changes"
            ),
            "reference": (
                "woof.hex.integration.recover_large_step_variables_compensated "
                "(CPU authority, zero rt_diabatic_tendency)"
            ),
            "nonclaims": [
                "the residual is not checkpointed: a restarted run restarts "
                "the compensation from zero",
                "native MPAS-A v8.4.1 carries no compensation; this lane is a "
                "declared divergence from native arithmetic",
                "the persistent residual, anchor and fold buffers are not "
                "counted in the forecast's GPU memory admission",
            ],
        }


def compensated_recovery_floor_seconds() -> float:
    """The smallest anchored timestep, below which the lane is on."""

    from .dt_admission import smallest_anchored_dt

    return float(smallest_anchored_dt())


def _refuse(reason: str, remedy: str) -> NoReturn:
    raise ConfigurationRefusal(
        "large_step_recovery", RECOVERY_MODE_COMPENSATED, reason, remedy
    )


def attach_compensated_recovery(
    driver: Any, *, dt_seconds: float | None = None
) -> CompensatedRecoveryAttachment | None:
    """Wrap ``driver._recover_candidate`` for a sub-anchor timestep, or do nothing.

    ``dt_seconds`` defaults to ``driver.config.config_dt``.  Returns ``None``
    and rebinds nothing unless it is strictly below the smallest anchored
    timestep (:func:`woof.hex.dt_admission.below_smallest_anchor`).
    """

    from .dt_admission import below_smallest_anchor

    dt = float(driver.config.config_dt if dt_seconds is None else dt_seconds)
    if not below_smallest_anchor(dt):
        return None
    if getattr(driver, "large_step_recovery", None) is not None:
        _refuse(
            "this driver already carries a compensated recovery attachment; "
            "a second wrap would fold every residual twice",
            "one attachment per driver",
        )
    if getattr(driver, "halo_exchanger_v841", None) is not None:
        _refuse(
            "the partitioned two-rank driver exchanges halo values after each "
            "recovery, and no rule keeps a peer's residual at a halo element "
            "equal to its owner's",
            "a single-device run",
        )
    original = getattr(driver, "_recover_candidate", None)
    if original is None:
        _refuse("the driver has no _recover_candidate to wrap", "a CUDA dycore driver")
    order = getattr(getattr(driver, "config", None), "config_time_integration_order", None)
    if order is not None:
        from .integration import RKSchedule

        stages = RKSchedule.from_mpas(dt, order=int(order)).stages
        if stages[-1].stage != 3:
            _refuse(
                f"config_time_integration_order={order} does not end on RK "
                "stage 3, the stage whose result the next subcycle carries",
                "config_time_integration_order=3",
            )
    original_step = getattr(driver, "_step_device_v841", None)
    attachment = CompensatedRecoveryAttachment(
        driver=driver,
        kernels=CompensatedRecoveryKernels(getattr(driver, "cache", None)),
        original_recover=original,
        recover_was_instance="_recover_candidate" in getattr(driver, "__dict__", {}),
        dt_seconds=dt,
        smallest_anchored_dt_seconds=compensated_recovery_floor_seconds(),
        original_step=original_step,
        step_was_instance="_step_device_v841" in getattr(driver, "__dict__", {}),
    )
    driver._recover_candidate = attachment.recover_candidate
    if original_step is not None:
        driver._step_device_v841 = attachment.step
    driver.large_step_recovery = attachment
    return attachment


def detach_compensated_recovery(attachment: CompensatedRecoveryAttachment) -> None:
    """Put ``_recover_candidate`` (and the step) back and drop the residual.

    Refuses unless this attachment is the outermost wrapper of both, so a
    wrapper attached later (local time stepping) is never silently unbound.
    """

    driver = attachment.driver
    bound = getattr(driver, "__dict__", {})
    if bound.get("_recover_candidate") != attachment.recover_candidate or (
        attachment.original_step is not None
        and bound.get("_step_device_v841") != attachment.step
    ):
        raise RuntimeError(
            "the compensated recovery is not the outermost wrapper of "
            "_recover_candidate/_step_device_v841; detach the wrapper attached "
            "after it (local time stepping) first"
        )
    if attachment.recover_was_instance:
        driver._recover_candidate = attachment.original_recover
    else:
        del driver._recover_candidate
    if attachment.original_step is not None:
        if attachment.step_was_instance:
            driver._step_device_v841 = attachment.original_step
        else:
            del driver._step_device_v841
    driver.large_step_recovery = None
    attachment.clear_residual()
    attachment.folded_buffers.clear()
    attachment.error_buffers.clear()


def stamp_large_step_recovery_history(
    path: Any, attachment: CompensatedRecoveryAttachment
) -> None:
    """Label a written history file ``large_step_recovery=compensated``."""

    from netCDF4 import Dataset

    with Dataset(path, "a") as dataset:
        dataset.setncattr(RECOVERY_MODE_KEY, RECOVERY_MODE_COMPENSATED)
        dataset.setncattr(
            RECOVERY_MODE_KEY + "_reason",
            f"dt={attachment.dt_seconds:g} s is below the smallest anchored "
            f"timestep ({attachment.smallest_anchored_dt_seconds:g} s); the "
            "carried binary32 large-step sums are TwoSum-compensated "
            "(woof.hex.cuda_compensated_recovery), experimental",
        )


__all__ = [
    "CARRIED_SUMS",
    "COMPILE_OPTIONS",
    "CUDA_COMPENSATED_RECOVERY_SOURCE",
    "CompensatedRecoveryAttachment",
    "CompensatedRecoveryKernels",
    "KERNEL_NAMES",
    "MODULE_KEY",
    "RECOVERY_MODE_COMPENSATED",
    "RECOVERY_MODE_KEY",
    "attach_compensated_recovery",
    "compensated_recovery_floor_seconds",
    "detach_compensated_recovery",
    "stamp_large_step_recovery_history",
]
