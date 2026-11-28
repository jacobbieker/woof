"""P3a's dry WRF window as the 4D-Var tangent-linear model (EXPERIMENTAL;
docs/dev/WRFDA-4DVAR-PLAN.md, PR 4D.3).

:class:`WindowModel` is a :class:`~woof.da.var.fourd.TLModel` over a
:class:`woof.grad.wrfplus.window.Window`:

- the TL state :func:`~woof.da.var.transfer4d.xa_to_tl` writes (WRFDA's
  ``g_*``, Fortran ``(i, j, k)``) goes into the window's layout
  (``(k, j, i)``, the window's device and type) and starts the window as
  WRFDA's TL model starts (:func:`~woof.grad.wrfplus.window.wrfda_tl_start`:
  ``g_p`` into both pressure carriers, ``g_mu`` into ``muts`` too, ``al``
  and ``alt`` zero).  ``g_ph_2`` goes into ``php`` (zero unless
  ``ph="hydrostatic"``);
- slot ``n`` is the window's state ``n * steps_per_slot`` (``var4d_bin`` over
  the time step), read back into WRFDA's names; ``g_p`` is WRF's
  perturbation pressure, the window's ``p_pert``;
- the adjoint puts each slot's forcing on the same names, runs
  :meth:`Window.ad` and restricts with
  :func:`~woof.grad.wrfplus.window.wrfda_tl_start_ad`.

**Moisture.**  On a moist window (``Window.inputs`` holds qv, qc and qr:
P3a's moist binding, mp_physics 0) ``moisture="advect"`` (the default there)
starts the window's ``qv`` from ``g_qv`` and reads each slot's ``g_qv``
back, as WRFDA's TL model advects ``g_moist``; qc and qr start at zero
(``cloud_cv_options = 0``: WRFDA's transfer gives them no perturbation) and
nothing observes them.  On a dry window ``moisture="persist"`` (the default
there, and the only option) holds the ``qv`` perturbation constant over the
window (its adjoint sums the slots' forcings): not WRFDA's moist TL, meant
for the dry OSSE (L2).  Each option is refused on the other kind of
window.

**Two trajectories.**  The window is linearised about ITS OWN base (its
start state run forward by :meth:`Window.forward`), but each slot's ``H'``
and ``wrftltoxa`` are linearised about THAT SLOT'S first guess ``xb``
(:class:`~woof.da.var.fourd.FourDObsTerm`).  Nothing here makes the two
the same: if the window's base at slot ``n`` is not ``fgNN``, the 4D-Var
gradient mixes two trajectories and is not WRFDA's.  WRFDA makes them one
(WRFPLUS writes the trajectory the slots read).  Pass ``base=`` (the slots'
``(xb, grid)``) to compare the window's base with each slot's ``grid`` at
construction and refuse a mismatch over ``base_rtol``; the L1/L0 lanes
should turn it on.

**Boundary tables** (a specified window, ``lbc="wrfda"``, the default
there).  WRFDA perturbs the lateral boundary tables even with
``var4d_lbc = .false.``: ``da_transfer_xatowrftl.inc:282`` calls
``da_transfer_wrftl_lbc_t0`` unconditionally, which couples ``g_u``,
``g_v``, ``g_t``, ``g_ph``, ``g_qv`` and ``g_mu`` about the start state
(``g_couple``, da_4dvar_lbc.inc: WRFDA's non-hybrid coupling, the face
mass of ``g_calc_mu_uv``, edge faces the adjacent cell, ``1/msf`` on u and
v only) and ``g_stuff_bdy``'s them into the value tables (``g_w_b`` and
``g_w_bt`` zero); ``da_transform_vtoy.inc:75`` then calls
``da_transfer_xatowrftl_lbc``, whose end-of-interval perturbation is zero
without ``var4d_lbc``, so each tendency table is ``(0 - g_b) /
interval_seconds``: the boundary perturbation decays linearly to zero over
the interval.  :func:`lbc_tl` builds those tables in the window's layout
(woof's resident tables', ``lateral_bc.extract_lateral_side``), and
:func:`lbc_ad` is its adjoint, which :meth:`WindowModel.ad` adds to the
state adjoint from ``Window.ad``'s table adjoints.  ``lbc="fixed"`` leaves
the tables unperturbed and drops their adjoint (not WRFDA's).

**Outer loops' boundaries.**  ``da_transfer_xatowrf.inc:665-727`` adds the
last TL call's table perturbation (``model_grid%g_*_b``, ``g_*_bt``: the
loop's own ``xhat``, ``da_minimise_cg.inc:258``) to ``grid``'s tables,
which the next loop's nonlinear run uses (``da_solve.inc:827-918``).
:meth:`WindowModel.boundary_increment` is that perturbation for a TL
state, and :meth:`WindowModel.add_boundary_increment` adds it to the
window's tables (the nonlinear rerun's and the TL's);
:func:`~woof.da.var.analysis.analyse` carries it on the analysis state
(``FirstGuess.lbc_increment``) for ``var4d.nl_model``.

The window must be binary64: a binary32 window's adjoint is only its
transpose to about 1e-6, and the minimiser needs 1e-12.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import transfer4d as t4

__all__ = ["WindowModel", "to_window", "from_window", "lbc_tl", "lbc_ad"]

#: WRFDA's TL names -> the window's names (the slot read-back; the start
#: goes through ``wrfda_tl_start``).
_NAMES = {"u_2": "u", "v_2": "v", "w_2": "w", "t_2": "thp", "mu_2": "mup", "ph_2": "php"}

#: ``moisture``: ``advect`` (a moist window carries g_qv) or ``persist`` (a dry
#: window: g_qv held over the window).
_MOISTURE = ("advect", "persist")

#: ``halo_start``: ``wrfplus`` (WRFPLUS's start_domain: the primal mu halo
#: reset to the edge row, its tangent seeded from g_mu) or ``carried`` (the
#: window's carried halo, the TL halo starting at zero; woof's own window).
_HALO_STARTS = ("wrfplus", "carried")


#: ``lbc``: ``wrfda`` (the tables perturbed as WRFDA's TL start perturbs
#: them, var4d_lbc off) or ``fixed`` (unperturbed, their adjoint dropped).
_LBCS = ("wrfda", "fixed")

#: Each boundary table field's state array (the window's names).
_LBC_STATE = {"u": "u", "v": "v", "theta": "thp", "phi": "php", "mu": "mup", "qv": "qv"}
_SIDES = ("west", "east", "south", "north")


def _faces(m: np.ndarray, axis: int) -> np.ndarray:
    """``g_calc_mu_uv``'s face mass (module_big_step_utilities_em_tl.F
    G_CALC_MU_UV, non-periodic): interior faces the two cells' mean, each
    domain-edge face its adjacent cell."""
    m = np.moveaxis(m, axis, -1)
    out = np.empty(m.shape[:-1] + (m.shape[-1] + 1,), dtype=np.float64)
    out[..., 1:-1] = 0.5 * (m[..., :-1] + m[..., 1:])
    out[..., 0] = m[..., 0]
    out[..., -1] = m[..., -1]
    return np.moveaxis(out, -1, axis)


def _faces_ad(a: np.ndarray, axis: int) -> np.ndarray:
    a = np.moveaxis(a, axis, -1)
    out = np.zeros(a.shape[:-1] + (a.shape[-1] - 1,), dtype=np.float64)
    out[..., :-1] += 0.5 * a[..., 1:-1]
    out[..., 1:] += 0.5 * a[..., 1:-1]
    out[..., 0] += a[..., 0]
    out[..., -1] += a[..., -1]
    return np.moveaxis(out, -1, axis)


def _side(f: np.ndarray, side: str, w: int) -> np.ndarray:
    """``extract_lateral_side``'s table (east and north outermost-first)."""
    if side == "west":
        return f[..., :w]
    if side == "east":
        return f[..., -w:][..., ::-1]
    if side == "south":
        return f[..., :w, :]
    return f[..., -w:, :][..., ::-1, :]


def _side_ad(acc: np.ndarray, a: np.ndarray, side: str, w: int) -> None:
    if side == "west":
        acc[..., :w] += a
    elif side == "east":
        acc[..., -w:] += a[..., ::-1]
    elif side == "south":
        acc[..., :w, :] += a
    else:
        acc[..., -w:, :] += a[..., ::-1, :]


def _table_fields(params) -> tuple:
    """The fields ``params`` (``Window.params``) holds tables for, in
    :data:`_LBC_STATE`'s order; each must have all eight tables."""
    fields = []
    for f in _LBC_STATE:
        have = [f"lbc_{f}_{s}{t}" in params for s in _SIDES for t in ("", "_t")]
        if any(have) and not all(have):
            raise ValueError(f"the window has some of {f}'s boundary tables, not all eight")
        if all(have):
            fields.append(f)
    unknown = sorted(set(params) - {f"lbc_{f}_{s}{t}" for f in fields for s in _SIDES for t in ("", "_t")})
    if unknown:
        raise ValueError(f"boundary tables of no known field: {unknown}")
    return tuple(fields)


def lbc_tl(g: dict, base: dict, consts: dict, fields, width: int, seconds: float) -> dict:
    """WRFDA's TL boundary tables (module docstring) from the TL state
    ``g`` about ``base`` (both the window's names and layout, host
    binary64; ``mup``, ``u``, ``v``, ``thp``, ``php`` and ``qv`` as
    ``fields`` need).  ``consts``: ``mub2d``, ``msfu``, ``msfv``.
    ``da_transfer_wrftl_lbc_t0``'s ``g_couple`` (non-hybrid, as WRFDA's):

    - u: ``g_u * muu / msfu + u * g_muu / msfu`` (``muu`` the face mass of
      ``mub + mu``), v likewise along y with ``msfv``;
    - theta, phi, qv: ``g_f * (mub + mu) + f * g_mu``;
    - mu: ``g_mu``.

    Value tables ``lbc_<field>_<side>``; tendencies ``lbc_<field>_<side>_t =
    -value / seconds`` (``da_transfer_xatowrftl_lbc``'s ``g_stuff_bdytend``
    with a zero end perturbation)."""
    mu = np.asarray(consts["mub2d"], dtype=np.float64) + base["mup"]
    gmu = g["mup"]
    out = {}
    for f in fields:
        if f == "u":
            c = (g["u"] * _faces(mu, 1) + base["u"] * _faces(gmu, 1)) / consts["msfu"]
        elif f == "v":
            c = (g["v"] * _faces(mu, 0) + base["v"] * _faces(gmu, 0)) / consts["msfv"]
        elif f == "mu":
            c = gmu[None]
        else:
            s = _LBC_STATE[f]
            c = g[s] * mu + base[s] * gmu
        for side in _SIDES:
            v = np.ascontiguousarray(_side(c, side, width))
            out[f"lbc_{f}_{side}"] = v
            out[f"lbc_{f}_{side}_t"] = -v / seconds
    return out


def lbc_ad(a: dict, base: dict, consts: dict, fields, width: int, seconds: float) -> dict:
    """The adjoint of :func:`lbc_tl`: the table adjoints ``a`` (``Window.ad``'s
    ``params``) to the TL state's, ``{name: array}`` over ``mup`` and each
    field's state array (new arrays, the window's layout)."""
    mu = np.asarray(consts["mub2d"], dtype=np.float64) + base["mup"]
    out = {"mup": np.zeros(mu.shape)}
    for f in fields:
        s = _LBC_STATE[f]
        shape = mu[None].shape if f == "mu" else base[s].shape
        ac = np.zeros(shape)
        for side in _SIDES:
            av = np.asarray(a[f"lbc_{f}_{side}"], dtype=np.float64)
            at = np.asarray(a[f"lbc_{f}_{side}_t"], dtype=np.float64)
            _side_ad(ac, av - at / seconds, side, width)
        if f == "u":
            ac = ac / consts["msfu"]
            out["u"] = ac * _faces(mu, 1)
            out["mup"] += _faces_ad(np.sum(ac * base["u"], axis=0), 1)
        elif f == "v":
            ac = ac / consts["msfv"]
            out["v"] = ac * _faces(mu, 0)
            out["mup"] += _faces_ad(np.sum(ac * base["v"], axis=0), 0)
        elif f == "mu":
            out["mup"] += ac[0]
        else:
            out[s] = ac * mu
            out["mup"] += np.sum(ac * base[s], axis=0)
    return out


def _halo_name():
    from woof.grad.wrfplus import model as M

    return getattr(M, "HALO", None)


#: WRF's state names (``wrftoxb``'s ``grid``) -> the window's, for the base check.
_BASE = {**_NAMES, "p": "p_pert"}


def to_window(a: np.ndarray) -> np.ndarray:
    """Fortran ``(i, j[, k])`` to the window's ``([k,] j, i)``."""
    return np.ascontiguousarray(np.asarray(a).transpose())


def from_window(a: np.ndarray) -> np.ndarray:
    """The window's ``([k,] j, i)`` to Fortran ``(i, j[, k])``, binary64."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64).transpose())


def _step_inputs() -> tuple:
    from woof.grad.wrfplus import model as M

    return M.STEP_INPUTS


def _refuse_metal() -> None:
    """The window on Metal: refused, as Metal has no binary64 (the 4D-Var
    window is binary64 only; PERF-QUEUE holds the Metal port)."""
    from woof import backend

    if backend.active() == "mlx":
        raise NotImplementedError("4D-Var's window model on Metal: the window must be binary64 and Metal has no "
                                  "binary64; run it on CUDA or the host backend (WOOF_BACKEND=host)")


def _default_to_device(a: np.ndarray, dtype):
    import cupy as cp

    return cp.asarray(np.ascontiguousarray(a, dtype=dtype))


def _default_to_host(a) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)


class WindowModel:
    """WRFDA's TL and AD model over ``window`` (a forwarded
    :class:`~woof.grad.wrfplus.window.Window`), one slot every
    ``steps_per_slot`` steps.

    ``to_device``/``to_host`` move arrays between the host and the window
    (cupy, or woof's facade, by default); tests pass their own.

    ``base`` (optional): the slots' ``(xb, grid)`` pairs, as
    :class:`~woof.da.var.fourd.FourDObsTerm` takes them; the window's base
    at each slot is compared with that slot's ``grid`` (see the module
    docstring) and a relative difference over ``base_rtol`` is refused.
    ``base_end``: the end state for a window on an external trajectory
    (:meth:`check_base`).

    The adjoint forces every state input the window has
    (``Window.inputs`` where it has one: a specified domain adds the
    carried mu halo, forced with zero, as nothing observes it).

    ``halo_start`` (specified windows only): ``"wrfplus"`` (default) starts
    as WRFPLUS's ``start_domain`` does, the primal mu halo reset to the edge
    row and the TL halo seeded from ``g_mu``
    (``wrfda_tl_start(..., specified=True)``); ``"carried"`` keeps the
    window's carried halo and starts the TL halo at zero (woof's own
    window, not WRFPLUS's: a warm window then differs from WRFPLUS's base).

    ``lbc`` (specified windows only): ``"wrfda"`` (default) perturbs the
    boundary tables as WRFDA's TL start does with ``var4d_lbc = .false.``
    (:func:`lbc_tl`, the module docstring) and adds their adjoint;
    ``"fixed"`` leaves them unperturbed and drops their adjoint.

    :meth:`relinearise` moves the base to a new nonlinear run (an outer
    loop's :class:`~woof.da.var.outer4d.Trajectory`, through
    ``Window.forward(trajectory=...)``).  Binary64 only: refused on Metal
    when ``to_device`` is the default.
    """

    def __init__(self, window, steps_per_slot: int, *, moisture: str | None = None, to_device=None, to_host=None,
                 base: Sequence | None = None, base_rtol: float = 1e-5, base_end: dict | None = None,
                 halo_start: str = "wrfplus", lbc: str | None = None):
        moist = "qv" in tuple(getattr(window, "inputs", None) or ())
        if moisture is None:
            moisture = "advect" if moist else "persist"
        if moisture not in _MOISTURE:
            raise ValueError(f"moisture must be one of {_MOISTURE}, not {moisture!r}")
        if moisture == "advect" and not moist:
            raise NotImplementedError("moisture='advect': the window is dry (no qv among its inputs); "
                                      "only 'persist' holds on a dry window")
        if moisture == "persist" and moist:
            raise ValueError("moisture='persist' on a moist window: its qv tangent moves with the flow, "
                             "so the slots' g_qv are the window's ('advect')")
        self.moisture = moisture
        if to_device is None:
            _refuse_metal()
        if steps_per_slot < 1:
            raise ValueError(f"steps_per_slot must be >= 1, not {steps_per_slot}")
        self.window = window
        self.sps = int(steps_per_slot)
        self.dtype = np.dtype(str(window._start["u"].dtype))
        if self.dtype != np.float64:
            raise TypeError(f"the window is {self.dtype}; 4D-Var needs a binary64 window "
                            "(a binary32 window's adjoint is its transpose to only ~1e-6)")
        self._dev = to_device or (lambda a: _default_to_device(a, self.dtype))
        self._host = to_host or _default_to_host
        if halo_start not in _HALO_STARTS:
            raise ValueError(f"halo_start must be one of {_HALO_STARTS}, not {halo_start!r}")
        self.halo = _halo_name() if _halo_name() in self._inputs() else None
        self.halo_start = halo_start
        if self.halo is not None and halo_start == "wrfplus":
            self._halo_to_edge()
        params = tuple(getattr(window, "params", None) or ())
        if lbc is None:
            lbc = "wrfda" if params else None
        if lbc is not None and lbc not in _LBCS:
            raise ValueError(f"lbc must be one of {_LBCS}, not {lbc!r}")
        if lbc is not None and not params:
            raise ValueError(f"lbc={lbc!r}: the window has no boundary tables (not a specified domain)")
        self.lbc = lbc
        self._lbc_fields = _table_fields(params) if lbc == "wrfda" else ()
        if "qv" in self._lbc_fields and moisture != "advect":
            raise ValueError("the window has qv boundary tables but no g_qv (moisture is not 'advect')")
        self._lbc_base = None
        if getattr(window, "_checkpoints", None) is None:
            window.forward()
        self.base_diff = None
        self._end = base_end
        if base is not None:
            self.base_diff = self.check_base(base, rtol=base_rtol, end=base_end)

    def relinearise(self, trajectory) -> dict:
        """An outer loop's new nonlinear run (:class:`~woof.da.var.outer4d.Trajectory`):
        the window linearises about its per-step ``states``
        (``Window.forward(trajectory=...)``, as WRFPLUS reads the rerun's
        states).  Returns the run's end state, which :meth:`check_base`
        then uses by default.  A trajectory without ``states`` is refused:
        the window would stay on the previous loop's base."""
        states = getattr(trajectory, "states", None)
        if states is None:
            raise ValueError("the window model needs the rerun's per-step states (Trajectory.states) to "
                             "relinearise; without them the TL and AD stay on the previous loop's trajectory")
        if self.halo is not None and self.halo_start == "wrfplus":
            # WRFPLUS's start on an external trajectory: checked before the base moves, not changed
            st = states[0]
            if not bool(np.array_equal(self._host(st[self.halo]), self._host(st["mup"]))):
                raise ValueError("halo_start='wrfplus': the rerun's state 0 carries a mu halo that is not mup "
                                 "(WRFPLUS's start resets it to the edge row); reset it, or pass "
                                 "halo_start='carried'")
        self._end = self.window.forward(trajectory=states)
        self.base_diff = None
        self._lbc_base = None
        return self._end

    def check_base(self, slots: Sequence, *, rtol: float = 1e-5, end: dict | None = None) -> dict:
        """The window's base at each slot against that slot's ``grid``:
        ``{(n, name): max|base - grid| / max|grid|}``; refuses any over
        ``rtol``.

        A window running its own base: reruns :meth:`Window.forward` (the
        base the TL and AD then use is unchanged) and needs a checkpoint at
        every slot state.  A window on an external trajectory
        (``Window.forward(trajectory=...)``, which holds every state but the
        last): reads the trajectory, never reruns ``forward`` (that would
        replace it), and compares the last slot with ``end``, the state
        ``forward(trajectory=...)`` returned."""
        states = self._slots(len(slots))
        nsteps = self.window.nsteps
        if getattr(self.window, "_external", False):
            end = end if end is not None else self._end
            if end is None:
                raise ValueError("the window runs on an external trajectory: pass end= (the state "
                                 "forward(trajectory=...) returned) for the last slot")
            final = end
        else:
            every = int(getattr(self.window, "every", 1))
            if self.sps % every:
                raise ValueError(f"the base check needs a checkpoint at every slot: checkpoint_every={every} "
                                 f"does not divide steps_per_slot={self.sps}")
            final = self.window.forward()
        out, bad = {}, []
        for n, (k, (_, grid)) in enumerate(zip(states, slots)):
            st = final if k == nsteps else self.window._checkpoints[k]
            for g, w in _BASE.items():
                ref = np.asarray(grid[g], dtype=np.float64)
                got = from_window(self._host(st[w]))
                if got.shape != ref.shape:
                    raise ValueError(f"slot {n}: the window's {w} is {got.shape[::-1]} (k, j, i), "
                                     f"the slot's {g} {ref.shape} (i, j, k)")
                scale = float(np.max(np.abs(ref))) or 1.0
                r = float(np.max(np.abs(got - ref))) / scale
                out[(n, g)] = r
                if not r <= rtol:
                    bad.append(f"slot {n} {g}: {r:.3g}")
        if bad:
            raise ValueError("the window's base is not the slots' first guesses (two trajectories): "
                             + "; ".join(bad) + f" > rtol {rtol:g}")
        return out

    def _halo_to_edge(self) -> None:
        """WRFPLUS's ``start_domain`` on a specified window: the primal mu
        halo starts as the edge row, which in woof's form (model.HALO, a copy
        of ``mup`` whose ring rk_step 1 reads) is ``mup`` itself.  Resets the
        window's start and reruns its base if it already ran one; a window
        on an external trajectory is checked, not changed (its state 0 is
        the caller's)."""
        w, h = self.window, self.halo
        if getattr(w, "_external", False):
            st = w._checkpoints[0]
            if not bool(np.array_equal(self._host(st[h]), self._host(st["mup"]))):
                raise ValueError("halo_start='wrfplus': the external trajectory's state 0 carries a mu halo "
                                 "that is not mup (WRFPLUS's start resets it to the edge row); reset it, or "
                                 "pass halo_start='carried'")
            return
        w._start[h][...] = w._start["mup"]
        if getattr(w, "_checkpoints", None) is not None:
            w.forward()
        self._lbc_base = None

    # ------------------------------------------------------- boundary tables

    def _lbc_args(self) -> tuple:
        """:func:`lbc_tl`'s ``base``, ``consts``, width and seconds: the base
        is the window's state 0 (WRFDA couples about ``grid``, the loop's
        start), host binary64, cached until the base moves."""
        if self._lbc_base is None:
            w = self.window
            ck = w._checkpoints.get(0) if w._checkpoints else None
            st = ck if ck is not None else w._start
            names = {"mup"} | {_LBC_STATE[f] for f in self._lbc_fields if f != "mu"}
            base = {n: np.asarray(self._host(st[n]), dtype=np.float64) for n in names}
            consts = {n: np.asarray(self._host(w._start[n]), dtype=np.float64) for n in ("mub2d", "msfu", "msfv")}
            width = int(w._start[f"lbc_{self._lbc_fields[0]}_west"].shape[-1])
            seconds = float(w.lbc_seconds)
            self._lbc_base = (base, consts, width, seconds)
        return self._lbc_base

    def _lbc_state(self, g0: dict) -> dict:
        """The TL state :func:`lbc_tl` reads, from WRFDA's ``g_*`` (Fortran)."""
        names = {"u_2": "u", "v_2": "v", "t_2": "thp", "ph_2": "php", "mu_2": "mup", "qv": "qv"}
        return {w: to_window(np.asarray(g0[k], dtype=np.float64)) for k, w in names.items()
                if k in g0 and (w == "mup" or any(_LBC_STATE[f] == w for f in self._lbc_fields))}

    def boundary_increment(self, g0: dict) -> dict:
        """WRFDA's boundary-table perturbation of the TL state ``g0``
        (:func:`lbc_tl`; host binary64, ``Window.params``' names): what
        ``da_transfer_xatowrf.inc:665-727`` adds to the tables after a
        loop, for ``g0`` the loop's own increment.  Empty unless
        ``lbc == "wrfda"``."""
        if self.lbc != "wrfda":
            return {}
        return lbc_tl(self._lbc_state(g0), *self._lbc_args()[:2], self._lbc_fields, *self._lbc_args()[2:])

    def add_boundary_increment(self, tables: dict) -> None:
        """Add ``tables`` (:meth:`boundary_increment`'s) to the window's
        boundary tables in place: the tables every later ``forward`` (the
        next loop's nonlinear run) and TL/AD step read, as WRFDA's next loop
        runs on ``grid``'s updated tables."""
        start = self.window._start
        for n, v in tables.items():
            if n not in start:
                raise ValueError(f"the window has no boundary table {n!r}")
            start[n][...] += self._dev(np.asarray(v, dtype=np.float64))

    def _tl_start(self, d: dict) -> dict:
        from woof.grad.wrfplus.window import wrfda_tl_start

        if self.halo is not None and self.halo_start == "wrfplus":
            return wrfda_tl_start(d, specified=True)
        return wrfda_tl_start(d)

    def _tl_start_ad(self, a: dict) -> dict:
        from woof.grad.wrfplus.window import wrfda_tl_start_ad

        if self.halo is not None and self.halo_start == "wrfplus":
            return wrfda_tl_start_ad(a, specified=True)
        return wrfda_tl_start_ad(a)

    def _inputs(self) -> tuple:
        """The window's state inputs: ``Window.inputs`` (STEP_INPUTS plus the
        mu halo on a specified domain) where the window has it."""
        return tuple(getattr(self.window, "inputs", None) or _step_inputs())

    def _slots(self, nslots: int) -> list:
        if nslots < 2:
            raise ValueError(f"{nslots} slot(s): 4D-Var over a window needs at least 2 slots "
                             "(the start and the end); one slot is 3D-Var")
        if self.window.nsteps != self.sps * (nslots - 1):
            raise ValueError(f"{nslots} slots {self.sps} steps apart need a window of "
                             f"{self.sps * (nslots - 1)} steps; it has {self.window.nsteps}")
        return [n * self.sps for n in range(nslots)]

    def tl(self, g0: dict, nslots: int) -> list:
        missing = [k for k in t4.TL_KEYS if k not in g0]
        if missing:
            raise ValueError(f"the TL state lacks {missing}")
        states = self._slots(nslots)
        dev = {"u": g0["u_2"], "v": g0["v_2"], "w": g0["w_2"], "thp": g0["t_2"], "mup": g0["mu_2"],
               "p": g0["p"]}
        dx0 = self._tl_start({k: self._dev(to_window(v)) for k, v in dev.items()})
        dx0["php"] = self._dev(to_window(g0["ph_2"]))
        if self.moisture == "advect":
            dx0["qv"] = self._dev(to_window(g0["qv"]))
        for n, v in self.boundary_increment(g0).items():
            dx0[n] = self._dev(v)
        out = self.window.tl(dx0, save_at=states, partial=True)
        res = []
        for k in states:
            d = out[k]
            g = {tl: from_window(self._host(d[w])) for tl, w in _NAMES.items()}
            g["p"] = from_window(self._host(d["p_pert"]))
            g["qv"] = (from_window(self._host(d["qv"])) if self.moisture == "advect"
                       else np.array(g0["qv"], dtype=np.float64))
            res.append(g)
        return res

    def ad(self, forcings: Sequence) -> dict:
        forcings = list(forcings)
        states = self._slots(len(forcings))
        start = self.window._start
        forcing = {}
        qv = None
        for k, f in zip(states, forcings):
            dk = {}
            for n in self._inputs():
                dk[n] = self._dev(np.zeros(tuple(start[n].shape)))
            for tl, w in _NAMES.items():
                if tl in f:
                    dk[w] = self._dev(to_window(f[tl]))
            if "p" in f:
                dk["p_pert"] = self._dev(to_window(f["p"]))
            if self.moisture == "advect" and "qv" in f:
                dk["qv"] = self._dev(to_window(f["qv"]))
            forcing[k] = dk
            if self.moisture == "persist" and "qv" in f:
                qv = np.array(f["qv"], dtype=np.float64) if qv is None else qv + f["qv"]
        a = self.window.ad(None, forcing)
        a0 = self._tl_start_ad(a)
        # php is not one of wrfda_tl_start's fields (WRFDA starts g_ph at
        # zero; ph="hydrostatic" sets it here, outside that map), so a0 has
        # no php: its adjoint is the window's own, a["php"]
        out = {"u_2": a0["u"], "v_2": a0["v"], "w_2": a0["w"], "t_2": a0["thp"], "mu_2": a0["mup"],
               "p": a0["p"], "ph_2": a["php"]}
        if self.moisture == "advect":
            out["qv"] = a["qv"]   # not one of wrfda_tl_start's fields either
        out = {k: from_window(self._host(v)) for k, v in out.items()}
        if self.lbc == "wrfda":   # the tables' adjoint, through lbc_tl's
            base, consts, width, seconds = self._lbc_args()
            at = {n: np.asarray(self._host(a[n]), dtype=np.float64) for n in self.window.params}
            names = {"u": "u_2", "v": "v_2", "thp": "t_2", "php": "ph_2", "mup": "mu_2", "qv": "qv"}
            for w, v in lbc_ad(at, base, consts, self._lbc_fields, width, seconds).items():
                out[names[w]] = out[names[w]] + from_window(v)
        if qv is not None:
            out["qv"] = qv
        return out

