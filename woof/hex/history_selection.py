"""Which variables a ``woof hex forecast`` history frame publishes.

``--history-vars a,b,c`` names them; ``--history-preset energy`` names a
row of :data:`HISTORY_PRESETS`.  Neither given is the full set, unchanged:
every array the frame carries, as before either option existed.  The two
are exclusive -- a preset plus a list is two answers to one question, and
the door does not guess which one the user meant.

The mesh coordinates (``indexToCellID``, ``latCell``, ``lonCell``, ``ter``,
and their edge counterparts) are written in every frame whatever is
selected, because a frame without them cannot be placed on a mesh; so is
``xtime``.  ``zgrid`` is selectable because it is the one large static.

A name the frame cannot carry is treated by how it was asked for:

* explicitly (``--history-vars``): a name no frame of the run can carry
  (a typo, a WRF or native-MPAS spelling) is refused before the run is
  built; an optional field the run turns out not to produce is refused at
  its first frame, before any step.  Both name the absent variables and
  the ones a frame does carry.  A list
  that silently lost a member would be found missing only by whoever
  reads the history later.
* by a preset: the frame publishes what it carries and the receipt names
  the rest.  A preset is written for every physics row and radiation
  scheme -- ``swddni``/``swddif`` exist only where the shortwave computes a
  direct beam, a two-moment row carries no ``qs``/``qg`` -- so absence is a
  property of the run, recorded rather than refused.

This mirrors the WRF side's ``energy`` preset (woof/energy, WRF history
names) in MPAS names: the fields the energy sampler
(``woof.energy.sample_hex``) reads, plus the thermodynamic profile.
"""

from __future__ import annotations

import re
from typing import Any, Collection, Mapping, Sequence

class HistorySelectionRefusal(ValueError):
    """A history selection names something the run cannot publish.

    A ``ValueError`` and deliberately NOT a ``woof.hex.errors.MpasPortError``
    (a ``RuntimeError``): the forecast loop reads a ``RuntimeError`` from a
    frame capture as the port refusing the model state, and a user's
    selection is a configuration mistake, not a numerical refusal.
    """


#: Written in every frame regardless of selection.
ALWAYS_WRITTEN = (
    "indexToCellID", "latCell", "lonCell", "ter",
    "indexToEdgeID", "latEdge", "lonEdge", "xtime",
)

#: Every frame array whose name does not depend on the physics row: the
#: dynamics fields the capture prepares, the adapter's surface/soil export
#: (required and optional), the GWDO diagnostics and refl10cm.  A run's
#: frame can carry only these plus its row's scalar names and precipitation
#: buckets (and ``zgrid``), so a ``--history-vars`` name outside that union
#: is refused before anything is built (:func:`precheck_names`).  Whether an
#: OPTIONAL name (``swddni``...) is produced is known only at the first
#: frame, where :func:`plan_frame` decides.
FRAME_NAMES = (
    "u_zonal", "v_meridional", "normal_u", "rho", "theta", "pressure",
    "surface_pressure", "w", "refl10cm",
    "tsk", "smois", "tslb", "hfx", "qfx", "lh",
    "t2", "q2", "pblh", "u10", "v10", "psfc",
    "swdown", "glw", "olr", "swddni", "swddif", "coszr",
    "rainc",
    "dusfcg", "dvsfcg", "dtaux3d", "dtauy3d", "rubldiff", "rvbldiff",
)

#: Named history selections.  ``full`` is the default and selects every
#: array a frame carries.
HISTORY_PRESETS: Mapping[str, tuple[str, ...] | None] = {
    "full": None,
    "energy": (
        "u_zonal", "v_meridional", "w", "theta", "pressure", "rho",
        "qv", "qc", "qr", "qi", "qs", "qg",
        "t2", "q2", "u10", "v10", "surface_pressure",
        "swdown", "swddni", "swddif", "coszr",
        "rainnc", "rainc",
        "zgrid", "xtime",
    ),
}

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def parse_history_vars(text: str) -> tuple[str, ...]:
    """``"a,b,c"`` -> ``("a", "b", "c")``; refuses blanks and repeats."""

    names = [part.strip() for part in str(text).split(",")]
    if not names or any(not name for name in names):
        raise HistorySelectionRefusal(
            f"--history-vars {text!r} has an empty name; give a comma-"
            "separated list such as u_zonal,v_meridional,t2"
        )
    bad = [name for name in names if not _NAME.match(name)]
    if bad:
        raise HistorySelectionRefusal(
            f"--history-vars names {bad} are not variable names (letters, "
            "digits and underscores, starting with a letter)"
        )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise HistorySelectionRefusal(
            f"--history-vars repeats {repeated}; name each variable once"
        )
    return tuple(names)


def resolve_history_selection(
    history_vars: str | None, history_preset: str | None
) -> dict[str, Any]:
    """The run's selection as a receipt-ready mapping.

    ``{"source": "default"|"preset"|"explicit", "preset": name|None,
    "variables": tuple|None, "strict": bool}``; ``variables`` None is the
    full set.
    """

    if history_vars is not None and history_preset is not None:
        raise HistorySelectionRefusal(
            "--history-vars and --history-preset are exclusive: a preset is "
            "a named list, so giving both is two answers to one question. "
            "Give the preset, or the list (spell the preset's members out "
            "and add to them)"
        )
    if history_vars is not None:
        return {
            "source": "explicit",
            "preset": None,
            "variables": parse_history_vars(history_vars),
            "strict": True,
        }
    if history_preset is not None:
        if history_preset not in HISTORY_PRESETS:
            raise HistorySelectionRefusal(
                f"--history-preset {history_preset!r} is not a history preset; "
                f"the presets are {sorted(HISTORY_PRESETS)}"
            )
        return {
            "source": "preset",
            "preset": history_preset,
            "variables": HISTORY_PRESETS[history_preset],
            "strict": False,
        }
    return {"source": "default", "preset": "full", "variables": None,
            "strict": False}


def selection_argv(selection: Mapping[str, Any]) -> list[str]:
    """The driver arguments that carry ``selection`` (empty for the default)."""

    if selection["source"] == "explicit":
        return ["--history-vars", ",".join(selection["variables"])]
    if selection["source"] == "preset" and selection["preset"] != "full":
        return ["--history-preset", str(selection["preset"])]
    return []


def wants_zgrid(selection: Mapping[str, Any]) -> bool:
    """Whether frames under ``selection`` write ``zgrid``."""

    requested = selection.get("variables")
    return requested is None or "zgrid" in requested


def precheck_names(
    selection: Mapping[str, Any], possible: Collection[str]
) -> None:
    """Refuse, before the run is built, an explicit name no frame can carry.

    ``possible`` is the run's own universe: :data:`FRAME_NAMES` plus its
    row's scalar names and precipitation buckets.  Presets are not checked
    (their members are recorded as absent at the frame instead).
    """

    if not selection.get("strict"):
        return
    universe = set(possible) | set(ALWAYS_WRITTEN) | {"zgrid"}
    unknown = [name for name in selection["variables"] if name not in universe]
    if unknown:
        raise HistorySelectionRefusal(
            f"--history-vars names {unknown}, which no history frame of this "
            f"run can carry.  The names a frame can carry are "
            f"{sorted(universe)} (MPAS names: t2 not t2m, u_zonal not "
            "uReconstructZonal)"
        )


def plan_frame(
    selection: Mapping[str, Any],
    carried: Collection[str],
    *,
    zgrid_available: bool,
) -> dict[str, Any]:
    """What one frame writes, given the arrays it ``carried``.

    Returns ``{"arrays": tuple|None, "zgrid": bool, "missing": tuple}``;
    ``arrays`` None writes every carried array.  Refuses (strict selection
    only) when a requested name is neither carried, ``zgrid`` (when the run
    holds one) nor always written.
    """

    carried = set(carried)
    requested: Sequence[str] | None = selection.get("variables")
    if requested is None:
        return {"arrays": None, "zgrid": bool(zgrid_available), "missing": ()}
    supplied = carried | set(ALWAYS_WRITTEN) | ({"zgrid"} if zgrid_available else set())
    missing = tuple(name for name in requested if name not in supplied)
    if missing and selection.get("strict"):
        raise HistorySelectionRefusal(
            f"--history-vars asks for {list(missing)}, which this run's "
            "history frame does not carry.  The frame carries "
            f"{sorted(supplied)}.  Drop the absent names, or select a "
            "physics/radiation configuration that produces them"
        )
    return {
        "arrays": tuple(name for name in requested if name in carried),
        "zgrid": "zgrid" in requested and bool(zgrid_available),
        "missing": missing,
    }


__all__ = [
    "ALWAYS_WRITTEN",
    "FRAME_NAMES",
    "HistorySelectionRefusal",
    "HISTORY_PRESETS",
    "parse_history_vars",
    "plan_frame",
    "precheck_names",
    "resolve_history_selection",
    "selection_argv",
    "wants_zgrid",
]
