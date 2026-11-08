"""GPU spectral nudging and prepared regional forecast integration."""
from .config import NudgingConfig, NudgingOptions


def build_nudger(config, bindings, **kwargs):
    """Off reads no fields, imports no CUDA code and allocates no buffers."""
    if not config.enabled:
        return None
    from .gpu import SpectralNudger
    return SpectralNudger(config, bindings, **kwargs)


def nudging_owner(state):
    """The SpectralNudger a prepared route bound to ``state``, or None.

    Kept here rather than in woof/core/model.py, whose clock audit
    (tests/test_clock.py) bans getattr."""
    return getattr(state, "_spectral_nudging_owner", None)


def surface_fdda_owner(state):
    """The surface FDDA owner a prepared route bound to ``state``, or None."""
    return getattr(state, "_surface_fdda_owner", None)


def px_inputs_owner(state):
    """The PX inputs owner a prepared route bound to ``state``, or None."""
    return getattr(state, "_px_inputs_owner", None)


def require_bound_nudging(model, experiment) -> None:
    """Refuse a [nudging] domain that no prepared route bound an owner to."""
    nudging = getattr(experiment, "nudging", None)
    if nudging is None:
        return
    for node in model.walk_parent_first():
        if (nudging.for_domain(node.cfg.grid_id) is not None
                and nudging_owner(node.state) is None
                and surface_fdda_owner(node.state) is None):
            raise ValueError("NUDGING_ANALYSES_UNBOUND: this domain has no prepared analysis forcing; use the prepared forecast route")


__all__ = ["NudgingConfig", "NudgingOptions", "build_nudger",
           "nudging_owner", "px_inputs_owner", "require_bound_nudging",
           "surface_fdda_owner"]
