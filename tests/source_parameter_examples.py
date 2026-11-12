"""Example values for the parameters a route needs the user to supply.

A route whose published object names carry a parameter with no default
(HAFS's ATCF storm id) cannot plan or fetch without one, so every test that
walks every source passes these through ``--source-parameters``.
"""
import re

_EXAMPLES = {"STORM": "09l"}


def source_parameters_value(source):
    """The ``KEY=VALUE,...`` a bare run of SOURCE needs, or None."""
    from woof import fetch_routes

    if source not in fetch_routes.route_ids():
        return None
    parts = []
    for key, row in (fetch_routes.route_for(source).parameters or {}).items():
        if row.get("default") is not None:
            continue
        example = _EXAMPLES[key]
        assert re.fullmatch(row["pattern"], example), (source, key, example)
        parts.append(f"{key}={example}")
    return ",".join(parts) or None


def source_parameter_args(source):
    """The CLI tokens for :func:`source_parameters_value`, possibly none."""
    value = source_parameters_value(source)
    return () if value is None else ("--source-parameters", value)
