"""A cycle a table route can plan and some host of it holds.

Shared by the walks over every fetch route.  They used to author every
route on one fixed cycle (2026-08-17 or 2026-09-09), which stopped being
true of the table as it grew: a historical archive row (GEFS v11) plans
only inside its publication era, a new product (ECMWF's 0.1-degree open
data) begins after the fixed cycle, and a rolling host's retention plus
an archive mirror's first run (ICON's source.coop copy, from
2026-08-25 12Z) leave a fixed cycle held by nobody once the wall clock
passes the rolling door's retention.  Each walk met the route's own,
correct refusal and read it as a failure of the door under test.

Decided from the route row's own declarations (publication eras, host
``serves_from`` bounds, ``first_lead`` and lead ladder), never from a
source name, and never from the wall clock: the cycle it returns is the
same on every day.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from woof import fetch_routes


def _route(source):
    if isinstance(source, fetch_routes.Route):
        return source
    try:
        return fetch_routes.table_route(source)
    except (KeyError, ValueError):
        return None


def _in_era(route, cycle: datetime) -> bool:
    if not route.publication_eras:
        return True
    try:
        era = fetch_routes.publication_era(route, cycle)
    except ValueError:
        return False
    return era is not None and not era.prep_refusal


def route_cycle(source, wanted: datetime) -> datetime:
    """``wanted``, or the nearest cycle at its hour the route can plan.

    1. Inside a publication era that can be prepared, else the route's
       planning cycle at the same hour (its newest preparable layout).
    2. Held by a host without asking the clock: a cycle before a mirror's
       first run (``serves_from``) is held only by rolling doors whose
       retention the wall clock ages it out of, so it moves to the first
       run the archive mirror holds, at the same hour.
    """

    route = _route(source)
    if route is None:
        return wanted
    cycle = wanted
    if not _in_era(route, cycle):
        planning = fetch_routes.planning_cycles(route)
        same_hour = [each for each in planning if each.hour == cycle.hour]
        cycle = (same_hour or list(planning))[0]
    bounds = [host.serves_from for host in route.hosts
              if host.archive and host.serves_from is not None]
    unbounded_archive = any(host.archive and host.serves_from is None
                            for host in route.hosts)
    if bounds and not unbounded_archive and cycle < min(bounds):
        first = min(bounds)
        moved = first.replace(hour=cycle.hour, minute=0, second=0, microsecond=0)
        if moved < first:
            moved += timedelta(days=1)
        cycle = moved
    return cycle


def route_cycle_text(source, wanted: str, fmt: str = "%Y-%m-%dT%H") -> str:
    """:func:`route_cycle` for a cycle spelled as text, in the same spelling."""

    return route_cycle(source, datetime.strptime(wanted, fmt)).strftime(fmt)


def first_window(route, hours: int) -> tuple[int, int]:
    """``(start_hour, hours)`` of the route's first window up to ``hours`` long.

    A route whose ladder does not publish f000 (the GEFS reforecast starts
    at f003) begins at its first lead; an analysis-only donor (one lead)
    is a window of that one frame.
    """

    start = route.first_lead
    ladders = [steps for _hours, steps in route.ladders]
    last = max(steps[-1][0] for steps in ladders)
    return start, max(0, min(int(hours), last - start))
