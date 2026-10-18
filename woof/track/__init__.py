"""3-D paths (time, lat, lon, height) for track-following simulations.

``load_track(path, mapping)`` reads one CSV, JSON or Parquet file whose
columns :class:`ColumnMapping` names, cleans it, and returns a
:class:`Track3D`: the fixes with their launch, landing, duration, peak,
height histogram, envelope and suggested boundary source.  See
:mod:`woof.track.ingest` for the readers and their quality control and
:mod:`woof.track.model` for the track model.
"""

from __future__ import annotations

from woof.track.ingest import ColumnMapping, load_track
from woof.track.model import Track3D, TrackEvent, boundary_source

__all__ = ["ColumnMapping", "Track3D", "TrackEvent", "boundary_source",
           "load_track"]
