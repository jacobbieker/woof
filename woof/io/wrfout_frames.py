"""The wrfout frame walk: which files under a history tree are frames.

Split out of :mod:`woof.io.wrfout` (which re-exports every name here) so a
module that only LISTS frames -- the live Icechunk exporter
(:mod:`woof.icechunk_export`), whose options the preparation-side config
loaders read -- does not import the history writer.  The RW-WPS wheel
(tools/build_rw_wps_release.py) refuses ``woof/io/wrfout.py`` as an
executor module, and this walk needs none of it.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import stat

from woof import progress_log, render_layout


#: Directory names under a wrfout tree whose contents are ABOUT frames.
#:
#: :func:`woof.progress_log.write_frame_marker` publishes
#: ``<outdir>/ready/<frame>.json``, a readiness receipt that
#: deliberately repeats the history basename, so ``wrfout*`` matches it
#: and ``wrfout*.tmp*`` matches its temporary while that is in flight.
#: Reading one as a frame costs both halves at once.  The sweep below
#: would file a valid receipt under ``ready/.quarantine`` as an
#: "incomplete-wrfout", which is false twice over (the receipt is
#: complete, and so is the frame it names) and which destroys the one
#: positive signal ``write_frame_marker`` documents a consumer may
#: trust: a marker that exists names a frame that is complete and
#: readable, and a consumer reads absence as "not yet".  A render feed
#: on this walk would hand the renderer JSON, which is the "NetCDF:
#: Unknown file format" exit 2 that ``woof.go_cli.WRFOUT_GLOB``
#: records itself as existing to prevent.
NON_FRAME_DIRNAMES = (progress_log.FRAME_MARKER_DIRNAME,)

#: Suffixes a history frame never carries.  ``woof.runplan``'s frame
#: count spells this exclusion its own way for the same tree
#: (``path.suffix != ".json"``, under the comment "Readiness JSON
#: receipts repeat the history basename under ready/"), so it is spelled
#: here beside the directory rule rather than instead of it: the two
#: catch the receipt whatever directory a future writer puts it in, and
#: a receipt in flight whatever suffix its temporary carries.
NON_FRAME_SUFFIXES = (".json",)


def is_wrfout_frame(relative) -> bool:
    """Does ``relative`` name a history frame, or something ABOUT one?

    ``relative`` is a path relative to a wrfout root, and only the shape
    of the name is read, never the bytes.  This is the whole definition
    of "frame" for :func:`iter_wrfout_files`, in one place and settled
    before anything opens a file, because the alternative is to infer
    it, and the inference is wrong: a walk that hands every ``wrfout*``
    match to a reader and calls whatever fails to open "incomplete"
    quarantines a valid readiness receipt under that label.  Discovery
    says what IS a frame; opening it decides only whether that frame is
    finished.

    What is excluded, and the file that would otherwise be misread:

    * Anything under a DOT-PREFIXED directory.
      :func:`woof.supervisor.quarantine_file` files a swept frame at
      ``<parent>/.quarantine/wrfout_....incomplete-wrfout.<stamp>.
      <pid>``, a name that still begins with ``wrfout``, so without this
      the sweep re-sweeps its own quarantine on every run and nests
      ``.quarantine`` inside ``.quarantine`` without bound.  Dot-
      prefixed FILES are kept, which is why only the parents are read:
      an orphan temporary is ``.wrfout_....tmp.<pid>.<n>`` and is
      exactly what the sweep is looking for.
    * Anything under a directory named in :data:`NON_FRAME_DIRNAMES`.
    * Any name whose suffix is in :data:`NON_FRAME_SUFFIXES`.

    The dot-directory rule is the one
    :func:`woof.render_layout.iter_rendered` makes for the PNG tree,
    deliberately spelled the same way.  The receipt rule has no
    counterpart there because nothing writes a receipt beside a picture
    under the picture's own name.
    """

    relative = Path(relative)
    parents = relative.parts[:-1]
    if any(part.startswith(".") for part in parents):
        return False
    if any(part in NON_FRAME_DIRNAMES for part in parents):
        return False
    return relative.suffix not in NON_FRAME_SUFFIXES


def iter_wrfout_files(directory, pattern: str = "wrfout*", *,
                      include_temporaries: bool = True) -> list[Path]:
    """Every history file under ``directory`` matching ``pattern``.

    THE reader for a wrfout directory, and the reason there is only one:
    this module WRITES ``d05/episode-002/wrfout_...`` when a domain
    declares a lifecycle (see :meth:`PerDomainWrfoutWriters.submit`,
    which takes the segment from ``render_layout.episode_segment``), and
    a one-level ``glob`` cannot see a single frame of it.  A flat reader
    therefore reports "no frames" for a run that published a full tape:
    the sweep below leaves every incomplete episode frame in place, and
    a render feed on the same spelling hands the renderer nothing.

    ``pattern`` is a filename glob; :func:`is_wrfout_frame`
    says which of them are frames, from the name alone and before
    anything is opened.  A match is dropped when it sits under a
    dot-prefixed directory (this walk's own ``.quarantine``, which would
    otherwise be re-swept and re-nested on every run) and when it is a
    receipt that repeats a frame's basename rather than a frame: the
    readiness markers at ``ready/<frame>.json``.  That second rule is
    what makes this walk safe to feed a RENDERER as well as the sweep,
    which is the other half of why there is only one of it.

    The walk goes through :func:`woof.render_layout.fs_path` and the
    results come back in the CALLER's spelling, for the same reason the
    picture walk does: a root's own length says nothing about its
    descendants', and a frame that cannot be enumerated is a frame that
    cannot be swept.

    The explicitly selected root may itself be a junction. Descendant
    symlinks and Windows reparse points are never followed: an episode
    link to another run must not let this sweep quarantine that run's files.

    Sorted by path, so two directories walk in the same order.
    Render consumers set ``include_temporaries=False``; the recovery sweep
    keeps the default because finding orphan temporaries is its job.
    """

    directory = Path(directory)
    walk_root = Path(render_layout.fs_path(directory, descend=True))
    if not walk_root.is_dir():
        return []
    def redirected(info):
        return (stat.S_ISLNK(info.st_mode)
                or bool(getattr(info, "st_file_attributes", 0)
                        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))

    found, pending = [], [walk_root]
    while pending:
        folder = pending.pop()
        try:
            if folder != walk_root and redirected(folder.lstat()):
                continue
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if redirected(info):
                        continue
                    path = Path(entry.path)
                    if stat.S_ISDIR(info.st_mode):
                        if not entry.name.startswith(".") and entry.name not in NON_FRAME_DIRNAMES:
                            pending.append(path)
                    elif stat.S_ISREG(info.st_mode) and fnmatch.fnmatch(entry.name, pattern):
                        if not include_temporaries and ".tmp" in entry.name:
                            continue
                        relative = path.relative_to(walk_root)
                        if is_wrfout_frame(relative):
                            found.append(directory / relative)
        except (FileNotFoundError, NotADirectoryError):
            continue
    return sorted(found, key=lambda path: path.as_posix())
