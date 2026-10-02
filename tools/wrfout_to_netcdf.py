#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from netCDF4 import Dataset


def _is_wrfout(path: Path) -> bool:
    return path.is_file() and path.name.startswith("wrfout_d")


def _verify_netcdf(path: Path) -> dict[str, object]:
    with Dataset(path, "r") as ds:
        dimensions = {name: len(dim) for name, dim in ds.dimensions.items()}
        variables = sorted(ds.variables.keys())
        return {
            "path": str(path),
            "dimensions": dimensions,
            "variables": variables,
            "format": getattr(ds, "file_format", "unknown"),
        }


def _target_name(source_name: str) -> str:
    if source_name.endswith(".nc"):
        return source_name
    return f"{source_name}.nc"


def convert_wrfout_directory(
    source_dir: Path,
    target_dir: Path,
    copy_files: bool,
) -> dict[str, object]:
    wrfouts = sorted(path for path in source_dir.iterdir() if _is_wrfout(path))
    if not wrfouts:
        raise FileNotFoundError(f"No wrfout files found in {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []

    for src in wrfouts:
        metadata = _verify_netcdf(src)
        dst = target_dir / _target_name(src.name)
        if dst.exists():
            dst.unlink()
        if copy_files:
            shutil.copy2(src, dst)
        else:
            os.link(src, dst)
        metadata["exported_path"] = str(dst)
        files.append(metadata)

    return {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "mode": "copy" if copy_files else "hardlink",
        "file_count": len(files),
        "files": files,
    }


def export_to_icechunk(
    source_dir: Path,
    repo_dir: Path,
    branch: str,
    message: str,
    compress_zstd_bitshuffle: bool,
    zstd_clevel: int,
) -> dict[str, object]:
    try:
        import xarray as xr
        import icechunk as ic
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Icechunk export requires xarray and icechunk in the active environment."
        ) from error
    if compress_zstd_bitshuffle:
        try:
            import numpy as np
            from zarr.codecs import BloscCodec, BloscShuffle
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Zstd+bitshuffle compression requires numpy and zarr codecs."
            ) from error

    wrfouts = sorted(path for path in source_dir.iterdir() if _is_wrfout(path))
    if not wrfouts:
        raise FileNotFoundError(f"No wrfout files found in {source_dir}")

    repo_dir.mkdir(parents=True, exist_ok=True)
    repo = ic.Repository.open_or_create(
        storage=ic.local_filesystem_storage(str(repo_dir))
    )
    paths = [str(path) for path in wrfouts]
    datasets = [xr.open_dataset(path, engine="netcdf4") for path in paths]
    dataset = xr.concat(datasets, dim="Time", data_vars="all", coords="minimal")
    session = repo.writable_session(branch)
    try:
        encoding: dict[str, dict[str, object]] = {}
        if compress_zstd_bitshuffle:
            compressor = BloscCodec(
                cname="zstd",
                clevel=int(zstd_clevel),
                shuffle=BloscShuffle.bitshuffle,
            )
            for name, variable in dataset.variables.items():
                dtype = getattr(variable, "dtype", None)
                if dtype is None:
                    continue
                if np.issubdtype(dtype, np.number):
                    encoding[name] = {"compressors": [compressor]}
        dataset.to_zarr(
            session.store,
            mode="w",
            consolidated=True,
            zarr_format=3,
            encoding=encoding or None,
        )
        snapshot = session.commit(message)
    finally:
        dataset.close()
        for item in datasets:
            item.close()

    return {
        "repo_dir": str(repo_dir),
        "branch": branch,
        "snapshot": str(snapshot),
        "compression": (
            {"codec": "blosc", "cname": "zstd", "shuffle": "bitshuffle", "clevel": int(zstd_clevel)}
            if compress_zstd_bitshuffle
            else None
        ),
        "source_file_count": len(paths),
        "source_files": paths,
    }


def stream_wrfout_to_icechunk(
    source_dir: Path,
    repo_dir: Path,
    branch: str,
    message_prefix: str,
    compress_zstd_bitshuffle: bool,
    zstd_clevel: int,
    delete_after_icechunk: bool,
    poll_seconds: float,
    max_idle_seconds: float | None,
    state_file: Path | None,
) -> dict[str, object]:
    try:
        import numpy as np
        import xarray as xr
        import icechunk as ic
        from zarr.codecs import BloscCodec, BloscShuffle
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Streaming Icechunk export requires xarray, icechunk, numpy, and zarr codecs."
        ) from error

    source_dir = source_dir.resolve()
    repo_dir = repo_dir.resolve()
    repo_dir.mkdir(parents=True, exist_ok=True)
    repo = ic.Repository.open_or_create(storage=ic.local_filesystem_storage(str(repo_dir)))

    if state_file is None:
        state_file = source_dir / ".icechunk-export-state.json"
    state_file = state_file.resolve()

    processed: set[str] = set()
    skipped: set[str] = set()
    initialized = False
    reference_variables: tuple[str, ...] = ()
    if state_file.exists():
        try:
            saved = json.loads(state_file.read_text(encoding="utf-8"))
            processed = set(saved.get("processed", []))
            skipped = set(saved.get("skipped", []))
            initialized = bool(saved.get("initialized", False))
            reference_variables = tuple(saved.get("reference_variables", ()))
        except (json.JSONDecodeError, OSError, TypeError):
            processed = set()
            skipped = set()
            initialized = False
            reference_variables = ()

    compressor = None
    if compress_zstd_bitshuffle:
        compressor = BloscCodec(
            cname="zstd",
            clevel=int(zstd_clevel),
            shuffle=BloscShuffle.bitshuffle,
        )

    exported = 0
    deleted = 0
    last_snapshot = None
    idle_started = time.monotonic()
    reference_set = set(reference_variables)

    while True:
        wrfouts = sorted(path for path in source_dir.iterdir() if _is_wrfout(path))
        pending = [path for path in wrfouts if path.name not in processed and path.name not in skipped]
        if pending:
            idle_started = time.monotonic()
        if not initialized and pending:
            if len(pending) >= 2:
                first, second = pending[0], pending[1]
                with xr.open_dataset(first, engine="netcdf4") as ds_first, xr.open_dataset(
                    second, engine="netcdf4"
                ) as ds_second:
                    first_vars = set(ds_first.variables.keys())
                    second_vars = set(ds_second.variables.keys())
                missing_from_first = sorted(second_vars - first_vars)
                if missing_from_first:
                    skipped.add(first.name)
                    print(
                        f"Skipping initial frame {first.name}: missing {len(missing_from_first)} "
                        "variable(s) present in the next timestep."
                    )
                    pending = [path for path in pending if path.name != first.name]
            if pending:
                seed = pending[0]
                with xr.open_dataset(seed, engine="netcdf4") as dataset:
                    session = repo.writable_session(branch)
                    encoding = None
                    if compressor is not None:
                        prepared: dict[str, dict[str, object]] = {}
                        for name, variable in dataset.variables.items():
                            dtype = getattr(variable, "dtype", None)
                            if dtype is None:
                                continue
                            if np.issubdtype(dtype, np.number):
                                prepared[name] = {"compressors": [compressor]}
                        encoding = prepared or None
                    dataset.to_zarr(
                        session.store,
                        mode="w",
                        consolidated=True,
                        zarr_format=3,
                        encoding=encoding,
                    )
                    last_snapshot = str(session.commit(f"{message_prefix}: {seed.name}"))
                    reference_variables = tuple(dataset.variables.keys())
                    reference_set = set(reference_variables)
                processed.add(seed.name)
                exported += 1
                initialized = True
                if delete_after_icechunk:
                    seed.unlink(missing_ok=True)
                    deleted += 1
                print(
                    f"Seeded Icechunk store with {seed.name} at snapshot {last_snapshot}"
                    + (" and deleted source file." if delete_after_icechunk else ".")
                )
                state_file.parent.mkdir(parents=True, exist_ok=True)
                state_file.write_text(
                    json.dumps(
                        {
                            "source_dir": str(source_dir),
                            "repo_dir": str(repo_dir),
                            "initialized": initialized,
                            "reference_variables": list(reference_variables),
                            "processed": sorted(processed),
                            "skipped": sorted(skipped),
                            "last_snapshot": last_snapshot,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                pending = [path for path in pending if path.name != seed.name]
        for wrfout in pending:
            with xr.open_dataset(wrfout, engine="netcdf4") as dataset:
                dataset_vars = set(dataset.variables.keys())
                missing = sorted(reference_set - dataset_vars)
                if missing:
                    skipped.add(wrfout.name)
                    print(
                        f"Skipping {wrfout.name}: missing {len(missing)} reference variable(s)."
                    )
                    continue
                if dataset_vars - reference_set:
                    dataset = dataset[list(reference_variables)]
                session = repo.writable_session(branch)
                dataset.to_zarr(
                    session.store,
                    mode="a",
                    append_dim="Time",
                    consolidated=True,
                    zarr_format=3,
                )
                last_snapshot = str(session.commit(f"{message_prefix}: {wrfout.name}"))
            processed.add(wrfout.name)
            exported += 1
            if delete_after_icechunk:
                wrfout.unlink(missing_ok=True)
                deleted += 1
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(
                json.dumps(
                    {
                        "source_dir": str(source_dir),
                        "repo_dir": str(repo_dir),
                        "initialized": initialized,
                        "reference_variables": list(reference_variables),
                        "processed": sorted(processed),
                        "skipped": sorted(skipped),
                        "last_snapshot": last_snapshot,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"Streamed {wrfout.name} to Icechunk snapshot {last_snapshot}"
                + (" and deleted source file." if delete_after_icechunk else ".")
            )

        if max_idle_seconds is not None:
            if time.monotonic() - idle_started >= max_idle_seconds:
                break
        time.sleep(max(0.1, float(poll_seconds)))

    return {
        "source_dir": str(source_dir),
        "repo_dir": str(repo_dir),
        "branch": branch,
        "state_file": str(state_file),
        "exported_files": exported,
        "deleted_files": deleted,
        "skipped_files": len(skipped),
        "reference_variable_count": len(reference_variables),
        "last_snapshot": last_snapshot,
        "compression": (
            {"codec": "blosc", "cname": "zstd", "shuffle": "bitshuffle", "clevel": int(zstd_clevel)}
            if compress_zstd_bitshuffle
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export wrfout files as explicit .nc NetCDF files."
    )
    parser.add_argument("source_dir", type=Path, help="Directory containing wrfout_d* files")
    parser.add_argument("target_dir", type=Path, help="Directory to write .nc outputs")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating hardlinks",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write a JSON manifest",
    )
    parser.add_argument(
        "--icechunk-repo",
        type=Path,
        default=None,
        help="Optional local Icechunk repository directory for Zarr export",
    )
    parser.add_argument(
        "--icechunk-branch",
        type=str,
        default="main",
        help="Icechunk branch name (default: main)",
    )
    parser.add_argument(
        "--icechunk-message",
        type=str,
        default="Import WRF history",
        help="Icechunk commit message",
    )
    parser.add_argument(
        "--compress-zstd-bitshuffle",
        action="store_true",
        help="When exporting to Icechunk, compress numeric arrays with Blosc(zstd+bitshuffle)",
    )
    parser.add_argument(
        "--zstd-clevel",
        type=int,
        default=5,
        help="Zstd compression level for --compress-zstd-bitshuffle (default: 5)",
    )
    parser.add_argument(
        "--watch-icechunk",
        action="store_true",
        help="Continuously watch wrfout files and append each one to Icechunk as it lands",
    )
    parser.add_argument(
        "--delete-after-icechunk",
        action="store_true",
        help="Delete each wrfout file after it is successfully committed to Icechunk (watch mode)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Watch mode polling interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--max-idle-seconds",
        type=float,
        default=None,
        help="Watch mode stops after this many seconds with no new wrfout files (default: never)",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Optional watch-mode JSON state file for processed wrfout tracking",
    )
    args = parser.parse_args()

    if args.watch_icechunk and args.icechunk_repo is None:
        raise SystemExit("--watch-icechunk requires --icechunk-repo")

    if args.watch_icechunk:
        summary = stream_wrfout_to_icechunk(
            source_dir=args.source_dir.resolve(),
            repo_dir=args.icechunk_repo.resolve(),
            branch=args.icechunk_branch,
            message_prefix=args.icechunk_message,
            compress_zstd_bitshuffle=bool(args.compress_zstd_bitshuffle),
            zstd_clevel=int(args.zstd_clevel),
            delete_after_icechunk=bool(args.delete_after_icechunk),
            poll_seconds=float(args.poll_seconds),
            max_idle_seconds=(
                None if args.max_idle_seconds is None else float(args.max_idle_seconds)
            ),
            state_file=(None if args.state_file is None else args.state_file.resolve()),
        )
        if args.manifest is not None:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"Watch export complete. Streamed {summary['exported_files']} file(s) to "
            f"{summary['repo_dir']}."
        )
        return 0

    summary = convert_wrfout_directory(
        source_dir=args.source_dir.resolve(),
        target_dir=args.target_dir.resolve(),
        copy_files=args.copy,
    )
    if args.icechunk_repo is not None:
        summary["icechunk"] = export_to_icechunk(
            source_dir=args.source_dir.resolve(),
            repo_dir=args.icechunk_repo.resolve(),
            branch=args.icechunk_branch,
            message=args.icechunk_message,
            compress_zstd_bitshuffle=bool(args.compress_zstd_bitshuffle),
            zstd_clevel=int(args.zstd_clevel),
        )

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"Exported {summary['file_count']} file(s) from {summary['source_dir']} to "
        f"{summary['target_dir']} ({summary['mode']})."
    )
    if "icechunk" in summary:
        icechunk = summary["icechunk"]
        print(
            "Committed Icechunk snapshot "
            f"{icechunk['snapshot']} on branch {icechunk['branch']} in {icechunk['repo_dir']}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
