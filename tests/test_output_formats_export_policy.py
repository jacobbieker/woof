"""The fork's live-export policy on upstream's store export.

``woof go --icechunk URL`` and ``[output] icechunk = {...}`` are retired:
upstream's ``--icechunk-store`` / ``[output] icechunk_store`` (with
``formats``) name the store, and the policy upstream had no word for rides
``[output] icechunk_delete_after_commit`` / ``icechunk_domains``
(:class:`woof.output_formats.ExportPolicy`): each committed frame is
deleted, the history writers wait for disk space while that runs
(woof.io.export_backpressure) and the disk review prices the backlog.
"""
from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from woof import output_formats as formats
from woof.io.export_backpressure import LIVE_EXPORT_DELETE_ENV


def test_policy_defaults_and_validation():
    assert formats.export_policy_from_table({}) == formats.ExportPolicy()
    policy = formats.export_policy_from_table(
        {"icechunk_delete_after_commit": True, "icechunk_domains": [3, 1]})
    assert policy.delete_after_commit and policy.domains == (1, 3)
    assert policy.exports(3) and not policy.exports(2) and policy.exports(None)
    assert json.loads(policy.env_value()) == {"domains": [1, 3]}
    for bad in ({"icechunk_delete_after_commit": "yes"},
                {"icechunk_domains": []}, {"icechunk_domains": [1, 1]},
                {"icechunk_domains": [0]}):
        with pytest.raises(ValueError):
            formats.export_policy_from_table(bad)


def test_the_output_table_takes_the_policy_and_a_domain_table_refuses_it():
    from woof.io.history_selection import HistorySelection

    HistorySelection.from_mapping({"formats": ["netcdf", "icechunk"],
                                   "icechunk_store": "s3://b/p",
                                   "icechunk_delete_after_commit": True,
                                   "icechunk_domains": [2]})
    with pytest.raises(ValueError, match="tree-wide"):
        HistorySelection.from_mapping({"icechunk_domains": [2]},
                                      tree_wide=False)
    with pytest.raises(ValueError, match="unknown key"):
        HistorySelection.from_mapping({"icechunk": "s3://b/p"})


def test_the_flag_turns_deletion_on_and_doors_forward_it(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text('[experiment]\nname = "alpha"\n[output]\n'
                      'formats = ["icechunk"]\nicechunk_store = "s3://b/{name}"\n'
                      'icechunk_domains = [2]\n')
    args = Namespace(config=config, output_format=None, icechunk_store=None,
                     output_config=None, icechunk_delete_after_commit=True)
    policy = formats.export_policy(args)
    assert policy.delete_after_commit and policy.domains == (2,)
    rows, store = formats.options(args)
    assert store == "s3://b/alpha"
    assert "--icechunk-delete-after-commit" in formats.argument_values(args)


def _publish(root: Path, name: str) -> Path:
    path = root / "run" / "wrfout" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"frame")
    with (path.parent / ".history-written").open("a") as journal:
        journal.write(name + "\n")
    return path


def test_delete_after_commit_frees_committed_frames_of_the_exported_domains(
        tmp_path, monkeypatch):
    monkeypatch.setattr(formats, "require_binary", lambda _: Path("/unused"))
    calls = []
    monkeypatch.setattr(formats.HistoryOutputs, "_invoke",
                        lambda self, row, binary, store, inputs, mode:
                        calls.append((store, [str(p) for p in inputs], mode)))
    monkeypatch.delenv(LIVE_EXPORT_DELETE_ENV, raising=False)
    row, = formats.selected("icechunk")
    policy = formats.ExportPolicy(delete_after_commit=True, domains=(2,))
    d01 = _publish(tmp_path, "wrfout_d01_2026-01-01_00:00:00")
    d02 = _publish(tmp_path, "wrfout_d02_2026-01-01_00:00:00")
    with formats.HistoryOutputs(tmp_path, [row], "s3://b/p",
                                policy=policy) as scope:
        # The history writers wait for space while deletion runs.
        assert json.loads(os.environ[LIVE_EXPORT_DELETE_ENV]) == {"domains": [2]}
    assert LIVE_EXPORT_DELETE_ENV not in os.environ
    appended = [c for c in calls if c[2] == "append"]
    assert [c[1] for c in appended] == [[str(d02)]]
    assert d01.exists() and not d02.exists()
    assert scope.deleted == 1
    receipt = json.loads((tmp_path / "output-formats" / "receipt.json").read_text())
    assert receipt["policy"] == {"delete_after_commit": True, "domains": [2],
                                 "frames_deleted": 1}


def test_without_the_policy_every_frame_stays(tmp_path, monkeypatch):
    monkeypatch.setattr(formats, "require_binary", lambda _: Path("/unused"))
    monkeypatch.setattr(formats.HistoryOutputs, "_invoke",
                        lambda *args: None)
    row, = formats.selected("icechunk")
    frame = _publish(tmp_path, "wrfout_d01_2026-01-01_00:00:00")
    with formats.HistoryOutputs(tmp_path, [row], None):
        assert LIVE_EXPORT_DELETE_ENV not in os.environ
    assert frame.exists()


def test_go_refuses_deletion_beside_its_own_frame_readers(tmp_path):
    from woof.go_cli import GoRefusal, _refuse_render_with_deleted_frames, icechunk_request

    config = tmp_path / "run.toml"
    config.write_text("[output]\nicechunk_delete_after_commit = true\n"
                      "[simulated_radar]\nenabled = true\n")
    with pytest.raises(GoRefusal, match="simulated_radar"):
        icechunk_request(Namespace(), config)
    config.write_text("[output]\nicechunk_delete_after_commit = true\n")
    policy = icechunk_request(Namespace(), config)
    with pytest.raises(GoRefusal, match="render stage"):
        _refuse_render_with_deleted_frames(policy, "t2m")
    _refuse_render_with_deleted_frames(policy, "none")
    config.write_text("[output]\nformats = [\"icechunk\"]\n")
    assert icechunk_request(Namespace(), config) is None


def test_go_has_upstreams_flag_and_not_the_forks():
    from woof.cli import build_parser

    parsed = build_parser().parse_args(
        ["go", "x.toml", "--icechunk-store", "s3://b/p",
         "--icechunk-delete-after-commit"])
    assert parsed.icechunk_store == "s3://b/p"
    assert parsed.icechunk_delete_after_commit is True
    assert not hasattr(parsed, "icechunk")


def test_a_track_plan_asks_upstreams_store_with_the_policy():
    from woof.track import plan as tp
    import inspect

    source = inspect.getsource(tp)
    assert '"icechunk_store": icechunk_url' in source
    assert '"icechunk_delete_after_commit": True' in source
