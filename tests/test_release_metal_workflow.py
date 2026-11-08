"""release-metal.yml: every merge into main becomes a metal-woof build.

The version step's own shell script is run against a scratch repository, so
the versions it plans from the tags are the ones the workflow would tag.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/release-metal.yml"

pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None,
                                reason="needs git and bash")


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps() -> list[dict]:
    return _workflow()["jobs"]["release"]["steps"]


def _version_script() -> str:
    (step,) = [s for s in _steps() if s.get("id") == "version"]
    return step["run"]


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    return subprocess.run(["git", *args], cwd=repo, env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, version: str, message: str) -> str:
    (repo / "pyproject.toml").write_text(f'[project]\nname = "recast-woof"\nversion = "{version}"\n')
    (repo / "f").write_text(message)
    _git(repo, "add", "pyproject.toml", "f")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _scratch(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "remote", "add", "origin", str(origin))
    return repo


def _push(repo: Path) -> None:
    _git(repo, "push", "-q", "origin", "HEAD:refs/heads/main", "--tags")


def _plan(repo: Path, tmp_path: Path) -> tuple[dict[str, str], list[list[str]]]:
    out = tmp_path / "github_output"
    out.write_text("")
    runner = tmp_path / "runner"
    runner.mkdir(exist_ok=True)
    subprocess.run(["bash", "-c", _version_script()], cwd=repo, check=True, capture_output=True,
                   text=True, env={**os.environ, "GITHUB_OUTPUT": str(out),
                                   "RUNNER_TEMP": str(runner)})
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)
    plan = Path(outputs["plan"])
    rows = [line.split(" ") for line in plan.read_text().splitlines()] if plan.exists() else []
    return outputs, rows


def test_first_build_after_an_upstream_release_is_dot_one(tmp_path):
    repo = _scratch(tmp_path)
    _commit(repo, "1.1.5", "old")
    _git(repo, "tag", "metal-v1.1.5.2")
    head = _commit(repo, "1.1.6", "sync 1.1.6")
    _push(repo)
    outputs, rows = _plan(repo, tmp_path)
    assert outputs["skip"] == "false"
    assert rows == [[head, "1.1.6.1", "metal-v1.1.5.2"]]


def test_merges_a_dropped_run_missed_are_released_in_order(tmp_path):
    repo = _scratch(tmp_path)
    _commit(repo, "1.1.6", "a")
    _git(repo, "tag", "metal-v1.1.6.9")
    b = _commit(repo, "1.1.6", "b")
    c = _commit(repo, "1.1.6", "c [skip release]")
    d = _commit(repo, "1.1.6", "d")
    _push(repo)
    _, rows = _plan(repo, tmp_path)
    assert c not in [r[0] for r in rows]
    assert rows == [[b, "1.1.6.10", "metal-v1.1.6.9"], [d, "1.1.6.11", "metal-v1.1.6.10"]]


def test_an_upstream_bump_mid_catch_up_restarts_the_build_number(tmp_path):
    repo = _scratch(tmp_path)
    _commit(repo, "1.1.6", "a")
    _git(repo, "tag", "metal-v1.1.6.3")
    b = _commit(repo, "1.1.6", "b")
    c = _commit(repo, "1.1.7", "sync 1.1.7")
    _push(repo)
    _, rows = _plan(repo, tmp_path)
    assert [r[:2] for r in rows] == [[b, "1.1.6.4"], [c, "1.1.7.1"]]
    assert rows[1][2] == "metal-v1.1.6.4"


def test_an_already_released_head_is_skipped(tmp_path):
    repo = _scratch(tmp_path)
    _commit(repo, "1.1.6", "a")
    _git(repo, "tag", "metal-v1.1.6.2")
    _push(repo)
    outputs, rows = _plan(repo, tmp_path)
    assert outputs["skip"] == "true" and rows == []


def test_no_previous_release_releases_only_head(tmp_path):
    repo = _scratch(tmp_path)
    _commit(repo, "1.1.6", "a")
    head = _commit(repo, "1.1.6", "b")
    _push(repo)
    _, rows = _plan(repo, tmp_path)
    assert rows == [[head, "1.1.6.1", ""]]


def test_runs_on_main_and_dispatches_publish_metal_for_every_planned_version():
    wf = _workflow()
    on = wf.get("on", wf.get(True))  # PyYAML reads the bare key `on` as True
    assert on["push"]["branches"] == ["main"]
    assert wf["concurrency"]["cancel-in-progress"] is False
    job = wf["jobs"]["release"]
    assert job["permissions"] == {"contents": "write", "actions": "write"}
    assert "refs/heads/main" in job["if"]
    checkout = next(s for s in _steps() if s.get("uses", "").startswith("actions/checkout"))
    assert checkout["with"]["persist-credentials"] is False
    loop = next(s["run"] for s in _steps() if "gh release create" in s.get("run", ""))
    assert "while read -r sha version previous" in loop
    assert '--target "$sha"' in loop and "--generate-notes" in loop
    assert "publish-metal.yml" in loop and "publish=true" in loop
    assert '--ref "metal-v${version}"' in loop
