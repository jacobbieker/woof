"""release-metal.yml: every push to main becomes the next metal-woof build.

The version step's own shell script is run against a scratch repository, so
the build number it derives from the tags is the one the workflow would tag.
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
                          capture_output=True, text=True).stdout


def _scratch(tmp_path: Path, version: str, tags: list[str]) -> Path:
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    repo.mkdir()
    _git(repo, "init", "-q")
    for i, tag in enumerate(tags):
        (repo / "f").write_text(str(i))
        _git(repo, "add", "f")
        _git(repo, "commit", "-q", "-m", f"c{i}")
        _git(repo, "tag", tag)
    (repo / "pyproject.toml").write_text(f'[project]\nname = "recast-woof"\nversion = "{version}"\n')
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-q", "-m", "head")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "HEAD:refs/heads/main", "--tags")
    return repo


def _run_version(repo: Path, tmp_path: Path) -> dict[str, str]:
    out = tmp_path / "github_output"
    out.write_text("")
    subprocess.run(["bash", "-c", _version_script()], cwd=repo, check=True, capture_output=True,
                   text=True, env={**os.environ, "GITHUB_OUTPUT": str(out)})
    return dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)


def test_first_build_after_an_upstream_release_is_dot_one(tmp_path):
    repo = _scratch(tmp_path, "1.1.6", ["metal-v1.1.5.1", "metal-v1.1.5.2"])
    got = _run_version(repo, tmp_path)
    assert got == {"skip": "false", "version": "1.1.6.1", "previous": "metal-v1.1.5.2"}


def test_build_number_follows_the_highest_tag_numerically(tmp_path):
    repo = _scratch(tmp_path, "1.1.6", ["metal-v1.1.6.1", "metal-v1.1.6.9", "metal-v1.1.6.10"])
    got = _run_version(repo, tmp_path)
    assert got["version"] == "1.1.6.11"
    assert got["previous"] == "metal-v1.1.6.10"


def test_an_already_released_head_is_skipped(tmp_path):
    repo = _scratch(tmp_path, "1.1.6", ["metal-v1.1.6.1"])
    _git(repo, "tag", "metal-v1.1.6.2")
    _git(repo, "push", "-q", "origin", "--tags")
    assert _run_version(repo, tmp_path) == {"skip": "true"}


def test_no_previous_release(tmp_path):
    repo = _scratch(tmp_path, "1.1.6", [])
    assert _run_version(repo, tmp_path) == {"skip": "false", "version": "1.1.6.1", "previous": ""}


def test_runs_on_main_and_dispatches_publish_metal_with_publish_on():
    wf = _workflow()
    on = wf.get("on", wf.get(True))  # PyYAML reads the bare key `on` as True
    assert on["push"]["branches"] == ["main"]
    assert wf["concurrency"]["cancel-in-progress"] is False
    job = wf["jobs"]["release"]
    assert job["permissions"] == {"contents": "write", "actions": "write"}
    dispatch = next(s["run"] for s in _steps() if "gh workflow run" in s.get("run", ""))
    assert "publish-metal.yml" in dispatch and "publish=true" in dispatch
    assert '--ref "metal-v${VERSION}"' in dispatch
    release = next(s["run"] for s in _steps() if "gh release create" in s.get("run", ""))
    assert "--generate-notes" in release and '--target "$GITHUB_SHA"' in release
