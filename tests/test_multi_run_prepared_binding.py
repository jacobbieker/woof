"""A module-form multi-run binds its preparation the way ``woof sim`` does.

A real field test packed a 1 km forecast from a sealed preparation beside
two running ones and failed twice in the child -- "a forecast binds its
preparation with --prepared-head-sha256, or with both --proof-sha256 and
--prepared-content-sha256", then "source-manifest-sha256 must be a
lowercase SHA-256 digest" -- until digests were copied out of proof.json by
hand.  Multi-run now derives them at plan load from the declared prepared
root (:func:`woof.stage_cli.sim_command`'s binding), keeps and verifies any
the plan gives, and refuses a declared config or namelist the preparation
was not made from.  CPU only, on the runner's own synthetic bundle.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import woof.multi_run as multi_run
from woof import stage_cli
from woof.multi_run import load_plan
from test_prepared_head_binding import _seal, _unseal
from test_prepared_single_domain_forecast import (_prepared_fixture,
                                                  _sha256)

RUNNER = "woof.prepared_single_domain_forecast"


def _plan(tmp_path: Path, fixture, *extra: str, source: str | None = "gfs",
          config: Path | None = None) -> Path:
    args = ([] if source is None else ["--source", source]) + [
        "--prepared-root", "{input0}",
        "--experiment-config", "{input1}",
        "--wps-namelist", "{input2}",
        *extra,
        "--io-mode", "history", "--outdir", "{outdir}"]
    inputs = [str(fixture.prepared), str(config or fixture.experiment),
              str(fixture.wps)]
    plan = tmp_path / "plan" / "plan.toml"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("\n".join([
        'schema = "gpuwm.multi-run-plan/v1"',
        'preflight = "off"',
        "[[run]]",
        'name = "prepared"',
        "device = 0",
        f'module = "{RUNNER}"',
        f"inputs = {json.dumps(inputs)}",
        f"args = {json.dumps(args)}",
        'outdir = "out"',
        'scratch = "scratch"',
        'cache = "cache"',
        "",
    ]), encoding="utf-8")
    return plan


def _sim_binding(fixture, bundle=None) -> dict[str, str]:
    bundle = bundle or stage_cli.resolve_bundle(fixture.prepared)
    command = stage_cli.sim_command(
        bundle, experiment_config=fixture.experiment,
        wps_namelist=fixture.wps, outdir=Path("unused"), runner="single")
    return multi_run._binding_from_command(
        command, multi_run._RUNNER_BINDING_FLAGS[RUNNER])


def _value(arguments, flag):
    return arguments[list(arguments).index(flag) + 1]


@pytest.fixture
def sealed(tmp_path):
    (tmp_path / "prep").mkdir()
    return _prepared_fixture(tmp_path / "prep", "gfs", physics_profile=None)


def test_a_plan_without_digests_is_bound_exactly_as_woof_sim_binds_it(
        tmp_path, sealed):
    spec = load_plan(_plan(tmp_path, sealed)).runs[0]
    expected = _sim_binding(sealed)
    assert set(expected) == {"--proof-sha256", "--source-manifest-sha256",
                             "--prepared-content-sha256"}
    assert expected["--proof-sha256"] == _sha256(sealed.proof)
    for flag, digest in expected.items():
        assert _value(spec.arguments, flag) == digest
    # Inserted right after the prepared root; the plan's argv is kept.
    prepared_at = spec.arguments.index("--prepared-root")
    assert spec.arguments[prepared_at + 2] == "--proof-sha256"
    assert "--proof-sha256" not in spec.plan_arguments
    binding = spec.prepared_binding
    assert binding["mode"] == "sealed_proof"
    assert binding["state"] == "sealed"
    assert [row["source"] for row in binding["digests"]] == ["derived"] * 3
    assert binding["source_option"]["source"] == "given, verified"
    # This route records no execution_inputs; that is said, not checked.
    assert binding["execution_inputs"]["experiment_config"]["status"] \
        == "NOT_RECORDED"
    # The worker command carries the derived digests to the runner.
    run = multi_run.PlannedRun(spec, multi_run.GPUIdentity(
        "GPU-alpha", "600.00", "card", 0))
    command = multi_run._run_command(run)
    assert _value(command, "--proof-sha256") == expected["--proof-sha256"]


def test_given_digests_are_kept_and_verified(tmp_path, sealed):
    expected = _sim_binding(sealed)
    spec = load_plan(_plan(
        tmp_path, sealed, "--source-manifest-sha256",
        expected["--source-manifest-sha256"])).runs[0]
    sources = {row["flag"]: row["source"]
               for row in spec.prepared_binding["digests"]}
    assert sources == {"--proof-sha256": "derived",
                       "--source-manifest-sha256": "given, verified",
                       "--prepared-content-sha256": "derived"}
    assert spec.arguments.count("--source-manifest-sha256") == 1


def test_a_wrong_digest_is_refused_by_name(tmp_path, sealed):
    with pytest.raises(ValueError, match=(
            r"--prepared-content-sha256 0{64} does not match the "
            r"preparation: expected [0-9a-f]{64}, derived from the sealed")):
        load_plan(_plan(tmp_path, sealed,
                        "--prepared-content-sha256", "0" * 64))


@pytest.mark.parametrize("extra,match", [
    (("--proof-sha256=" + "a" * 64,), "use the separate form"),
    (("--proof", "a" * 64), "abbreviates"),
    (("--proof-sha256", "ABC"), "must be a lowercase SHA-256 digest"),
    (("--proof-sha256", "a" * 64, "--proof-sha256", "a" * 64),
     "2 times"),
])
def test_binding_digests_take_one_canonical_form(tmp_path, sealed, extra,
                                                 match):
    with pytest.raises(ValueError, match=match):
        load_plan(_plan(tmp_path, sealed, *extra))


def test_source_is_derived_when_absent_and_refused_when_wrong(
        tmp_path, sealed):
    spec = load_plan(_plan(tmp_path / "absent", sealed, source=None)).runs[0]
    assert spec.arguments[:2] == ("--source", "gfs")
    assert spec.prepared_binding["source_option"]["source"] == "derived"
    with pytest.raises(ValueError, match="was prepared from 'gfs'"):
        load_plan(_plan(tmp_path / "wrong", sealed, source="era5"))


def test_an_unbindable_root_is_refused_before_anything_is_claimed(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    fixture = type("F", (), {"prepared": root,
                             "experiment": tmp_path / "x.toml",
                             "wps": tmp_path / "x.wps"})
    fixture.experiment.write_text("[experiment]\n", encoding="utf-8")
    fixture.wps.write_text("&share\n/\n", encoding="utf-8")
    plan = _plan(tmp_path, fixture)
    with pytest.raises(ValueError,
                       match="prepared root .* cannot be bound: .*none of"):
        load_plan(plan)
    assert not (plan.parent / "scratch").exists()


def test_a_sealed_chained_preparation_accepts_its_recorded_head(
        tmp_path, sealed):
    """The field plan's hand-pinned head binding stays valid, verified."""

    chain = _unseal(sealed)
    _seal(sealed, chain)
    head = chain.head["head_sha256"]
    manifest = _sha256(sealed.source_manifest)
    spec = load_plan(_plan(
        tmp_path / "pinned", sealed, "--prepared-head-sha256", head,
        "--source-manifest-sha256", manifest)).runs[0]
    binding = spec.prepared_binding
    assert binding["mode"] == "prepared_head"
    assert binding["state"] == "sealed"
    assert {row["flag"]: (row["sha256"], row["source"])
            for row in binding["digests"]} == {
        "--prepared-head-sha256": (head, "given, verified"),
        "--source-manifest-sha256": (manifest, "given, verified")}
    assert "--proof-sha256" not in spec.arguments

    # Only the head given: the manifest the head names is filled in.
    spec = load_plan(_plan(tmp_path / "head-only", sealed,
                           "--prepared-head-sha256", head)).runs[0]
    assert _value(spec.arguments, "--source-manifest-sha256") == manifest

    # Nothing given: the sealed proof is bound, as woof sim binds it.
    spec = load_plan(_plan(tmp_path / "derived", sealed)).runs[0]
    assert spec.prepared_binding["mode"] == "sealed_proof"

    with pytest.raises(ValueError, match="is not the head .* records"):
        load_plan(_plan(tmp_path / "wrong", sealed,
                        "--prepared-head-sha256", "1" * 64))


def test_a_sealed_unchained_preparation_has_no_head_to_bind(
        tmp_path, sealed):
    with pytest.raises(ValueError, match="records no chained head"):
        load_plan(_plan(tmp_path, sealed,
                        "--prepared-head-sha256", "1" * 64))


def test_a_preparation_still_being_prepared_binds_its_head(tmp_path, sealed):
    chain = _unseal(sealed)
    spec = load_plan(_plan(tmp_path / "live", sealed)).runs[0]
    binding = spec.prepared_binding
    assert binding["state"] == "being_prepared"
    assert binding["mode"] == "prepared_head"
    assert _value(spec.arguments, "--prepared-head-sha256") \
        == chain.head["head_sha256"]
    assert "--proof-sha256" not in spec.arguments
    with pytest.raises(ValueError, match="still being prepared"):
        load_plan(_plan(tmp_path / "proof", sealed,
                        "--proof-sha256", "a" * 64))


def _mapped_bundle(tmp_path: Path, config: Path, wps: Path) -> dict:
    return {"payload": {"execution_inputs": {
        "experiment_config": {"path": str(config), "bytes": 1,
                              "sha256": _sha256(config)},
        "wps_namelist": {"path": str(wps), "bytes": 1,
                         "sha256": _sha256(wps)}}}}


def test_declared_config_must_be_the_one_the_preparation_records(tmp_path):
    """A mapped preparation records its config/namelist by digest; the
    runner refuses any other bytes, so multi-run refuses them first and
    points at the file that still matches."""

    original = tmp_path / "maui.toml"
    original.write_text("[experiment]\nname = 'maui'\n", encoding="utf-8")
    wps = tmp_path / "maui.namelist.wps"
    wps.write_text("&share\n/\n", encoding="utf-8")
    bundle = _mapped_bundle(tmp_path, original, wps)
    check = multi_run._execution_input_check
    row = check(bundle, "experiment_config", original, _sha256(original),
                "run 'x'", tmp_path)
    assert row["status"] == "MATCH"

    edited = tmp_path / "maui-edited.toml"
    edited.write_text("[experiment]\nname = 'other'\n", encoding="utf-8")
    with pytest.raises(ValueError, match=(
            r"declared experiment config .*maui-edited.toml .* is not the "
            r"file the preparation .* was made from.* .*maui.toml still "
            r"holds those bytes: declare it")):
        check(bundle, "experiment_config", edited, _sha256(edited),
              "run 'x'", tmp_path)
    original.unlink()
    with pytest.raises(ValueError, match="No file holding those bytes"):
        check(bundle, "experiment_config", edited, _sha256(edited),
              "run 'x'", tmp_path)


def test_dry_run_prints_the_bound_plan_and_claims_nothing(
        tmp_path, sealed, capsys, monkeypatch):
    monkeypatch.setattr(multi_run, "query_gpus",
                        lambda: pytest.fail("dry run must not ask devices"))
    plan = _plan(tmp_path, sealed)
    assert multi_run.multi_run_main(Namespace(
        plan=plan, summary=None, preflight=None, dry_run=True,
        kernel_cache="per-run")) == 0
    document = json.loads(capsys.readouterr().out)
    run = document["runs"][0]
    assert document["schema"] == multi_run.DRY_RUN_SCHEMA
    assert document["kernel_cache"]["mode"] == "per-run"
    assert _value(run["command"], "--proof-sha256") == _sha256(sealed.proof)
    assert run["prepared_binding"]["mode"] == "sealed_proof"
    assert not (plan.parent / "scratch").exists()
    assert not (plan.parent / "plan.summary.json").exists()
