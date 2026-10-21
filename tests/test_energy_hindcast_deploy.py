"""``woof energy deploy render``: the four backends, the container, refusals.

Every backend is rendered from the shipped examples and from flags alone,
then checked the way its consumer would read it: JSON request bodies against
the keys the AWS Batch and GCP Batch APIs require, YAML for Kubernetes
(``kubectl apply --dry-run=client`` when kubectl is installed), ``bash -n``
for every script.  Nothing contacts a cloud.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from woof.cli import build_parser
from woof.energy import deploy
from woof.energy.cli import _DEPLOY_BACKENDS
from woof.energy.deploy import (
    DEPLOY_SCHEMA, DeployRefused, config_from_mapping, load_config_mapping,
    render, write_rendered)

REPO = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO / "deploy"
AWS_EXAMPLE = DEPLOY_DIR / "examples" / "aws.toml"
GCP_EXAMPLE = DEPLOY_DIR / "examples" / "gcp.toml"
BASH = shutil.which("bash")


def _config(path: Path, **overrides):
    mapping = load_config_mapping(path)
    mapping.update(overrides)
    return config_from_mapping(mapping)


def _minimal(**overrides):
    mapping = {"name": "wf", "image": "reg.example.com/woof:1.0.3-cu12",
               "queue_url": "s3://bkt/queue", "output_url": "s3://bkt/out"}
    mapping.update(overrides)
    return config_from_mapping(mapping)


def _bash_n(path: Path) -> None:
    if BASH is None:
        pytest.skip("bash is not installed")
    result = subprocess.run([BASH, "-n", str(path)], capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# Registry and determinism


def test_the_parser_offers_exactly_the_backends_the_renderer_knows():
    assert _DEPLOY_BACKENDS == deploy.BACKENDS == tuple(deploy._RENDERERS)
    args = build_parser().parse_args(
        ["energy", "deploy", "render", "--backend", "slurm", "-o", "x"])
    assert args.backend == "slurm" and args.spot is None


@pytest.mark.parametrize("example,backend", [
    (AWS_EXAMPLE, "aws-batch"), (AWS_EXAMPLE, "k8s"), (AWS_EXAMPLE, "slurm"),
    (GCP_EXAMPLE, "gcp-batch"), (GCP_EXAMPLE, "k8s")])
def test_the_examples_render_strictly_and_deterministically(
        tmp_path, example, backend):
    config = _config(example)
    first = render(config, backend, strict=True)
    second = render(_config(example), backend, strict=True)
    assert first.files == second.files
    assert first.placeholders == ()
    manifest = write_rendered(first, tmp_path)
    assert manifest["schema"] == DEPLOY_SCHEMA
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == sorted([*first.files, "deploy.json"])
    on_disk = json.loads((tmp_path / "deploy.json").read_text())
    assert on_disk == manifest
    for name, digest in manifest["files"].items():
        import hashlib
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() \
            == digest
    assert os.access(tmp_path / "submit.sh", os.X_OK)
    for name, text in first.files.items():
        assert not re.search(r"AKIA|ASIA[0-9A-Z]{16}|BEGIN .*PRIVATE",
                             text), name


# --------------------------------------------------------------------------
# AWS Batch


def test_aws_batch_request_bodies_carry_the_required_keys(tmp_path):
    config = _config(AWS_EXAMPLE)
    files = render(config, "aws-batch", strict=True).files
    ce = json.loads(files["compute-environment.json"])
    queue = json.loads(files["job-queue.json"])
    jobdef = json.loads(files["job-definition.json"])
    submit = json.loads(files["submit-job.json"])
    template = json.loads(files["launch-template.json"])

    # CreateComputeEnvironment
    assert {"computeEnvironmentName", "type", "computeResources"} <= set(ce)
    assert ce["type"] == "MANAGED"
    resources = ce["computeResources"]
    assert {"type", "maxvCpus", "subnets", "instanceTypes",
            "instanceRole"} <= set(resources)
    assert resources["type"] == "SPOT"
    assert resources["allocationStrategy"] == "SPOT_PRICE_CAPACITY_OPTIMIZED"
    assert resources["instanceTypes"] == ["p5.48xlarge", "p5e.48xlarge"]
    assert resources["minvCpus"] == 0
    assert resources["maxvCpus"] == 32 * 192
    assert resources["ec2Configuration"] == [{"imageType": "ECS_AL2023_NVIDIA"}]
    assert resources["launchTemplate"]["launchTemplateName"] == \
        template["LaunchTemplateName"]

    # CreateJobQueue
    assert {"jobQueueName", "priority", "computeEnvironmentOrder"} <= set(queue)
    assert queue["computeEnvironmentOrder"] == [
        {"order": 1, "computeEnvironment": ce["computeEnvironmentName"]}]

    # RegisterJobDefinition
    assert {"jobDefinitionName", "type", "containerProperties"} <= set(jobdef)
    container = jobdef["containerProperties"]
    assert container["image"] == config.image
    requirements = {r["type"]: r["value"]
                    for r in container["resourceRequirements"]}
    assert requirements["GPU"] == "8"
    assert int(requirements["VCPU"]) <= 192
    assert int(requirements["MEMORY"]) <= 2048 * 1024
    assert container["command"][:4] == ["woof", "energy", "worker", "--queue"]
    assert container["command"][4] == config.queue_url
    assert "--exit-when-empty" in container["command"]
    env = {e["name"]: e["value"] for e in container["environment"]}
    assert env["WOOF_ENERGY_OUTPUT_URL"] == config.output_url
    assert env["WOOF_GEOG_URL"] == config.geog_url
    assert env["AWS_REGION"] == "eu-west-2"
    assert container["jobRoleArn"].endswith(":role/woof-energy-worker")
    assert 1 <= jobdef["retryStrategy"]["attempts"] <= 10
    assert {"onExitCode": "2", "action": "EXIT"} in \
        jobdef["retryStrategy"]["evaluateOnExit"]
    assert jobdef["timeout"]["attemptDurationSeconds"] >= 60

    # SubmitJob: an array of N workers
    assert {"jobName", "jobQueue", "jobDefinition"} <= set(submit)
    assert submit["arrayProperties"] == {"size": 64}
    assert submit["jobQueue"] == queue["jobQueueName"]
    assert submit["jobDefinition"] == jobdef["jobDefinitionName"]

    # The launch template's user data is MIME with the NVMe script inside.
    userdata = base64.b64decode(
        template["LaunchTemplateData"]["UserData"]).decode("ascii")
    assert userdata.startswith("MIME-Version: 1.0")
    assert deploy.NVME_USERDATA in userdata
    for name in ("submit.sh", "scratch-userdata.sh"):
        path = tmp_path / name
        path.write_text(files[name])
        _bash_n(path)


def test_aws_batch_one_worker_is_not_an_array_and_on_demand_is_ec2():
    files = render(_minimal(spot=False, aws_batch={"region": "us-east-1",
                                                   "instance_types":
                                                   ["g6e.xlarge"]}),
                   "aws-batch").files
    submit = json.loads(files["submit-job.json"])
    assert "arrayProperties" not in submit
    ce = json.loads(files["compute-environment.json"])
    assert ce["computeResources"]["type"] == "EC2"
    jobdef = json.loads(files["job-definition.json"])
    reqs = {r["type"]: r["value"]
            for r in jobdef["containerProperties"]["resourceRequirements"]}
    assert reqs["GPU"] == "1" and reqs["VCPU"] == "4"


def test_aws_batch_without_nvme_still_opens_scratch_to_the_worker():
    rendered = render(_minimal(aws_batch={"nvme_scratch": False}),
                      "aws-batch")
    template = json.loads(rendered.files["launch-template.json"])
    userdata = base64.b64decode(
        template["LaunchTemplateData"]["UserData"]).decode("ascii")
    assert deploy.SCRATCH_USERDATA in userdata
    assert "mdadm" not in userdata
    ce = json.loads(rendered.files["compute-environment.json"])
    assert ce["computeResources"]["launchTemplate"]["launchTemplateName"] \
        == template["LaunchTemplateName"]


def test_aws_submit_stops_on_an_invalid_compute_environment():
    script = render(_config(AWS_EXAMPLE), "aws-batch").files["submit.sh"]
    assert "INVALID*)" in script and "after 15 minutes" in script


def test_rerendering_a_directory_drops_the_earlier_files(tmp_path):
    write_rendered(render(_minimal(workers=4), "aws-batch"), tmp_path)
    assert (tmp_path / "job-definition.json").exists()
    write_rendered(render(_minimal(gpus_per_worker=1), "k8s"), tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == \
        ["deploy.json", "job.yaml", "submit.sh"]
    other = tmp_path / "other"
    other.mkdir()
    (other / "deploy.json").write_text('{"schema": "something-else"}')
    with pytest.raises(DeployRefused, match="not a woof-energy.deploy.v1"):
        write_rendered(render(_minimal(gpus_per_worker=1), "k8s"), other)


def test_aws_batch_missing_identifiers_become_placeholders_and_guard(tmp_path):
    rendered = render(_minimal(workers=4), "aws-batch")
    assert "REPLACE_ME_AWS_JOB_ROLE_ARN" in rendered.placeholders
    assert "REPLACE_ME_AWS_SUBNET_ID" in rendered.placeholders
    write_rendered(rendered, tmp_path)
    if BASH is None:
        pytest.skip("bash is not installed")
    result = subprocess.run([BASH, str(tmp_path / "submit.sh")],
                            capture_output=True, text=True,
                            env={"PATH": os.environ.get("PATH", "")})
    assert result.returncode == 2
    assert "REFUSED: placeholders" in result.stderr
    with pytest.raises(DeployRefused, match="--strict"):
        render(_minimal(workers=4), "aws-batch", strict=True)


# --------------------------------------------------------------------------
# GCP Batch


def test_gcp_batch_job_has_the_documented_shape(tmp_path):
    config = _config(GCP_EXAMPLE)
    files = render(config, "gcp-batch", strict=True).files
    job = json.loads(files["job.json"])
    assert set(job) >= {"taskGroups", "allocationPolicy", "logsPolicy"}
    (group,) = job["taskGroups"]
    assert group["taskCount"] == 64 and group["parallelism"] == 32
    assert group["taskCountPerNode"] == 1
    spec = group["taskSpec"]
    container = spec["runnables"][-1]["container"]
    assert container["imageUri"] == config.image
    assert container["commands"][:5] == ["woof", "energy", "worker",
                                         "--queue", config.queue_url]
    assert spec["environment"]["variables"]["WOOF_ENERGY_OUTPUT_URL"] == \
        config.output_url
    assert spec["computeResource"]["cpuMilli"] > 0
    assert spec["maxRunDuration"].endswith("s")
    assert spec["lifecyclePolicies"][0]["actionCondition"]["exitCodes"] == [2]
    (instance,) = job["allocationPolicy"]["instances"]
    assert instance["installGpuDrivers"] is True
    policy = instance["policy"]
    assert policy["machineType"] == "a3-highgpu-8g"
    assert policy["provisioningModel"] == "SPOT"
    assert policy["accelerators"] == [{"type": "nvidia-h100-80gb",
                                       "count": 8}]
    assert job["allocationPolicy"]["location"]["allowedLocations"] == \
        ["regions/europe-west4"]
    assert job["allocationPolicy"]["serviceAccount"]["email"].startswith(
        "woof-energy-worker@")
    path = tmp_path / "submit.sh"
    path.write_text(files["submit.sh"])
    _bash_n(path)


def test_gcp_batch_refuses_a_gpu_count_the_machine_does_not_have():
    with pytest.raises(DeployRefused, match="carries 8 GPU"):
        render(_config(GCP_EXAMPLE, gpus_per_worker=4), "gcp-batch")
    with pytest.raises(DeployRefused, match="not a known GPU machine"):
        render(_config(GCP_EXAMPLE, gcp_batch={"machine_type": "n2-standard-8"}),
               "gcp-batch")


# --------------------------------------------------------------------------
# Kubernetes


def _k8s_documents(text: str) -> list[dict]:
    yaml = pytest.importorskip("yaml")
    return [doc for doc in yaml.safe_load_all(text) if doc]


@pytest.mark.parametrize("example,annotation", [
    (AWS_EXAMPLE, "eks.amazonaws.com/role-arn"),
    (GCP_EXAMPLE, "iam.gke.io/gcp-service-account")])
def test_k8s_indexed_job_and_workload_identity(tmp_path, example, annotation):
    config = _config(example)
    files = render(config, "k8s", strict=True).files
    sa, job = _k8s_documents(files["job.yaml"])
    assert sa["kind"] == "ServiceAccount"
    assert annotation in sa["metadata"]["annotations"]
    assert job["apiVersion"] == "batch/v1" and job["kind"] == "Job"
    spec = job["spec"]
    assert spec["completionMode"] == "Indexed"
    assert spec["completions"] == 64 and spec["parallelism"] == 32
    pod = spec["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    assert pod["serviceAccountName"] == sa["metadata"]["name"]
    assert pod["securityContext"]["runAsNonRoot"] is True
    (container,) = pod["containers"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "8"
    assert container["args"][:3] == ["woof", "energy", "worker"]
    assert any(t["key"] == "nvidia.com/gpu" for t in pod["tolerations"])
    spot_key = ("cloud.google.com/gke-spot" if example is GCP_EXAMPLE
                else "eks.amazonaws.com/capacityType")
    assert spot_key in pod["nodeSelector"]
    rules = spec["podFailurePolicy"]["rules"]
    assert rules[0]["action"] == "FailIndex"
    assert rules[1]["onPodConditions"] == [{"type": "DisruptionTarget"}]
    path = tmp_path / "job.yaml"
    path.write_text(files["job.yaml"])
    kubectl = shutil.which("kubectl")
    if kubectl:
        result = subprocess.run(
            [kubectl, "apply", "--dry-run=client", "--validate=false",
             "-f", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_the_yaml_writer_round_trips_awkward_strings():
    yaml = pytest.importorskip("yaml")
    document = {"a": "yes", "b": "0123", "c": ["*x", "#y", "k: v", ""],
                "d": {}, "e": [], "f": {"nested/key.io": [1, True, None]},
                "g": "line\twith\ttabs \"quoted\" é"}
    (loaded,) = yaml.safe_load_all(deploy.yaml_documents([document]))
    assert loaded == document


def test_k8s_and_slurm_refuse_a_missing_gpu_count():
    for backend in ("k8s", "slurm"):
        with pytest.raises(DeployRefused, match="needs gpus_per_worker"):
            render(_minimal(), backend)


def test_k8s_host_path_scratch_is_opened_to_the_worker():
    files = render(_config(AWS_EXAMPLE), "k8s").files
    _, job = _k8s_documents(files["job.yaml"])
    pod = job["spec"]["template"]["spec"]
    (init,) = pod["initContainers"]
    assert init["command"][-1] == "chmod 1777 /scratch"
    assert init["securityContext"]["runAsUser"] == 0
    assert {"name": "scratch", "hostPath": {
        "path": "/mnt/k8s-disks/0", "type": "DirectoryOrCreate"}} \
        in pod["volumes"]
    plain = render(_minimal(gpus_per_worker=1), "k8s").files["job.yaml"]
    assert "initContainers" not in plain


def test_yaml_keys_that_yaml_1_1_reads_as_booleans_are_quoted():
    text = deploy.yaml_documents([{"on": "x", "yes": 1, "No": 2, "y": 3,
                                   "plain": 4}])
    assert '"on": "x"' in text and '"yes": 1' in text
    assert '"No": 2' in text and '"y": 3' in text and "plain: 4" in text


def test_k8s_generic_spot_and_file_urls_are_refused():
    with pytest.raises(DeployRefused, match="spot node label"):
        render(_minimal(spot=True), "k8s")
    with pytest.raises(DeployRefused, match="file://"):
        render(_minimal(queue_url="file:///shared/queue"), "k8s")


# --------------------------------------------------------------------------
# Slurm


@pytest.mark.parametrize("runtime", ["apptainer", "pyxis", "native"])
def test_slurm_array_script(tmp_path, runtime):
    slurm = {"runtime": runtime, "partition": "gpu"}
    if runtime == "native":
        slurm["woof_root"] = "/shared/woof"
    config = _minimal(workers=10, parallelism=4, gpus_per_worker=8,
                      queue_url="file:///shared/queue",
                      output_url="s3://bkt/out", slurm=slurm)
    rendered = render(config, "slurm", strict=True)
    script = rendered.files["worker.sbatch"]
    assert "#SBATCH --array=0-9%4\n" in script
    assert "#SBATCH --gres=gpu:8\n" in script
    assert "#SBATCH --nodes=1\n" in script
    assert "#SBATCH --partition=gpu\n" in script
    assert "export WOOF_ENERGY_OUTPUT_URL=s3://bkt/out\n" in script
    assert "woof energy worker --queue file:///shared/queue --gpus all " \
           "--exit-when-empty" in script
    if runtime == "pyxis":
        assert "--container-image='reg.example.com#woof:1.0.3-cu12'" in script
    for name in ("worker.sbatch", "submit.sh"):
        path = tmp_path / name
        path.write_text(rendered.files[name])
        _bash_n(path)


def test_slurm_time_limit_follows_max_runtime():
    script = render(_minimal(max_runtime_hours=30.5, gpus_per_worker=4),
                    "slurm").files["worker.sbatch"]
    assert "#SBATCH --time=1-06:30:00\n" in script
    with pytest.raises(DeployRefused, match="time_limit"):
        render(_minimal(gpus_per_worker=4,
                        slurm={"time_limit": "forever"}), "slurm")


# --------------------------------------------------------------------------
# Refusals shared by every backend


@pytest.mark.parametrize("overrides,match", [
    ({"image": "reg.example.com/woof"}, "no tag or digest"),
    ({"image": "reg.example.com/woof:latest"}, "latest"),
    ({"name": "Wind_Farm"}, "lower-case"),
    ({"queue_url": "https://bkt/queue"}, "scheme"),
    ({"queue_url": "s3://key:secret@bkt/q"}, "no credentials"),
    ({"workers": 0}, "at least 1"),
    ({"workers": 2, "parallelism": 0}, "parallelism"),
    ({"env": {"AWS_SECRET_ACCESS_KEY": "x"}}, "credential"),
    ({"env": {"NOTE": "AKIAABCDEFGHIJKLMNOP"}}, "access key"),
    ({"env": {"WOOF_ENERGY_QUEUE_URL": "s3://x/y"}}, "renderer sets"),
    ({"worker_args": ["--queue=s3://x/y"]}, "--queue"),
    ({"worker_args": ["--exit-when-empty"]}, "exit_when_empty"),
    ({"worker_gpus": "some"}, "worker_gpus"),
    ({"labels": {"Project": "x"}}, "label"),
    ({"labels": {"team": "gpu-"}}, "label"),
    ({"labels": {"team_": "gpu"}}, "label"),
    ({"retries": 12}, "retries"),
    ({"spot": "yes"}, "true or false"),
    ({"surprise": 1}, "unknown key"),
    ({"aws_batch": {"instance_type": "p5.48xlarge"}}, "unknown key"),
])
def test_bad_configurations_are_refused_by_name(overrides, match):
    with pytest.raises(DeployRefused, match=match):
        render(_minimal(**overrides), "aws-batch")


def test_cloud_backends_refuse_the_other_clouds_urls():
    with pytest.raises(DeployRefused, match="gs://"):
        render(_minimal(), "gcp-batch")
    with pytest.raises(DeployRefused, match="s3://"):
        render(_config(GCP_EXAMPLE), "aws-batch")
    with pytest.raises(DeployRefused, match="s3://"):
        render(_minimal(geog_url="gs://other/WPS_GEOG"), "aws-batch")


def test_aws_instance_shapes_are_checked():
    with pytest.raises(DeployRefused, match="does not fit"):
        render(_minimal(gpus_per_worker=8,
                        aws_batch={"instance_types": ["p5.48xlarge",
                                                      "g6e.12xlarge"]}),
               "aws-batch")
    with pytest.raises(DeployRefused, match="not a known GPU instance"):
        render(_minimal(aws_batch={"instance_types": ["x9.huge"]}),
               "aws-batch")
    declared = render(_minimal(aws_batch={
        "instance_types": ["x9.huge"],
        "instance_shapes": {"x9.huge": {"gpus": 4, "vcpus": 64,
                                        "memory_gib": 512}}}), "aws-batch")
    jobdef = json.loads(declared.files["job-definition.json"])
    assert jobdef["containerProperties"]["resourceRequirements"][0] == \
        {"type": "GPU", "value": "4"}
    # One GPU of an 8-GPU instance asks for an eighth of it, so eight pack.
    eighth = render(_minimal(gpus_per_worker=1,
                             aws_batch={"instance_types": ["p5.48xlarge"]}),
                    "aws-batch")
    reqs = {r["type"]: int(r["value"]) for r in json.loads(
        eighth.files["job-definition.json"])["containerProperties"][
            "resourceRequirements"]}
    assert reqs == {"GPU": 1, "VCPU": 24,
                    "MEMORY": int(2048 * 1024 * 0.9 / 8)}
    with pytest.raises(DeployRefused, match="at most 10000"):
        render(_minimal(workers=10_001), "aws-batch")


def test_missing_required_keys_are_named():
    with pytest.raises(DeployRefused, match="missing required key.*image"):
        config_from_mapping({"name": "wf", "queue_url": "s3://b/q",
                             "output_url": "s3://b/o"})


# --------------------------------------------------------------------------
# CLI


def _cli(*argv):
    return build_parser().parse_args(["energy", "deploy", "render", *argv])


def test_cli_renders_from_flags_and_prints_a_summary(tmp_path, capsys):
    out = tmp_path / "gcp"
    args = _cli("--backend", "gcp-batch", "--config", str(GCP_EXAMPLE),
                "--workers", "8", "--on-demand", "--strict", "-o", str(out))
    assert deploy.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == DEPLOY_SCHEMA
    assert summary["workers"] == 8 and summary["placeholders"] == []
    assert summary["parallelism"] == 8  # the config's 32, capped
    job = json.loads((out / "job.json").read_text())
    assert job["taskGroups"][0]["taskCount"] == 8
    assert job["taskGroups"][0]["parallelism"] == 8
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert policy["provisioningModel"] == "STANDARD"


def test_cli_refusal_exits_2_with_a_reason_on_stderr(tmp_path, capsys):
    args = _cli("--backend", "aws-batch", "--name", "wf",
                "--image", "reg.example.com/woof:latest",
                "--queue", "s3://b/q", "--output-url", "s3://b/o",
                "-o", str(tmp_path / "out"))
    assert deploy.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "refused" in captured.err and "latest" in captured.err
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------
# Container


def test_entrypoint_is_valid_bash_and_stages_a_file_url(tmp_path):
    entrypoint = DEPLOY_DIR / "entrypoint.sh"
    _bash_n(entrypoint)
    source = tmp_path / "src" / "WPS_GEOG"
    (source / "topo").mkdir(parents=True)
    (source / "topo" / "index").write_text("type=continuous\n")
    scratch = tmp_path / "scratch"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
           "WOOF_SCRATCH": str(scratch), "WOOF_GEOG_URL": f"file://{source}",
           "AWS_BATCH_JOB_ARRAY_INDEX": "7"}
    for _ in range(2):
        result = subprocess.run(
            [BASH, str(entrypoint), "sh", "-c",
             'echo "$GPUWM_WPS_GEOG|$WOOF_WORKER_INDEX|$XDG_CACHE_HOME"'],
            capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
    geog, index, cache = result.stdout.strip().split("|")
    assert Path(geog) == scratch / "case-data" / "WPS_GEOG"
    assert index == "7"
    assert Path(cache) == scratch / "cache"
    assert (Path(geog) / "topo" / "index").read_text() == "type=continuous\n"
    assert "already staged" in result.stderr
    moved = subprocess.run(
        [BASH, str(entrypoint), "sh", "-c", 'echo "$TMPDIR"'],
        capture_output=True, text=True,
        env={**env, "TMPDIR": "/proc/woof-not-creatable"})
    assert moved.returncode == 0, moved.stderr
    assert Path(moved.stdout.strip()) == scratch / "tmp"
    refused = subprocess.run([BASH, str(entrypoint)], capture_output=True,
                             text=True, env={**env, "WOOF_GEOG_URL": ""})
    assert refused.returncode == 2
    assert "WOOF_ENERGY_QUEUE_URL" in refused.stderr


def test_the_dockerfile_matches_what_the_renderers_assume():
    text = (DEPLOY_DIR / "Dockerfile").read_text()
    assert "ARG CUDA=12" in text and "cuda13" in text
    assert "pixi install --locked" in text
    assert "build-native" in text and "woof fetch-tables" in text
    assert f"USER {deploy.CONTAINER_UID}:{deploy.CONTAINER_UID}" in text
    assert f"{deploy.CONTAINER_ENTRYPOINT}\"" in text
    assert re.search(r"HEALTHCHECK .*\n.*--healthcheck", text)
    assert f"WOOF_SCRATCH={deploy.CONTAINER_SCRATCH}" in text
    ignore = (DEPLOY_DIR / "Dockerfile.dockerignore").read_text().split()
    assert ".git" in ignore and ".pixi" in ignore
    assert "**/target" not in ignore  # vendored crates carry src/target


@pytest.mark.slow
def test_docker_build_check_passes():
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker is not installed")
    result = subprocess.run(
        [docker, "build", "--check", "-f", str(DEPLOY_DIR / "Dockerfile"),
         str(REPO)], capture_output=True, text=True, timeout=600)
    if result.returncode != 0 and "Cannot connect" in result.stderr:
        pytest.skip("no docker daemon")
    assert result.returncode == 0, result.stderr


def test_build_native_builds_every_documented_native_workspace():
    """The image builds its bridges with ``pixi run build-native``; a crate
    the documented build loop names but the task leaves out is missing from
    every deployed worker (the Zarr reader was, so ERA5 from the ARCO store
    could not be fetched in the image)."""
    import tomllib

    root = DEPLOY_DIR.parent
    task = tomllib.loads((root / "pyproject.toml").read_text())[
        "tool"]["pixi"]["feature"]["base"]["tasks"]["build-native"]
    task = task if isinstance(task, str) else task["cmd"]
    built = set(re.findall(r"cd (?:tools/|\.\./)([\w-]+) ", task))
    loop = re.search(r"for workspace in ([^;]+);",
                     (root / "docs/public/FIRST-LIGHT.md").read_text())
    documented = {item.removeprefix("tools/") for item in loop.group(1).split()}
    assert "zarr_bridge" in documented
    assert documented <= built, sorted(documented - built)
