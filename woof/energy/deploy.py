"""``woof energy deploy render``: run ``woof energy worker`` fleets in the cloud.

A year-long hindcast at every public wind farm is thousands of independent
jobs (``woof energy hindcast plan``) on an object-store queue that GPU
workers drain (``woof energy worker --queue URL --gpus all
--exit-when-empty``).  This module renders the deployment that starts those
workers on one of four backends, from one typed :class:`DeployConfig`:

``aws-batch``
    ``launch-template.json`` (instance-store NVMe RAIDed to ``/scratch``),
    ``compute-environment.json`` (managed EC2 or SPOT, GPU instance types,
    the ECS GPU AMI), ``job-queue.json``, ``job-definition.json`` (GPU,
    vCPU and memory resource requirements, retry on spot reclaim, never on
    a refusal), ``submit-job.json`` (an array job of N workers) and
    ``submit.sh`` (the ``aws`` CLI calls, in order).  Each JSON file is the
    ``--cli-input-json`` request body of the call that consumes it.
``gcp-batch``
    ``job.json`` (a Batch ``Job``: ``taskCount`` workers, ``parallelism``
    at once, one per VM, accelerators matching the machine type, ``SPOT`` or
    ``STANDARD`` provisioning, a scratch disk) and ``submit.sh``.
``k8s``
    ``job.yaml``: an optional ServiceAccount carrying the workload-identity
    annotation (EKS IRSA or GKE Workload Identity) and an Indexed ``Job``
    (``completions``/``parallelism``, ``nvidia.com/gpu`` limits, node
    selectors and tolerations, spot evictions ignored by the pod failure
    policy, refusals fail their index).  Plus ``submit.sh``.
``slurm``
    ``worker.sbatch``: an array job, one worker per node using every GPU it
    was given (``--gres=gpu:N``), run through Apptainer, Pyxis/enroot or a
    native pixi checkout.  Plus ``submit.sh``.

**Several jobs per GPU.**  ``slots_per_gpu`` (``"1"``, a count or
``"auto"``) and ``mps`` reach every backend's worker command as
``--slots-per-gpu`` and ``--mps`` (docs/energy-hindcast/queue-worker.md):
the worker packs jobs whose declared memory fits each card and, with MPS,
lets their kernels run concurrently.  The worker starts its own MPS control
daemon inside the container; on Kubernetes the NVIDIA device plugin can
instead own MPS through its sharing configuration, and the k8s backend
writes a commented ``mps-sharing.example.yaml`` showing it.

Every backend also gets ``deploy.json`` (``woof-energy.deploy.v1``): the
normalized configuration, its SHA-256, the worker command, the files with
their SHA-256 and the placeholders left in them.

**Credentials.**  None are ever written.  Workers authenticate with the
instance role (AWS Batch, Slurm on cloud VMs), the VM service account (GCP
Batch) or workload identity (Kubernetes).  A configuration whose ``env``
names a secret-looking variable, or carries an AWS access key id or a PEM
block, is refused.

**Placeholders.**  An account-specific identifier the configuration does not
give (a subnet, a role ARN, a GCP project) is written as
``REPLACE_ME_<WHAT>`` and listed in the summary; ``submit.sh`` refuses to
run while any remain, and ``--strict`` refuses to render them at all.

**Pinned.**  Rendering is deterministic: the same configuration and woof
version produce byte-identical files (no timestamps, no host names, keys in a
fixed order).  The GPU machine tables (:data:`AWS_GPU_INSTANCES`,
:data:`GCP_GPU_MACHINES`) are the published shapes as of 2026; an
instance type outside them is refused unless the configuration declares its
shape.  Nothing here contacts a cloud.
"""

from __future__ import annotations

import base64
from dataclasses import MISSING, asdict, dataclass, field, fields
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys
import types
import typing
from typing import Any, Mapping

DEPLOY_SCHEMA = "woof-energy.deploy.v1"
BACKENDS = ("aws-batch", "gcp-batch", "k8s", "slurm")
PLACEHOLDER_PREFIX = "REPLACE_ME_"

#: Where the container's entrypoint and every backend put the worker scratch
#: (WPS_GEOG, the high-resolution cache, run directories).
CONTAINER_SCRATCH = "/scratch"
CONTAINER_ENTRYPOINT = "/usr/local/bin/woof-entrypoint"
#: The image's non-root user (deploy/Dockerfile).
CONTAINER_UID = 10001

#: Variables the renderers set; a configuration's ``env`` may not override them.
RESERVED_ENV = (
    "WOOF_ENERGY_QUEUE_URL", "WOOF_ENERGY_OUTPUT_URL", "WOOF_ENERGY_GPUS",
    "WOOF_GEOG_URL", "WOOF_HIGHRES_CACHE_URL", "WOOF_SCRATCH",
    "WOOF_DEPLOY_BACKEND", "WOOF_DEPLOY_NAME",
)
_SECRET_NAME = re.compile(
    r"(SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE_KEY|ACCESS_KEY|CREDENTIAL|"
    r"API_KEY|APIKEY)", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"((?:AKIA|ASIA)[0-9A-Z]{16}|-----BEGIN [A-Z ]+-----)")
_NAME = re.compile(r"^[a-z]([-a-z0-9]{0,38}[a-z0-9])?$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LABEL_KEY = re.compile(r"^[a-z]([-_a-z0-9]{0,61}[a-z0-9])?$")
_LABEL_VALUE = re.compile(r"^([a-z0-9]([-_a-z0-9]{0,61}[a-z0-9])?)?$")
#: Plain words a YAML 1.1 reader (kubectl) turns into booleans or null.
_YAML_11_WORDS = frozenset(
    "y yes n no true false on off null ~".split())
_SCHEMES = ("s3", "gs", "file")
#: Schemes a cloud's workers can reach with that cloud's ambient identity.
_CLOUD_SCHEME = {"aws": "s3", "gcp": "gs"}

#: AWS Batch array size bounds (SubmitJob arrayProperties.size).
AWS_ARRAY_MAX = 10_000
#: AWS Batch retryStrategy.attempts bounds.
AWS_MAX_ATTEMPTS = 10
#: Slurm's default MaxArraySize; a larger array needs the site to raise it.
SLURM_DEFAULT_MAX_ARRAY = 1001
#: Exit status woof uses for a refusal: never worth a retry.
REFUSAL_EXIT = 2


class DeployRefused(ValueError):
    """The configuration cannot be rendered as asked; the message says why."""


@dataclass(frozen=True)
class GpuMachine:
    """One GPU machine shape: GPUs, vCPUs and memory a worker may claim."""

    gpus: int
    gpu_model: str
    vcpus: int
    memory_gib: float
    #: GCP accelerator type (``accelerators[].type``); empty on AWS.
    accelerator: str = ""


def _shapes(gpu_model: str, rows, accelerator: str = ""):
    return {name: GpuMachine(gpus, gpu_model, vcpus, mem, accelerator)
            for name, gpus, vcpus, mem in rows}


#: AWS EC2 GPU instance types (published shapes).
AWS_GPU_INSTANCES: dict[str, GpuMachine] = {
    **_shapes("H100 80GB", [("p5.48xlarge", 8, 192, 2048)]),
    **_shapes("H200 141GB", [("p5e.48xlarge", 8, 192, 2048),
                             ("p5en.48xlarge", 8, 192, 2048)]),
    **_shapes("A100 40GB", [("p4d.24xlarge", 8, 96, 1152)]),
    **_shapes("A100 80GB", [("p4de.24xlarge", 8, 96, 1152)]),
    **_shapes("L40S 48GB", [
        ("g6e.xlarge", 1, 4, 32), ("g6e.2xlarge", 1, 8, 64),
        ("g6e.4xlarge", 1, 16, 128), ("g6e.8xlarge", 1, 32, 256),
        ("g6e.16xlarge", 1, 64, 512), ("g6e.12xlarge", 4, 48, 384),
        ("g6e.24xlarge", 4, 96, 768), ("g6e.48xlarge", 8, 192, 1536)]),
    **_shapes("A10G 24GB", [
        ("g5.xlarge", 1, 4, 16), ("g5.2xlarge", 1, 8, 32),
        ("g5.4xlarge", 1, 16, 64), ("g5.8xlarge", 1, 32, 128),
        ("g5.16xlarge", 1, 64, 256), ("g5.12xlarge", 4, 48, 192),
        ("g5.24xlarge", 4, 96, 384), ("g5.48xlarge", 8, 192, 768)]),
    **_shapes("L4 24GB", [
        ("g6.xlarge", 1, 4, 16), ("g6.2xlarge", 1, 8, 32),
        ("g6.4xlarge", 1, 16, 64), ("g6.8xlarge", 1, 32, 128),
        ("g6.16xlarge", 1, 64, 256), ("g6.12xlarge", 4, 48, 192),
        ("g6.24xlarge", 4, 96, 384), ("g6.48xlarge", 8, 192, 768)]),
}

#: GCP accelerator-optimized machine types; the GPUs come with the machine
#: and ``accelerators`` must name the same type and count.
GCP_GPU_MACHINES: dict[str, GpuMachine] = {
    **_shapes("H100 80GB", [("a3-highgpu-1g", 1, 26, 234),
                            ("a3-highgpu-2g", 2, 52, 468),
                            ("a3-highgpu-4g", 4, 104, 936),
                            ("a3-highgpu-8g", 8, 208, 1872)],
              "nvidia-h100-80gb"),
    **_shapes("H100 80GB Mega", [("a3-megagpu-8g", 8, 208, 1872)],
              "nvidia-h100-mega-80gb"),
    **_shapes("A100 40GB", [("a2-highgpu-1g", 1, 12, 85),
                            ("a2-highgpu-2g", 2, 24, 170),
                            ("a2-highgpu-4g", 4, 48, 340),
                            ("a2-highgpu-8g", 8, 96, 680),
                            ("a2-megagpu-16g", 16, 96, 1360)],
              "nvidia-tesla-a100"),
    **_shapes("A100 80GB", [("a2-ultragpu-1g", 1, 12, 170),
                            ("a2-ultragpu-2g", 2, 24, 340),
                            ("a2-ultragpu-4g", 4, 48, 680),
                            ("a2-ultragpu-8g", 8, 96, 1360)],
              "nvidia-a100-80gb"),
    **_shapes("L4 24GB", [("g2-standard-4", 1, 4, 16),
                          ("g2-standard-8", 1, 8, 32),
                          ("g2-standard-12", 1, 12, 48),
                          ("g2-standard-16", 1, 16, 64),
                          ("g2-standard-32", 1, 32, 128),
                          ("g2-standard-24", 2, 24, 96),
                          ("g2-standard-48", 4, 48, 192),
                          ("g2-standard-96", 8, 96, 384)],
              "nvidia-l4"),
}


# ---------------------------------------------------------------------------
# The typed configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AwsBatchOptions:
    """``[aws_batch]``: the compute environment, queue and job definition."""

    region: str | None = None
    instance_types: tuple[str, ...] = ("p5.48xlarge",)
    #: Shapes for instance types outside :data:`AWS_GPU_INSTANCES`:
    #: ``{type: {gpus, vcpus, memory_gib, gpu_model}}``.
    instance_shapes: dict[str, dict] = field(default_factory=dict)
    subnets: tuple[str, ...] = ()
    security_group_ids: tuple[str, ...] = ()
    #: The EC2 instance profile ARN the container instances run as.
    instance_role_arn: str | None = None
    #: The IAM role the worker container assumes (S3 queue and output).
    job_role_arn: str | None = None
    #: Optional Batch service role; absent, Batch uses its service-linked role.
    service_role_arn: str | None = None
    #: Ceiling on the environment's vCPUs; default ``parallelism`` times the
    #: largest instance's vCPUs.
    max_vcpus: int | None = None
    #: RAID0 the instance-store NVMe drives to ``/scratch`` via a launch
    #: template (p5/p4d/g6e/g5 all carry them).
    nvme_scratch: bool = True
    root_volume_gib: int = 200


@dataclass(frozen=True)
class GcpBatchOptions:
    """``[gcp_batch]``: the Batch job's allocation policy."""

    project: str | None = None
    region: str | None = None
    machine_type: str = "a3-highgpu-8g"
    #: The service account the VMs (and so the workers) run as.
    service_account_email: str | None = None
    network: str | None = None
    subnetwork: str | None = None
    no_external_ip: bool = False
    boot_disk_gb: int = 200
    scratch_disk_gb: int = 2000
    scratch_disk_type: str = "pd-balanced"


@dataclass(frozen=True)
class K8sOptions:
    """``[k8s]``: the Indexed Job and its identity."""

    #: ``aws`` (EKS: IRSA annotation, s3:// URLs), ``gcp`` (GKE: Workload
    #: Identity annotation, gs:// URLs) or ``generic`` (no annotation).
    cloud: str = "generic"
    namespace: str = "woof-energy"
    service_account: str = "woof-energy-worker"
    create_service_account: bool = True
    #: EKS: the IAM role ARN the service account assumes (IRSA).
    iam_role_arn: str | None = None
    #: GKE: the Google service account the KSA impersonates.
    gcp_service_account: str | None = None
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: tuple[dict, ...] = ()
    cpu: str | None = None
    memory_gib: int | None = None
    #: A node path for scratch (local NVMe); absent, an emptyDir.
    scratch_host_path: str | None = None
    scratch_size_gib: int = 1000
    shm_gib: int = 64
    ttl_seconds_after_finished: int = 7 * 24 * 3600


@dataclass(frozen=True)
class SlurmOptions:
    """``[slurm]``: the sbatch array script."""

    partition: str | None = None
    account: str | None = None
    #: ``HH:MM:SS`` or ``D-HH:MM:SS``; default from ``max_runtime_hours``.
    time_limit: str | None = None
    cpus_per_task: int | None = None
    #: ``apptainer`` (``docker://IMAGE`` or ``sif_path``), ``pyxis``
    #: (``srun --container-image``) or ``native`` (a pixi checkout).
    runtime: str = "apptainer"
    sif_path: str | None = None
    #: ``native``: the woof checkout whose pixi environment runs the worker.
    woof_root: str | None = None
    pixi_env: str = "default"
    #: Node-local scratch; shell expansions are allowed.
    #: Stable across jobs so a node stages WPS_GEOG once (.woof-synced).
    scratch_dir: str = "/tmp/woof-energy-${USER}"
    log_dir: str = "logs"
    exclusive: bool = True
    extra_directives: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeployConfig:
    """Everything a backend renderer needs; see the module docstring."""

    name: str
    image: str
    queue_url: str
    output_url: str
    workers: int = 1
    #: Workers running at once; default (and at most) all of them.
    parallelism: int | None = None
    #: GPUs each worker gets; default the machine's on AWS/GCP, required
    #: on k8s and Slurm.
    gpus_per_worker: int | None = None
    #: The worker's ``--gpus``: ``all`` or a count.
    worker_gpus: str = "all"
    #: The worker's ``--slots-per-gpu``: ``"1"`` (one job per card), a
    #: count, or ``"auto"``.
    slots_per_gpu: str = "1"
    #: The worker's ``--mps`` (CUDA MPS for packed slots).
    mps: bool = False
    exit_when_empty: bool = True
    worker_args: tuple[str, ...] = ()
    spot: bool = False
    #: WPS_GEOG tree the entrypoint syncs to local scratch at start.
    geog_url: str | None = None
    #: High-resolution static cache synced the same way.
    highres_cache_url: str | None = None
    max_runtime_hours: float = 72.0
    #: Retries per worker after the first attempt (never on a refusal).
    retries: int = 3
    env: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    aws_batch: AwsBatchOptions = field(default_factory=AwsBatchOptions)
    gcp_batch: GcpBatchOptions = field(default_factory=GcpBatchOptions)
    k8s: K8sOptions = field(default_factory=K8sOptions)
    slurm: SlurmOptions = field(default_factory=SlurmOptions)

    @property
    def effective_parallelism(self) -> int:
        """Workers at once: ``parallelism`` capped at ``workers``."""
        if self.parallelism is None:
            return self.workers
        return min(self.parallelism, self.workers)


_SECTIONS = {"aws_batch": AwsBatchOptions, "gcp_batch": GcpBatchOptions,
             "k8s": K8sOptions, "slurm": SlurmOptions}


def _coerce(value: Any, hint: Any, where: str) -> Any:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(value, inner, where)
    if hint is bool:
        if not isinstance(value, bool):
            raise DeployRefused(f"{where} must be true or false; got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise DeployRefused(f"{where} must be an integer; got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DeployRefused(f"{where} must be a number; got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise DeployRefused(f"{where} must be a string; got {value!r}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise DeployRefused(f"{where} must be a list; got {value!r}")
        return tuple(_coerce(item, args[0], f"{where}[{i}]")
                     for i, item in enumerate(value))
    if origin is dict:
        if not isinstance(value, Mapping):
            raise DeployRefused(f"{where} must be a table; got {value!r}")
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DeployRefused(f"{where} keys must be strings")
            out[key] = _coerce(item, args[1], f"{where}.{key}")
        return out
    if hint is dict:
        if not isinstance(value, Mapping):
            raise DeployRefused(f"{where} must be a table; got {value!r}")
        return dict(value)
    raise AssertionError(f"unhandled hint {hint!r} at {where}")


def _build(cls, mapping: Mapping[str, Any], where: str):
    if not isinstance(mapping, Mapping):
        raise DeployRefused(f"{where} must be a table")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise DeployRefused(
            f"{where}: unknown key(s) {', '.join(unknown)}; "
            f"known: {', '.join(sorted(known))}")
    kwargs = {}
    for name, value in mapping.items():
        if name in _SECTIONS and cls is DeployConfig:
            kwargs[name] = _build(_SECTIONS[name], value, f"[{name}]")
        else:
            label = name if where == "config" else f"{where}.{name}"
            kwargs[name] = _coerce(value, hints[name], label)
    try:
        return cls(**kwargs)
    except TypeError as error:
        missing = [f.name for f in fields(cls)
                   if f.name not in kwargs and f.default is MISSING
                   and f.default_factory is MISSING]
        raise DeployRefused(
            f"{where}: missing required key(s) {', '.join(missing)}") from error


def config_from_mapping(mapping: Mapping[str, Any]) -> DeployConfig:
    """A :class:`DeployConfig` from a parsed TOML/JSON table (refuses
    unknown keys and wrong types by name)."""

    return _build(DeployConfig, mapping, "config")


def load_config_mapping(path: Path) -> dict:
    """The raw table in a ``.toml`` or ``.json`` configuration file."""

    text = Path(path).read_text(encoding="utf-8")
    try:
        if Path(path).suffix.lower() == ".json":
            data = json.loads(text)
        else:
            import tomllib
            data = tomllib.loads(text)
    except ValueError as error:
        raise DeployRefused(f"{path}: not valid "
                            f"{'JSON' if path.suffix == '.json' else 'TOML'}: "
                            f"{error}") from error
    if not isinstance(data, dict):
        raise DeployRefused(f"{path}: the top level must be a table")
    return data


def config_to_dict(config: DeployConfig) -> dict:
    """The normalized configuration (tuples as lists, keys in field order)."""

    return json.loads(json.dumps(asdict(config)))


def config_sha256(config: DeployConfig) -> str:
    text = json.dumps(config_to_dict(config), sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Validation shared by every backend
# ---------------------------------------------------------------------------

def _scheme(url: str, what: str) -> str:
    match = re.match(r"^([a-z0-9]+)://(.*)$", url)
    if not match:
        raise DeployRefused(f"{what} must be a URL (s3://, gs:// or file://); "
                            f"got {url!r}")
    scheme, rest = match.groups()
    if scheme not in _SCHEMES:
        raise DeployRefused(f"{what}: scheme {scheme}:// is not one of "
                            f"{', '.join(s + '://' for s in _SCHEMES)}")
    if scheme in ("s3", "gs"):
        bucket = rest.split("/", 1)[0]
        if not bucket or "@" in bucket or ":" in bucket:
            raise DeployRefused(f"{what}: {url!r} needs a bucket name and no "
                                "credentials in the URL")
    elif not rest.startswith("/"):
        raise DeployRefused(f"{what}: a file:// URL needs an absolute path; "
                            f"got {url!r}")
    if any(c.isspace() for c in url):
        raise DeployRefused(f"{what}: {url!r} contains whitespace")
    return scheme


def _urls(config: DeployConfig) -> dict[str, str]:
    out = {"queue_url": config.queue_url, "output_url": config.output_url}
    if config.geog_url:
        out["geog_url"] = config.geog_url
    if config.highres_cache_url:
        out["highres_cache_url"] = config.highres_cache_url
    return out


def _check_image(image: str) -> None:
    if not image or any(c.isspace() for c in image):
        raise DeployRefused(f"image {image!r} is empty or contains whitespace")
    if "@sha256:" in image:
        return
    last = image.rsplit("/", 1)[-1]
    if ":" not in last:
        raise DeployRefused(
            f"image {image!r} has no tag or digest; pin one (e.g. "
            f"{image}:1.0.3-cu12) so every worker runs the same build")
    if last.rsplit(":", 1)[1] == "latest":
        raise DeployRefused(
            f"image {image!r} is tagged latest, which moves; pin a version "
            "tag or a digest so every worker runs the same build")


def _slots(config: DeployConfig) -> int | str | None:
    """``slots_per_gpu`` parsed as the worker parses it (None if invalid)."""

    from woof.energy.packing import parse_slots_per_gpu

    try:
        return parse_slots_per_gpu(config.slots_per_gpu)
    except ValueError:
        return None


def _check_common(config: DeployConfig) -> None:
    if not _NAME.match(config.name):
        raise DeployRefused(
            f"name {config.name!r} must be 1-40 characters of lower-case "
            "letters, digits and '-', starting with a letter (it names cloud "
            "resources with suffixes such as -queue)")
    _check_image(config.image)
    for what, url in _urls(config).items():
        _scheme(url, what)
    if config.workers < 1:
        raise DeployRefused(f"workers must be at least 1; got {config.workers}")
    if config.parallelism is not None and config.parallelism < 1:
        raise DeployRefused(
            f"parallelism must be at least 1; got {config.parallelism}")
    if config.gpus_per_worker is not None and config.gpus_per_worker < 1:
        raise DeployRefused("gpus_per_worker must be at least 1")
    if config.worker_gpus != "all" and not (
            config.worker_gpus.isdigit() and int(config.worker_gpus) >= 1):
        raise DeployRefused(f"worker_gpus must be 'all' or a positive count; "
                            f"got {config.worker_gpus!r}")
    if _slots(config) is None:
        raise DeployRefused(f"slots_per_gpu must be 'auto' or a positive "
                            f"count (as a string); got "
                            f"{config.slots_per_gpu!r}")
    if config.mps and _slots(config) == 1:
        raise DeployRefused(
            "mps = true shares a GPU between packed slots; set "
            "slots_per_gpu to a count above 1 or 'auto' (one slot per GPU "
            "has nothing to share)")
    if not (0.0 < config.max_runtime_hours <= 24.0 * 14):
        raise DeployRefused("max_runtime_hours must be in (0, 336]; got "
                            f"{config.max_runtime_hours}")
    if not (0 <= config.retries < AWS_MAX_ATTEMPTS):
        raise DeployRefused(f"retries must be 0..{AWS_MAX_ATTEMPTS - 1}; "
                            f"got {config.retries}")
    for arg in config.worker_args:
        flag = arg.split("=", 1)[0]
        if flag in ("--queue", "--gpus"):
            raise DeployRefused(
                f"worker_args may not carry {flag}: the renderer passes it "
                "from queue_url / worker_gpus")
        if flag in ("--slots-per-gpu", "--mps"):
            raise DeployRefused(
                f"worker_args may not carry {flag}: set slots_per_gpu / mps "
                "so every backend renders it")
        if flag == "--exit-when-empty":
            raise DeployRefused("worker_args may not carry --exit-when-empty; "
                                "set exit_when_empty = true instead")
        if "\n" in arg or "\0" in arg:
            raise DeployRefused(f"worker_args entry {arg!r} has a newline")
    for key, value in config.env.items():
        if not _ENV_NAME.match(key):
            raise DeployRefused(f"env name {key!r} is not a variable name")
        if key in RESERVED_ENV:
            raise DeployRefused(f"env may not set {key}: the renderer sets it "
                                "from the configuration")
        if _SECRET_NAME.search(key):
            raise DeployRefused(
                f"env {key} looks like a credential; no secrets go in "
                "deployment files -- workers use the instance role, VM "
                "service account or workload identity")
        if _SECRET_VALUE.search(value):
            raise DeployRefused(
                f"env {key} carries what looks like an access key or private "
                "key; no secrets go in deployment files")
        if "\n" in value or "\0" in value:
            raise DeployRefused(f"env {key} has a newline")
    for key, value in config.labels.items():
        if not _LABEL_KEY.match(key) or not _LABEL_VALUE.match(value):
            raise DeployRefused(
                f"label {key}={value!r}: keys and values must be lower-case "
                "letters, digits, '-' and '_', starting and ending with a "
                "letter or digit (keys start with a letter), 63 characters "
                "at most, so every cloud accepts them")


def _require_scheme(config: DeployConfig, scheme: str, backend: str) -> None:
    for what, url in _urls(config).items():
        got = _scheme(url, what)
        if got != scheme:
            raise DeployRefused(
                f"{backend} workers reach {scheme}:// with their ambient "
                f"identity; {what} is {url!r}.  Use a {scheme}:// URL (a "
                "cross-cloud copy would need a credential in the job, which "
                "these files never carry)")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

class _Placeholders:
    def __init__(self) -> None:
        self.names: list[str] = []

    def need(self, value, what: str):
        if value:
            return value
        token = PLACEHOLDER_PREFIX + what
        if token not in self.names:
            self.names.append(token)
        return token


def worker_command(config: DeployConfig) -> list[str]:
    """``woof energy worker --queue URL --gpus G [--exit-when-empty] ...``."""

    argv = ["woof", "energy", "worker", "--queue", config.queue_url,
            "--gpus", config.worker_gpus]
    if config.exit_when_empty:
        argv.append("--exit-when-empty")
    slots = _slots(config)
    if slots is not None and slots != 1:
        argv.extend(["--slots-per-gpu", str(slots)])
    if config.mps:
        argv.append("--mps")
    argv.extend(config.worker_args)
    return argv


def worker_env(config: DeployConfig, backend: str,
               scratch: str = CONTAINER_SCRATCH) -> dict[str, str]:
    """The variables every worker gets: URLs, scratch, identity of the
    deployment, then the configuration's own ``env`` (sorted)."""

    env = {
        "WOOF_DEPLOY_BACKEND": backend,
        "WOOF_DEPLOY_NAME": config.name,
        "WOOF_ENERGY_QUEUE_URL": config.queue_url,
        "WOOF_ENERGY_OUTPUT_URL": config.output_url,
        "WOOF_ENERGY_GPUS": config.worker_gpus,
        "WOOF_SCRATCH": scratch,
    }
    if config.geog_url:
        env["WOOF_GEOG_URL"] = config.geog_url
    if config.highres_cache_url:
        env["WOOF_HIGHRES_CACHE_URL"] = config.highres_cache_url
    for key in sorted(config.env):
        env[key] = config.env[key]
    return env


def _json(document: Any) -> str:
    return json.dumps(document, indent=2, ensure_ascii=True) + "\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    # A JSON string is a valid YAML double-quoted scalar.
    return json.dumps(str(value), ensure_ascii=True)


def _yaml(value: Any, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        if not value:
            return [pad + "{}"]
        for key, item in value.items():
            plain = (re.match(r"^[A-Za-z_][\w.\-/]*$", str(key))
                     and str(key).lower() not in _YAML_11_WORDS)
            k = str(key) if plain else _yaml_scalar(str(key))
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}{k}:")
                lines.extend(_yaml(item, indent + 1))
            else:
                empty = "{}" if isinstance(item, dict) else "[]"
                text = empty if isinstance(item, (dict, list)) else \
                    _yaml_scalar(item)
                lines.append(f"{pad}{k}: {text}")
        return lines
    if isinstance(value, list):
        if not value:
            return [pad + "[]"]
        for item in value:
            if isinstance(item, (dict, list)) and item:
                sub = _yaml(item, indent + 1)
                lines.append(f"{pad}- {sub[0].lstrip()}")
                lines.extend(sub[1:])
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return lines
    return [pad + _yaml_scalar(value)]


def yaml_documents(documents: list[dict]) -> str:
    """A deterministic multi-document YAML stream (strings double-quoted)."""

    return "".join("---\n" + "\n".join(_yaml(doc)) + "\n" for doc in documents)


def _quote(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


_PLACEHOLDER_GUARD = f"""\
here="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
if grep -rlE --exclude=deploy.json '{PLACEHOLDER_PREFIX}[A-Z0-9_]+' "$here" >/dev/null 2>&1; then
  echo "submit.sh: REFUSED: placeholders ({PLACEHOLDER_PREFIX}...) remain in:" >&2
  grep -rlE --exclude=deploy.json '{PLACEHOLDER_PREFIX}[A-Z0-9_]+' "$here" >&2
  echo "fill them in (or re-render with the values in the config) first" >&2
  exit 2
fi
"""


def _script(header: str, body: str) -> str:
    return ("#!/usr/bin/env bash\n"
            f"# {header}\n"
            "# Rendered by `woof energy deploy render`; carries no credentials.\n"
            "# It uses whatever identity the caller's CLI is already logged in as.\n"
            "set -euo pipefail\n" + _PLACEHOLDER_GUARD + body)


def _hours_seconds(hours: float) -> int:
    return max(60, int(round(hours * 3600)))


# ---------------------------------------------------------------------------
# AWS Batch
# ---------------------------------------------------------------------------

def _aws_shape(options: AwsBatchOptions, name: str) -> GpuMachine:
    if name in options.instance_shapes:
        raw = options.instance_shapes[name]
        try:
            return GpuMachine(int(raw["gpus"]), str(raw.get("gpu_model", "")),
                              int(raw["vcpus"]), float(raw["memory_gib"]))
        except (KeyError, TypeError, ValueError) as error:
            raise DeployRefused(
                f"[aws_batch.instance_shapes.{name}] needs gpus, vcpus and "
                f"memory_gib numbers") from error
    if name not in AWS_GPU_INSTANCES:
        raise DeployRefused(
            f"aws_batch instance type {name!r} is not a known GPU instance "
            f"({', '.join(sorted(AWS_GPU_INSTANCES))}); declare its shape in "
            f"[aws_batch.instance_shapes.\"{name}\"] if it is one")
    return AWS_GPU_INSTANCES[name]


SCRATCH_USERDATA = """\
#!/bin/bash
# Make /scratch writable for the container's non-root user (woof energy deploy).
set -euo pipefail
mkdir -p /scratch
chmod 1777 /scratch
"""

NVME_USERDATA = """\
#!/bin/bash
# RAID0 every instance-store NVMe drive to /scratch (woof energy deploy).
set -euo pipefail
mkdir -p /scratch
mapfile -t devices < <(lsblk -dpno NAME,MODEL | awk '/Instance Storage/ {print $1}')
if [ "${#devices[@]}" -gt 1 ]; then
  dnf install -y mdadm || yum install -y mdadm
  mdadm --create /dev/md/woof-scratch --level=0 --force --run \\
    --raid-devices="${#devices[@]}" "${devices[@]}"
  target=/dev/md/woof-scratch
elif [ "${#devices[@]}" -eq 1 ]; then
  target="${devices[0]}"
else
  target=""
fi
if [ -n "$target" ] && ! mountpoint -q /scratch; then
  mkfs.xfs -f "$target"
  mount -o noatime "$target" /scratch
fi
chmod 1777 /scratch
"""


def _mime_userdata(script: str) -> str:
    boundary = "==WOOFENERGYDEPLOY=="
    return (f'MIME-Version: 1.0\nContent-Type: multipart/mixed; '
            f'boundary="{boundary}"\n\n--{boundary}\n'
            'Content-Type: text/x-shellscript; charset="us-ascii"\n\n'
            f"{script}\n--{boundary}--\n")


def render_aws_batch(config: DeployConfig, ph: _Placeholders,
                     notes: list[str]) -> dict[str, str]:
    options = config.aws_batch
    _require_scheme(config, "s3", "aws-batch")
    if not options.instance_types:
        raise DeployRefused("aws_batch.instance_types is empty")
    shapes = {t: _aws_shape(options, t) for t in options.instance_types}
    gpus = config.gpus_per_worker or min(s.gpus for s in shapes.values())
    small = [t for t, s in shapes.items() if s.gpus < gpus]
    if small:
        raise DeployRefused(
            f"gpus_per_worker={gpus} does not fit on {', '.join(small)} "
            f"({', '.join(f'{t}: {shapes[t].gpus} GPU' for t in small)}); "
            "drop those types or lower gpus_per_worker")
    if config.workers > AWS_ARRAY_MAX:
        raise DeployRefused(f"an AWS Batch array holds at most {AWS_ARRAY_MAX} "
                            f"children; workers={config.workers}")
    # The job must fit on the smallest listed type; leave the ECS agent
    # ~10 % of memory.
    # A worker given g of an instance's G GPUs asks for g/G of its vCPUs
    # and memory, so G/g workers pack onto one instance.
    vcpus = min(max(1, s.vcpus * gpus // s.gpus) for s in shapes.values())
    memory_mib = min(int(s.memory_gib * 1024 * 0.9 * gpus / s.gpus)
                     for s in shapes.values())
    shm_mib = min(65536, memory_mib // 4)
    max_vcpus = options.max_vcpus or (
        config.effective_parallelism * max(s.vcpus for s in shapes.values()))
    region = ph.need(options.region, "AWS_REGION")
    labels = {"woof-deploy": config.name, **config.labels}

    ce_name, queue_name = f"{config.name}-ce", f"{config.name}-queue"
    jobdef_name, lt_name = f"{config.name}-worker", f"{config.name}-nvme"

    resources: dict[str, Any] = {
        "type": "SPOT" if config.spot else "EC2",
        "allocationStrategy": ("SPOT_PRICE_CAPACITY_OPTIMIZED" if config.spot
                               else "BEST_FIT_PROGRESSIVE"),
        "minvCpus": 0,
        "maxvCpus": max_vcpus,
        "instanceTypes": list(options.instance_types),
        "subnets": list(options.subnets) or [ph.need(None, "AWS_SUBNET_ID")],
        "securityGroupIds": (list(options.security_group_ids)
                             or [ph.need(None, "AWS_SECURITY_GROUP_ID")]),
        "instanceRole": ph.need(options.instance_role_arn,
                                "AWS_INSTANCE_PROFILE_ARN"),
        "ec2Configuration": [{"imageType": "ECS_AL2023_NVIDIA"}],
        "tags": labels,
    }
    resources["launchTemplate"] = {"launchTemplateName": lt_name,
                                   "version": "$Latest"}
    compute_env: dict[str, Any] = {
        "computeEnvironmentName": ce_name,
        "type": "MANAGED",
        "state": "ENABLED",
        "computeResources": resources,
    }
    if options.service_role_arn:
        compute_env["serviceRole"] = options.service_role_arn
    compute_env["tags"] = labels

    job_queue = {
        "jobQueueName": queue_name,
        "state": "ENABLED",
        "priority": 1,
        "computeEnvironmentOrder": [{"order": 1,
                                     "computeEnvironment": ce_name}],
        "tags": labels,
    }
    env = worker_env(config, "aws-batch")
    env["AWS_REGION"] = region
    container: dict[str, Any] = {
        "image": config.image,
        "command": worker_command(config),
        "jobRoleArn": ph.need(options.job_role_arn, "AWS_JOB_ROLE_ARN"),
        "resourceRequirements": [
            {"type": "GPU", "value": str(gpus)},
            {"type": "VCPU", "value": str(vcpus)},
            {"type": "MEMORY", "value": str(memory_mib)},
        ],
        "environment": [{"name": k, "value": v} for k, v in env.items()],
        "volumes": [{"name": "scratch",
                     "host": {"sourcePath": CONTAINER_SCRATCH}}],
        "mountPoints": [{"sourceVolume": "scratch",
                         "containerPath": CONTAINER_SCRATCH,
                         "readOnly": False}],
        "linuxParameters": {"sharedMemorySize": shm_mib},
        "ulimits": [
            {"name": "nofile", "softLimit": 1048576, "hardLimit": 1048576},
            {"name": "memlock", "softLimit": -1, "hardLimit": -1},
        ],
    }
    job_definition = {
        "jobDefinitionName": jobdef_name,
        "type": "container",
        "platformCapabilities": ["EC2"],
        "containerProperties": container,
        "retryStrategy": {
            "attempts": config.retries + 1,
            "evaluateOnExit": [
                {"onExitCode": str(REFUSAL_EXIT), "action": "EXIT"},
                {"onStatusReason": "Host EC2*", "action": "RETRY"},
            ],
        },
        "timeout": {"attemptDurationSeconds":
                    _hours_seconds(config.max_runtime_hours)},
        "propagateTags": True,
        "tags": labels,
    }
    submit_job: dict[str, Any] = {
        "jobName": config.name,
        "jobQueue": queue_name,
        "jobDefinition": jobdef_name,
    }
    if config.workers > 1:
        submit_job["arrayProperties"] = {"size": config.workers}
    submit_job["propagateTags"] = True
    submit_job["tags"] = labels

    files = {
        "compute-environment.json": _json(compute_env),
        "job-queue.json": _json(job_queue),
        "job-definition.json": _json(job_definition),
        "submit-job.json": _json(submit_job),
    }
    userdata = NVME_USERDATA if options.nvme_scratch else SCRATCH_USERDATA
    launch_template = {
        "LaunchTemplateName": lt_name,
        "LaunchTemplateData": {
            "BlockDeviceMappings": [{
                "DeviceName": "/dev/xvda",
                "Ebs": {"VolumeSize": options.root_volume_gib,
                        "VolumeType": "gp3", "Encrypted": True,
                        "DeleteOnTermination": True}}],
            "MetadataOptions": {"HttpTokens": "required",
                                "HttpPutResponseHopLimit": 2},
            "UserData": base64.b64encode(
                _mime_userdata(userdata).encode("ascii")).decode("ascii"),
        },
    }
    files["launch-template.json"] = _json(launch_template)
    files["scratch-userdata.sh"] = userdata
    lt_step = ('aws ec2 create-launch-template --region "$region" '
               '--cli-input-json "file://$here/launch-template.json"\n')
    if not options.nvme_scratch:
        notes.append("aws-batch: nvme_scratch=false; /scratch is a "
                     "directory on the instance's root volume")
    body = (
        f"region={shlex.quote(region)}\n"
        f"ce={shlex.quote(ce_name)}\n"
        + lt_step +
        'aws batch create-compute-environment --region "$region" '
        '--cli-input-json "file://$here/compute-environment.json"\n'
        'for attempt in $(seq 1 90); do\n'
        '  status="$(aws batch describe-compute-environments --region "$region" '
        '--compute-environments "$ce" '
        "--query 'computeEnvironments[0].[status,statusReason]' --output text)\"\n"
        '  case "$status" in\n'
        '    VALID*) break ;;\n'
        '    INVALID*) echo "submit.sh: $ce is INVALID: $status" >&2; exit 1 ;;\n'
        '  esac\n'
        '  if [ "$attempt" -eq 90 ]; then\n'
        '    echo "submit.sh: $ce not VALID after 15 minutes: $status" >&2; exit 1\n'
        '  fi\n'
        '  echo "waiting for $ce to become VALID ($status)" >&2; sleep 10\n'
        "done\n"
        'aws batch create-job-queue --region "$region" '
        '--cli-input-json "file://$here/job-queue.json"\n'
        'aws batch register-job-definition --region "$region" '
        '--cli-input-json "file://$here/job-definition.json"\n'
        'aws batch submit-job --region "$region" '
        '--cli-input-json "file://$here/submit-job.json"\n')
    files["submit.sh"] = _script(
        f"AWS Batch: {config.workers} woof energy worker(s) on "
        f"{', '.join(options.instance_types)}", body)
    return files


# ---------------------------------------------------------------------------
# GCP Batch
# ---------------------------------------------------------------------------

GCP_SCRATCH_MOUNT = "/mnt/disks/woof-scratch"


def render_gcp_batch(config: DeployConfig, ph: _Placeholders,
                     notes: list[str]) -> dict[str, str]:
    options = config.gcp_batch
    _require_scheme(config, "gs", "gcp-batch")
    machine = GCP_GPU_MACHINES.get(options.machine_type)
    if machine is None:
        raise DeployRefused(
            f"gcp_batch.machine_type {options.machine_type!r} is not a known "
            f"GPU machine type ({', '.join(sorted(GCP_GPU_MACHINES))})")
    gpus = config.gpus_per_worker or machine.gpus
    if gpus != machine.gpus:
        raise DeployRefused(
            f"{options.machine_type} carries {machine.gpus} GPU(s) and runs "
            f"one worker per VM; gpus_per_worker={gpus} would leave GPUs idle "
            "or not fit.  Pick the machine type with that many GPUs")
    if options.scratch_disk_gb < 10 or options.boot_disk_gb < 30:
        raise DeployRefused("gcp_batch.scratch_disk_gb must be >= 10 and "
                            "boot_disk_gb >= 30")
    region = ph.need(options.region, "GCP_REGION")
    project = ph.need(options.project, "GCP_PROJECT")
    env = worker_env(config, "gcp-batch")
    env["GOOGLE_CLOUD_PROJECT"] = project
    labels = {"woof-deploy": config.name, **config.labels}
    shm_gb = max(1, min(64, int(machine.memory_gib // 4)))
    task_spec = {
        "runnables": [
            {"displayName": "scratch-permissions",
             "script": {"text": f"mkdir -p {GCP_SCRATCH_MOUNT} && "
                                f"chmod 1777 {GCP_SCRATCH_MOUNT}"}},
            {"displayName": "woof-energy-worker",
             "container": {
                 "imageUri": config.image,
                 "commands": worker_command(config),
                 "volumes": [f"{GCP_SCRATCH_MOUNT}:{CONTAINER_SCRATCH}"],
                 "options": (f"--shm-size={shm_gb}g --ulimit memlock=-1 "
                             "--ulimit nofile=1048576:1048576"),
             }},
        ],
        "computeResource": {
            "cpuMilli": max(1000, (machine.vcpus - 1) * 1000),
            "memoryMib": int(machine.memory_gib * 1024 * 0.9),
        },
        "maxRetryCount": config.retries,
        "maxRunDuration": f"{_hours_seconds(config.max_runtime_hours)}s",
        "lifecyclePolicies": [{
            "action": "FAIL_TASK",
            "actionCondition": {"exitCodes": [REFUSAL_EXIT]},
        }],
        "environment": {"variables": env},
        "volumes": [{"deviceName": "woof-scratch",
                     "mountPath": GCP_SCRATCH_MOUNT,
                     "mountOptions": ["rw", "async"]}],
    }
    policy = {
        "machineType": options.machine_type,
        "provisioningModel": "SPOT" if config.spot else "STANDARD",
        "accelerators": [{"type": machine.accelerator, "count": machine.gpus}],
        "bootDisk": {"type": "pd-balanced", "sizeGb": options.boot_disk_gb},
        "disks": [{"deviceName": "woof-scratch",
                   "newDisk": {"type": options.scratch_disk_type,
                               "sizeGb": options.scratch_disk_gb}}],
    }
    allocation: dict[str, Any] = {
        "location": {"allowedLocations": [f"regions/{region}"]},
        "instances": [{"installGpuDrivers": True, "policy": policy}],
        "serviceAccount": {"email": ph.need(options.service_account_email,
                                            "GCP_SERVICE_ACCOUNT_EMAIL")},
        "labels": labels,
    }
    if options.network or options.subnetwork or options.no_external_ip:
        interface: dict[str, Any] = {}
        if options.network:
            interface["network"] = options.network
        if options.subnetwork:
            interface["subnetwork"] = options.subnetwork
        interface["noExternalIpAddress"] = options.no_external_ip
        allocation["network"] = {"networkInterfaces": [interface]}
    job = {
        "taskGroups": [{
            "taskSpec": task_spec,
            "taskCount": config.workers,
            "parallelism": config.effective_parallelism,
            "taskCountPerNode": 1,
        }],
        "allocationPolicy": allocation,
        "labels": labels,
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }
    body = (
        f"project={shlex.quote(project)}\n"
        f"region={shlex.quote(region)}\n"
        f"gcloud batch jobs submit {shlex.quote(config.name)} "
        '--project "$project" --location "$region" '
        '--config "$here/job.json"\n')
    return {
        "job.json": _json(job),
        "submit.sh": _script(
            f"GCP Batch: {config.workers} woof energy worker(s) on "
            f"{options.machine_type}", body),
    }


# ---------------------------------------------------------------------------
# Kubernetes
# ---------------------------------------------------------------------------

def _explicit_gpus(config: DeployConfig, backend: str) -> int:
    if config.gpus_per_worker is None:
        raise DeployRefused(
            f"{backend} needs gpus_per_worker: the number of GPUs each "
            "worker is given (on a dedicated GPU node, its GPU count).  "
            "There is no machine table to read it from")
    return config.gpus_per_worker


def render_k8s(config: DeployConfig, ph: _Placeholders,
               notes: list[str]) -> dict[str, str]:
    options = config.k8s
    if options.cloud not in ("aws", "gcp", "generic"):
        raise DeployRefused(f"k8s.cloud must be aws, gcp or generic; got "
                            f"{options.cloud!r}")
    if options.cloud in _CLOUD_SCHEME:
        _require_scheme(config, _CLOUD_SCHEME[options.cloud],
                        f"k8s ({options.cloud})")
    else:
        for what, url in _urls(config).items():
            if _scheme(url, what) == "file":
                raise DeployRefused(
                    f"k8s: {what} is {url!r}; a file:// URL is only "
                    "reachable on one node's disk.  Use s3:// or gs://")
    for label, value in (("namespace", options.namespace),
                         ("service_account", options.service_account)):
        if not re.match(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$", value):
            raise DeployRefused(f"k8s.{label} {value!r} is not a DNS-1123 "
                                "label")
    if config.spot and options.cloud == "generic":
        raise DeployRefused(
            "spot=true with k8s.cloud=generic: the spot node label differs "
            "per provider.  Set k8s.cloud to aws or gcp, or put your "
            "cluster's spot label in k8s.node_selector and spot=false")
    gpus = _explicit_gpus(config, "k8s")
    labels = {"app.kubernetes.io/name": "woof-energy-worker",
              "app.kubernetes.io/instance": config.name,
              **config.labels}
    documents: list[dict] = []
    if options.create_service_account:
        annotations: dict[str, str] = {}
        if options.cloud == "aws":
            annotations["eks.amazonaws.com/role-arn"] = ph.need(
                options.iam_role_arn, "AWS_IRSA_ROLE_ARN")
        elif options.cloud == "gcp":
            annotations["iam.gke.io/gcp-service-account"] = ph.need(
                options.gcp_service_account, "GCP_SERVICE_ACCOUNT_EMAIL")
        sa_meta: dict[str, Any] = {"name": options.service_account,
                                   "namespace": options.namespace,
                                   "labels": labels}
        if annotations:
            sa_meta["annotations"] = annotations
        documents.append({"apiVersion": "v1", "kind": "ServiceAccount",
                          "metadata": sa_meta})
    elif options.iam_role_arn or options.gcp_service_account:
        notes.append("k8s: create_service_account=false, so the workload "
                     "identity in the config is not written; annotate "
                     f"{options.service_account} yourself")

    node_selector = dict(sorted(options.node_selector.items()))
    tolerations = [{"key": "nvidia.com/gpu", "operator": "Exists",
                    "effect": "NoSchedule"}]
    if config.spot and options.cloud == "gcp":
        node_selector.setdefault("cloud.google.com/gke-spot", "true")
        tolerations.append({"key": "cloud.google.com/gke-spot",
                            "operator": "Equal", "value": "true",
                            "effect": "NoSchedule"})
    elif config.spot and options.cloud == "aws":
        node_selector.setdefault("eks.amazonaws.com/capacityType", "SPOT")
    tolerations.extend(dict(t) for t in options.tolerations)

    limits: dict[str, Any] = {"nvidia.com/gpu": str(gpus)}
    requests: dict[str, Any] = {"nvidia.com/gpu": str(gpus)}
    if options.cpu:
        limits["cpu"] = requests["cpu"] = options.cpu
    if options.memory_gib:
        limits["memory"] = requests["memory"] = f"{options.memory_gib}Gi"
    if options.scratch_host_path:
        scratch_volume = {"name": "scratch",
                          "hostPath": {"path": options.scratch_host_path,
                                       "type": "DirectoryOrCreate"}}
    else:
        scratch_volume = {"name": "scratch",
                          "emptyDir": {"sizeLimit":
                                       f"{options.scratch_size_gib}Gi"}}
    env = worker_env(config, "k8s")
    container = {
        "name": "worker",
        "image": config.image,
        "imagePullPolicy": "IfNotPresent",
        "args": worker_command(config),
        "env": [{"name": k, "value": v} for k, v in env.items()],
        "resources": {"limits": limits, "requests": requests},
        "volumeMounts": [{"name": "scratch", "mountPath": CONTAINER_SCRATCH},
                         {"name": "dshm", "mountPath": "/dev/shm"}],
        "securityContext": {"allowPrivilegeEscalation": False,
                            "capabilities": {"drop": ["ALL"]}},
    }
    pod_spec: dict[str, Any] = {
        "serviceAccountName": options.service_account,
        "restartPolicy": "Never",
        "activeDeadlineSeconds": _hours_seconds(config.max_runtime_hours),
        "terminationGracePeriodSeconds": 120,
    }
    if node_selector:
        pod_spec["nodeSelector"] = node_selector
    pod_spec["tolerations"] = tolerations
    pod_spec["securityContext"] = {"runAsNonRoot": True,
                                   "runAsUser": CONTAINER_UID,
                                   "runAsGroup": CONTAINER_UID,
                                   "fsGroup": CONTAINER_UID}
    if options.scratch_host_path:
        # fsGroup does not apply to hostPath: open the node directory to
        # the non-root worker before it starts.
        pod_spec["initContainers"] = [{
            "name": "scratch-permissions",
            "image": config.image,
            "command": ["/bin/sh", "-c", f"chmod 1777 {CONTAINER_SCRATCH}"],
            "volumeMounts": [{"name": "scratch",
                              "mountPath": CONTAINER_SCRATCH}],
            "securityContext": {"runAsUser": 0, "runAsNonRoot": False,
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"],
                                                 "add": ["FOWNER"]}},
        }]
    pod_spec["containers"] = [container]
    pod_spec["volumes"] = [scratch_volume,
                           {"name": "dshm",
                            "emptyDir": {"medium": "Memory",
                                         "sizeLimit": f"{options.shm_gib}Gi"}}]
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": config.name, "namespace": options.namespace,
                     "labels": labels},
        "spec": {
            "completionMode": "Indexed",
            "completions": config.workers,
            "parallelism": config.effective_parallelism,
            "backoffLimitPerIndex": config.retries,
            "maxFailedIndexes": config.workers,
            "podFailurePolicy": {"rules": [
                {"action": "FailIndex",
                 "onExitCodes": {"containerName": "worker",
                                 "operator": "In",
                                 "values": [REFUSAL_EXIT]}},
                {"action": "Ignore",
                 "onPodConditions": [{"type": "DisruptionTarget"}]},
            ]},
            "ttlSecondsAfterFinished": options.ttl_seconds_after_finished,
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }
    documents.append(job)
    files: dict[str, str] = {}
    if config.mps:
        files["mps-sharing.example.yaml"] = _k8s_mps_example(config, gpus)
        notes.append(
            "k8s: mps = true starts an MPS control daemon inside each worker "
            "pod (the image needs the driver's nvidia-cuda-mps-control; the "
            "worker refuses without it).  If the cluster's NVIDIA device "
            "plugin owns MPS instead, apply its sharing configuration "
            "(mps-sharing.example.yaml, commented) and the worker uses that "
            "daemon through CUDA_MPS_PIPE_DIRECTORY")
    body = (
        f"namespace={shlex.quote(options.namespace)}\n"
        'kubectl create namespace "$namespace" --dry-run=client -o yaml '
        "| kubectl apply -f -\n"
        'kubectl apply -f "$here/job.yaml"\n')
    files.update({
        "job.yaml": yaml_documents(documents),
        "submit.sh": _script(
            f"Kubernetes: Indexed Job of {config.workers} woof energy "
            "worker(s)", body),
    })
    return files


def _k8s_mps_example(config: DeployConfig, gpus: int) -> str:
    """A commented NVIDIA device plugin MPS sharing configuration."""

    from woof.energy.packing import AUTO_MAX_SLOTS_PER_GPU

    slots = _slots(config)
    replicas = AUTO_MAX_SLOTS_PER_GPU if slots == "auto" else int(slots)
    lines = [
        "# NVIDIA device plugin: CUDA MPS sharing (EXAMPLE, every line "
        "commented).",
        "#",
        "# The worker command in job.yaml carries --mps, so each pod starts "
        "its own",
        "# MPS control daemon for the GPU(s) it was given and packs "
        f"--slots-per-gpu {config.slots_per_gpu} jobs on them.  That needs "
        "the driver's",
        "# nvidia-cuda-mps-control in the container and a GPU not in "
        "EXCLUSIVE_PROCESS",
        "# mode held by another daemon.",
        "#",
        "# On clusters where the device plugin (k8s-device-plugin >= 0.15 or "
        "the GPU",
        "# Operator) owns MPS, the plugin runs the control daemon per node "
        "and splits",
        "# each GPU into replicas with a memory and SM-share limit each.  "
        "Configure it",
        "# like this, then give each worker one replica per slot (or run "
        "--slots-per-gpu 1",
        "# workers, one per replica); the worker finds the plugin's daemon "
        "through",
        "# CUDA_MPS_PIPE_DIRECTORY and neither starts nor stops it.  See "
        "the plugin's",
        "# README, 'Shared Access to GPUs' (sharing.mps).",
        "#",
        "# apiVersion: v1",
        "# kind: ConfigMap",
        "# metadata:",
        "#   name: nvidia-device-plugin-config",
        "#   namespace: nvidia-device-plugin   # or gpu-operator",
        "# data:",
        "#   mps-sharing: |-",
        "#     version: v1",
        "#     sharing:",
        "#       mps:",
        "#         renameByDefault: false",
        "#         resources:",
        "#         - name: nvidia.com/gpu",
        f"#           replicas: {replicas}"
        + ("   # auto starts at most this many slots per GPU; match "
           "the slot count you run" if slots == "auto" else ""),
        "#",
        "# Select it per node (GPU Operator: label the nodes",
        "# nvidia.com/device-plugin.config=mps-sharing) and request",
        f"# nvidia.com/gpu: {gpus * replicas} per pod for {gpus} GPU(s) x "
        f"{replicas} replicas.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Slurm
# ---------------------------------------------------------------------------

_SLURM_TIME = re.compile(r"^(\d+-)?\d{1,3}:\d{2}:\d{2}$")
_UNSAFE_SHELL_WORD = re.compile(r'["`\\\n]|\$\(')


def _slurm_time(hours: float) -> str:
    minutes = int(math.ceil(hours * 60))
    days, rem = divmod(minutes, 24 * 60)
    text = f"{rem // 60:02d}:{rem % 60:02d}:00"
    return f"{days}-{text}" if days else text


def _pyxis_image(image: str) -> str:
    """``registry/repo:tag`` -> ``registry#repo:tag`` (enroot's spelling)."""

    first, sep, rest = image.partition("/")
    if sep and ("." in first or ":" in first or first == "localhost"):
        return f"{first}#{rest}"
    return image


def render_slurm(config: DeployConfig, ph: _Placeholders,
                 notes: list[str]) -> dict[str, str]:
    options = config.slurm
    if config.spot:
        notes.append("slurm: spot=true is not rendered; the partition "
                     "decides capacity (pick a spot partition if the "
                     "cluster has one)")
    if options.runtime not in ("apptainer", "pyxis", "native"):
        raise DeployRefused(f"slurm.runtime must be apptainer, pyxis or "
                            f"native; got {options.runtime!r}")
    time_limit = options.time_limit or _slurm_time(config.max_runtime_hours)
    if not _SLURM_TIME.match(time_limit):
        raise DeployRefused(f"slurm.time_limit {time_limit!r} is not "
                            "HH:MM:SS or D-HH:MM:SS")
    for label, value in (("scratch_dir", options.scratch_dir),
                         ("log_dir", options.log_dir),
                         ("woof_root", options.woof_root or ""),
                         ("sif_path", options.sif_path or "")):
        if _UNSAFE_SHELL_WORD.search(value):
            raise DeployRefused(f"slurm.{label} {value!r} may not contain a "
                                'double quote, backtick, backslash or $( ')
    for directive in options.extra_directives:
        if "\n" in directive or not directive.startswith("--"):
            raise DeployRefused(f"slurm.extra_directives entry {directive!r} "
                                "must be one --option[=value]")
    gpus = _explicit_gpus(config, "slurm")
    if config.workers > SLURM_DEFAULT_MAX_ARRAY:
        notes.append(f"slurm: an array of {config.workers} exceeds Slurm's "
                     f"default MaxArraySize ({SLURM_DEFAULT_MAX_ARRAY}); the "
                     "cluster must allow it")
    directives = [
        f"--job-name={config.name}",
        f"--array=0-{config.workers - 1}%{config.effective_parallelism}",
        "--nodes=1",
        "--ntasks=1",
        f"--gres=gpu:{gpus}",
    ]
    if options.cpus_per_task:
        directives.append(f"--cpus-per-task={options.cpus_per_task}")
    directives += ["--mem=0", f"--time={time_limit}"]
    if options.exclusive:
        directives.append("--exclusive")
    if options.partition:
        directives.append(f"--partition={options.partition}")
    if options.account:
        directives.append(f"--account={options.account}")
    directives += [f"--output={options.log_dir}/%x-%A_%a.out",
                   "--requeue", "--signal=B:TERM@300"]
    directives += list(options.extra_directives)

    env = worker_env(config, "slurm")
    env.pop("WOOF_SCRATCH")
    exports = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
    argv = worker_command(config)
    if options.runtime == "apptainer":
        image = options.sif_path or f"docker://{config.image}"
        if not options.sif_path:
            notes.append(
                "slurm: every array task converts docker://"
                f"{config.image} itself; build one SIF on shared storage "
                "(apptainer pull) and set slurm.sif_path")
        launch = (
            'export WOOF_SCRATCH=/scratch\n'
            f'exec apptainer exec --nv --bind "$scratch_host:/scratch" '
            f'"{image}" {CONTAINER_ENTRYPOINT} {_quote(argv)}\n')
    elif options.runtime == "pyxis":
        launch = (
            'export WOOF_SCRATCH=/scratch\n'
            f'exec srun --ntasks=1 --container-image={shlex.quote(_pyxis_image(config.image))} '
            '--container-mounts="$scratch_host:/scratch" '
            f'{CONTAINER_ENTRYPOINT} {_quote(argv)}\n')
    else:
        root = ph.need(options.woof_root, "WOOF_ROOT")
        launch = (
            'export WOOF_SCRATCH="$scratch_host"\n'
            f'exec pixi run --manifest-path "{root}/pyproject.toml" '
            f'-e {shlex.quote(options.pixi_env)} '
            f'bash "{root}/deploy/entrypoint.sh" {_quote(argv)}\n')
    script = (
        "#!/bin/bash\n"
        + "".join(f"#SBATCH {d}\n" for d in directives)
        + "# Rendered by `woof energy deploy render`; carries no credentials.\n"
          "# Workers reach the object store with the nodes' own identity\n"
          "# (instance role / VM service account).\n"
          "set -euo pipefail\n"
        + exports
        + f'scratch_host="{options.scratch_dir}"\n'
          'mkdir -p "$scratch_host"\n'
          'echo "woof energy worker: array task ${SLURM_ARRAY_TASK_ID:-0} on '
          '$(hostname), scratch $scratch_host" >&2\n'
        + launch)
    body = (f'mkdir -p "$here/{options.log_dir}"\n'
            if not options.log_dir.startswith("/") else
            f'mkdir -p "{options.log_dir}"\n')
    body += 'cd "$here"\nsbatch "$here/worker.sbatch"\n'
    return {
        "worker.sbatch": script,
        "submit.sh": _script(
            f"Slurm: array of {config.workers} woof energy worker(s)", body),
    }


_RENDERERS = {"aws-batch": render_aws_batch, "gcp-batch": render_gcp_batch,
              "k8s": render_k8s, "slurm": render_slurm}
assert tuple(_RENDERERS) == BACKENDS


# ---------------------------------------------------------------------------
# Rendering and writing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rendered:
    backend: str
    config: DeployConfig
    files: dict[str, str]
    placeholders: tuple[str, ...]
    notes: tuple[str, ...]

    def manifest(self) -> dict:
        from woof import __version__ as woof_version

        return {
            "schema": DEPLOY_SCHEMA,
            "backend": self.backend,
            "name": self.config.name,
            "woof_version": woof_version,
            "config_sha256": config_sha256(self.config),
            "worker_command": worker_command(self.config),
            "files": {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                      for name, text in sorted(self.files.items())},
            "placeholders": list(self.placeholders),
            "notes": list(self.notes),
            "config": config_to_dict(self.config),
        }


def render(config: DeployConfig, backend: str, *,
           strict: bool = False) -> Rendered:
    """Validate ``config`` and render ``backend``'s files in memory."""

    if backend not in _RENDERERS:
        raise DeployRefused(f"unknown backend {backend!r}; one of "
                            f"{', '.join(BACKENDS)}")
    _check_common(config)
    ph = _Placeholders()
    notes: list[str] = []
    files = _RENDERERS[backend](config, ph, notes)
    if strict and ph.names:
        raise DeployRefused(
            "--strict: these values are missing from the configuration and "
            f"would be written as placeholders: {', '.join(ph.names)}")
    return Rendered(backend, config, files, tuple(ph.names), tuple(notes))


_EXECUTABLE = (".sh", ".sbatch")


def write_rendered(rendered: Rendered, out_dir: Path) -> dict:
    """Write every file and ``deploy.json`` atomically; return the manifest."""

    from woof.fetch_guard import atomic_write_text

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = out_dir / "deploy.json"
    if previous.is_file():
        try:
            old = json.loads(previous.read_text(encoding="utf-8"))
        except ValueError:
            old = {}
        if not isinstance(old, dict) or old.get("schema") != DEPLOY_SCHEMA:
            raise DeployRefused(
                f"{previous} is not a {DEPLOY_SCHEMA} manifest; render into "
                "an empty directory or one this command wrote")
        # Files the earlier render wrote and this one does not are removed,
        # so the directory (and submit.sh's placeholder guard) holds one
        # deployment.
        for name in old.get("files", {}):
            if name not in rendered.files and "/" not in name:
                (out_dir / name).unlink(missing_ok=True)
    for name, text in sorted(rendered.files.items()):
        path = out_dir / name
        atomic_write_text(path, text)
        if name.endswith(_EXECUTABLE):
            path.chmod(0o755)
    manifest = rendered.manifest()
    atomic_write_text(out_dir / "deploy.json", _json(manifest))
    return manifest


def _overrides(args) -> dict:
    out: dict[str, Any] = {}
    for attribute, key in (("name", "name"), ("image", "image"),
                           ("queue", "queue_url"),
                           ("output_url", "output_url"),
                           ("workers", "workers"),
                           ("parallelism", "parallelism"),
                           ("gpus_per_worker", "gpus_per_worker"),
                           ("slots_per_gpu", "slots_per_gpu"),
                           ("geog_url", "geog_url"),
                           ("highres_cache_url", "highres_cache_url")):
        value = getattr(args, attribute, None)
        if value is not None:
            out[key] = value
    if getattr(args, "spot", None) is not None:
        out["spot"] = bool(args.spot)
    if getattr(args, "mps", None) is not None:
        out["mps"] = bool(args.mps)
    return out


def main(args) -> int:
    try:
        mapping = (load_config_mapping(Path(args.config))
                   if args.config else {})
        mapping.update(_overrides(args))
        config = config_from_mapping(mapping)
        rendered = render(config, args.backend, strict=bool(args.strict))
        manifest = write_rendered(rendered, Path(args.output))
    except DeployRefused as error:
        print(f"woof energy deploy render: refused: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"woof energy deploy render: failed: {error}", file=sys.stderr)
        return 1
    for note in rendered.notes:
        print(f"woof energy deploy render: note: {note}", file=sys.stderr)
    if rendered.placeholders:
        print("woof energy deploy render: note: fill in "
              f"{', '.join(rendered.placeholders)} before running submit.sh "
              "(it refuses while any remain)", file=sys.stderr)
    summary = {
        "command": "woof energy deploy render",
        "schema": DEPLOY_SCHEMA,
        "backend": rendered.backend,
        "output": str(args.output),
        "files": sorted([*rendered.files, "deploy.json"]),
        "workers": config.workers,
        "parallelism": config.effective_parallelism,
        "worker_command": manifest["worker_command"],
        "placeholders": manifest["placeholders"],
        "config_sha256": manifest["config_sha256"],
    }
    print(json.dumps(summary, indent=2))
    return 0


__all__ = [
    "AWS_GPU_INSTANCES", "AwsBatchOptions", "BACKENDS", "DEPLOY_SCHEMA",
    "DeployConfig", "DeployRefused", "GCP_GPU_MACHINES", "GcpBatchOptions",
    "GpuMachine", "K8sOptions", "PLACEHOLDER_PREFIX", "Rendered",
    "SlurmOptions", "config_from_mapping", "config_sha256", "config_to_dict",
    "load_config_mapping", "main", "render", "worker_command", "worker_env",
    "write_rendered", "yaml_documents",
]
