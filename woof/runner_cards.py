"""What a runner IS: its card, its host, and where each fact came from.

``woof estimate`` prices a run for a RUNNER -- a GPU (or an Apple
Silicon GPU), the host around it and how many runs share it -- and this
module is the half that says what that runner is.  Three sources, in
the order a caller layers them:

1. :data:`RUNNER_CARDS`, the built-in card table.  Every figure is a
   vendor nameplate number with its source named on the row: memory,
   memory bandwidth, SM (or GPU-core) count and FP32 throughput, and
   FP64 only where the vendor's datasheet states it.  The exceptions
   are this project's own readings, and each row that carries one says
   so: memory on the cards this project has run on is what
   ``nvidia-smi`` reported there, and some capacity and SM figures are
   the ones ``woof/hex/device_admission.py`` ``KNOWN_CARDS`` measured.
   No TIMING is in this table; the timing measurements live in
   :mod:`woof.runtime_estimate` and name their card.
2. :func:`detect_local_runner`, which reads THIS box without creating a
   CUDA context: ``nvidia-smi`` for an NVIDIA card (name, total memory,
   PCIe generation and width), ``sysctl`` and ``system_profiler`` for
   an Apple chip, and the host's RAM and CPU count.  The card name is
   matched to a table row; a card with no row keeps the memory it
   reported and NO bandwidth, and the estimator then says it is pricing
   against a substitute.
3. A ``[runner]`` TOML (:func:`load_runner_toml`), whose keys override
   both.  It is the way to price a box you are not sitting at, or to
   correct a row.

The FP64 column is informational: the model state is FP32 throughout
(``docs/public/HARDWARE.md``), so the time model never reads it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

MIB = 1 << 20
GIB = 1 << 30

CUDA = "cuda"
METAL = "metal"
BACKENDS = (CUDA, METAL)

_NV = "NVIDIA datasheet"
_APPLE = "Apple newsroom / tech specs"


@dataclass(frozen=True)
class RunnerCard:
    """One GPU model's nameplate.  ``memory_mib`` is ``None`` on Apple
    chips, whose GPU shares the host's unified memory (the runner reads
    it from the host)."""

    key: str
    name: str
    backend: str
    memory_mib: int | None
    bandwidth_gbs: float
    sm_count: int | None
    fp32_tflops: float | None
    fp64_tflops: float | None
    pcie: str | None
    source: str
    aliases: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["aliases"] = list(self.aliases)
        return out


def _card(key, name, backend, memory_mib, bandwidth_gbs, sm_count,
          fp32_tflops, fp64_tflops, pcie, source, *aliases) -> RunnerCard:
    return RunnerCard(key, name, backend, memory_mib, float(bandwidth_gbs),
                      sm_count, fp32_tflops, fp64_tflops, pcie, source,
                      tuple(aliases))


#: The built-in card table.  ``sm_count`` is streaming multiprocessors on
#: NVIDIA and GPU cores on Apple.  ``memory_mib`` on a card this project
#: has run on is what nvidia-smi reported there (the row says which).
RUNNER_CARDS: Mapping[str, RunnerCard] = MappingProxyType({
    card.key: card for card in (
        # -- workstation / consumer NVIDIA --------------------------------
        _card("rtx-a6000", "NVIDIA RTX A6000 48GB", CUDA, 49_140, 768.0, 84,
              38.7, None, "PCIe 4.0 x16",
              f"{_NV} (RTX A6000); 49,140 MiB as nvidia-smi reports it on "
              "this project's pb host", "a6000", "rtxa6000"),
        _card("rtx-3080", "NVIDIA GeForce RTX 3080 10GB", CUDA, 10_240,
              760.0, 68, 29.8, None, "PCIe 4.0 x16",
              f"{_NV} (GeForce RTX 3080); capacity and SMs as "
              "woof/hex/device_admission.py KNOWN_CARDS measured", "3080"),
        _card("rtx-3090", "NVIDIA GeForce RTX 3090 24GB", CUDA, 24_576,
              936.0, 82, 35.6, None, "PCIe 4.0 x16",
              f"{_NV} (GeForce RTX 3090)", "3090"),
        _card("rtx-4080", "NVIDIA GeForce RTX 4080 16GB", CUDA, 16_376,
              716.8, 76, 48.7, None, "PCIe 4.0 x16",
              f"{_NV} (GeForce RTX 4080)", "4080"),
        _card("rtx-4090", "NVIDIA GeForce RTX 4090 24GB", CUDA, 24_564,
              1008.0, 128, 82.6, None, "PCIe 4.0 x16",
              f"{_NV} (GeForce RTX 4090); capacity and SMs as "
              "woof/hex/device_admission.py KNOWN_CARDS measured", "4090"),
        _card("rtx-5070-ti", "NVIDIA GeForce RTX 5070 Ti 16GB", CUDA, 16_303,
              896.0, 70, 43.9, None, "PCIe 5.0 x16",
              f"{_NV} (GeForce RTX 5070 Ti); capacity and SMs as "
              "woof/hex/device_admission.py KNOWN_CARDS measured",
              "5070ti"),
        _card("rtx-5090", "NVIDIA GeForce RTX 5090 32GB", CUDA, 32_607,
              1792.0, 170, 104.8, None, "PCIe 5.0 x16",
              f"{_NV} (GeForce RTX 5090); capacity and SMs as "
              "woof/hex/device_admission.py KNOWN_CARDS measured", "5090"),
        _card("rtx-pro-6000", "NVIDIA RTX PRO 6000 Blackwell 96GB", CUDA,
              97_887, 1792.0, 188, None, None, "PCIe 5.0 x16",
              f"{_NV} (RTX PRO 6000 Blackwell); capacity and SMs as "
              "woof/hex/device_admission.py KNOWN_CARDS measured",
              "rtxpro6000", "pro6000", "rtxpro6000blackwell",
              "rtxpro6000blackwellserveredition"),
        # -- datacentre NVIDIA (capacities as woof.card_catalog) ----------
        _card("l4", "NVIDIA L4 24GB", CUDA, 23_034, 300.0, 58, 30.3, None,
              "PCIe 4.0 x16", f"{_NV} (L4)", "l424gb"),
        _card("l40s", "NVIDIA L40S 48GB", CUDA, 46_068, 864.0, 142, 91.6,
              None, "PCIe 4.0 x16", f"{_NV} (L40S)", "l40s48gb"),
        _card("a100-40gb", "NVIDIA A100 SXM4 40GB", CUDA, 40_960, 1555.0, 108,
              19.5, 9.7, "PCIe 4.0 host link", f"{_NV} (A100)",
              "a100", "a100sxm440gb", "a10040gb"),
        _card("a100-80gb", "NVIDIA A100 SXM4 80GB", CUDA, 81_920, 2039.0, 108,
              19.5, 9.7, "PCIe 4.0 host link", f"{_NV} (A100)",
              "a10080gb", "a100sxm480gb", "a100sxm4"),
        _card("h100-pcie", "NVIDIA H100 PCIe 80GB", CUDA, 81_559, 2000.0, 114,
              51.0, 26.0, "PCIe 5.0 x16", f"{_NV} (H100)",
              "h100", "h100pcie", "h100pcie80gb"),
        _card("h100-sxm", "NVIDIA H100 SXM5 80GB", CUDA, 81_559, 3350.0, 132,
              67.0, 34.0, "PCIe 5.0 host link", f"{_NV} (H100)",
              "h100sxm", "h100sxm5", "h10080gbhbm3", "h100sxm580gb"),
        # -- Apple Silicon (unified memory; GPU cores in sm_count) --------
        _card("m1-max", "Apple M1 Max (32-core GPU)", METAL, None, 400.0, 32,
              10.4, None, None, f"{_APPLE} (M1 Pro and M1 Max, 2021-10)"),
        _card("m1-ultra", "Apple M1 Ultra (64-core GPU)", METAL, None, 800.0,
              64, 21.0, None, None, f"{_APPLE} (M1 Ultra, 2022-03)"),
        _card("m2-max", "Apple M2 Max (38-core GPU)", METAL, None, 400.0, 38,
              13.6, None, None, f"{_APPLE} (M2 Pro and M2 Max, 2023-01)"),
        _card("m2-ultra", "Apple M2 Ultra (76-core GPU)", METAL, None, 800.0,
              76, 27.2, None, None, f"{_APPLE} (M2 Ultra, 2023-06)"),
        _card("m3-max", "Apple M3 Max (40-core GPU)", METAL, None, 400.0, 40,
              None, None, None,
              f"{_APPLE} (M3 family, 2023-10: 400 GB/s on the 40-core GPU "
              "part; Apple publishes no TFLOPS figure)"),
        _card("m3-ultra", "Apple M3 Ultra (80-core GPU)", METAL, None, 819.0,
              80, None, None, None, f"{_APPLE} (Mac Studio, 2025-03)"),
        _card("m4-pro", "Apple M4 Pro (20-core GPU)", METAL, None, 273.0, 20,
              None, None, None, f"{_APPLE} (M4 Pro and M4 Max, 2024-10)"),
        _card("m4-max", "Apple M4 Max (40-core GPU)", METAL, None, 546.0, 40,
              None, None, None, f"{_APPLE} (M4 Pro and M4 Max, 2024-10)"),
    )
})

_VENDOR_WORDS = ("nvidia", "geforce", "quadro", "tesla", "apple")


def _normalise(spelling: str) -> str:
    token = str(spelling).strip().lower()
    for word in _VENDOR_WORDS:
        token = token.replace(word, " ")
    token = re.sub(r"\b(rtx|gpu|graphics)\b", " ", token)
    return re.sub(r"[^a-z0-9]+", "", token)


def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for key, card in RUNNER_CARDS.items():
        for alias in (key, card.name, *card.aliases):
            token = _normalise(alias)
            if index.get(token, key) != key:
                raise AssertionError(f"runner card alias {alias!r} is "
                                     "ambiguous")
            index[token] = key
    return index


_ALIASES = _alias_index()


def card_for(spelling: str | None) -> RunnerCard | None:
    """The table row a spelling names, or ``None``.

    Exact after normalising (vendor words, ``RTX``, spaces and
    punctuation dropped), so ``"NVIDIA RTX A6000"``, ``rtx-a6000`` and
    ``a6000`` are one row.  An Apple chip name with a core count
    (``"Apple M3 Max"``) matches its row; nvidia-smi's long datacentre
    spellings (``"NVIDIA A100-SXM4-80GB"``, ``"NVIDIA H100 80GB HBM3"``)
    match theirs.
    """

    if not spelling:
        return None
    key = _ALIASES.get(_normalise(spelling))
    if key is None:
        # "Apple M3 Max" -> m3max; table name carries "(40-core GPU)".
        token = _normalise(spelling)
        for k, card in RUNNER_CARDS.items():
            if card.backend == METAL and _normalise(k) == token:
                key = k
                break
    return RUNNER_CARDS[key] if key is not None else None


@dataclass(frozen=True)
class Runner:
    """A resolved runner: the card, the host, the concurrency, and the
    provenance of every field (``detected``, ``table``, ``toml``,
    ``flag`` or ``unknown``)."""

    name: str
    backend: str
    card_key: str | None
    vram_bytes: int | None
    bandwidth_gbs: float | None
    sm_count: int | None = None
    fp32_tflops: float | None = None
    fp64_tflops: float | None = None
    pcie: str | None = None
    pinned_gbps: float | None = None
    host_ram_bytes: int | None = None
    cpu_cores: int | None = None
    gpus: int = 1
    slots_per_gpu: int | str = 1
    mps: bool = False
    free_vram_bytes: int | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    @property
    def card(self) -> RunnerCard | None:
        return RUNNER_CARDS.get(self.card_key) if self.card_key else None

    @property
    def vram_gib(self) -> float | None:
        return None if self.vram_bytes is None else self.vram_bytes / GIB

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if k != "provenance"}
        out["provenance"] = dict(self.provenance)
        out["vram_gib"] = (None if self.vram_bytes is None
                           else round(self.vram_bytes / GIB, 2))
        out["host_ram_gib"] = (None if self.host_ram_bytes is None
                               else round(self.host_ram_bytes / GIB, 1))
        return out


def runner_from_card(card: RunnerCard, *, source: str = "table",
                     host_ram_bytes: int | None = None) -> Runner:
    """A runner that is just ``card`` (host facts unknown)."""

    vram = (card.memory_mib * MIB if card.memory_mib is not None
            else host_ram_bytes)
    prov = {k: source for k in ("backend", "vram_bytes", "bandwidth_gbs",
                                "sm_count", "fp32_tflops")}
    if card.memory_mib is None:
        prov["vram_bytes"] = ("unified memory: the host's RAM"
                              if host_ram_bytes else "unknown")
    return Runner(name=card.name, backend=card.backend, card_key=card.key,
                  vram_bytes=vram, bandwidth_gbs=card.bandwidth_gbs,
                  sm_count=card.sm_count, fp32_tflops=card.fp32_tflops,
                  fp64_tflops=card.fp64_tflops, pcie=card.pcie,
                  host_ram_bytes=host_ram_bytes, provenance=prov)


# --------------------------------------------------------------------------
# detection (no CUDA context)
# --------------------------------------------------------------------------
def _run(argv: list[str], timeout: float = 10.0) -> str | None:
    if shutil.which(argv[0]) is None:
        return None
    try:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _host_ram_bytes() -> int | None:
    try:
        from tilestream.autoplan import _host_memtotal

        found = _host_memtotal()
        if found:
            return int(found[0] if isinstance(found, tuple) else found)
    except Exception:  # noqa: BLE001 - detection never raises
        pass
    if sys.platform == "darwin":
        text = _run(["sysctl", "-n", "hw.memsize"])
        if text and text.strip().isdigit():
            return int(text.strip())
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Rows of ``nvidia-smi --query-gpu=name,memory.total,memory.free,
    pcie.link.gen.max,pcie.link.width.max --format=csv,noheader,nounits``."""

    rows = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue

        def number(value: str) -> int | None:
            return int(value) if value.isdigit() else None

        rows.append({"name": parts[0], "memory_mib": number(parts[1]),
                     "free_mib": number(parts[2]),
                     "pcie_gen": number(parts[3]) if len(parts) > 3 else None,
                     "pcie_width": number(parts[4]) if len(parts) > 4 else None})
    return rows


def detect_local_runner(*, smi_text: str | None = None,
                        apple_chip: str | None = None,
                        platform: str | None = None) -> Runner | None:
    """This box as a :class:`Runner`, or ``None`` when no GPU is visible.

    Never creates a CUDA context and never raises.  ``smi_text`` /
    ``apple_chip`` / ``platform`` are for tests.
    """

    platform = platform or sys.platform
    host = _host_ram_bytes()
    cores = os.cpu_count()
    if smi_text is None and platform != "darwin":
        smi_text = _run(["nvidia-smi", "--query-gpu=name,memory.total,"
                         "memory.free,pcie.link.gen.max,pcie.link.width.max",
                         "--format=csv,noheader,nounits"])
    rows = parse_nvidia_smi(smi_text or "")
    if rows:
        first = rows[0]
        card = card_for(first["name"])
        if card is not None:
            runner = runner_from_card(card, source="table",
                                      host_ram_bytes=host)
        else:
            runner = Runner(name=first["name"], backend=CUDA, card_key=None,
                            vram_bytes=None, bandwidth_gbs=None,
                            provenance={"bandwidth_gbs": "unknown: no table "
                                        "row names this card"})
        prov = dict(runner.provenance)
        vram = runner.vram_bytes
        if first["memory_mib"]:
            vram = first["memory_mib"] * MIB
            prov["vram_bytes"] = "detected (nvidia-smi)"
        pcie = runner.pcie
        if first["pcie_gen"] and first["pcie_width"]:
            pcie = f"PCIe {first['pcie_gen']}.0 x{first['pcie_width']}"
            prov["pcie"] = "detected (nvidia-smi)"
        prov.update(name="detected (nvidia-smi)",
                    host_ram_bytes="detected", cpu_cores="detected",
                    gpus="detected (nvidia-smi)")
        return replace(runner, name=first["name"], vram_bytes=vram,
                       pcie=pcie, host_ram_bytes=host, cpu_cores=cores,
                       gpus=len(rows),
                       free_vram_bytes=(None if first["free_mib"] is None
                                        else first["free_mib"] * MIB),
                       provenance=prov)
    if platform == "darwin":
        chip = apple_chip
        if chip is None:
            chip = (_run(["sysctl", "-n", "machdep.cpu.brand_string"])
                    or "").strip()
        if chip.startswith("Apple"):
            gpu_cores = None
            profile = _run(["system_profiler", "SPDisplaysDataType"],
                           timeout=15.0) if apple_chip is None else None
            match = re.search(r"Total Number of Cores:\s*(\d+)",
                              profile or "")
            if match:
                gpu_cores = int(match.group(1))
            card = card_for(chip)
            if card is not None:
                runner = runner_from_card(card, host_ram_bytes=host)
            else:
                runner = Runner(name=chip, backend=METAL, card_key=None,
                                vram_bytes=host, bandwidth_gbs=None,
                                provenance={"bandwidth_gbs": "unknown: no "
                                            "table row names this chip"})
            prov = dict(runner.provenance)
            prov.update(name="detected (sysctl)", host_ram_bytes="detected",
                        cpu_cores="detected",
                        vram_bytes="unified memory: the host's RAM")
            if gpu_cores is not None:
                prov["sm_count"] = "detected (system_profiler GPU cores)"
            note = ""
            if (card is not None and gpu_cores is not None
                    and card.sm_count and gpu_cores != card.sm_count):
                # A binned part: Apple quotes a lower bandwidth for some
                # (M3 Max 30-core: 300 GB/s).  Say so; do not guess it.
                note = (f"; this chip has {gpu_cores} GPU cores and the row "
                        f"is the {card.sm_count}-core part, whose bandwidth "
                        "may differ -- give bandwidth_gbs in a [runner] "
                        "TOML if Apple quotes another figure for yours")
                prov["bandwidth_gbs"] = "table" + note
            return replace(runner, name=chip, vram_bytes=host,
                           sm_count=gpu_cores or runner.sm_count,
                           host_ram_bytes=host, cpu_cores=cores,
                           provenance=prov)
    return None


# --------------------------------------------------------------------------
# [runner] TOML
# --------------------------------------------------------------------------
#: Keys a ``[runner]`` table may carry, and their types.
RUNNER_KEYS: Mapping[str, type] = MappingProxyType({
    "card": str, "name": str, "backend": str, "vram_gib": float,
    "bandwidth_gbs": float, "sm_count": int, "fp32_tflops": float,
    "fp64_tflops": float, "pinned_gbps": float, "host_ram_gib": float,
    "cpu_cores": int, "gpus": int, "slots_per_gpu": object, "mps": bool,
})


class RunnerSpecError(ValueError):
    """A runner description that cannot be used; the message says why."""


def runner_from_mapping(table: Mapping[str, Any], *, base: Runner | None,
                        source: str = "toml") -> Runner:
    """Layer a ``[runner]`` mapping over ``base`` (or over its ``card``)."""

    unknown = sorted(set(table) - set(RUNNER_KEYS))
    if unknown:
        raise RunnerSpecError(f"[runner] does not read {unknown}; it reads "
                              f"{sorted(RUNNER_KEYS)}")
    runner = base
    if "card" in table:
        card = card_for(str(table["card"]))
        if card is None:
            raise RunnerSpecError(
                f"[runner] card = {table['card']!r} names no row of the card "
                f"table ({', '.join(RUNNER_CARDS)}); omit it and give "
                "bandwidth_gbs and vram_gib instead")
        host = None if base is None else base.host_ram_bytes
        runner = runner_from_card(card, source=source, host_ram_bytes=host)
        if base is not None:
            runner = replace(runner, cpu_cores=base.cpu_cores,
                             gpus=base.gpus)
    if runner is None:
        runner = Runner(name=str(table.get("name", "runner")),
                        backend=str(table.get("backend", CUDA)),
                        card_key=None, vram_bytes=None, bandwidth_gbs=None)
    prov = dict(runner.provenance)
    changes: dict[str, Any] = {}

    def number(key: str, kind: type) -> Any:
        value = table[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RunnerSpecError(f"[runner] {key} must be a number, not "
                                  f"{value!r}")
        if value <= 0:
            raise RunnerSpecError(f"[runner] {key} must be positive")
        return kind(value)

    for key in ("bandwidth_gbs", "fp32_tflops", "fp64_tflops", "pinned_gbps"):
        if key in table:
            changes[key] = number(key, float)
    for key in ("sm_count", "cpu_cores", "gpus"):
        if key in table:
            changes[key] = number(key, int)
    if "vram_gib" in table:
        changes["vram_bytes"] = int(number("vram_gib", float) * GIB)
    if "host_ram_gib" in table:
        changes["host_ram_bytes"] = int(number("host_ram_gib", float) * GIB)
    if "name" in table:
        changes["name"] = str(table["name"])
    if "backend" in table:
        backend = str(table["backend"]).lower()
        if backend not in BACKENDS:
            raise RunnerSpecError(f"[runner] backend must be one of "
                                  f"{BACKENDS}, not {table['backend']!r}")
        changes["backend"] = backend
    if "mps" in table:
        if not isinstance(table["mps"], bool):
            raise RunnerSpecError("[runner] mps must be true or false")
        changes["mps"] = table["mps"]
    if "slots_per_gpu" in table:
        from woof.gpu_packing import parse_count_or_auto

        try:
            changes["slots_per_gpu"] = parse_count_or_auto(
                table["slots_per_gpu"], flag="[runner] slots_per_gpu")
        except ValueError as error:
            raise RunnerSpecError(str(error)) from None
    for key in changes:
        prov[key] = source
    if "vram_bytes" not in changes and runner.vram_bytes is None \
            and runner.backend == METAL and "host_ram_bytes" in changes:
        changes["vram_bytes"] = changes["host_ram_bytes"]
        prov["vram_bytes"] = "unified memory: the host's RAM"
    return replace(runner, provenance=prov, **changes)


def load_runner_toml(path: str | Path, *, base: Runner | None = None
                     ) -> Runner:
    """Read ``[runner]`` from ``path`` (a bare table at top level works too)."""

    try:
        document = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RunnerSpecError(f"runner spec {path} does not exist") from None
    except tomllib.TOMLDecodeError as error:
        raise RunnerSpecError(f"runner spec {path}: {error}") from None
    table = document.get("runner", document)
    if not isinstance(table, Mapping):
        raise RunnerSpecError(f"runner spec {path}: [runner] must be a table")
    return runner_from_mapping(table, base=base, source=f"toml ({path})")


__all__ = [
    "BACKENDS", "CUDA", "METAL", "RUNNER_CARDS", "RUNNER_KEYS", "Runner",
    "RunnerCard", "RunnerSpecError", "card_for", "detect_local_runner",
    "load_runner_toml", "parse_nvidia_smi", "runner_from_card",
    "runner_from_mapping",
]
