"""Card catalog: memory, SM count and memory bandwidth by ``--card`` spelling.

Two kinds of row live here and the ``provenance`` field keeps them apart:

* ``"measured"`` -- the capacity and SM count were read off a real device
  by this project (the rows of :data:`woof.hex.device_admission.KNOWN_CARDS`
  and the receipts they cite).  Bandwidth is still the vendor's nameplate.
* ``"declared"`` -- **declared, not measured locally**: every figure is the
  vendor's published specification, and the capacity is the total an
  ``nvidia-smi`` on that part commonly reports (ECC on where the part
  has it).  No GPU of that model was ever probed by this tree.  These are
  the cloud datacentre parts a hindcast campaign rents.

What is pinned
--------------

* :func:`cloud_card_row` is what :func:`woof.domain_wizard.card_capacity_gib`
  consults before reading a size out of a spelling, so ``--card h100-80gb``,
  ``--card l40s`` or ``--card a10g`` size against the capacity the part
  presents (an L40S shows 46,068 MiB, not 48 GiB) and never probe the local
  card.  Only ``declared`` rows are looked up this way; consumer spellings
  keep the existing :data:`woof.domain_wizard.KNOWN_CARD_VRAM_GIB` path
  byte for byte.
* A spelling that names a model in two variants and no variant word picks
  the conservative one: bare ``h100`` is the PCIe part (lower bandwidth,
  same memory), bare ``a100`` is the 40 GB part.
* The SM count and bandwidth feed :mod:`woof.energy.cost`'s throughput
  model only.  Sizing a declared card keeps the wizard's conservative
  170-SM reference device profile (see
  :func:`woof.domain_wizard.resolve_sizing_budget`): every declared part
  here has fewer SMs than that, so the reference over-prices the
  SM-proportional workspaces, which is the safe direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

MIB = 1024 ** 2
GIB = 1024 ** 3

#: The two provenance labels a row may carry.
MEASURED = "measured"
DECLARED = "declared"
DECLARED_NOTE = "declared, not measured locally"


@dataclass(frozen=True)
class CardRow:
    """One GPU model: what a sizing or cost door may read about it."""

    key: str
    name: str
    memory_mib: int
    sm_count: int
    bandwidth_gbs: float
    compute_capability: str
    provenance: str
    source: str
    aliases: tuple[str, ...] = ()
    #: A datacentre part a cloud provider rents (the rows the sizing door
    #: looks up by exact spelling); consumer rows are calibration cards.
    datacentre: bool = False

    @property
    def memory_gib(self) -> float:
        return self.memory_mib * MIB / GIB

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "memory_mib": self.memory_mib,
            "memory_gib": round(self.memory_gib, 3),
            "sm_count": self.sm_count,
            "bandwidth_gbs": self.bandwidth_gbs,
            "compute_capability": self.compute_capability,
            "provenance": self.provenance,
            "provenance_note": (DECLARED_NOTE if self.provenance == DECLARED
                                else "SM count and capacity measured on a "
                                     "device by this project; bandwidth is "
                                     "the vendor nameplate"),
            "source": self.source,
            "datacentre": self.datacentre,
        }


_NVIDIA_SPECS = ("NVIDIA datasheets: H100 (nvidia.com/en-us/data-center/h100), "
                 "H200, A100, L40S, L4; AWS EC2 G5 (A10G) instance page")

#: Every card the catalog knows, keyed by a canonical ``--card`` spelling.
CARD_CATALOG: Mapping[str, CardRow] = MappingProxyType({
    # -- declared, not measured locally: cloud datacentre parts ----------
    "h100-sxm-80gb": CardRow(
        "h100-sxm-80gb", "NVIDIA H100 SXM5 80GB HBM3", 81_559, 132, 3350.0,
        "9.0", DECLARED, _NVIDIA_SPECS + "; AWS p5.48xlarge, GCP a3-highgpu-8g",
        ("h100sxm", "h100sxm80gb", "h100sxm5", "h100sxm580gb", "h10080gbhbm3"), True),
    "h100-pcie-80gb": CardRow(
        "h100-pcie-80gb", "NVIDIA H100 PCIe 80GB", 81_559, 114, 2000.0,
        "9.0", DECLARED, _NVIDIA_SPECS,
        ("h100", "h10080gb", "h100pcie", "h100pcie80gb"), True),
    "h200-141gb": CardRow(
        "h200-141gb", "NVIDIA H200 SXM 141GB HBM3e", 143_771, 132, 4800.0,
        "9.0", DECLARED, _NVIDIA_SPECS + "; AWS p5en.48xlarge, GCP a3-ultragpu-8g",
        ("h200", "h200141gb", "h200sxm", "h200sxm141gb"), True),
    "a100-80gb": CardRow(
        "a100-80gb", "NVIDIA A100 SXM4 80GB", 81_920, 108, 2039.0,
        "8.0", DECLARED, _NVIDIA_SPECS + "; GCP a2-ultragpu-1g, AWS p4de",
        ("a10080gb", "a100sxm480gb", "a100sxm80gb"), True),
    "a100-40gb": CardRow(
        "a100-40gb", "NVIDIA A100 SXM4 40GB", 40_960, 108, 1555.0,
        "8.0", DECLARED, _NVIDIA_SPECS + "; AWS p4d.24xlarge, GCP a2-highgpu",
        ("a100", "a10040gb", "a100sxm440gb", "a100sxm40gb"), True),
    "l40s-48gb": CardRow(
        "l40s-48gb", "NVIDIA L40S 48GB GDDR6 (ECC on)", 46_068, 142, 864.0,
        "8.9", DECLARED, _NVIDIA_SPECS + "; AWS g6e",
        ("l40s", "l40s48gb"), True),
    "l4-24gb": CardRow(
        "l4-24gb", "NVIDIA L4 24GB GDDR6 (ECC on)", 23_034, 58, 300.0,
        "8.9", DECLARED, _NVIDIA_SPECS + "; GCP g2-standard, AWS g6",
        ("l4", "l424gb"), True),
    "a10g-24gb": CardRow(
        "a10g-24gb", "NVIDIA A10G 24GB GDDR6 (ECC on)", 23_028, 80, 600.0,
        "8.6", DECLARED, _NVIDIA_SPECS + "; AWS g5",
        ("a10g", "a10g24gb"), True),
    # -- measured by this project (capacity, SMs); bandwidth nameplate --
    "rtx-5090": CardRow(
        "rtx-5090", "RTX 5090 (32,607 MiB, sm_120)", 32_607, 170, 1792.0,
        "12.0", MEASURED, "woof/hex/device_admission.py KNOWN_CARDS['32gib-170sm']"),
    "rtx-5070-ti": CardRow(
        "rtx-5070-ti", "RTX 5070 Ti (16,303 MiB, sm_120)", 16_303, 70, 896.0,
        "12.0", MEASURED, "woof/hex/device_admission.py KNOWN_CARDS['16gib-70sm']"),
    "rtx-3080": CardRow(
        "rtx-3080", "RTX 3080 (10,240 MiB, sm_86)", 10_240, 68, 760.0,
        "8.6", MEASURED, "woof/hex/device_admission.py KNOWN_CARDS['10gib-68sm']"),
    "rtx-4070": CardRow(
        "rtx-4070", "RTX 4070 12GB", 12_282, 46, 504.0,
        "8.9", DECLARED, "NVIDIA RTX 4070 specification; timing receipt in "
        "docs/public/FIRST-LIGHT.md (the card ran, its SMs were not probed "
        "into a row)"),
})

#: The same vendor words woof.domain_wizard.card_capacity_gib strips.
_VENDOR_WORDS = ("nvidia", "geforce", "quadro", "tesla", "rtx", "gtx")


def _normalise(spelling: str) -> str:
    token = str(spelling).strip().lower()
    for word in _VENDOR_WORDS:
        token = token.replace(word, " ")
    return re.sub(r"[\s_\-]+", "", token)


def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for key, row in CARD_CATALOG.items():
        for alias in (key, *row.aliases):
            normalised = _normalise(alias)
            if normalised in index and index[normalised] != key:
                raise AssertionError(f"card alias {alias!r} is ambiguous")
            index[normalised] = key
    return index


_ALIASES = _alias_index()


def card_row(spelling: str) -> CardRow | None:
    """The catalog row an exact ``--card`` spelling names, or ``None``.

    Matching is exact after lower-casing and stripping vendor words,
    spaces, hyphens and underscores, so ``"NVIDIA H100 SXM"``,
    ``h100-sxm`` and ``h100sxm`` are one row and ``"h100 foo"`` is none.
    """

    key = _ALIASES.get(_normalise(spelling))
    return CARD_CATALOG[key] if key is not None else None


def cloud_card_row(spelling: str) -> CardRow | None:
    """The datacentre (cloud, ``declared``) row a spelling names, or ``None``."""

    row = card_row(spelling)
    return row if row is not None and row.datacentre else None
