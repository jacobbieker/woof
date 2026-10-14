"""Global wind-farm inventory: four public sources and a cross-source merge.

``woof energy import --format {uswtdb,gem,emodnet,osm-pbf}`` reads one
source into a ``woof-energy.assets.v1`` document, and ``woof energy
inventory merge`` folds several such documents (these, REPD, an Overpass
fetch, earlier merges) into one inventory of wind farms, deduplicated
across sources.  The output is still ``woof-energy.assets.v1``; everything
this module adds lives in string ``tags`` under the ``woof:`` prefix, so an
older reader loads it unchanged.

Sources (column layouts pinned as read; licences recorded per asset)
--------------------------------------------------------------------

``uswtdb``
    The U.S. Wind Turbine Database (USGS / LBNL / ACP), CSV download or the
    JSON list its API returns.  One row per turbine: ``case_id, t_state,
    t_county, p_name, p_year, p_tnum, p_cap, t_manu, t_model, t_cap, t_hh,
    t_rd, t_rsa, t_ttlh, t_conf_atr, t_conf_loc, xlong, ylat`` (and more).
    ``t_cap`` is kW; ``p_cap`` MW; ``t_hh``/``t_rd``/``t_ttlh`` metres.
    Values of ``-9999`` or below are the database's "missing" and read as
    null.  Each row becomes a ``generator`` Point with ``generator_source
    wind``, ``hub_height_m = t_hh``, ``rotor_diameter_m = t_rd`` and
    ``capacity_mw = t_cap / 1000``; ``p_name`` is the asset name and the
    ``woof:project`` tag.  Every row is an installed turbine
    (``woof:status=operating``); an ``offshore`` or ``t_offshore`` column,
    when present and truthy, marks ``woof:installation=offshore``.  Public
    domain (``license`` ``public-domain``).

``gem``
    Global Energy Monitor's Global Wind Power Tracker, the ``.xlsx`` the
    download form hands out or a CSV export of its data sheet.  Columns are
    matched case- and whitespace-insensitively: required ``Project Name,
    Capacity (MW), Latitude, Longitude, Status``; read when present
    ``Phase Name, Installation Type, Country/Area`` (or ``Country``),
    ``Start Year, Retired Year, Operator, Owner, Location Accuracy, GEM
    location ID, GEM phase ID, Other Name(s), State/Province, Wiki URL``.
    Every sheet whose header row (one of the first five rows) holds the
    required columns is read, so the "below threshold" sheet comes along.
    Rows are project phases; after the status filter, phases sharing a
    ``GEM location ID`` and coordinates become one ``plant`` Point whose
    capacity is the sum of its phases' and whose status is the most
    advanced of them (order of :data:`STATUSES`).  ``Status`` values are
    GEM's (``operating``, ``construction``, ``pre-construction``,
    ``announced``, ``shelved``, ``cancelled``, ``mothballed``, ``retired``;
    an ``" - inferred ..."`` suffix is dropped).  ``Installation Type``
    ``onshore`` / ``offshore ...`` sets ``woof:installation``.  CC-BY-4.0.
    The ``.xlsx`` is read by :func:`read_xlsx` (zip + XML, standard library
    only) so no spreadsheet package is needed.

``emodnet``
    EMODnet Human Activities offshore wind farms, GeoJSON from the WFS
    (layers ``emodnet:windfarmspoly`` / ``emodnet:windfarms``).  Properties
    matched case-insensitively: ``NAME, COUNTRY, STATUS, POWER_MW,
    N_TURBINES, YEAR, DIST_COAST``.  Each feature becomes a ``plant`` with
    its Polygon, MultiPolygon or Point geometry, ``woof:installation=
    offshore``.  ``STATUS``: ``Production`` and ``Test site`` -> operating,
    ``Construction`` -> construction, ``Approved`` -> pre-construction,
    ``Planned`` -> announced, ``Dismantled`` -> retired.  A ``crs`` member
    other than CRS84 / ``EPSG:4326`` (including the latitude-first
    ``urn:ogc:def:crs:EPSG::4326``) is refused, not reordered.  CC-BY-4.0.

``osm-pbf``
    An OpenStreetMap planet or regional extract (``.osm.pbf``, ``.osm``,
    ``.osm.bz2``) read with pyosmium when it is installed; without it the
    import is refused with the install line.  ``power=generator`` with a
    ``generator:source`` naming ``wind`` -> ``generator`` (turbines);
    ``power=plant`` with ``plant:source`` naming ``wind`` -> ``plant``.
    Nodes are Points, closed ways and multipolygon relations polygons,
    converted by the same :func:`woof.energy.osm.element_to_asset` the
    Overpass fetch uses, so ``asset_id`` (``osm:node/123``) is identical
    across the two routes.  Features carrying a primary ``power`` tag
    describe built infrastructure: ``woof:status=operating`` (a
    ``--status`` without ``operating`` is refused before the file is
    read).  ODbL-1.0.
    For the planet, cut it first with osmium-tool
    (``osmium tags-filter planet.osm.pbf nwr/power=generator,plant -o
    power.osm.pbf``) and pass ``--osm-index sparse_file_array,FILE`` when
    the location index will not fit in memory.

Status
------

:data:`STATUSES` is the vocabulary of ``woof:status``.  Imports of the
four formats above keep ``operating`` only unless ``--status`` widens it;
the merge applies the same filter to every input asset through
:func:`asset_status`, which also maps REPD's ``Development Status
(short)`` and calls Overpass-fetched OSM assets operating.  An asset no
rule can place is ``unknown`` and is kept only when ``unknown`` is asked
for.

Merge rules (:class:`MergeRules`, every threshold a flag)
---------------------------------------------------------

Only wind assets take part (``generator_source == "wind"``, kinds
``plant``/``generator``); others are dropped unless ``--keep-non-wind``,
which passes them through unmatched.  Assets with an ``asset_id`` or a
``(source, source_ref)`` already seen are dropped as exact duplicates.  A
merge never joins two assets of the same source, and a cluster holds at
most one asset per source, so a chain of near neighbours cannot fold two
farms of one database together.  Missing a duplicate costs one extra
site; merging two farms loses one, so every rule leans toward keeping
both.

Turbines (``generator`` Points)
    Two turbines of different sources within ``turbine_match_m`` (50 m)
    are one turbine; pairs are taken nearest first.
Plants
    Distance is between centroids, zero when one plant's centroid lies
    inside the other's polygon.  Onshore and offshore never match.  Two
    plants match when (1) they lie within ``plant_match_m`` (5 km), their
    names agree (similarity >= ``name_threshold``, 0.7) and their
    capacities do not disagree (relative difference over
    ``capacity_tolerance``, 0.25, when both are known), or (2) they lie
    within ``plant_near_m`` (1 km), their names do not disagree
    (similarity >= ``name_floor``, 0.3, or a name is missing) and their
    capacities do not disagree.  Rule-1 pairs are taken before rule-2
    pairs, nearest first.
Names
    Lower-cased, accents stripped, split on non-alphanumerics; number
    words and Roman numerals become digits, compass words a direction
    code.  Digits, directions and ``extension`` are *distinguishing*:
    names whose distinguishing sets differ disagree outright ("Hornsea 1"
    vs "Hornsea 2", "Walney" vs "Walney Extension").  The rest, minus
    :data:`NAME_STOPWORDS`, is compared as the larger of the token Jaccard
    index and :class:`difflib.SequenceMatcher` ratio of the sorted tokens.
Winner and fill
    In each cluster the winner is the asset with a polygon (plants), then
    the source earliest in ``prefer`` (default :data:`DEFAULT_PREFER`):
    its ``asset_id``, kind and geometry are kept; null ``name``,
    ``operator``, ``capacity_mw``, ``hub_height_m`` and
    ``rotor_diameter_m`` are filled from the others in the same order, and
    tags they carry that the winner lacks are added.
Turbine-level over plant-level
    Each merged turbine is offered to the merged wind plants near it, best
    first: inside the plant's polygon; within ``plant_link_m`` (10 km) with
    a ``woof:project`` name agreeing with the plant's; within
    ``plant_link_near_m`` (2 km) with no disagreeing project name.  A plant
    whose linked turbines cover at least ``turbine_coverage`` (0.8) of its
    capacity (a turbine without capacity counts as
    :data:`COVERAGE_TURBINE_MW`; a plant without capacity is covered only
    when its ``woof:turbines`` count is known and reached) is
    *turbine-level*: each linked turbine gets ``tags["plant"] = plant
    asset_id`` (what ``woof energy sites`` reads to drop the plant site and
    its double-counted capacity) and the plant ``woof:turbine_level=yes``.
    Any other plant with linked turbines records ``woof:turbines_partial``
    (count) and keeps its own site (``woof energy sites`` honours the tag
    even for a polygon with turbines inside); its turbines stay sites.  A
    turbine ``plant`` tag naming a record now inside another cluster is
    re-pointed at that cluster's winner.

Provenance
    Every output wind asset carries ``woof:provenance`` (compact JSON list
    of ``{"source", "ref", "asset_id", "license"}``, winner first) and
    ``woof:licenses`` (``;``-joined, sorted); ``license`` becomes the SPDX
    ``AND`` of every contributing licence.  Re-merging a merged document
    extends these lists rather than restarting them: a single-record input
    asset another input's provenance already lists is dropped as already
    merged; assets whose provenance shares a record are joined first; and
    a cluster tracks every record of every member's provenance, so two
    clusters join only when each source they share names the same
    records.  The document's
    ``sources`` are the inputs' records plus one
    ``woof-energy.inventory-merge`` record with the rules and counts.

Python boundary: per-asset bookkeeping over at most a few hundred thousand
points; neighbour queries go through one scipy KD-tree per pass and
polygon tests through :func:`woof.energy.geometry.point_in_polygon`.  No
gridded field is touched.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import difflib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from woof.energy import geometry as geo
from woof.energy.contracts import (
    Asset,
    AssetCollection,
    ContractError,
    dump_assets,
    file_ref,
    load_assets,
)

INVENTORY_FORMATS = ("uswtdb", "gem", "emodnet", "osm-pbf")

#: ``woof:status`` vocabulary, most advanced first.
STATUSES = ("operating", "construction", "pre-construction", "announced",
            "shelved", "mothballed", "cancelled", "retired", "unknown")
DEFAULT_STATUSES = ("operating",)

TAG_STATUS = "woof:status"
TAG_INSTALLATION = "woof:installation"
TAG_PROVENANCE = "woof:provenance"
TAG_LICENSES = "woof:licenses"
TAG_PROJECT = "woof:project"
TAG_START_YEAR = "woof:start_year"
TAG_TURBINES = "woof:turbines"
TAG_COUNTRY = "woof:country"
TAG_TURBINE_LEVEL = "woof:turbine_level"
TAG_TURBINES_PARTIAL = "woof:turbines_partial"
TAG_TURBINES_LINKED = "woof:turbines_linked"
INSTALLATIONS = ("onshore", "offshore")

MERGE_RECORD_SOURCE = "woof-energy.inventory-merge"
SUMMARY_SCHEMA = "woof-energy.inventory-summary.v1"

USWTDB_SOURCE = "uswtdb"
USWTDB_LICENSE = "public-domain"
USWTDB_ATTRIBUTION = (
    "U.S. Wind Turbine Database (USGS, Lawrence Berkeley National "
    "Laboratory, American Clean Power Association); public domain; Hoen et "
    "al., doi:10.5066/F7TX3DN0")
GEM_SOURCE = "gem"
GEM_LICENSE = "CC-BY-4.0"
GEM_ATTRIBUTION = ("Global Energy Monitor, Global Wind Power Tracker, "
                   "CC BY 4.0")
EMODNET_SOURCE = "emodnet"
EMODNET_LICENSE = "CC-BY-4.0"
EMODNET_ATTRIBUTION = ("EMODnet Human Activities, Offshore wind farms, "
                       "CC BY 4.0")

#: Source order for picking a cluster's winner: OSM geometry is mapped
#: from imagery, EMODnet carries farm outlines, national registers next,
#: GEM's points are often approximate.
DEFAULT_PREFER = ("osm", "uswtdb", "emodnet", "repd", "gem")

#: Capacity assumed for a linked turbine with none, only to decide whether
#: turbines cover a plant (never written to an asset).  Deliberately low,
#: so an unknown turbine set reads as partial and the plant keeps its site.
COVERAGE_TURBINE_MW = 2.0

#: The USWTDB writes this (and lower) for a missing value.
_USWTDB_MISSING = -9999.0

NAME_STOPWORDS = frozenset({
    "wind", "winds", "farm", "farms", "windfarm", "windfarms", "windpark",
    "windparks", "park", "parc", "parque", "power", "plant", "project",
    "energy", "energie", "station", "offshore", "onshore", "the", "of",
    "and", "de", "del", "der", "des", "di", "du", "la", "le", "les", "el",
    "eolico", "eolica", "eolien", "eolienne", "eoliennes", "eolicos",
    "windenergie", "windkraft", "windenergieanlage", "centrale", "central",
    "llc", "ltd", "limited", "inc", "gmbh", "co", "company", "corp", "sa",
    "sas", "srl", "phase", "repower", "repowering", "facility",
    "renewable", "renewables", "wf", "wpp", "owf", "ofw", "turbine",
    "turbines", "generation", "generating", "complex", "cluster",
})
_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4",
                 "five": "5", "six": "6", "seven": "7", "eight": "8",
                 "nine": "9", "ten": "10", "first": "1", "second": "2",
                 "third": "3"}
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6",
          "vii": "7", "viii": "8", "ix": "9", "x": "10"}
_DIRECTIONS = {"north": "n", "northern": "n", "nord": "n", "norte": "n",
               "south": "s", "southern": "s", "sud": "s",
               "east": "e", "eastern": "e", "ost": "e",
               "west": "w", "western": "w", "ouest": "w",
               "oeste": "w", "northeast": "ne", "northwest": "nw",
               "southeast": "se", "southwest": "sw"}
_EXTENSION = {"extension": "extension", "ext": "extension"}

_EARTH_RADIUS_M = 6371008.8


class InventoryRefused(ValueError):
    """The inventory importers or merge cannot do what was asked."""


# --------------------------------------------------------------------------
# small helpers


def _importers():
    from woof.energy import importers
    return importers


def _norm_header(name: str) -> str:
    return " ".join(str(name).split()).lower()


def _missing_number(value: Any) -> float | None:
    """A finite float, ``None`` for blanks and the USWTDB ``-9999``."""

    number = _importers()._float(value)
    if number is None or number <= _USWTDB_MISSING:
        return None
    return number


def _positive_or_none(value: Any) -> float | None:
    number = _missing_number(value)
    return number if number is not None and number > 0.0 else None


def _year(value: Any) -> str | None:
    number = _missing_number(value)
    if number is None or number != int(number) or not 1800 <= number <= 2200:
        return None
    return str(int(number))


def _text(value: Any) -> str | None:
    if _importers()._blank(value):
        return None
    text = " ".join(str(value).split())
    return text or None


def _haversine_m(lon1, lat1, lon2, lat2):
    return _importers()._haversine_m(lon1, lat1, lon2, lat2)


def _xyz(lon, lat) -> np.ndarray:
    lon_r = np.radians(np.asarray(lon, dtype=np.float64))
    lat_r = np.radians(np.asarray(lat, dtype=np.float64))
    cos_lat = np.cos(lat_r)
    return _EARTH_RADIUS_M * np.column_stack(
        (cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)))


def _chord(distance_m: float) -> float:
    """Chord length for a great-circle distance (KD-tree radius)."""

    return 2.0 * _EARTH_RADIUS_M * math.sin(
        min(distance_m / (2.0 * _EARTH_RADIUS_M), math.pi / 2.0))


def _tree(points: np.ndarray):
    from scipy.spatial import cKDTree

    return cKDTree(points if len(points) else np.zeros((0, 3)))


def _check_statuses(statuses: Iterable[str] | None) -> tuple[str, ...] | None:
    if statuses is None:
        return None
    statuses = tuple(statuses)
    bad = [s for s in statuses if s not in STATUSES]
    if bad or not statuses:
        raise InventoryRefused(f"status filter {list(statuses)} must name "
                               f"some of {', '.join(STATUSES)}")
    return statuses


def _ledger():
    return _importers()._Ledger()


def _wind(asset: Asset) -> bool:
    return (asset.kind in ("plant", "generator")
            and asset.generator_source == "wind")


# --------------------------------------------------------------------------
# status and installation of any asset


_REPD_STATUS = {
    "operational": "operating",
    "under construction": "construction",
    "awaiting construction": "pre-construction",
    "appeal granted": "pre-construction",
    "no application required": "pre-construction",
    "application submitted": "announced",
    "revised": "announced",
    "appeal lodged": "announced",
    "application refused": "cancelled",
    "appeal refused": "cancelled",
    "appeal withdrawn": "cancelled",
    "application withdrawn": "cancelled",
    "abandoned": "cancelled",
    "planning permission expired": "cancelled",
    "secretary of state - refusal": "cancelled",
    "decommissioned": "retired",
}

_GEM_STATUS = {status: status for status in STATUSES}
_GEM_STATUS.update({"pre construction": "pre-construction",
                    "preconstruction": "pre-construction",
                    "canceled": "cancelled"})

_EMODNET_STATUS = {
    "production": "operating",
    "test site": "operating",
    "construction": "construction",
    "approved": "pre-construction",
    "planned": "announced",
    "dismantled": "retired",
}


def _gem_status(text: str | None) -> str | None:
    if not text:
        return None
    lowered = text.strip().lower().split(" - ")[0].strip()
    return _GEM_STATUS.get(lowered)


def asset_status(asset: Asset) -> str:
    """``woof:status`` of any asset; see the module docstring."""

    tagged = asset.tags.get(TAG_STATUS)
    if tagged in STATUSES:
        return tagged
    if asset.source == "repd":
        return _REPD_STATUS.get((asset.tags.get("repd:status") or "")
                                .strip().lower(), "unknown")
    if asset.source == "osm":
        return "operating"
    if asset.source == USWTDB_SOURCE:
        return "operating"
    return "unknown"


def asset_installation(asset: Asset) -> str | None:
    """``onshore``, ``offshore`` or ``None`` when no source says."""

    tagged = asset.tags.get(TAG_INSTALLATION)
    if tagged in INSTALLATIONS:
        return tagged
    if asset.tags.get("offshore") == "yes":
        return "offshore"
    if asset.source == "repd":
        technology = asset.tags.get("repd:technology", "").lower()
        if technology == "wind onshore":
            return "onshore"
        if technology == "wind offshore":
            return "offshore"
    return None


def filter_status(collection: AssetCollection, statuses: Sequence[str]
                  ) -> tuple[AssetCollection, Counter]:
    """Assets whose :func:`asset_status` is in ``statuses``; drops counted
    by status."""

    statuses = _check_statuses(statuses) or DEFAULT_STATUSES
    kept, dropped = [], Counter()
    for asset in collection.assets:
        status = asset_status(asset)
        if status in statuses:
            kept.append(asset)
        else:
            dropped[status] += 1
    return AssetCollection(assets=kept, sources=list(collection.sources)), \
        dropped


# --------------------------------------------------------------------------
# xlsx (standard library)


_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL_NS = ("{http://schemas.openxmlformats.org/officeDocument/2006/"
           "relationships}id")
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_CELL_REF = re.compile(r"([A-Z]+)(\d+)")


def _column_index(letters: str) -> int:
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - 64)
    return index - 1


def _string_item(element: ET.Element) -> str:
    """Text of a shared-string or inline-string item, rich runs joined,
    phonetic runs skipped."""

    main = _XLSX_NS["m"]
    parts = []
    for child in element:
        if child.tag == f"{{{main}}}t":
            parts.append(child.text or "")
        elif child.tag == f"{{{main}}}r":
            for run in child:
                if run.tag == f"{{{main}}}t":
                    parts.append(run.text or "")
    return "".join(parts)


def read_xlsx(path: str | Path) -> list[tuple[str, list[list[str]]]]:
    """``[(sheet name, rows)]`` of an ``.xlsx`` workbook, cells as text.

    ``rows[k]`` is sheet row ``k + 1``; rows the sheet does not store are
    empty lists.

    Shared strings, inline strings, numbers (their stored text), booleans
    and cached formula results are read; errors and styles are not.
    Pinned to the Office Open XML SpreadsheetML layout (ECMA-376):
    ``xl/workbook.xml`` lists the sheets, ``xl/_rels/workbook.xml.rels``
    maps them to parts.  Anything else is refused.
    """

    path = Path(path)
    main = _XLSX_NS["m"]
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as error:
        raise InventoryRefused(f"{path} is not an .xlsx workbook (not a zip "
                               "archive)") from error
    with archive:
        names = set(archive.namelist())
        if "xl/workbook.xml" not in names:
            raise InventoryRefused(f"{path} is not an .xlsx workbook (no "
                                   "xl/workbook.xml)")
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            with archive.open("xl/sharedStrings.xml") as handle:
                for _, element in ET.iterparse(handle):
                    if element.tag == f"{{{main}}}si":
                        shared.append(_string_item(element))
                        element.clear()
        targets: dict[str, str] = {}
        if "xl/_rels/workbook.xml.rels" in names:
            rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            for rel in rels.iter(f"{_PKG_REL}Relationship"):
                target = rel.get("Target") or ""
                target = target.lstrip("/")
                if not target.startswith("xl/"):
                    target = "xl/" + target
                targets[rel.get("Id") or ""] = target
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = []
        for sheet in workbook.iter(f"{{{main}}}sheet"):
            part = targets.get(sheet.get(_REL_NS) or "")
            if part is None or part not in names:
                raise InventoryRefused(f"{path}: sheet {sheet.get('name')!r} "
                                       "has no worksheet part")
            sheets.append((sheet.get("name") or part, part))
        out = []
        for name, part in sheets:
            rows: list[list[str]] = []
            with archive.open(part) as handle:
                for _, element in ET.iterparse(handle):
                    if element.tag != f"{{{main}}}row":
                        continue
                    cells: dict[int, str] = {}
                    for position, cell in enumerate(
                            element.iter(f"{{{main}}}c")):
                        ref = _CELL_REF.match(cell.get("r") or "")
                        column = (_column_index(ref.group(1)) if ref
                                  else position)
                        kind = cell.get("t") or "n"
                        value = cell.find(f"{{{main}}}v")
                        text = value.text if value is not None else None
                        if kind == "s" and text is not None:
                            try:
                                text = shared[int(text)]
                            except (ValueError, IndexError) as error:
                                raise InventoryRefused(
                                    f"{path}: sheet {name!r} cell "
                                    f"{cell.get('r')} names shared string "
                                    f"{text!r} that does not exist") from error
                        elif kind == "inlineStr":
                            inline = cell.find(f"{{{main}}}is")
                            text = (_string_item(inline)
                                    if inline is not None else None)
                        elif kind == "b":
                            text = {"1": "TRUE", "0": "FALSE"}.get(text or "")
                        elif kind == "e":
                            text = None
                        if text is not None:
                            cells[column] = text
                    try:
                        number = int(element.get("r") or len(rows) + 1)
                    except ValueError:
                        number = len(rows) + 1
                    element.clear()
                    # keep list index == sheet row - 1 so messages can
                    # name the spreadsheet row
                    while len(rows) < number - 1:
                        rows.append([])
                    width = max(cells) + 1 if cells else 0
                    rows.append([cells.get(i, "") for i in range(width)])
            out.append((name, rows))
    return out


# --------------------------------------------------------------------------
# USWTDB


_USWTDB_REQUIRED = ("case_id", "xlong", "ylat")
_USWTDB_TAGS = ("t_state", "t_county", "t_fips", "p_name", "p_year",
                "p_tnum", "p_cap", "t_manu", "t_model", "t_cap", "t_hh",
                "t_rd", "t_rsa", "t_ttlh", "retrofit", "retrofit_year",
                "t_conf_atr", "t_conf_loc", "t_img_date", "t_img_srce",
                "eia_id", "faa_ors", "faa_asn", "usgs_pr_id", "t_offshore",
                "offshore")


def _uswtdb_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.suffix.lower() == ".json":
        text, _ = _importers()._read_text(path)
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise InventoryRefused(f"{path} is not JSON: {error}") from error
        if not isinstance(document, list) or not all(
                isinstance(row, dict) for row in document):
            raise InventoryRefused(f"{path}: a USWTDB JSON download is a list "
                                   "of turbine objects")
        return document, "json"
    header, rows, encoding = _importers()._read_csv(path, quotechars=('"',))
    return rows, encoding


def import_uswtdb(path: str | Path, *, statuses: Sequence[str] = DEFAULT_STATUSES
                  ) -> AssetCollection:
    """USWTDB turbines -> ``generator`` Points (see the module docstring)."""

    path = Path(path)
    statuses = _check_statuses(statuses)
    rows, encoding = _uswtdb_rows(path)
    header = set().union(*(row.keys() for row in rows)) if rows else set()
    missing = [c for c in _USWTDB_REQUIRED if c not in header]
    if missing:
        raise InventoryRefused(
            f"{path.name} is not a USWTDB download: lacks columns {missing}; "
            f"expected {list(_USWTDB_REQUIRED)} plus t_cap (kW), t_hh, t_rd "
            "(m), p_name, p_year, p_cap (MW)")
    ledger = _ledger()
    ledger.notes.append("t_cap read as kW; -9999 read as missing")
    status = "operating"
    assets, seen = [], set()
    for number, row in enumerate(rows, start=1):
        ledger.rows += 1
        ref = _text(row.get("case_id"))
        if ref is not None and ref.endswith(".0") and ref[:-2].isdigit():
            ref = ref[:-2]
        where = f"row {number}" + (f" (case_id {ref})" if ref else "")
        if ref is None:
            ledger.refuse(where, "no case_id")
            continue
        if ref in seen:
            ledger.refuse(where, "duplicate case_id")
            continue
        seen.add(ref)
        if status not in statuses:
            ledger.skipped[f"status {status}"] += 1
            continue
        lon, lat = _missing_number(row.get("xlong")), \
            _missing_number(row.get("ylat"))
        if lon is None or lat is None:
            ledger.refuse(where, "no xlong/ylat")
            continue
        tags = {}
        for key in _USWTDB_TAGS:
            value = _text(row.get(key))
            number = _importers()._float(value)
            if value is None or (number is not None
                                 and number <= _USWTDB_MISSING):
                continue
            tags[f"uswtdb:{key}"] = value
        tags[TAG_STATUS] = status
        offshore = next((row[k] for k in ("t_offshore", "offshore")
                         if k in row and _text(row[k]) is not None), None)
        flag = None if offshore is None else _importers()._bool(offshore)
        if flag is not None:
            tags[TAG_INSTALLATION] = "offshore" if flag else "onshore"
        elif offshore is not None:
            ledger.derived[f"offshore flag {offshore!r} not read"] += 1
        project = _text(row.get("p_name"))
        if project:
            tags[TAG_PROJECT] = project
        year = _year(row.get("p_year"))
        if year:
            tags[TAG_START_YEAR] = year
        tags[TAG_COUNTRY] = "US"
        t_cap_kw = _positive_or_none(row.get("t_cap"))
        try:
            assets.append(Asset(
                asset_id=f"{USWTDB_SOURCE}:{ref}", kind="generator",
                geometry={"type": "Point",
                          "coordinates": [round(lon, 7), round(lat, 7)]},
                source=USWTDB_SOURCE, source_ref=ref, name=project,
                generator_source="wind",
                capacity_mw=None if t_cap_kw is None else t_cap_kw / 1000.0,
                hub_height_m=_positive_or_none(row.get("t_hh")),
                rotor_diameter_m=_positive_or_none(row.get("t_rd")),
                license=USWTDB_LICENSE, tags=tags))
        except ContractError as error:
            ledger.refuse(where, _importers()._refusal(error))
    record = {"source": USWTDB_SOURCE, "license": USWTDB_LICENSE,
              "attribution": USWTDB_ATTRIBUTION,
              "files": [_importers()._file_record(path)],
              "encoding": encoding, "imported_utc": _importers()._now_utc(),
              "status_filter": list(statuses)}
    record.update(ledger.record())
    return AssetCollection(assets=assets, sources=[record])


# --------------------------------------------------------------------------
# GEM Global Wind Power Tracker


_GEM_REQUIRED = ("project name", "capacity (mw)", "latitude", "longitude",
                 "status")
_GEM_OPTIONAL = {
    "phase_name": ("phase name",),
    "installation": ("installation type",),
    "country": ("country/area", "country"),
    "start": ("start year",),
    "retired": ("retired year",),
    "operator": ("operator",),
    "owner": ("owner",),
    "accuracy": ("location accuracy",),
    "location_id": ("gem location id",),
    "phase_id": ("gem phase id",),
    "other_names": ("other name(s)",),
    "state": ("state/province",),
    "wiki": ("wiki url",),
}


def _gem_tables(path: Path) -> tuple[list[tuple[str, list[str],
                                                list[list[str]], int]], str]:
    """``[(sheet, header, rows, first data row number)]`` holding the
    required GEM columns."""

    if path.suffix.lower() in (".xlsx", ".xlsm"):
        sheets = read_xlsx(path)
        encoding = "xlsx"
    else:
        header, rows, encoding = _importers()._read_csv(path,
                                                        quotechars=('"',))
        sheets = [(path.name, [header] + [[row[h] for h in header]
                                          for row in rows])]
    tables = []
    for name, rows in sheets:
        for start in range(min(5, len(rows))):
            header = [_norm_header(c) for c in rows[start]]
            if all(c in header for c in _GEM_REQUIRED):
                # first data row's number: the sheet row for xlsx, the
                # CSV line otherwise
                tables.append((name, header, rows[start + 1:], start + 2))
                break
    if not tables:
        raise InventoryRefused(
            f"{path.name} is not a GEM Global Wind Power Tracker download: "
            f"no sheet has the columns {list(_GEM_REQUIRED)} (matched "
            "case-insensitively) in its first five rows")
    return tables, encoding


def _gem_installation(text: str | None) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("offshore"):
        return "offshore"
    if lowered.startswith("onshore"):
        return "onshore"
    return None


def import_gem(path: str | Path, *, statuses: Sequence[str] = DEFAULT_STATUSES
               ) -> AssetCollection:
    """GEM wind phases -> one ``plant`` Point per location (see above)."""

    path = Path(path)
    statuses = _check_statuses(statuses)
    tables, encoding = _gem_tables(path)
    ledger = _ledger()
    ledger.notes.append("phases sharing a GEM location ID and coordinates "
                        "merged into one plant; capacity summed")
    groups: dict[tuple, list[dict[str, Any]]] = {}
    order: list[tuple] = []
    status_counts = Counter()
    for sheet, header, rows, first_row in tables:
        index = {name: header.index(name) for name in header}

        def cell(row, *names):
            for name in names:
                if name in index and index[name] < len(row):
                    return _text(row[index[name]])
            return None

        for number, row in enumerate(rows, start=first_row):
            if not any(str(c).strip() for c in row):
                continue
            ledger.rows += 1
            phase_id = cell(row, *_GEM_OPTIONAL["phase_id"])
            location_id = cell(row, *_GEM_OPTIONAL["location_id"])
            where = f"{sheet} row {number}" + (
                f" ({phase_id or location_id})"
                if phase_id or location_id else "")
            raw_status = cell(row, "status")
            status = _gem_status(raw_status)
            if status is None:
                status = "unknown"
                ledger.derived[f"status {raw_status!r} read as unknown"] += 1
            status_counts[status] += 1
            if status not in statuses:
                ledger.skipped[f"status {status}"] += 1
                continue
            lat = _importers()._float(cell(row, "latitude"))
            lon = _importers()._float(cell(row, "longitude"))
            if lat is None or lon is None:
                ledger.refuse(where, "no Latitude/Longitude")
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                ledger.refuse(where, f"Latitude/Longitude out of range: "
                                     f"{lat:g}, {lon:g}")
                continue
            project = cell(row, "project name")
            if project is None:
                ledger.refuse(where, "no Project Name")
                continue
            phase = {
                "project": project, "status": status, "raw_status": raw_status,
                "lat": lat, "lon": lon,
                "capacity": _importers()._positive(
                    cell(row, "capacity (mw)")),
                "phase_id": phase_id, "location_id": location_id,
                "where": where,
                **{key: cell(row, *names) for key, names in
                   _GEM_OPTIONAL.items()
                   if key not in ("phase_id", "location_id")},
            }
            if location_id:
                key = ("location", location_id, round(lat, 6), round(lon, 6))
            else:
                key = ("row", where)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(phase)
    assets: list[Asset] = []
    used: Counter = Counter()
    for key in order:
        phases = groups[key]
        first = phases[0]
        ref = first["location_id"] or first["phase_id"] or first["where"]
        used[ref] += 1
        if used[ref] > 1:
            ref = f"{ref}#{used[ref]}"
        capacities = [p["capacity"] for p in phases if p["capacity"]]
        status = min((p["status"] for p in phases), key=STATUSES.index)
        tags: dict[str, str] = {TAG_STATUS: status}
        installations = {_gem_installation(p["installation"])
                         for p in phases} - {None}
        if len(installations) == 1:
            tags[TAG_INSTALLATION] = installations.pop()
        raw_installations = sorted({p["installation"] for p in phases
                                    if p["installation"]})
        if raw_installations:
            tags["gem:installation_type"] = "; ".join(raw_installations)
        tags["gem:status"] = "; ".join(sorted({p["raw_status"] or ""
                                                for p in phases}))
        if len(phases) > 1 or first["phase_name"]:
            tags["gem:phases"] = json.dumps(
                [{"phase_id": p["phase_id"], "phase_name": p["phase_name"],
                  "capacity_mw": p["capacity"], "status": p["status"],
                  "start_year": _year(p["start"])} for p in phases],
                separators=(",", ":"))
        starts = [y for y in (_year(p["start"]) for p in phases) if y]
        if starts:
            tags[TAG_START_YEAR] = min(starts)
        for key_name, tag in (("country", TAG_COUNTRY),
                              ("owner", "gem:owner"),
                              ("accuracy", "gem:location_accuracy"),
                              ("other_names", "gem:other_names"),
                              ("state", "gem:state_province"),
                              ("wiki", "gem:wiki_url"),
                              ("location_id", "gem:location_id"),
                              ("retired", "gem:retired_year")):
            values = sorted({p[key_name] for p in phases if p[key_name]})
            if values:
                tags[tag] = "; ".join(values)
        operators = sorted({p["operator"] for p in phases if p["operator"]})
        try:
            assets.append(Asset(
                asset_id=f"{GEM_SOURCE}:{ref}", kind="plant",
                geometry={"type": "Point",
                          "coordinates": [round(first["lon"], 7),
                                          round(first["lat"], 7)]},
                source=GEM_SOURCE, source_ref=ref, name=first["project"],
                operator="; ".join(operators) or None,
                generator_source="wind",
                capacity_mw=sum(capacities) if capacities else None,
                license=GEM_LICENSE, tags=tags))
        except ContractError as error:
            ledger.refuse(first["where"], _importers()._refusal(error))
    record = {"source": GEM_SOURCE, "license": GEM_LICENSE,
              "attribution": GEM_ATTRIBUTION,
              "files": [_importers()._file_record(path)],
              "encoding": encoding, "sheets": [t[0] for t in tables],
              "imported_utc": _importers()._now_utc(),
              "status_filter": list(statuses),
              "status_counts": dict(status_counts)}
    record.update(ledger.record())
    return AssetCollection(assets=assets, sources=[record])


# --------------------------------------------------------------------------
# EMODnet offshore wind farms


_EMODNET_CRS_OK = {"urn:ogc:def:crs:ogc:1.3:crs84", "epsg:4326",
                   "urn:ogc:def:crs:ogc::crs84", "crs84"}


def _emodnet_crs(document: Mapping[str, Any], path: Path) -> None:
    crs = document.get("crs")
    if crs is None:
        return
    name = ((crs.get("properties") or {}).get("name")
            if isinstance(crs, Mapping) else None)
    if not isinstance(name, str) or name.strip().lower() not in _EMODNET_CRS_OK:
        raise InventoryRefused(
            f"{path.name} declares crs {name!r}; only longitude-first "
            "WGS84 (CRS84 or EPSG:4326 short form, or no crs member) is "
            "read.  Request the WFS with srsName=urn:ogc:def:crs:OGC:1.3:"
            "CRS84; urn:ogc:def:crs:EPSG::4326 is latitude-first")


def import_emodnet(path: str | Path, *,
                   statuses: Sequence[str] = DEFAULT_STATUSES
                   ) -> AssetCollection:
    """EMODnet wind farm features -> ``plant`` assets (see above)."""

    path = Path(path)
    statuses = _check_statuses(statuses)
    text, _ = _importers()._read_text(path)
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise InventoryRefused(f"{path} is not JSON: {error}") from error
    if not isinstance(document, dict) or \
            document.get("type") != "FeatureCollection":
        raise InventoryRefused(f"{path.name} is not a GeoJSON "
                               "FeatureCollection")
    _emodnet_crs(document, path)
    ledger = _ledger()
    status_counts = Counter()
    assets, seen = [], set()
    for number, feature in enumerate(document.get("features") or [], start=1):
        ledger.rows += 1
        where = f"feature {number}"
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            ledger.refuse(where, "not a GeoJSON Feature")
            continue
        props = {_norm_header(k): v
                 for k, v in (feature.get("properties") or {}).items()}
        ref = (_text(feature.get("id")) or _text(props.get("objectid"))
               or _text(props.get("id")) or str(number))
        where = f"feature {number} ({ref})"
        if ref in seen:
            ledger.refuse(where, "duplicate id")
            continue
        raw_status = _text(props.get("status"))
        status = _EMODNET_STATUS.get((raw_status or "").lower())
        if status is None:
            status = "unknown"
            ledger.derived[f"STATUS {raw_status!r} read as unknown"] += 1
        status_counts[status] += 1
        if status not in statuses:
            ledger.skipped[f"status {status}"] += 1
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") not in (
                "Point", "Polygon", "MultiPolygon"):
            ledger.refuse(where, "geometry is not a Point, Polygon or "
                                 "MultiPolygon")
            continue
        tags = {TAG_STATUS: status, TAG_INSTALLATION: "offshore"}
        if raw_status:
            tags["emodnet:status"] = raw_status
        turbines = _importers()._count(props.get("n_turbines"))
        if turbines is not None:
            tags[TAG_TURBINES] = str(turbines)
        year = _year(props.get("year"))
        if year:
            tags[TAG_START_YEAR] = year
        country = _text(props.get("country"))
        if country:
            tags[TAG_COUNTRY] = country
        for key in ("dist_coast", "area_sqkm", "notes", "name_field"):
            value = _text(props.get(key))
            if value:
                tags[f"emodnet:{key}"] = value
        try:
            assets.append(Asset(
                asset_id=f"{EMODNET_SOURCE}:{ref}", kind="plant",
                geometry=geometry, source=EMODNET_SOURCE, source_ref=ref,
                name=_text(props.get("name")), generator_source="wind",
                capacity_mw=_importers()._positive(props.get("power_mw")),
                license=EMODNET_LICENSE, tags=tags))
            seen.add(ref)
        except ContractError as error:
            ledger.refuse(where, _importers()._refusal(error))
    record = {"source": EMODNET_SOURCE, "license": EMODNET_LICENSE,
              "attribution": EMODNET_ATTRIBUTION,
              "files": [_importers()._file_record(path)],
              "imported_utc": _importers()._now_utc(),
              "status_filter": list(statuses),
              "status_counts": dict(status_counts)}
    record.update(ledger.record())
    return AssetCollection(assets=assets, sources=[record])


# --------------------------------------------------------------------------
# OSM .pbf via pyosmium


OSM_PBF_INSTALL = ("pip install osmium  (pyosmium; conda-forge: "
                   "pixi add pyosmium)")


def _wind_source(tags: Mapping[str, str], key: str) -> bool:
    return any(part.strip().lower() == "wind"
               for part in str(tags.get(key, "")).split(";"))


def is_osm_wind(tags: Mapping[str, str]) -> bool:
    """``power=generator`` + ``generator:source`` wind, or ``power=plant``
    + ``plant:source`` wind."""

    power = tags.get("power")
    if power == "generator":
        return _wind_source(tags, "generator:source")
    if power == "plant":
        return _wind_source(tags, "plant:source")
    return False


def osm_elements_to_collection(elements: Iterable[Mapping[str, Any]],
                               path: Path | None = None
                               ) -> AssetCollection:
    """Overpass-shaped elements (``type``, ``id``, ``tags`` and ``lat``/
    ``lon``, ``geometry`` or ``members``) of OSM wind features -> assets,
    through :func:`woof.energy.osm.element_to_asset`."""

    from woof.energy import osm

    ledger = _ledger()
    assets, seen = [], set()
    for element in elements:
        ledger.rows += 1
        where = f"{element.get('type')}/{element.get('id')}"
        tags = element.get("tags") or {}
        if not is_osm_wind(tags):
            ledger.skipped["not a wind generator or plant"] += 1
            continue
        try:
            asset, why = osm.element_to_asset(element)
        except ContractError as error:
            ledger.refuse(where, _importers()._refusal(error))
            continue
        if asset is None:
            ledger.refuse(where, str(why))
            continue
        if why == osm.CENTROID_FALLBACK:
            ledger.fallbacks[str(why)] += 1
        if asset.asset_id in seen:
            ledger.refuse(where, "duplicate element")
            continue
        seen.add(asset.asset_id)
        asset = replace(asset, tags={**asset.tags, TAG_STATUS: "operating"})
        assets.append(asset)
    record = {"source": osm.SOURCE, "license": osm.LICENSE,
              "attribution": osm.ATTRIBUTION,
              "imported_utc": _importers()._now_utc(),
              "status_filter": ["operating"]}
    if path is not None:
        record["files"] = [_importers()._file_record(path)]
    record.update(ledger.record())
    return AssetCollection(assets=assets, sources=[record])


def _require_osmium():
    try:
        import osmium  # noqa: F401
    except ImportError as error:
        raise InventoryRefused(
            "--format osm-pbf reads .osm.pbf extracts with pyosmium, which "
            f"is not installed in this environment: {OSM_PBF_INSTALL}.  "
            "Alternatively fetch a bounded area with `woof energy fetch` "
            "(Overpass)") from error
    return osmium


def _osm_element_stream(path: Path, idx: str) -> list[dict[str, Any]]:
    """Wind features of a pbf as Overpass-shaped elements (pyosmium)."""

    osmium = _require_osmium()
    elements: list[dict[str, Any]] = []

    def tag_dict(taglist) -> dict[str, str]:
        return {tag.k: tag.v for tag in taglist}

    def located(refs) -> list[dict[str, float]] | None:
        out = []
        for ref in refs:
            location = ref.location
            if not location.valid():
                return None
            out.append({"lon": location.lon, "lat": location.lat})
        return out

    class Handler(osmium.SimpleHandler):
        def node(self, n):
            tags = tag_dict(n.tags)
            if not is_osm_wind(tags):
                return
            if not n.location.valid():
                elements.append({"type": "node", "id": n.id, "tags": tags})
                return
            elements.append({"type": "node", "id": n.id, "tags": tags,
                             "lon": n.location.lon, "lat": n.location.lat})

        def way(self, w):
            tags = tag_dict(w.tags)
            if not is_osm_wind(tags):
                return
            refs = list(w.nodes)
            elements.append({"type": "way", "id": w.id, "tags": tags,
                             "nodes": [r.ref for r in refs],
                             "geometry": located(refs) or []})

        def area(self, a):
            if a.from_way():
                return
            tags = tag_dict(a.tags)
            if not is_osm_wind(tags):
                return
            members = []
            for outer in a.outer_rings():
                ring = located(outer)
                if ring:
                    members.append({"type": "way", "role": "outer",
                                    "geometry": ring})
                for inner in a.inner_rings(outer):
                    ring = located(inner)
                    if ring:
                        members.append({"type": "way", "role": "inner",
                                        "geometry": ring})
            elements.append({"type": "relation", "id": a.orig_id(),
                             "tags": tags, "members": members})

    Handler().apply_file(str(path), locations=True, idx=idx)
    return elements


def import_osm_pbf(path: str | Path, *, statuses: Sequence[str] =
                   DEFAULT_STATUSES, idx: str = "flex_mem"
                   ) -> AssetCollection:
    """OSM wind turbines and plants from a pbf (pyosmium; see above)."""

    path = Path(path)
    statuses = _check_statuses(statuses)
    if "operating" not in statuses:
        raise InventoryRefused(
            f"--status {','.join(statuses)} keeps nothing from an OSM "
            "extract: every mapped power=generator/plant is operating")
    return osm_elements_to_collection(_osm_element_stream(path, idx), path)


# --------------------------------------------------------------------------
# names and capacities


@dataclass(frozen=True)
class _Name:
    base: frozenset
    distinguishing: frozenset
    joined: str


def name_key(text: str | None) -> _Name | None:
    """Normalised name tokens (see the module docstring), or ``None``."""

    if text is None or not str(text).strip():
        return None
    folded = unicodedata.normalize("NFKD", str(text))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    tokens = re.split(r"[^0-9a-z]+", folded.lower())
    base, distinguishing = set(), set()
    for token in tokens:
        if not token:
            continue
        if token in _NUMBER_WORDS:
            token = _NUMBER_WORDS[token]
        elif token in _ROMAN:
            token = _ROMAN[token]
        if token.isdigit():
            distinguishing.add(str(int(token)))
        elif token in _DIRECTIONS:
            distinguishing.add(f"dir:{_DIRECTIONS[token]}")
        elif token in _EXTENSION:
            distinguishing.add("extension")
        elif token not in NAME_STOPWORDS:
            base.add(token)
    if not base and not distinguishing:
        return None
    return _Name(frozenset(base), frozenset(distinguishing),
                 " ".join(sorted(base)))


def name_similarity(a: _Name | None, b: _Name | None) -> float | None:
    """0..1, or ``None`` when either name is missing; differing
    distinguishing tokens give 0."""

    if a is None or b is None:
        return None
    if a.distinguishing != b.distinguishing:
        return 0.0
    if not a.base and not b.base:
        return 1.0
    if not a.base or not b.base:
        return None
    jaccard = len(a.base & b.base) / len(a.base | b.base)
    ratio = difflib.SequenceMatcher(None, a.joined, b.joined).ratio()
    return max(jaccard, ratio)


def _capacity_verdict(a: float | None, b: float | None,
                      tolerance: float) -> str:
    if a is None or b is None:
        return "unknown"
    return "agree" if abs(a - b) <= tolerance * max(a, b) else "disagree"


# --------------------------------------------------------------------------
# merge


@dataclass(frozen=True)
class MergeRules:
    """Thresholds of :func:`merge_inventory` (module docstring)."""

    turbine_match_m: float = 50.0
    plant_near_m: float = 1000.0
    plant_match_m: float = 5000.0
    name_threshold: float = 0.7
    name_floor: float = 0.3
    capacity_tolerance: float = 0.25
    plant_link_m: float = 10000.0
    plant_link_near_m: float = 2000.0
    turbine_coverage: float = 0.8
    prefer: tuple[str, ...] = DEFAULT_PREFER

    def __post_init__(self):
        for name in ("turbine_match_m", "plant_near_m", "plant_match_m",
                     "plant_link_m", "plant_link_near_m"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value)
                    and value > 0.0):
                raise InventoryRefused(f"{name} must be a positive distance, "
                                       f"got {value!r}")
        for name in ("name_threshold", "name_floor", "capacity_tolerance",
                     "turbine_coverage"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and 0.0 <= value <= 1.0):
                raise InventoryRefused(f"{name} must lie in [0, 1], got "
                                       f"{value!r}")
        if self.name_floor > self.name_threshold:
            raise InventoryRefused("name_floor must not exceed name_threshold")
        if self.plant_near_m > self.plant_match_m:
            raise InventoryRefused("plant_near_m must not exceed "
                                   "plant_match_m")
        if self.plant_link_near_m > self.plant_link_m:
            raise InventoryRefused("plant_link_near_m must not exceed "
                                   "plant_link_m")


def _provenance(asset: Asset) -> list[dict[str, Any]]:
    raw = asset.tags.get(TAG_PROVENANCE)
    if raw:
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError as error:
            raise InventoryRefused(f"{asset.asset_id}: {TAG_PROVENANCE} is "
                                   f"not JSON: {error}") from error
        if isinstance(entries, list) and all(isinstance(e, dict)
                                             for e in entries):
            return entries
        raise InventoryRefused(f"{asset.asset_id}: {TAG_PROVENANCE} must be "
                               "a JSON list of objects")
    return [{"source": asset.source, "ref": asset.source_ref,
             "asset_id": asset.asset_id, "license": asset.license}]


def _licenses(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    found = set()
    for entry in entries:
        license_ = entry.get("license")
        if isinstance(license_, str) and license_.strip():
            for part in license_.split(" AND "):
                found.add(part.strip())
    return sorted(found)


@dataclass
class _Item:
    asset: Asset
    lon: float
    lat: float
    name: _Name | None
    project: _Name | None
    status: str
    installation: str | None
    polygons: list | None


def _item(asset: Asset) -> _Item:
    from woof.energy.sites import _centroid, _polygons_of

    lon, lat, _ = _centroid(asset.geometry)
    polygons = (_polygons_of(asset.geometry)
                if asset.geometry["type"] in ("Polygon", "MultiPolygon")
                else None)
    return _Item(asset=asset, lon=lon, lat=lat, name=name_key(asset.name),
                 project=name_key(asset.tags.get(TAG_PROJECT)),
                 status=asset_status(asset),
                 installation=asset_installation(asset), polygons=polygons)


class _Clusters:
    """Union-find whose clusters hold at most one record per source.

    A merged asset stands for every record in its provenance, so a cluster
    tracks ``source -> record asset_ids``; two clusters may join only when
    every source they share names the same records (two merged assets that
    both carry GEM record G are the same farm; one carrying G and one
    carrying G' are not joined by a distance rule).
    """

    def __init__(self, items: Sequence[_Item]):
        self.parent = list(range(len(items)))
        self.records: list[dict[str, set]] = []
        for item in items:
            records: dict[str, set] = defaultdict(set)
            for entry in _provenance(item.asset):
                records[str(entry.get("source"))].add(entry.get("asset_id"))
            self.records.append(records)

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> bool:
        a, b = self.find(i), self.find(j)
        if a == b:
            return False
        ra, rb = self.records[a], self.records[b]
        if any(ra[source] != rb[source] for source in set(ra) & set(rb)):
            return False
        self._link(a, b)
        return True

    def _link(self, a: int, b: int) -> None:
        self.parent[b] = a
        for source, ids in self.records[b].items():
            self.records[a][source] |= ids

    def join_shared_records(self, items: Sequence[_Item], counts: Counter,
                            what: str) -> None:
        """Join assets whose provenance names the same record (re-merges
        of documents that both absorbed it)."""

        holders: dict[Any, list[int]] = defaultdict(list)
        for index, item in enumerate(items):
            for entry in _provenance(item.asset):
                holders[entry.get("asset_id")].append(index)
        for indices in holders.values():
            for other in indices[1:]:
                a, b = self.find(indices[0]), self.find(other)
                if a != b:
                    self._link(a, b)
                    counts[f"{what} joined by a shared provenance record"] \
                        += 1

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return [sorted(members) for _, members in sorted(out.items())]


def _inside(item: _Item, lon: float, lat: float) -> bool:
    if item.polygons is None:
        return False
    return bool(np.asarray(geo.point_in_polygon(
        np.array([lon]), np.array([lat]), item.polygons))[0])


def _plant_distance(a: _Item, b: _Item) -> float:
    if _inside(a, b.lon, b.lat) or _inside(b, a.lon, a.lat):
        return 0.0
    return float(_haversine_m(a.lon, a.lat, b.lon, b.lat))


def _rank(rules: MergeRules, source: str) -> int:
    try:
        return rules.prefer.index(source)
    except ValueError:
        return len(rules.prefer)


def _combine(items: Sequence[_Item], rules: MergeRules, kind_counter: Counter
             ) -> Asset:
    """Winner of a cluster with fields filled and provenance joined."""

    ordered = sorted(items, key=lambda it: (it.polygons is None,
                                            _rank(rules, it.asset.source),
                                            it.asset.asset_id))
    winner = ordered[0].asset
    updates: dict[str, Any] = {}
    for name in ("name", "operator", "capacity_mw", "hub_height_m",
                 "rotor_diameter_m"):
        if getattr(winner, name) is None:
            for other in ordered[1:]:
                value = getattr(other.asset, name)
                if value is not None:
                    updates[name] = value
                    kind_counter[f"{name} filled from {other.asset.source}"] \
                        += 1
                    break
    tags = dict(winner.tags)
    status = ordered[0].status
    if status == "unknown":
        status = next((it.status for it in ordered if it.status != "unknown"),
                      "unknown")
    installation = next((it.installation for it in ordered
                         if it.installation is not None), None)
    for other in ordered[1:]:
        for key, value in other.asset.tags.items():
            if key.startswith("woof:") or key == "plant":
                continue
            tags.setdefault(key, value)
        for key in (TAG_PROJECT, TAG_START_YEAR, TAG_TURBINES, TAG_COUNTRY):
            if key in other.asset.tags:
                tags.setdefault(key, other.asset.tags[key])
    entries: list[dict[str, Any]] = []
    for item in ordered:
        for entry in _provenance(item.asset):
            if entry not in entries:
                entries.append(entry)
    licenses = _licenses(entries)
    tags[TAG_STATUS] = status
    if installation is not None:
        tags[TAG_INSTALLATION] = installation
    tags[TAG_PROVENANCE] = json.dumps(entries, separators=(",", ":"),
                                      sort_keys=True)
    if licenses:
        tags[TAG_LICENSES] = ";".join(licenses)
    updates["tags"] = tags
    updates["license"] = " AND ".join(licenses) if licenses else None
    return replace(winner, **updates)


def _cluster_turbines(items: list[_Item], rules: MergeRules,
                      counts: Counter) -> list[list[int]]:
    clusters = _Clusters(items)
    clusters.join_shared_records(items, counts, "turbines")
    if len(items) > 1:
        tree = _tree(_xyz([it.lon for it in items], [it.lat for it in items]))
        pairs = []
        for i, j in tree.query_pairs(_chord(rules.turbine_match_m)):
            a, b = items[i], items[j]
            if a.asset.source == b.asset.source:
                continue
            d = float(_haversine_m(a.lon, a.lat, b.lon, b.lat))
            if d <= rules.turbine_match_m:
                pairs.append((d, min(i, j), max(i, j)))
        for d, i, j in sorted(pairs):
            if clusters.union(i, j):
                counts["turbines joined"] += 1
    return clusters.groups()


def _plant_pair_rule(a: _Item, b: _Item, distance: float,
                     rules: MergeRules) -> int | None:
    if (a.installation and b.installation
            and a.installation != b.installation):
        return None
    capacity = _capacity_verdict(a.asset.capacity_mw, b.asset.capacity_mw,
                                 rules.capacity_tolerance)
    if capacity == "disagree":
        return None
    similarity = name_similarity(a.name, b.name)
    if (similarity is not None and similarity >= rules.name_threshold
            and distance <= rules.plant_match_m):
        return 1
    if distance <= rules.plant_near_m and (similarity is None
                                           or similarity >= rules.name_floor):
        return 2
    return None


def _cluster_plants(items: list[_Item], rules: MergeRules,
                    counts: Counter) -> list[list[int]]:
    clusters = _Clusters(items)
    clusters.join_shared_records(items, counts, "plants")
    if len(items) > 1:
        xyz = _xyz([it.lon for it in items], [it.lat for it in items])
        tree = _tree(xyz)
        candidates = set(tree.query_pairs(_chord(rules.plant_match_m)))
        # A centroid inside another plant's polygon lies within that
        # polygon's radius of its centroid: search each polygon by its own
        # radius, not the whole input by the largest one.
        for k, it in enumerate(items):
            if it.polygons is None:
                continue
            ring = np.asarray([p for poly in it.polygons for p in poly[0]])
            reach = float(np.max(_haversine_m(it.lon, it.lat,
                                              ring[:, 0], ring[:, 1])))
            if reach <= rules.plant_match_m:
                continue
            for other in tree.query_ball_point(xyz[k], _chord(reach)):
                if other != k:
                    candidates.add((min(k, other), max(k, other)))
        pairs = []
        for i, j in sorted(candidates):
            a, b = items[i], items[j]
            if a.asset.source == b.asset.source:
                continue
            distance = _plant_distance(a, b)
            rule = _plant_pair_rule(a, b, distance, rules)
            if rule is not None:
                pairs.append((rule, distance, min(i, j), max(i, j)))
        for rule, distance, i, j in sorted(pairs):
            if clusters.union(i, j):
                counts[f"plants joined by rule {rule}"] += 1
    return clusters.groups()


def _link_turbines(plants: list[_Item], turbines: list[_Item],
                   rules: MergeRules, counts: Counter
                   ) -> tuple[dict[int, int], dict[int, list[int]]]:
    """``turbine index -> plant index`` for the best plant of each turbine,
    and the turbines offered to each plant."""

    best: dict[int, tuple] = {}
    if not plants or not turbines:
        return {}, {}
    tree = _tree(_xyz([t.lon for t in turbines], [t.lat for t in turbines]))
    for p_index, plant in enumerate(plants):
        reach = rules.plant_link_m
        if plant.polygons is not None:
            ring = np.asarray([p for poly in plant.polygons for p in poly[0]])
            reach += float(np.max(_haversine_m(plant.lon, plant.lat,
                                               ring[:, 0], ring[:, 1])))
        candidates = tree.query_ball_point(
            _xyz([plant.lon], [plant.lat])[0], _chord(reach))
        if not candidates:
            continue
        inside = np.zeros(len(candidates), dtype=bool)
        if plant.polygons is not None:
            inside = np.asarray(geo.point_in_polygon(
                np.array([turbines[c].lon for c in candidates]),
                np.array([turbines[c].lat for c in candidates]),
                plant.polygons), dtype=bool)
        similarity_cache: dict[Any, float | None] = {}
        for c, is_inside in zip(candidates, inside):
            turbine = turbines[c]
            if (plant.installation and turbine.installation
                    and plant.installation != turbine.installation):
                continue
            distance = float(_haversine_m(plant.lon, plant.lat,
                                          turbine.lon, turbine.lat))
            key = turbine.project
            if key not in similarity_cache:
                similarity_cache[key] = name_similarity(plant.name, key)
            similarity = similarity_cache[key]
            if is_inside and (similarity is None
                              or similarity >= rules.name_floor):
                strength = 0
            elif (similarity is not None
                  and similarity >= rules.name_threshold
                  and distance <= rules.plant_link_m):
                strength = 1
            elif (distance <= rules.plant_link_near_m
                  and (similarity is None
                       or similarity >= rules.name_floor)):
                strength = 2
            else:
                continue
            offer = (strength, distance, p_index)
            if c not in best or offer < best[c]:
                best[c] = offer
    links = {t: offer[2] for t, offer in best.items()}
    offered: dict[int, list[int]] = defaultdict(list)
    for t, p in links.items():
        offered[p].append(t)
    counts["turbines linked to a plant"] = len(links)
    return links, offered


def _coverage(plant: Asset, turbines: Sequence[Asset], rules: MergeRules
              ) -> bool:
    if not turbines:
        return False
    if plant.capacity_mw is None:
        count = plant.tags.get(TAG_TURBINES)
        if count and count.isdigit():
            return len(turbines) >= rules.turbine_coverage * int(count)
        # neither capacity nor turbine count: coverage cannot be shown, so
        # the plant keeps its site (a duplicate site, never a lost farm)
        return False
    total = sum(t.capacity_mw if t.capacity_mw is not None
                else COVERAGE_TURBINE_MW for t in turbines)
    return total >= rules.turbine_coverage * plant.capacity_mw


def merge_inventory(collections: Sequence[AssetCollection], *,
                    rules: MergeRules | None = None,
                    statuses: Sequence[str] = DEFAULT_STATUSES,
                    keep_non_wind: bool = False,
                    inputs: Sequence[Mapping[str, Any]] = ()
                    ) -> tuple[AssetCollection, dict[str, Any]]:
    """One deduplicated wind inventory from several assets documents.

    Rules and their order: module docstring.  ``inputs`` are file
    references recorded in the merge provenance record.
    """

    rules = rules or MergeRules()
    statuses = _check_statuses(statuses)
    if not collections:
        raise InventoryRefused("nothing to merge: no input documents")
    counts: Counter = Counter()
    by_status: Counter = Counter()
    seen_ids: set[str] = set()
    seen_refs: set[tuple[str, str]] = set()
    wind: list[Asset] = []
    passthrough: list[Asset] = []
    sources: list[dict] = []
    offered = 0
    absorbed = {entry.get("asset_id") for collection in collections
                for asset in collection.assets
                for entry in _provenance(asset)
                if entry.get("asset_id") != asset.asset_id}
    for collection in collections:
        sources.extend(dict(r) for r in collection.sources)
        for asset in collection.assets:
            offered += 1
            if (asset.asset_id in absorbed
                    and len(_provenance(asset)) == 1):
                counts["dropped: already merged into another asset"] += 1
                continue
            if asset.asset_id in seen_ids:
                counts["dropped: same asset_id"] += 1
                continue
            ref = (asset.source, asset.source_ref)
            if asset.source_ref is not None and ref in seen_refs:
                counts["dropped: same source and source_ref"] += 1
                continue
            seen_ids.add(asset.asset_id)
            if asset.source_ref is not None:
                seen_refs.add(ref)
            if not _wind(asset):
                if keep_non_wind:
                    passthrough.append(asset)
                else:
                    counts["dropped: not a wind plant or generator"] += 1
                continue
            status = asset_status(asset)
            if status not in statuses:
                by_status[status] += 1
                continue
            wind.append(asset)
    items = [_item(a) for a in wind]
    turbine_items = [it for it in items if it.asset.kind == "generator"
                     and it.asset.geometry["type"] == "Point"]
    plant_items = [it for it in items if it.asset.kind == "plant"]
    # Wind generators mapped as areas: kept as they are, never matched.
    other_items = [it for it in items if it.asset.kind == "generator"
                   and it.asset.geometry["type"] != "Point"]
    fills: Counter = Counter()
    merged_turbines = [
        _combine([turbine_items[i] for i in group], rules, fills)
        for group in _cluster_turbines(turbine_items, rules, counts)]
    merged_plants = [
        _combine([plant_items[i] for i in group], rules, fills)
        for group in _cluster_plants(plant_items, rules, counts)]
    merged_plants += [_combine([it], rules, fills) for it in other_items]
    plant_view = [_item(a) for a in merged_plants if a.kind == "plant"]
    turbine_view = [_item(a) for a in merged_turbines]
    links, offered_to = _link_turbines(plant_view, turbine_view, rules, counts)
    covered: dict[str, list[int]] = {}
    partial: dict[str, int] = {}
    for p_index, t_indices in offered_to.items():
        plant = plant_view[p_index].asset
        members = [turbine_view[t].asset for t in t_indices]
        if _coverage(plant, members, rules):
            covered[plant.asset_id] = t_indices
        else:
            partial[plant.asset_id] = len(t_indices)
    turbine_plant: dict[int, str] = {}
    for plant_id, t_indices in covered.items():
        for t in t_indices:
            turbine_plant[t] = plant_id
    # A turbine's existing ``plant`` tag may name a record that is now part
    # of another cluster's winner: point it at the winner.
    plant_out_id: dict[Any, str] = {}
    for asset in merged_plants:
        for entry in _provenance(asset):
            plant_out_id[entry.get("asset_id")] = asset.asset_id
    output_plants = {a.asset_id for a in merged_plants}
    tagged: Counter = Counter()
    out_turbines = []
    for t, asset in enumerate(merged_turbines):
        tags = dict(asset.tags)
        existing = tags.get("plant")
        if existing is not None and plant_out_id.get(existing,
                                                     existing) != existing:
            existing = tags["plant"] = plant_out_id[existing]
            counts["turbine plant tag re-pointed at its merged plant"] += 1
        plant_id = turbine_plant.get(t)
        if plant_id is not None:
            if (existing is None or existing == plant_id
                    or existing not in output_plants):
                tags["plant"] = plant_id
                tagged[plant_id] += 1
            else:
                counts["turbine plant tag kept (names another plant)"] += 1
        out_turbines.append(asset if tags == asset.tags
                            else replace(asset, tags=tags))
    out_plants = []
    turbine_level = partly = 0
    stale = (TAG_TURBINE_LEVEL, TAG_TURBINES_LINKED, TAG_TURBINES_PARTIAL)
    for asset in merged_plants:
        tags = {k: v for k, v in asset.tags.items() if k not in stale}
        if tagged[asset.asset_id]:
            tags[TAG_TURBINE_LEVEL] = "yes"
            tags[TAG_TURBINES_LINKED] = str(tagged[asset.asset_id])
            turbine_level += 1
        elif asset.asset_id in covered or asset.asset_id in partial:
            tags[TAG_TURBINES_PARTIAL] = str(
                len(covered.get(asset.asset_id, ()))
                or partial[asset.asset_id])
            partly += 1
        out_plants.append(asset if tags == asset.tags
                          else replace(asset, tags=tags))
    counts["plants turbine-level"] = turbine_level
    counts["plants partly covered by turbines"] = partly
    assets = out_plants + out_turbines + passthrough
    report = {
        "offered": offered,
        "wind_after_status_filter": len(wind),
        "status_filter": list(statuses),
        "dropped_by_status": dict(by_status),
        "rules": {**{k: getattr(rules, k) for k in (
            "turbine_match_m", "plant_near_m", "plant_match_m",
            "name_threshold", "name_floor", "capacity_tolerance",
            "plant_link_m", "plant_link_near_m", "turbine_coverage")},
                  "prefer": list(rules.prefer)},
        "counts": dict(counts),
        "attributes_filled": dict(fills),
        "assets": len(assets),
        "by_kind": dict(Counter(a.kind for a in assets)),
        "by_source": dict(Counter(a.source for a in assets)),
        "by_status": dict(Counter(asset_status(a) for a in assets
                                  if _wind(a))),
        "by_installation": dict(Counter(asset_installation(a) or "unknown"
                                        for a in assets if _wind(a))),
        "multi_source_assets": sum(
            1 for a in assets if len(json.loads(
                a.tags.get(TAG_PROVENANCE, "[]"))) > 1),
        "keep_non_wind": bool(keep_non_wind),
    }
    record = {"source": MERGE_RECORD_SOURCE, "license": None,
              "imported_utc": _importers()._now_utc(),
              "inputs": [dict(i) for i in inputs], **report}
    return AssetCollection(assets=assets, sources=sources + [record]), report


# --------------------------------------------------------------------------
# downloads (network; optional, untested)


DOWNLOADS = {
    "uswtdb": "https://energy.usgs.gov/api/uswtdb/v1/turbines",
    "emodnet": ("https://ows.emodnet-humanactivities.eu/wfs?service=WFS&"
                "version=2.0.0&request=GetFeature&typeNames=emodnet:"
                "windfarmspoly&outputFormat=application/json&srsName="
                "urn:ogc:def:crs:OGC:1.3:CRS84"),
}
DOWNLOAD_GUIDANCE = {
    "gem": ("GEM hands the Global Wind Power Tracker out through a form: "
            "https://globalenergymonitor.org/projects/global-wind-power-"
            "tracker/download-data/ ; save the .xlsx and import it with "
            "--format gem"),
    "osm": ("download a regional extract from https://download.geofabrik.de"
            "/ or the planet from https://planet.openstreetmap.org/ , cut it "
            "with `osmium tags-filter IN.osm.pbf nwr/power=generator,plant "
            "-o power.osm.pbf` and import it with --format osm-pbf"),
}


def download(dataset: str, output: str | Path, *, timeout_s: float = 300.0
             ) -> dict[str, Any]:
    """Fetch a dataset to ``output`` (``.partial`` then rename).  GEM and
    OSM are refused with how to get them."""

    if dataset in DOWNLOAD_GUIDANCE:
        raise InventoryRefused(DOWNLOAD_GUIDANCE[dataset])
    if dataset not in DOWNLOADS:
        raise InventoryRefused(f"unknown dataset {dataset!r}")
    url = DOWNLOADS[dataset]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".partial")
    request = urllib.request.Request(url, headers={
        "User-Agent": "woof-energy-inventory (+https://github.com/"
                      "recastsystems/woof)"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response, \
            open(staging, "wb") as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
    staging.replace(output)
    return {"dataset": dataset, "url": url, "output": str(output),
            "sha256": file_ref(output)["sha256"],
            "retrieved_utc": _importers()._now_utc()}


# --------------------------------------------------------------------------
# CLI (``woof energy inventory``)


def _merge_main(args) -> int:
    rules = MergeRules(
        turbine_match_m=args.turbine_match_m, plant_near_m=args.plant_near_m,
        plant_match_m=args.plant_match_m, name_threshold=args.name_threshold,
        name_floor=args.name_floor,
        capacity_tolerance=args.capacity_tolerance,
        plant_link_m=args.plant_link_m,
        plant_link_near_m=args.plant_link_near_m,
        turbine_coverage=args.turbine_coverage,
        prefer=tuple(args.prefer) if args.prefer else DEFAULT_PREFER)
    collections, inputs = [], []
    for name in args.assets:
        path = Path(name)
        if not path.is_file():
            raise InventoryRefused(f"{path} does not exist")
        collections.append(load_assets(path))
        inputs.append(file_ref(path))
    merged, report = merge_inventory(
        collections, rules=rules, statuses=args.status,
        keep_non_wind=args.keep_non_wind, inputs=inputs)
    if not merged.assets:
        raise InventoryRefused(
            f"the merge kept no assets (dropped by status: "
            f"{report['dropped_by_status']}; counts: {report['counts']})")
    dump_assets(merged, args.output)
    print(json.dumps({"schema": SUMMARY_SCHEMA, "command": "merge",
                      "output": str(args.output), **report},
                     indent=2, default=str))
    return 0


def _download_main(args) -> int:
    record = download(args.dataset, args.output, timeout_s=args.timeout_s)
    print(json.dumps({"schema": SUMMARY_SCHEMA, "command": "download",
                      **record}, indent=2))
    return 0


def main(args) -> int:
    command = getattr(args, "inventory_command", None)
    try:
        if command == "merge":
            return _merge_main(args)
        if command == "download":
            return _download_main(args)
        print("woof energy inventory: choose merge or download",
              file=sys.stderr)
        return 2
    except (InventoryRefused, ContractError, OSError) as error:
        print(f"woof energy inventory {command}: refused: {error}",
              file=sys.stderr)
        print(json.dumps({"schema": SUMMARY_SCHEMA, "command": command,
                          "refusal": str(error)}, indent=2))
        return 2


__all__ = [
    "INVENTORY_FORMATS", "STATUSES", "DEFAULT_STATUSES", "DEFAULT_PREFER",
    "TAG_STATUS", "TAG_INSTALLATION", "TAG_PROVENANCE", "TAG_LICENSES",
    "TAG_PROJECT", "TAG_START_YEAR", "TAG_TURBINES", "TAG_COUNTRY",
    "TAG_TURBINE_LEVEL", "TAG_TURBINES_PARTIAL", "TAG_TURBINES_LINKED",
    "MERGE_RECORD_SOURCE", "SUMMARY_SCHEMA", "COVERAGE_TURBINE_MW",
    "NAME_STOPWORDS", "InventoryRefused", "MergeRules", "asset_status",
    "asset_installation", "filter_status", "read_xlsx", "import_uswtdb",
    "import_gem", "import_emodnet", "import_osm_pbf", "is_osm_wind",
    "osm_elements_to_collection", "name_key", "name_similarity",
    "merge_inventory", "download", "main",
]
