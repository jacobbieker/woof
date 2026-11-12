"""TUV-x input data on demand: pinned NCAR TUV-x v0.17.0 files, fetched once and cached under ``~/.woof``.

WOOF's TUV-x port (photolysis for MICM, ``photolysis = "tuvx"``) reads the cross sections, quantum yields,
wavelength grid and extraterrestrial flux files TUV-x's own configuration names.  They are NCAR's
(Apache-2.0) and are neither committed nor packaged in a wheel: each is fetched on first use from the TUV-x
repository at the commit the oracle (tools/tuvx_oracle) builds, verified against its size and SHA-256 pin
before an atomic install (woof.table_assets, the Rust fetch bridge), and kept under
``~/.woof/tuvx/<commit>/`` with TUV-x's own relative paths, so a configuration's ``data/...`` paths resolve
against that root unchanged.

``WOOF_TUVX_DATA`` names another root (a mirror or an offline copy); it is verified file by file under the
same pins and never written.  ``WOOF_TUVX_DATA_URL_BASE`` replaces the download base (the relative path is
appended).  Offline, :func:`ensure` with ``from_dir`` stages a local copy under the same verification.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

#: NCAR/tuv-x v0.17.0, the commit tools/tuvx_oracle/build.sh checks out.
TUVX_COMMIT = "ebec1cb47084fd102e338fb896c6d2b80fbcd292"
URL_BASE = f"https://raw.githubusercontent.com/NCAR/tuv-x/{TUVX_COMMIT}"
ENV_ROOT = "WOOF_TUVX_DATA"
ENV_URL_BASE = "WOOF_TUVX_DATA_URL_BASE"

#: relative path -> (bytes, sha256): the files tools/tuvx_oracle/photolysis.json reads (O2, O3 -> O1D, O3 -> O3P
#: and NO2 photolysis on TUV-x's tuv_5_4 configuration), measured at TUVX_COMMIT.
FILES = {
    "data/cross_sections/NO2_1.nc": (11042, "1edf2c7c3fa447a8c411c741750be105202f05444d7ff4814d125a74a3d6004b"),
    "data/cross_sections/O2_1.nc": (9458, "bda552277f0099ef3218968fa58ac84c41e2852bef2b726b268b85bb9a2f1b34"),
    "data/cross_sections/O2_parameters.txt": (
        16824, "c410c4db24620128aa97fde8e92a229ce0911d4c6b753dda89950fcf100be468"),
    "data/cross_sections/O3_1.nc": (1027060, "7440a28625d4efa5d1e6fb572481b4d56a2311d8b15182257372540bdf20904a"),
    "data/cross_sections/O3_2.nc": (610512, "1434585d9c54cb3592ccadeadf635a4b9682e13d2243a6d851184c0b083c161a"),
    "data/cross_sections/O3_3.nc": (14193, "f3f349c0878d32f1ea9c9c48afa9ecf2f8d7a8cb29986f53325ee86c125e4095"),
    "data/cross_sections/O3_4.nc": (14421, "7676bd8464258361db1e6411fbbfc2536cf0fd3564f7bcf3760b625ffa1f50ec"),
    "data/grids/wavelength/combined.grid": (
        1732, "740bec041ab0e56d4bbea06234542abc2c25c497bf7fca98c0749e706ed3b4d2"),
    "data/profiles/solar/atlas3_1994_317_a.dat": (
        134282, "7f58fb1a36af1cbbb211742c9b81a6d86996841a8744e172837ddf36eeec06dc"),
    "data/profiles/solar/neckel.flx": (12915, "727c29b2367bb5b09cbdc3e29885c45f6002cee7d22b5c5bdc7eccaac89aa0fa"),
    "data/profiles/solar/sao2010.solref.converted": (
        2082665, "b2dcafe6bd98fd9ef7227677f8ee5dca6f077ff1ae281eb3a9e5e2bb1bfe71f0"),
    "data/profiles/solar/susim_hi.flx": (145612, "1abb673db750d1654680fe70223f320f22cff1022dfa506dd301824cb08ffcdb"),
    "data/quantum_yields/NO2_1.nc": (10610, "b6882fc1e7b547590d48d2386e96032bed3a2f2c794670579dbcd9f3b7a0c937"),
}


class TuvxDataError(RuntimeError):
    pass


def _asset(relative):
    from woof.core.thompson_contract import TableAsset
    size, sha256 = FILES[relative]
    return TableAsset(Path(relative).name, size, sha256)


def cache_root() -> Path:
    """``~/.woof/tuvx/<commit>``: the fetch target (beside ``~/.woof/tables``)."""
    return Path.home() / ".woof" / "tuvx" / TUVX_COMMIT


def _verify(path: Path, relative: str) -> None:
    size, sha256 = FILES[relative]
    if not path.is_file():
        raise TuvxDataError(f"{relative}: {path} does not exist")
    if path.stat().st_size != size:
        raise TuvxDataError(f"{relative}: {path} has {path.stat().st_size} bytes, the pin {size}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise TuvxDataError(f"{relative}: {path} SHA-256 {digest} differs from the pin {sha256}")


def ensure(files=None, *, root=None, from_dir=None) -> Path:
    """The verified data root holding ``files`` (default: every pinned file), fetching what is missing.

    ``WOOF_TUVX_DATA`` (or ``root`` naming an existing read-only copy) is only verified.  Otherwise each file
    lands in :func:`cache_root` from ``from_dir`` (a local copy laid out as TUV-x's tree) or the pinned URL."""
    from woof import table_assets
    names = tuple(FILES) if files is None else tuple(files)
    unknown = sorted(set(names) - set(FILES))
    if unknown:
        raise TuvxDataError(f"{unknown} are not pinned TUV-x data files; pinned: {sorted(FILES)}")
    explicit = root if root is not None else os.environ.get(ENV_ROOT, "").strip() or None
    if explicit is not None:
        base = Path(explicit).expanduser().resolve()
        for relative in names:
            _verify(base / relative, relative)
        return base
    base = cache_root()
    url_base = os.environ.get(ENV_URL_BASE, "").strip().rstrip("/") or URL_BASE
    for relative in names:
        directory = base / Path(relative).parent
        directory.mkdir(parents=True, exist_ok=True)
        try:
            if from_dir is not None:
                table_assets.fetch_asset_from_dir(directory, _asset(relative), Path(from_dir) / Path(relative).parent)
            else:
                table_assets.fetch_asset_from_url(directory, _asset(relative), f"{url_base}/{relative}")
        except table_assets.TableAssetError as error:
            raise TuvxDataError(f"TUV-x data {relative}: {error}") from error
        _verify(base / relative, relative)
    return base
