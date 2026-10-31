"""Print the test files one CI leg runs, one per line.

    python .github/ci/select_tests.py shard INDEX COUNT   # tests/test_*.py, leg INDEX of COUNT
    python .github/ci/select_tests.py hex                 # hex Tier 1 (paths relative to components/hex)
    python .github/ci/select_tests.py check COUNT         # every tests/ file is in exactly one leg

The tests/ tree is split into COUNT legs by a hash of each file's path, so a
file's leg depends only on its own name: adding or removing a file never moves
any other file to a different leg, and two runs of the same commit select the
same files.

The hex list is components/hex/tools/battery/cpu_files.txt, read as written:
nobody retypes it into the workflow (that README's rule).  Its paths are
relative to components/hex, which is where its pytest runs.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEX = ROOT / "components" / "hex"


def tests_tree() -> list[str]:
    return sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("test_*.py"))


def leg(path: str, count: int) -> int:
    return int.from_bytes(hashlib.sha256(path.encode()).digest()[:8], "big") % count


def shard(index: int, count: int) -> list[str]:
    if not 0 <= index < count:
        raise SystemExit(f"shard index {index} is outside 0..{count - 1}")
    return [p for p in tests_tree() if leg(p, count) == index]


def hex_tier1() -> list[str]:
    lines = (HEX / "tools" / "battery" / "cpu_files.txt").read_text(encoding="utf-8").splitlines()
    files = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    missing = [f for f in files if not (HEX / f).is_file()]
    if missing:
        raise SystemExit("cpu_files.txt names files that do not exist under components/hex: "
                         + ", ".join(missing))
    return files


def main(argv: list[str]) -> int:
    if argv[:1] == ["shard"] and len(argv) == 3:
        print("\n".join(shard(int(argv[1]), int(argv[2]))))
    elif argv == ["hex"]:
        print("\n".join(hex_tier1()))
    elif argv[:1] == ["check"] and len(argv) == 2:
        count = int(argv[1])
        legs = [shard(i, count) for i in range(count)]
        seen = [p for files in legs for p in files]
        if sorted(seen) != tests_tree() or len(seen) != len(set(seen)):
            raise SystemExit("the legs do not partition tests/")
        print(" ".join(f"leg{i}={len(files)}" for i, files in enumerate(legs)), f"total={len(seen)}")
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
