"""Sum pytest junit XML into one line per file, for the job summary.

    python .github/ci/junit_summary.py LABEL junit/*.xml >> "$GITHUB_STEP_SUMMARY"

Prints passed / failed / errors / skipped and the wall time pytest recorded,
so a leg whose tests all skipped reads as such rather than as green.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def counts(path: str) -> tuple[int, int, int, int, float]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = failures = errors = skipped = 0
    seconds = 0.0
    for s in suites:
        tests += int(s.get("tests", 0))
        failures += int(s.get("failures", 0))
        errors += int(s.get("errors", 0))
        skipped += int(s.get("skipped", 0))
        seconds += float(s.get("time", 0.0))
    return tests - failures - errors - skipped, failures, errors, skipped, seconds


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    label, paths = argv[0], argv[1:]
    print(f"### {label}\n\n| junit | passed | failed | errors | skipped | pytest s |\n|---|---|---|---|---|---|")
    for path in paths:
        p, f, e, s, t = counts(path)
        print(f"| {path} | {p} | {f} | {e} | {s} | {t:.0f} |")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
