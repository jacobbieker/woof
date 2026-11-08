"""Prepared complete generic bytes against independently captured old source."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from ishmael_retirement import requested_by_own_number, without_retired_line
from test_mix_selection_byte_controls import build_experiment, import_documents
from wrf_clock_lines import without_wrf_clock_lines
from woof_engine_spelling import engine_spelling
from test_audit_parent_generic_bytes import without_import_capability_lines


PARENT = "1077c6e8abe502aab59de103d7e6d70dbb8588a1"
MATRIX_SHA256 = "47f581f453640588e8cf72a132780570761ca60b8ce8cf627fc730c4c66a2583"
CASES = ("explicit_false", "short_false", "mixed_false_true", "omitted_theta")
FIXTURES = Path(__file__).parent / "fixtures/mix_additional_parent_controls"


@pytest.mark.parametrize("parsed", (False, True))
@pytest.mark.parametrize("case", CASES)
def test_generic_false_and_omitted_theta_bytes_match_original_parent(case, parsed, tmp_path):
    matrix_bytes = (FIXTURES / "matrix.json").read_bytes()
    assert hashlib.sha256(matrix_bytes).hexdigest() == MATRIX_SHA256
    matrix = json.loads(matrix_bytes)
    assert matrix["schema"] == "mix-parent-additional-inputs-v1"
    assert matrix["audit_parent"] == PARENT
    assert set(matrix["cases"]) == set(CASES)
    pins = json.loads((FIXTURES / "historical-pins.json").read_bytes())
    assert pins["mode"] == "baseline" and pins["audit_parent"] == PARENT
    assert pins["matrix_sha256"] == MATRIX_SHA256
    pin = pins["cases"][case]
    relative = Path(pin["file"])
    assert len(relative.parts) == 1 and relative.suffix == ".toml"
    expected = (FIXTURES / relative).read_bytes()
    assert len(expected) == pin["bytes"]
    assert hashlib.sha256(expected).hexdigest() == pin["sha256"]
    row = deepcopy(matrix["cases"][case])
    row["input"] = requested_by_own_number(row["input"])
    emitted, _ = import_documents(tmp_path, row, parsed=parsed)
    # The pinned parent predates the 2.8.9 D8 fixed-step clock block; the
    # sibling byte controls take out exactly those lines (wrf_clock_lines),
    # the engine-era comment header says gpuwm (woof_engine_spelling), and
    # the import capability lanes since added exactly three default-on lines.
    expected = without_retired_line(expected)
    if case == "omitted_theta":
        # 1.1.6 ports use_theta_m = 1 (WRF's default), so the omitted key is
        # no longer rewritten to the dry form: its report line goes, and the
        # imported value is stated after km_opt.
        theta = (b"#   use_theta_m 1 (moist potential temperature (theta_m) "
                 b"prognostic) -> use_theta_m 0 (dry potential temperature)\n")
        assert expected.count(theta) == 1 and expected.count(b"\nkm_opt = 4\n") == 1
        expected = expected.replace(theta, b"").replace(
            b"\nkm_opt = 4\n", b"\nkm_opt = 4\nuse_theta_m = 1\n")
    assert engine_spelling(without_import_capability_lines(
        without_wrf_clock_lines(emitted))).encode("utf-8") == expected
    exp = build_experiment(tomllib.loads(emitted), source="original generic bytes")
    assert [domain.run.mix_full_fields for domain in exp.domains] == [True, True]
