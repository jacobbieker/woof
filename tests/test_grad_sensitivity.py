"""woof.grad.sensitivity: directions, perturbed checkpoints and central differences."""
from __future__ import annotations

import json

import numpy as np
import pytest

from woof.grad import sensitivity as sens


def _checkpoint(path, shape=(4, 5, 6)):
    nz, ny, nx = shape
    payload = {"state/thp": np.zeros(shape, np.float32),
               "state/u": np.zeros((nz, ny, nx + 1), np.float32),
               "state/v": np.zeros((nz, ny + 1, nx), np.float32),
               "meta/elapsed_seconds": np.asarray(10800.0)}
    np.savez(path, **payload)
    return path


def test_a_blob_peaks_at_its_centre_and_falls_off_as_a_gaussian():
    blob = sens.gaussian_blob((5, 9, 11), k=2, j=4, i=5, radius_cells=2.0, depth_levels=1.0, amplitude=3.0)
    assert blob.shape == (5, 9, 11)
    assert blob[2, 4, 5] == pytest.approx(3.0)
    assert np.unravel_index(np.argmax(blob), blob.shape) == (2, 4, 5)
    assert blob[2, 4, 7] == pytest.approx(3.0 * np.exp(-0.5))
    assert blob[3, 4, 5] == pytest.approx(3.0 * np.exp(-0.5))
    with pytest.raises(sens.SensitivityError):
        sens.gaussian_blob((5, 9), k=0, j=0, i=0, radius_cells=1, depth_levels=1)


def test_directions_round_trip_and_refuse_tampering(tmp_path):
    fields = {"thp": np.arange(6.0).reshape(1, 2, 3)}
    path = tmp_path / "d.npz"
    sha = sens.save_direction(path, fields, {"kind": "test"})
    loaded, meta = sens.load_direction(path)
    assert meta["sha256"] == sha and meta["kind"] == "test"
    np.testing.assert_array_equal(loaded["thp"], fields["thp"])
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    payload["field/thp"] = payload["field/thp"] + 1.0
    np.savez(path, **payload)
    with pytest.raises(sens.SensitivityError, match="digest"):
        sens.load_direction(path)


def test_perturb_writes_scale_times_direction_through_the_analysis_writer(tmp_path):
    source = _checkpoint(tmp_path / "x0.npz")
    blob = sens.gaussian_blob((4, 5, 6), k=1, j=2, i=3, radius_cells=1.5, depth_levels=1.0)
    receipt = sens.perturb_checkpoint(source, {"thp": blob}, -0.25, tmp_path / "x1.npz")
    assert receipt["scale"] == -0.25 and receipt["direction_sha256"] == sens.direction_digest({"thp": blob})
    fields = sens.checkpoint_fields(tmp_path / "x1.npz")
    np.testing.assert_allclose(fields["thp"], (-0.25 * blob).astype(np.float32))
    np.testing.assert_array_equal(fields["u"], 0.0)
    assert sens.checkpoint_fields(source, ["thp"])["thp"].max() == 0.0, "the source is never modified"
    with pytest.raises(ValueError):
        sens.perturb_checkpoint(source, {"thp": blob[:, :, :5]}, 1.0, tmp_path / "x2.npz")
    with pytest.raises(sens.SensitivityError):
        sens.checkpoint_fields(source, ["qv"])


def _summary(J, accepted="ref.npz", lost=0):
    return {"J": J, "obs_sha256": "o", "error_table_sha256": "t", "accepted_from": accepted,
            "counts": {"lost_from_reference": lost}}


def test_central_differences_recover_a_quadratic_exactly():
    def J(x):
        return 5.0 + 2.0 * x + 3.0 * x * x
    legs = [(eps, _summary(J(eps)), _summary(J(-eps))) for eps in (0.4, 0.2, 0.1)]
    out = sens.fd_summary(legs, reference=_summary(J(0.0)), zero=_summary(J(0.0) + 1e-6))
    assert out["dJ"] == pytest.approx(2.0) and out["dJ_scale"] == 0.1 and out["converged"] is True
    for leg in out["legs"]:
        assert leg["dJ_central"] == pytest.approx(2.0)
        assert leg["curvature"] == pytest.approx(6.0)
        assert leg["dJ_forward"] == pytest.approx(2.0 + 3.0 * leg["scale"])
    assert out["repair_offset"] == pytest.approx(1e-6)
    assert out["legs"][1]["relative_change_from_larger_scale"] == pytest.approx(0.0, abs=1e-9)


def test_fd_refuses_runs_scored_with_different_definitions():
    with pytest.raises(sens.SensitivityError, match="--accept"):
        sens.fd_summary([(0.1, _summary(1.0), _summary(0.5, accepted=None))])
    with pytest.raises(sens.SensitivityError, match="lost"):
        sens.fd_summary([(0.1, _summary(1.0), _summary(0.5, lost=2))])
    other = _summary(0.5)
    other["obs_sha256"] = "x"
    with pytest.raises(sens.SensitivityError, match="obs_sha256"):
        sens.fd_summary([(0.1, _summary(1.0), other)])
    assert sens.fd_summary([(0.5, 2.0, 1.0)])["dJ"] == 1.0
    with pytest.raises(sens.SensitivityError, match="--accept"):
        sens.fd_summary([(0.1, _summary(1.0), _summary(0.5))], reference=_summary(0.7, accepted=None))
    with pytest.raises(sens.SensitivityError, match="bare numbers"):
        sens.fd_summary([(0.2, _summary(1.0), _summary(0.5)), (0.1, 3.0, 1.0)])


def test_fd_headline_and_offsets(tmp_path):
    from woof.file_digest import sha256_file

    pairs = tmp_path / "ref.npz"
    pairs.write_bytes(b"pairs")
    # A summary written before accepted_sha256 existed names the file; one written after carries its digest.
    old = dict(_summary(1.0), accepted_from=str(pairs))
    new = dict(_summary(0.0), accepted_from=None, accepted_sha256=sha256_file(pairs))
    out = sens.fd_summary([(0.5, old, new)], reference=new, zero=new)
    assert out["converged"] is None and out["legs"][0]["difference_over_repair_offset"] is None
    json.dumps(out, allow_nan=False)
    # Noise at the small scale: the headline stays at the largest scale and says it did not converge.
    legs = [(0.8, 1.6, 0.0), (0.4, 0.8, 0.0), (0.1, 5.0, 0.0)]
    out = sens.fd_summary(legs)
    assert out["dJ_scale"] == 0.4 and out["converged"] is True
    out = sens.fd_summary([(0.8, 1.6, 0.0), (0.1, 5.0, 0.0)])
    assert out["dJ_scale"] == 0.8 and out["converged"] is False


def test_blob_rim_and_direction_files(tmp_path):
    blob = sens.gaussian_blob((2, 12, 12), k=0, j=1, i=6, radius_cells=3, depth_levels=1, rim=2)
    assert not blob[:, :2].any() and not blob[:, -2:].any() and not blob[:, :, :2].any() and blob[:, 2].any()
    with pytest.raises(sens.SensitivityError, match="rim"):
        sens.gaussian_blob((2, 4, 4), k=0, j=1, i=1, radius_cells=1, depth_levels=1, rim=2)
    sha = sens.save_direction(tmp_path / "p", {"thp": blob})
    assert sens.load_direction(tmp_path / "p.npz")[1]["sha256"] == sha
    assert not list(tmp_path.glob(".*partial"))


def test_the_direction_perturb_and_fd_doors(tmp_path, capsys):
    from woof.grad.cli import main

    source = _checkpoint(tmp_path / "x0.npz")
    d = tmp_path / "d.npz"
    assert main(["direction", "--checkpoint", str(source), "--field", "u", "--blob", "1", "2", "3",
                 "--radius-cells", "2", "--rim", "1", "--out", str(d)]) == 0
    assert main(["direction", "--checkpoint", str(source), "--field", "thp", "--blob", "1", "2", "3",
                 "--rim", "1", "--add-to", str(d), "--out", str(d)]) == 0
    fields, meta = sens.load_direction(d)
    assert sorted(fields) == ["thp", "u"] and fields["u"].shape == (4, 5, 7)
    assert [blob["field"] for blob in meta["blobs"]] == ["u", "thp"]
    assert main(["perturb", "--checkpoint", str(source), "--direction", str(d), "--scale", "0.5",
                 "--out", str(tmp_path / "p.npz"), "--quiet"]) == 0
    assert sens.checkpoint_fields(tmp_path / "p.npz", ["u"])["u"].max() == pytest.approx(0.5)
    for name, J in (("p", 3.0), ("m", 1.0)):
        (tmp_path / f"{name}.json").write_text(json.dumps(_summary(J)))
    capsys.readouterr()
    assert main(["fd", "--leg", "0.5", str(tmp_path / "p.json"), str(tmp_path / "m.json")]) == 0
    assert json.loads(capsys.readouterr().out)["dJ"] == pytest.approx(2.0)


def _mpas_state(path, lat, lon, nz=4, bdy=None):
    import netCDF4

    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("Time", None)
        d.createDimension("nCells", len(lat))
        d.createDimension("nVertLevels", nz)
        d.createVariable("latCell", "f8", ("nCells",))[:] = np.radians(lat)
        d.createVariable("lonCell", "f8", ("nCells",))[:] = np.radians(lon)
        if bdy is not None:
            d.createVariable("bdyMaskCell", "i4", ("nCells",))[:] = bdy
        theta = d.createVariable("theta", "f4", ("Time", "nCells", "nVertLevels"))
        theta[0] = np.full((len(lat), nz), 290.0)
    return path


def test_hex_blobs_and_perturbed_analysis_files(tmp_path):
    import netCDF4

    lat = np.array([55.0, 55.0, 55.5, 56.0])
    lon = np.array([-3.0, -2.5, -3.0, -3.0])
    src = _mpas_state(tmp_path / "bg.nc", lat, lon, bdy=np.array([0, 0, 0, 3]))
    cells = sens.mesh_cells(src)
    assert cells["nz"] == 4 and cells["bdy"].tolist() == [0, 0, 0, 3]
    blob = sens.cell_blob(cells["lat"], cells["lon"], 4, lat0=55.0, lon0=-3.0, level=1, radius_km=30.0,
                          depth_levels=1.0, amplitude=2.0, interior=cells["bdy"] == 0)
    assert blob[0, 1] == pytest.approx(2.0)
    assert blob[2, 1] == pytest.approx(2.0 * np.exp(-0.5 * (sens.great_circle_km(55.5, -3.0, 55.0, -3.0) / 30.0) ** 2))
    assert np.all(blob[3] == 0.0), "relaxation-ring cells are never perturbed"
    receipt = sens.perturb_mpas_file(src, {"theta": blob}, -0.5, tmp_path / "an.nc")
    assert receipt["fields"][0]["max_abs_increment"] == pytest.approx(1.0)
    with netCDF4.Dataset(tmp_path / "an.nc") as d, netCDF4.Dataset(src) as s:
        np.testing.assert_allclose(d["theta"][0], 290.0 - 0.5 * blob, rtol=1e-6)
        assert np.all(s["theta"][0] == 290.0)
    with pytest.raises(sens.SensitivityError):
        sens.perturb_mpas_file(src, {"theta": blob[:, :3]}, 1.0, tmp_path / "x.nc")
    with pytest.raises(sens.SensitivityError):
        sens.perturb_mpas_file(src, {"qv": blob}, 1.0, tmp_path / "x.nc")
    assert not (tmp_path / "x.nc").exists()


def test_fd_splits_the_difference_by_observation_type():
    plus, minus = _summary(5.0), _summary(1.0)
    plus["by_type"] = {"synop/t": {"J": 3.0}, "sonde/t": {"J": 2.0}}
    minus["by_type"] = {"synop/t": {"J": 0.0}, "sonde/t": {"J": 1.0}}
    leg = sens.fd_summary([(0.5, plus, minus)])["legs"][0]
    assert leg["dJ_central_by_type"] == {"synop/t": 3.0, "sonde/t": 1.0}
