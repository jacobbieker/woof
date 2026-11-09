"""The hex chemistry config's wrf_chem block: WRF's names, WRF's sets, refusals by name."""
import json

import numpy as np
import pytest

from woof.hex import chem_wrf
from woof.hex.chem_transport import configure, history_metadata, history_units, load_config, product_slugs


def config(tmp_path, block, **extra):
    path = tmp_path / "chemistry.json"
    path.write_text(json.dumps({"schema": "gpuwm-hex.chemistry/v1", "wrf_chem": block, **extra}))
    return path


def test_chem_opt_selects_the_wrf_package_set(tmp_path):
    row = configure(config(tmp_path, {"chem_opt": 401, "dust_opt": 1}))
    assert row.appended_scalar_names == ("dust_1", "dust_2", "dust_3", "dust_4", "dust_5")
    assert history_units("dust_1") == "ug kg-1"
    assert history_units("dust_3_column_mass") == "kg m-2"
    assert history_units("dust_3_surface_concentration") == "ug m-3"
    assert history_units("edust2") == "kg m-2"


def test_chem_sets_by_name(tmp_path):
    row = configure(config(tmp_path, {"chem_sets": "gocart_primary", "seas_opt": 1}))
    assert {"seas_1", "bc1", "oc2", "dust_5"} <= set(row.appended_scalar_names)


@pytest.mark.parametrize("block,match", [
    ({"chem_opt": 300}, "chem.sulfur"),
    ({"chem_sets": "smoke", "fire_emission_mode": 3}, "must be a string"),
    ({"chem_sets": "sfire_smoke"}, "fire grid"),
    ({"chem_opt": 1}, "RADM2"),
    ({"chem_opt": 7}, "PNNL"),
    ({"chem_opt": 401, "biomass_burn_opt": True}, "integer"),
    ({"chem_opt": 401, "emiss_opt": 1}, "not carried on the hex route"),
    ({"chem_opt": 401, "dust_opt": 1.5}, "integer"),
    ({"chem_opt": 401, "chem_sets": "dust"}, "exactly one"),
    ({}, "exactly one"),
    ({"chem_sets": "dust,smoke"}, "one set"),
])
def test_unported_and_malformed_refused_by_name(tmp_path, block, match):
    # (fire smoke with plume rise is carried: tests/test_hex_chem_wrf_gpu.py, the smoke case)
    with pytest.raises(ValueError, match=match):
        load_config(config(tmp_path, block))


def test_species_and_wrf_chem_are_exclusive(tmp_path):
    path = config(tmp_path, {"chem_opt": 401}, species=[])
    with pytest.raises(ValueError, match=r"not \['species', 'wrf_chem'\]"):
        load_config(path)


def test_inputs_block_refused(tmp_path):
    with pytest.raises(ValueError, match="inputs block"):
        load_config(config(tmp_path, {"chem_opt": 401}, inputs={"valid_time": 0}))


def test_history_metadata_and_product_slugs(tmp_path):
    import hashlib
    path = config(tmp_path, {"chem_opt": 401, "dust_opt": 1})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = json.loads(history_metadata(path, digest))
    assert meta["mode"] == "wrf_chem" and meta["chem_set"] == "dust"
    assert meta["rows"][0] == {"name": "dust_1", "units": "ug kg-1"}
    names, _units = chem_wrf.product_names(chem_wrf.load_table(json.loads(path.read_text()))[1])
    (tmp_path / "chemistry-receipt.json").write_text(json.dumps(
        {"schema": "gpuwm-hex.chemistry/v1", "mode": "wrf_chem", "products": list(names)}))
    assert product_slugs(tmp_path) == tuple("var:wrf_" + n for n in names)


def _geog(tmp_path, name, planes, missing=None):
    """A 1-byte regular_ll WPS GEOG dataset of 10 x 10 one-degree points from (0.5N, 0.5E)."""
    directory = tmp_path / name
    directory.mkdir()
    lines = ["type=continuous", "projection=regular_ll", "dx=1.0", "dy=1.0", "known_x=1.0",
             "known_y=1.0", "known_lat=0.5", "known_lon=0.5", "wordsize=1", "tile_x=10", "tile_y=10",
             f"tile_z={len(planes)}", "scale_factor=0.01"]
    if missing is not None:
        lines.append(f"missing_value={missing}")
    (directory / "index").write_text("\n".join(lines) + "\n")
    np.stack(planes).astype(np.uint8).tofile(directory / "00001-00010.00001-00010")


def test_dust_statics_follow_each_rows_interpolation_sequence(tmp_path):
    from woof.chem_table import catalog
    rows = {v: (src.grid["dataset"], tuple(spec["selector"]["interp_options"]))
            for src in catalog().sources.values() if src.kind == "static" and src.format == "wps_geog"
            for v, spec in src.variables.items()}
    assert rows["EROD"] == ("erod", ("average_4pt",))
    assert rows["CLAYFRAC"][1][:2] == ("four_pt", "average_4pt")
    y, x = np.mgrid[0:10, 0:10]
    erod = [(x + 10 * y) % 90, (2 * x + y) % 80, (x * y) % 70]
    erod[0][3, 4] = 255
    clay = x * 5
    clay[5, 6] = 255
    _geog(tmp_path, "erod", erod, missing=255)
    _geog(tmp_path, rows["CLAYFRAC"][0], [clay], missing=255)
    _geog(tmp_path, rows["SANDFRAC"][0], [y * 7])
    lat = np.array([2.0, 4.0, 4.6, 6.25, 6.0, 9.8])
    lon = np.array([3.0, 5.0, 5.3, 7.75, 7.0, 3.0])
    land = np.array([1, 1, 1, 0, 1, 1])
    out = chem_wrf.sample_dust_statics(tmp_path, lat, lon, land)
    assert out["EROD"].shape == (3, 6)

    def corners(la, lo):
        xx, yy = lo - 0.5, la - 0.5   # 0-based fractional source indices
        i, j = int(np.floor(xx)), int(np.floor(yy))
        return i, j, xx - i, yy - j

    def mean4(plane, la, lo, skip=None):
        i, j, _, _ = corners(la, lo)
        vals = [plane[j + b, i + a] * 0.01 for a in (0, 1) for b in (0, 1) if (j + b, i + a) != skip]
        return sum(vals) / len(vals)

    def bilinear(plane, la, lo):
        i, j, fx, fy = corners(la, lo)
        p = plane.astype(float) * 0.01
        return ((p[j, i] * (1 - fx) + p[j, i + 1] * fx) * (1 - fy)
                + (p[j + 1, i] * (1 - fx) + p[j + 1, i + 1] * fx) * fy)

    for c in (0, 1, 2):
        assert out["EROD"][0, c] == pytest.approx(mean4(erod[0], lat[c], lon[c], skip=(3, 4)))
        assert out["CLAYFRAC"][c] == pytest.approx(bilinear(clay, lat[c], lon[c]))
        assert out["SANDFRAC"][c] == pytest.approx(bilinear(y * 7, lat[c], lon[c]))
    # (6.0N, 7.0E) touches the missing clay point: four_pt fails, average_4pt takes the other three.
    assert out["CLAYFRAC"][4] == pytest.approx(mean4(clay, lat[4], lon[4], skip=(5, 6)))
    assert out["EROD"][:, 3].tolist() == [0.0, 0.0, 0.0]   # water takes the fill
    assert out["CLAYFRAC"][3] == 0.0
    # (9.8N) straddles the dataset's top row; the rows beyond it are padding, not zeros (SANDFRAC declares
    # no missing value): four_pt fails and average_4pt takes the two real points.
    assert out["SANDFRAC"][5] == pytest.approx(9 * 7 * 0.01)
    receipt = chem_wrf.statics_receipt(tmp_path)
    assert receipt["EROD"]["interp_options"] == ["average_4pt"]
    assert receipt["CLAYFRAC"]["interp_options"][:2] == ["four_pt", "average_4pt"]
    before = receipt["SANDFRAC"]["files_sha256"]
    (tmp_path / rows["SANDFRAC"][0] / "00001-00010.00001-00010").write_bytes(b"\0" * 100)
    assert chem_wrf.statics_receipt(tmp_path)["SANDFRAC"]["files_sha256"] != before


def test_suite_route_refuses_wrf_chem(tmp_path):
    from woof.hex.mpasphys.chemistry_v842 import SuiteChemistry
    with pytest.raises(ValueError, match="hex WOOF column route"):
        SuiteChemistry(config(tmp_path, {"chem_opt": 401}), registry_names=("qv",), mesh=None, nlev=3,
                       grid_path=None, start_text="2021-06-01_18:00:00")


def test_backend_runs_the_wrf_step_after_the_physics_and_settles_it(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from woof.hex import chem_transport as chem
    from woof.hex import cuda_arwen_physics_v841 as base
    from woof.hex import cuda_physics_prep_v841 as prep
    calls = []
    fields = {"exch_h": np.ones((3, 1, 2))}

    class Base:
        def __init__(self, **kwargs):
            self._prep_geometry = object()
            self._kernel_cache = object()
            self._seam = SimpleNamespace(
                _driver=SimpleNamespace(fields=fields, microphysics="micro"), _cfg="cfg",
                _start_time="t0", _state=SimpleNamespace(fnm="fnm", fnp="fnp", ht="ht"))

        def finish_step(self, **kwargs):
            calls.append("base")
            return "update"

        def commit_step(self):
            calls.append("base-commit")

        def abort_step(self):
            calls.append("base-abort")

    endpoint = SimpleNamespace(**{n: n for n in ("pres_p", "th_p", "rho_dry", "z_p", "qv_p", "qc_p",
                                                 "qr_p", "u_p", "v_p", "w_p")})

    def prepare(atmosphere, **kwargs):
        calls.append(("prepare", kwargs["post_rk_wsm6"], kwargs["phase2_only"], kwargs["phase2_winds"]))
        return endpoint

    class Runtime:
        feedback = False
        needs_phase_one = False
        phase2_winds = True
        columns = chem_wrf.HexWrfChemistry.columns

        def __init__(self, path, raw, digest, **kwargs):
            self.kwargs = kwargs

        def bind_seam(self, seam):
            pass

        def step(self, scalars, columns, **kwargs):
            calls.append(("chem", columns["rho_dry"], columns["u"], kwargs["physics_fields"] is fields,
                          kwargs["microphysics"], kwargs["elapsed_seconds"]))

        def commit(self):
            calls.append("commit")

        def abort(self):
            calls.append("abort")

    monkeypatch.setattr(base, "PersistentTwoPhaseCudaPhysicsBackendV841", Base)
    monkeypatch.setattr(prep, "prepare_mpas_to_phys_cuda_v841", prepare)
    monkeypatch.setattr(chem_wrf, "HexWrfChemistry", Runtime)
    row = configure(config(tmp_path, {"chem_opt": 401, "dust_opt": 1}))
    backend = chem.build_column_backend(row, constructor=SimpleNamespace(n_levels=3, n_columns=2),
                                        prep_geometry=object(), kernel_cache=object(),
                                        cell_area_m2=np.ones(2))
    assert backend.chemistry.kwargs["offset"] == len(backend.chemistry.species_row.species) - 5
    atmosphere = SimpleNamespace(state=SimpleNamespace(scalars="scalars", time_seconds=120.0))
    assert backend.finish_step(atmosphere=atmosphere, scalar_names=(), dt=60.0) == "update"
    backend.commit_step()
    backend.finish_step(atmosphere=atmosphere, scalar_names=(), dt=60.0)
    backend.abort_step()
    step = ("chem", "rho_dry", "u_p", True, "micro", 120.0)
    assert calls == ["base", ("prepare", True, True, True), step, "base-commit", "commit",
                     "base", ("prepare", True, True, True), step, "abort", "base-abort"]


def test_feedback_prepares_the_radiation_before_phase_one(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from woof.hex import chem_transport as chem
    from woof.hex import cuda_arwen_physics_v841 as base
    from woof.hex import cuda_physics_prep_v841 as prep
    calls = []

    class Base:
        def __init__(self, **kwargs):
            self._prep_geometry = object()
            self._kernel_cache = object()
            self._seam = self._new_seam()

        def _new_seam(self):
            calls.append("new-seam")
            return SimpleNamespace(_driver=SimpleNamespace(fields={}), _cfg="cfg", _start_time="t0",
                                   _state=SimpleNamespace(fnm="fnm", fnp="fnp", ht="ht"))

        def begin_step(self, **kwargs):
            calls.append("base-begin")
            return "raw"

    endpoint = SimpleNamespace(**{n: n for n in ("pres_p", "pres_hyd_p", "th_p", "rho_dry", "z_p", "qv_p",
                                                 "qc_p", "qr_p", "u_p", "v_p", "w_p")})

    def prepare(atmosphere, **kwargs):
        calls.append(("prepare", kwargs["post_rk_wsm6"]))
        return endpoint

    class Runtime:
        feedback = True
        needs_phase_one = True
        phase2_winds = True
        columns = chem_wrf.HexWrfChemistry.columns

        def __init__(self, *args, **kwargs):
            pass

        def bind_seam(self, seam):
            calls.append("bind")

        def before_phase_one(self, scalars, columns, *, elapsed_seconds):
            calls.append(("radiation", columns["pressure"], elapsed_seconds))

    monkeypatch.setattr(base, "PersistentTwoPhaseCudaPhysicsBackendV841", Base)
    monkeypatch.setattr(prep, "prepare_mpas_to_phys_cuda_v841", prepare)
    monkeypatch.setattr(chem_wrf, "HexWrfChemistry", Runtime)
    row = configure(config(tmp_path, {"chem_opt": 401, "dust_opt": 1, "aer_ra_feedback": 1}))
    backend = chem.build_column_backend(row, constructor=SimpleNamespace(n_levels=3, n_columns=2),
                                        prep_geometry=object(), kernel_cache=object(),
                                        cell_area_m2=np.ones(2))
    atmosphere = SimpleNamespace(state=SimpleNamespace(scalars="scalars", time_seconds=60.0))
    assert backend.begin_step(atmosphere=atmosphere, scalar_names=(), dt=60.0) == "raw"
    backend._seam = backend._new_seam()   # a rollback's fresh seam is bound again
    assert calls == ["new-seam", "bind", ("prepare", False), ("radiation", "pres_hyd_p", 60.0),
                     "base-begin", "new-seam", "bind"]


def test_fire_smoke_takes_its_sources_and_the_sets_defaults(tmp_path):
    from woof.hex.chem_wrf import chem_sources, load_table
    raw = {"wrf_chem": {"chem_sets": "smoke", "biomass_burn_opt": 1}}
    assert chem_sources("smoke", raw["wrf_chem"]) == ("rave-3km",)
    name, table, values = load_table(raw)
    # The smoke set's reference defaults (RRFS-SD: washout, an hourly plume) fill what the block leaves unset.
    assert values["plumerisefire_frq"] == 60 and values["wetscav_onoff"] == -1
    assert {"emission.fire", "plumerise.freitas"} <= set(table.processes)
    stated = load_table({"wrf_chem": {"chem_sets": "smoke", "biomass_burn_opt": 1, "wetscav_onoff": 0}})[2]
    assert stated["wetscav_onoff"] == 0
