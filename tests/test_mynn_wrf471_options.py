"""Ported 4.7.1 controls must reach the same solver from public doors."""
from dataclasses import replace
import pytest
from woof.config import RunConfig,validate_run_config
from woof.core.mynn471_gpu import diagnostic_shapes,schema

ROWS=[("bl_mynn_edmf",0),("icloud_bl",0),("tke_budget",1),
      ("bl_mynn_cloudpdf",0),("bl_mynn_cloudpdf",1),
      ("bl_mynn_mixlength",0),("bl_mynn_edmf_mom",0),
      ("bl_mynn_edmf_tke",1),("bl_mynn_cloudmix",0),
      ("bl_mynn_mixqt",1),("bl_mynn_output",1),
      ("bl_mynn_closure",2.5),("bl_mynn_closure",3.0),
      ("bl_mynn_mixscalars",1)]

def config(**kw):
    return RunConfig(nx=8,ny=8,nz=49,dx=3000.,dy=3000.,ztop=20000.,
                     dt=12.,run_seconds=120.,moist=True,mp_physics=8,
                     bl_pbl_physics=5,sf_sfclay_physics=1,**kw)

@pytest.mark.parametrize("key,value",ROWS)
def test_each_new_control_selects_471_and_validates(key,value):
    cfg=config(**{key:value})
    validate_run_config(cfg)
    assert cfg.bl_mynn_version=="wrf_471"
    assert getattr(cfg,key)==value

def test_default_generation_and_optional_outputs_stay_off():
    cfg=config()
    assert cfg.bl_mynn_version=="wrf_461"
    assert diagnostic_shapes(cfg)=={}
    budget=diagnostic_shapes(replace(cfg,tke_budget=1))
    assert set(budget)=={"qshear","qbuoy","qdiss","qwt","dqke"}
    assert budget["dqke"]==(49,8,8)
    assert budget["qwt"]==(50,8,8)

def test_vertical_schema_has_aligned_nonoverlapping_argument_slabs():
    for nz in (20,49,50,64,128):
        meta=schema(nz)
        assert meta["scratch_bytes"]%4==0
        for slab,count in meta["slab_fields"].items():
            ranges=sorted((f["offset"],f["offset"]+f["count"])
                          for f in meta["arguments"].values() if f.get("slab")==slab)
            assert ranges[0][0]==0
            assert ranges[-1][1]==count
            assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))


def test_metal_text_builds_the_single_step_kernel():
    """The Metal build of MYNN 4.7.1 (mynn471_gpu.metal_text): the step
    kernel only, its phase-carried scalars locals, phase 8's re-pointed
    ktop dropped, and M471Ref's narrowing stores spelled.  On the A6000's
    strict words and Metal's, a 3-step full-physics run agrees in all 91
    state fields bit for bit."""
    from woof.core.mynn471_gpu import metal_text, source
    text = source(49)
    out = metal_text(text)
    assert "mynn471_step" in out and "mynn471_phase0(" not in out
    assert "=M471_PHASE<0 ? local_" not in out
    assert "ktop=M471Ptr<integer>(&m471_local_top);" not in out
    assert "return *this=(T)(*p*x);" in out
    assert out.count("\n") < text.count("\n")


@pytest.mark.parametrize("nz", [12, 16, 24, 30, 48, 49, 50, 60, 100])
def test_the_private_array_arena_holds_every_array_at_any_depth(nz):
    """The arena offsets fitted through the 49- and 50-level lowerings went
    negative below 24 levels and overlapped away from 49/50 (each array
    rounds to 8 bytes, so the fit's slopes are 0 or 8 where arrays grow by
    4): the 16-level non-finite dqv and the 30-level fast-mode RAINNC.
    arena_layout keeps every array at least its 49-level size, growing by
    at least 8 bytes a level above 49, and the 49-level text unchanged."""
    from woof.core.mynn471_gpu import _ARENA_OFFSET, _arena_rows, arena_layout, schema, source
    rows, total49, _ = _arena_rows()
    offsets, total = arena_layout(nz)
    starts = [offsets[(slope, base)] for base, slope in rows]
    ends = starts[1:] + [total]
    sizes49 = [b - a for (a, _), b in zip(rows, [b for b, _ in rows[1:]] + [total49])]
    assert starts[0] == 0
    for start, end, size49 in zip(starts, ends, sizes49):
        assert end - start >= size49 + 8 * max(nz - 49, 0)
    assert schema(nz)["scratch_bytes"] == total
    if nz == 49:
        assert all(offsets[(slope, base)] == base for base, slope in rows)
        assert _ARENA_OFFSET.search(source(49))
    else:
        assert not _ARENA_OFFSET.search(source(nz))
