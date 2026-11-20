"""Generate every-output RK fixtures with the compiled WRF routines."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

F32 = np.float32
DRY_FIELDS = ("U", "V", "W", "PH", "T")
MAP_FIELDS = ("MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY",
              "MAPFAC_VX", "MF_VX_INV", "MAPFAC_VY")
SCALAR_FIELDS = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")


def pad3(value, nx, ny, nz, *, sentinel=False):
    result = np.full((ny + 8, nz + 1, nx + 8), F32(-73.125))
    if not sentinel:
        # Four horizontal halos are edge extensions of the real crop.
        padded = np.pad(value, ((0, nz + 1 - value.shape[0]), (4, ny + 4 - value.shape[1]),
                                (4, nx + 4 - value.shape[2])), mode="edge")
        result[:] = padded.transpose(1, 0, 2)
    else:
        nk, nj, ni = value.shape
        result[4:4+nj, :nk, 4:4+ni] = value.transpose(1, 0, 2)
    return np.ascontiguousarray(result)


def pad2(value, nx, ny):
    return np.ascontiguousarray(np.pad(value, ((4, ny+4-value.shape[0]),
                                              (4, nx+4-value.shape[1])), mode="edge"))


def generate(lib, state, out):
    p = np.load(state)
    nz, ny, nx = p["T"].shape
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    library = ctypes.CDLL(str(lib.resolve()))
    dry = library.oracle_rk_dry
    dry.argtypes = [ctypes.c_int]*4 + [ptr]*8
    scalar = library.oracle_rk_scalar
    scalar.argtypes = [ctypes.c_int]*9 + [ctypes.c_float] + [ptr]*9
    c1 = np.pad(p["C1H"], (0,1), mode="edge")
    c2 = np.pad(p["C2H"], (0,1), mode="edge")
    maps = np.stack([pad2(p[k], nx, ny) for k in MAP_FIELDS])
    mut = pad2(p["MU"]+p["MUB"],nx,ny)
    zeros3 = np.zeros((nz,ny,nx),np.float32)
    a = np.stack([pad3(p[k]*F32(1/64), nx,ny,nz,sentinel=True) for k in DRY_FIELDS])
    b = np.stack([pad3(p[k]*F32(1/2048), nx,ny,nz,sentinel=True) for k in DRY_FIELDS])
    # PH has no held physics tendency in the production dycore.
    b[3] = pad3(np.zeros_like(p["PH"]),nx,ny,nz,sentinel=True)
    saves = np.stack([pad3(np.zeros_like(p[k]),nx,ny,nz) for k in DRY_FIELDS]
                     +[pad3(p["T"]*F32(1/4096),nx,ny,nz)])
    mu = np.stack([pad2(p["MU"]*F32(1/64),nx,ny),pad2(p["MU"]*F32(1/2048),nx,ny)])
    arrays = {"nx":np.array(nx), "ny":np.array(ny), "nz":np.array(nz),
              "c1":c1, "c2":c2}
    case_meta=[]
    for case,(label,step,extreme,zero) in enumerate([
        ("real-rk1",1,False,False),("real-rk2",2,False,False),("real-rk3",3,False,False),
        ("map-extremes",1,True,False),("zero",2,False,True),
    ]):
        aa,bb,cc,mm,uu = (x.copy() for x in (a,b,saves,maps,mu))
        if extreme:
            factor = np.tile(np.array([0.25,0.5,1,2,4,8],np.float32),
                             (ny+8,(nx+13)//6))[:,:nx+8]
            for n in (0,1,2,3,4,6): mm[n]=factor
            mm[5]=F32(1)/mm[4]
        if zero:
            for x in (aa,bb,cc,uu): x.fill(0)
        for k,v in {"a":aa,"b":bb,"saves":cc,"maps":mm,"mut":mut,"mu":uu}.items():
            arrays[f"dry{case}_{k}"]=v.copy()
        dry(nx,ny,nz,step,aa,bb,cc,mm,c1,c2,mut,uu)
        for k,v in {"a":aa,"b":bb,"mu":uu}.items(): arrays[f"dry{case}_{k}_wrf"]=v
        case_meta.append({"kind":"dry","id":case,"name":label,"step":step})
    s2 = np.stack([pad3(p[k],nx,ny,nz,sentinel=True) for k in SCALAR_FIELDS])
    s1 = s2*F32(0.75)
    # Source rates preserve the real state's spatial structure and positivity.
    st = np.stack([pad3(p[k]*F32(1/128),nx,ny,nz) for k in SCALAR_FIELDS])
    adv = pad3(p["QVAPOR"]*F32(1/256),nx,ny,nz)
    decomp = np.stack([pad3(zeros3,nx,ny,nz)]*4)
    smaps = maps[:2].copy()
    smu = np.stack([pad2(p["MU"],nx,ny),pad2(p["MU"]+F32(0.125),nx,ny),pad2(p["MUB"],nx,ny)])
    # No advection decomposition diagnostics are requested by the default run.
    for case,(label,step,specified,nested,periodic,extreme,tiny) in enumerate([
        ("real-rk1",1,0,0,1,False,False),("real-rk2",2,0,0,1,False,False),
        ("real-rk3",3,0,0,1,False,False),("specified-ring",2,1,0,0,False,False),
        ("nested-ring",2,0,1,0,False,False),
        ("map-extremes",2,0,0,1,True,False),("near-zero",2,0,0,1,False,True),
    ]):
        q1,q2,t,ad,dd,mp,cm=(x.copy() for x in (s1,s2,st,adv,decomp,smaps,smu))
        if extreme:
            mp[:]=np.tile(np.array([0.25,0.5,1,2,4,8],np.float32),
                         (ny+8,(nx+13)//6))[:,:nx+8]
        if tiny:
            for x in (q1,q2,t,ad): x*=F32(1e-25)
        for k,v in {"s1":q1,"s2":q2,"st":t,"adv":ad,"decomp":dd,"maps":mp,"mu":cm}.items():
            arrays[f"scalar{case}_{k}"]=v.copy()
        scalar(nx,ny,nz,len(SCALAR_FIELDS),step,specified,nested,periodic,1,F32(20),
               q1,q2,t,ad,dd,mp,c1,c2,cm)
        for k,v in {"s1":q1,"s2":q2,"decomp":dd}.items(): arrays[f"scalar{case}_{k}_wrf"]=v
        case_meta.append({"kind":"scalar","id":case,"name":label,"step":step,
                          "specified":bool(specified),"nested":bool(nested),
                          "periodic_x":bool(periodic),"spec_zone":1,"dt":20.0})
    np.savez_compressed(out,**arrays)
    meta={"schema":"bigstep-rk-wrf471-v1","fixture_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
          "state_sha256":hashlib.sha256(state.read_bytes()).hexdigest(),"cases":case_meta,
          "scope":{"dry":"zero boundary save arrays; physics held PH tendency zero; all memory outputs",
                   "scalar":"six moisture species; tenddec disabled; all memory outputs"},
          "tendencies":"dry initial state/64, held state/2048, theta heating T/4096; scalar source species/128 and common QVAPOR/256 advection"}
    out.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")


#: The specified zone and relaxation zone of the no-mixing relaxation cases.
RELAX_SPEC_ZONE, RELAX_ZONE = 1, 4


def _relax_ring(value, nz, ny, nx):
    """``value`` on the relaxation rows of a crop, zero elsewhere.

    relax_bdy_dry writes ``u_save`` etc. only between the specified zone
    and the relaxation zone's inner edge; the spec rows and interior stay 0.
    """
    nk, nj, ni = value.shape
    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing="ij")
    edge = np.minimum(np.minimum(jj, nj - 1 - jj), np.minimum(ii, ni - 1 - ii))
    ring = (edge >= RELAX_SPEC_ZONE) & (edge < RELAX_SPEC_ZONE + RELAX_ZONE)
    return np.where(ring[None], value, F32(0)).astype(F32)


def generate_relax_fold(lib, state, out):
    """No-mixing specified domain: rk_addtend_dry's fold of the relaxation.

    With km_opt = 0 and no physics the held ``*_tendf`` are zero, so the
    stage-1 relaxation (``u_save``/``v_save``/``w_save``/``t_save``) is the
    whole held tendency.  rk_addtend_dry folds it on rk_step 1 and adds
    the held tendency on every stage.  The three stages run in sequence on
    the same held arrays, each from a fresh stage tendency, as solve_em
    does.  ph_save is zero: woof relaxes phi outside the held slots.
    """
    p = np.load(state)
    nz, ny, nx = p["T"].shape
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    library = ctypes.CDLL(str(lib.resolve()))
    dry = library.oracle_rk_dry
    dry.argtypes = [ctypes.c_int]*4 + [ptr]*8
    c1 = np.pad(p["C1H"], (0,1), mode="edge")
    c2 = np.pad(p["C2H"], (0,1), mode="edge")
    maps = np.stack([pad2(p[k], nx, ny) for k in MAP_FIELDS])
    mut = pad2(p["MU"]+p["MUB"],nx,ny)
    held = np.stack([pad3(np.zeros_like(p[k]),nx,ny,nz,sentinel=True) for k in DRY_FIELDS])
    save_scale = {"U":F32(1/512),"V":F32(1/512),"W":F32(1/8),"PH":F32(0),"T":F32(1/256)}
    saves = np.stack([pad3(_relax_ring(p[k]*save_scale[k],*p[k].shape),nx,ny,nz)
                      for k in DRY_FIELDS]+[pad3(np.zeros_like(p["T"]),nx,ny,nz)])
    arrays = {"nx":np.array(nx), "ny":np.array(ny), "nz":np.array(nz),
              "c1":c1, "c2":c2}
    case_meta=[]
    for case,(label,extreme) in enumerate([("nomix-relax",False),
                                            ("nomix-relax-map-extremes",True)]):
        mm = maps.copy()
        if extreme:
            factor = np.tile(np.array([0.25,0.5,1,2,4,8],np.float32),
                             (ny+8,(nx+13)//6))[:,:nx+8]
            for n in (0,1,2,3,4,6): mm[n]=factor
            mm[5]=F32(1)/mm[4]
        bb = held.copy()
        prefix = f"relax{case}_"
        for k,v in {"saves":saves,"maps":mm,"mut":mut}.items():
            arrays[prefix+k]=v.copy()
        for step in (1,2,3):
            aa = np.stack([pad3(p[k]*F32(1/(32*step)),nx,ny,nz,sentinel=True)
                           for k in DRY_FIELDS])
            uu = np.stack([pad2(p["MU"]*F32(1/64),nx,ny),pad2(np.zeros_like(p["MU"]),nx,ny)])
            arrays[f"{prefix}a{step}"]=aa.copy()
            arrays[f"{prefix}mu{step}"]=uu.copy()
            dry(nx,ny,nz,step,aa,bb,saves.copy(),mm,c1,c2,mut,uu)
            arrays[f"{prefix}a{step}_wrf"]=aa
        case_meta.append({"kind":"relax","id":case,"name":label,
                          "spec_zone":RELAX_SPEC_ZONE,"relax_zone":RELAX_ZONE})
    np.savez_compressed(out,**arrays)
    meta={"schema":"bigstep-rk-relax-fold-wrf471-v1",
          "fixture_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
          "state_sha256":hashlib.sha256(state.read_bytes()).hexdigest(),"cases":case_meta,
          "scope":"km_opt = 0, no physics: zero held tendf, relaxation-ring saves "
                  "U/512, V/512, W/8, T/256 (ph_save and h_diabatic zero); RK "
                  "stages 1-3 in sequence on one held set, stage tendency state/(32*step)"}
    out.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library",type=Path)
    parser.add_argument("state",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--relax-fold",action="store_true",
                        help="write the no-mixing relaxation-fold cases instead")
    args=parser.parse_args()
    (generate_relax_fold if args.relax_fold else generate)(args.library,args.state,args.output)
