"""Cut WRF v4.7.1's multi_perturb pattern arithmetic out of module_stoch.F.

    python extract_stoch.py WRF_ROOT OUT_DIR

Writes stoch_cut.F90, one module ``stoch_cut`` holding, byte-unmodified:

  * module_stoch's module-scope declarations (``implicit none`` .. ``save``:
    LENSAV, wavenumber_k/l, WSAVE1/2, RPI, CP, T0);
  * SETUP_RAND_PERTURB, findindex, do_fftback_along_x, do_fftback_along_y;
  * UPDATE_STOCH's AR(1) loops (:913-947), in a stand-in subroutine
    ``ar1_cut`` whose arguments are the draws ZRANDNOSS/ZRANDNOSC -- the
    part of UPDATE_STOCH before those loops is the compiler's RANDOM_NUMBER
    stream, which the port does not reproduce (woof draws Philox);
  * RAND_PERT_UPDATE's variable 'T' fill (:1189-1196), its vertical
    structure (:1222-1233) and its clip (:1333, :1337-1344), in stand-in
    subroutines over a stand-in ``grid`` carrying RAND_REAL/RAND_IMAG and
    one tile's bounds (``ij = 1``).

Only the stand-in wrappers' declarations and loop headers are written here.
"""
import pathlib
import re
import sys

root, out = map(pathlib.Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
text = (root / "dyn_em/module_stoch.F").read_text()
lines = text.splitlines(keepends=True)


def block(first, last, starts, ends):
    b = "".join(lines[first - 1:last])
    assert b.lstrip().startswith(starts), (first, b[:80])
    assert b.rstrip().endswith(ends), (last, b[-80:])
    return b


m = re.search(r"(?ms)^      implicit none\n      public ::.*?\n(      INTEGER :: LMINFORC.*?^      save\n)", text)
assert m, "module header"
# ``implicit none`` and the declarations, without the public list naming
# routines this cut leaves out.
header = "      implicit none\n" + m.group(1)
subs = []
for name in ("SETUP_RAND_PERTURB", "findindex", "do_fftback_along_x", "do_fftback_along_y"):
    s = re.search(rf"(?msi)^ *subroutine {name}\(.*?^ *end subroutine {name}\n", text)
    assert s, name
    subs.append(s.group(0))

ar1 = block(913, 947, "DO IK=jts,jte", "ENDDO")
fill = block(1189, 1196, "DO IK=grid%j_start(ij), grid%j_end(ij)", "ENDDO")
vert = block(1222, 1233, "IF (skebs_vertstruc.ne.0) then", "ENDIF")
thresh = block(1333, 1333, "thresh=thresh_fact*gridpt_stddev", "gridpt_stddev")
clip = block(1337, 1344, "DO k=kts,min(kte,grid%num_stoch_levels)", "ENDDO")

grid_type = """
      type stoch_grid
        real, allocatable :: RAND_REAL(:,:,:), RAND_IMAG(:,:,:)
        integer :: i_start(1), i_end(1), j_start(1), j_end(1)
        integer :: num_stoch_levels
      end type stoch_grid
"""

wrappers = f"""
      subroutine ar1_cut(SPFORCS,SPFORCC,SP_AMP,ALPH,ZRANDNOSS,ZRANDNOSC, &
                         ids, ide, jds, jde, ims, ime, jms, jme, its, ite, jts, jte)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: ids, ide, jds, jde, ims, ime, jms, jme, its, ite, jts, jte
      REAL, DIMENSION( ids:ide,jds:jde)      :: ZRANDNOSS,ZRANDNOSC
      REAL, DIMENSION (ims:ime,jms:jme)      :: SPFORCS,SPFORCC,SP_AMP
      REAL :: ALPH
      INTEGER ::IL, IK,LMAX,KMAX
      KMAX=(jde-jds)+1 !NLAT
      LMAX=(ide-ids)+1 !NATX
{ar1}      end subroutine ar1_cut

      subroutine fill_cut(grid, SPFORCC, SPFORCS, kts, kte_stoch, ims, ime, jms, jme)
      IMPLICIT NONE
      type(stoch_grid), intent(inout) :: grid
      INTEGER, INTENT(IN) :: kts, kte_stoch, ims, ime, jms, jme
      REAL, DIMENSION(ims:ime,jms:jme) :: SPFORCS,SPFORCC
      INTEGER :: IK, IL, ILEV, ij
      ij = 1
{fill}      end subroutine fill_cut

      subroutine vert_cut(grid, skebs_vertstruc, VERTSTRUCC, VERTSTRUCS, VERTAMP, &
                          kts, kte_stoch, ims, ime, kms, kme, jms, jme)
      IMPLICIT NONE
      type(stoch_grid), intent(inout) :: grid
      INTEGER, INTENT(IN) :: skebs_vertstruc, kts, kte_stoch, ims, ime, kms, kme, jms, jme
      REAL, DIMENSION(ims:ime,kms:kme, jms:jme),INTENT(IN)    :: VERTSTRUCC,VERTSTRUCS
      REAL, DIMENSION(kms:kme )                ,INTENT(IN)    :: VERTAMP
      INTEGER :: IK, IL, ILEV, ij
      ij = 1
{vert}      end subroutine vert_cut

      subroutine clip_cut(grid, RAND_PERT, thresh_fact, gridpt_stddev, kts, kte, &
                          ims, ime, kme_stoch, jms, jme)
      IMPLICIT NONE
      type(stoch_grid), intent(inout) :: grid
      INTEGER, INTENT(IN) :: kts, kte, ims, ime, kme_stoch, jms, jme
      REAL, INTENT(IN) :: thresh_fact, gridpt_stddev
      REAL, DIMENSION(ims:ime,1:kme_stoch, jms:jme) :: RAND_PERT
      REAL :: thresh
      INTEGER :: I, J, K, ij
      ij = 1
{thresh}{clip}      end subroutine clip_cut
"""

(out / "stoch_cut.F90").write_text(
    "module stoch_cut\n" + header + grid_type + "contains\n" + "".join(subs)
    + wrappers + "end module stoch_cut\n")
print("cut", sum(len(s.splitlines()) for s in subs), "routine lines and",
      sum(len(b.splitlines()) for b in (ar1, fill, vert, thresh, clip)), "block lines")
