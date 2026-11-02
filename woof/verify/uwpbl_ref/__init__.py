"""CPU reference of the WRF v4.7.1 UW moist-turbulence PBL (bl_pbl_physics=9).

A literal Python transcription of module_bl_camuwpbl_driver.F and the CAM
modules it calls, in binary64 Python floats, graded bit for bit against the
gfortran oracle in tools/uwpbl_wrf471_oracle.  Its transcendentals
(exp/pow/cos/acos) come through fortran.libm: the oracle's own functions
on any host where the CPU bridge carries the opt-in glibc 2.39
transcription, so the reference is exact there; otherwise they are the host
C library's (math.exp/pow/cos/acos), exact only on a glibc x86-64 host with
FMA.  The GPU
port is woof/core/uwpbl.py.
"""
