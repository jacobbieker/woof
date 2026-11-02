"""CPU reference of the WRF v4.7.1 UW moist-turbulence PBL (bl_pbl_physics=9).

A literal Python transcription of module_bl_camuwpbl_driver.F and the CAM
modules it calls, in binary64 Python floats, graded bit for bit against the
gfortran oracle in tools/uwpbl_wrf471_oracle.  Its transcendentals
(exp/pow/cos/acos) are glibc's words through fortran.libm: the oracle's own
functions wherever the CPU bridge's glibc 2.39 transcription is installed,
on any host, so the reference is exact there; without the bridge they are
the host C library's, exact only on a glibc x86-64 host with FMA.  The GPU
port is woof/core/uwpbl.py.
"""
