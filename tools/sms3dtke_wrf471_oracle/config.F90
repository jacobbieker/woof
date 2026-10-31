! The configuration record and the two framework calls the cut routines
! (extract.py) reference.  Only the fields they read; no arithmetic.  The
! species slots are the fixture's layout: moist(:,:,:,1) is unused, qv, qc
! and qi are slots 2, 3 and 4 (PARAM_FIRST_SCALAR = 2), and the scalar
! number-concentration slots are named so vertical_diffusion_implicit's
! skip list compiles (n_scalar = 1, so its scalar loop is empty).
module module_oracle_config
implicit none
type grid_config_rec_type
logical :: open_xs=.false., open_xe=.false., open_ys=.false., open_ye=.false.
logical :: symmetric_xs=.false., symmetric_xe=.false., symmetric_ys=.false., symmetric_ye=.false.
logical :: periodic_x=.false., periodic_y=.false., specified=.false., nested=.false., polar=.false.
logical :: mix_full_fields=.false.
integer :: use_theta_m=0
integer :: km_opt=5, diff_opt=2, sfs_opt=0, bl_pbl_physics=0
integer :: isfflx=1, cu_physics=0, shcu_physics=0, spec_bdy_width=5
real :: c_s=.25, c_k=.15, tke_drag_coefficient=0., tke_heat_flux=0.
end type
integer, parameter :: param_first_scalar=2, p_qv=2, p_qc=3, p_qi=4
integer, parameter :: p_m11=1,p_m22=2,p_m33=3,p_m12=4,p_m13=5,p_m23=6
integer, parameter :: p_r12=1,p_r13=2,p_r23=3
integer, parameter :: p_qns=4,p_qnr=5,p_qng=6,p_qt=7,p_qnh=8,p_qvolg=9
end module module_oracle_config

subroutine wrf_error_fatal(message)
character(*) :: message
write(0, '(a)') trim(message)
error stop 7
end subroutine wrf_error_fatal
