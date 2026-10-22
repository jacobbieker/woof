! Fixture writer shared by every smoke oracle driver (run_*.F90).
!
! The layout of tools/urban_wrf471_oracle/oracle_io.F90, with REAL(8) added
! for module_gocart_settling.F (whose settling routine is REAL*8):
!
! One case = one directory <outdir>/<case>/ holding one little-endian binary
! file per array (<name>.bin, raw float32, float64 or int32 in Fortran
! column-major order, no record markers) and a text MANIFEST.txt with one
! line per array:
!
!     <name> <f4|f8|i4> <rank> <extent_1> ... <extent_rank>
!
! Extents are Fortran's, first index fastest.  woof.verify.smoke_oracle
! reads them back with order='F', so a Fortran (k, n) array arrives as a
! NumPy array indexed [k, n].
!
! Nothing here computes anything.
module oracle_io
  implicit none
  private
  public :: oracle_open, oracle_close, oracle_put, oracle_root

  character(len=1024), save :: case_dir = ''
  character(len=1024), save :: root_dir = '.'
  integer, save :: manifest = -1

  interface oracle_put
    module procedure put_r0, put_r1, put_r2, put_r3
    module procedure put_d0, put_d1, put_d2, put_d3
    module procedure put_i0, put_i1, put_i2
  end interface oracle_put

contains

  ! The output root, from the first command-line argument (default '.').
  subroutine oracle_root(dir)
    character(len=*), intent(out) :: dir
    integer :: n
    call get_command_argument(1, dir, length=n)
    if (n == 0) dir = '.'
    root_dir = dir
  end subroutine oracle_root

  subroutine oracle_open(case)
    character(len=*), intent(in) :: case
    integer :: stat
    if (manifest /= -1) call oracle_close()
    case_dir = trim(root_dir) // '/' // trim(case)
    call execute_command_line('mkdir -p "' // trim(case_dir) // '"', &
                              exitstat=stat)
    if (stat /= 0) error stop 'oracle_io: cannot create case directory'
    open(newunit=manifest, file=trim(case_dir) // '/MANIFEST.txt', &
         status='replace', action='write', form='formatted')
  end subroutine oracle_open

  subroutine oracle_close()
    if (manifest /= -1) close(manifest)
    manifest = -1
  end subroutine oracle_close

  subroutine header(name, kind, shp)
    character(len=*), intent(in) :: name, kind
    integer, intent(in) :: shp(:)
    integer :: d
    if (manifest == -1) error stop 'oracle_io: oracle_put before oracle_open'
    write(manifest, '(A,1X,A,1X,I0)', advance='no') trim(name), kind, size(shp)
    do d = 1, size(shp)
      write(manifest, '(1X,I0)', advance='no') shp(d)
    end do
    write(manifest, '(A)') ''
  end subroutine header

  subroutine open_bin(name, u)
    character(len=*), intent(in) :: name
    integer, intent(out) :: u
    open(newunit=u, file=trim(case_dir) // '/' // trim(name) // '.bin', &
         status='replace', action='write', access='stream', &
         form='unformatted', convert='little_endian')
  end subroutine open_bin

  subroutine put_r0(name, a)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a
    integer :: u
    integer :: none(0)
    call header(name, 'f4', none)
    call open_bin(name, u); write(u) a; close(u)
  end subroutine put_r0

  subroutine put_r1(name, a)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(:)
    integer :: u
    call header(name, 'f4', shape(a))
    call open_bin(name, u); write(u) a; close(u)
  end subroutine put_r1

  subroutine put_r2(name, a)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(:, :)
    integer :: u
    call header(name, 'f4', shape(a))
    call open_bin(name, u); write(u) a; close(u)
  end subroutine put_r2

  subroutine put_r3(name, a)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(:, :, :)
    integer :: u
    call header(name, 'f4', shape(a))
    call open_bin(name, u); write(u) a; close(u)
  end subroutine put_r3

  subroutine put_d0(name, a)
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a
    integer :: u
    integer :: none(0)
    call header(name, 'f8', none)
    call open_bin(name, u); write(u) a; close(u)
  end subroutine put_d0

  subroutine put_d1(name, a)
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:)
    integer :: u
    call header(name, 'f8', shape(a))
    call open_bin(name, u); write(u) a; close(u)
  end subroutine put_d1

  subroutine put_d2(name, a)
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:, :)
    integer :: u
    call header(name, 'f8', shape(a))
    call open_bin(name, u); write(u) a; close(u)
  end subroutine put_d2

  subroutine put_d3(name, a)
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:, :, :)
    integer :: u
    call header(name, 'f8', shape(a))
    call open_bin(name, u); write(u) a; close(u)
  end subroutine put_d3

  subroutine put_i0(name, a)
    character(len=*), intent(in) :: name
    integer(4), intent(in) :: a
    integer :: u
    integer :: none(0)
    call header(name, 'i4', none)
    call open_bin(name, u); write(u) a; close(u)
  end subroutine put_i0

  subroutine put_i1(name, a)
    character(len=*), intent(in) :: name
    integer(4), intent(in) :: a(:)
    integer :: u
    call header(name, 'i4', shape(a))
    call open_bin(name, u); write(u) a; close(u)
  end subroutine put_i1

  subroutine put_i2(name, a)
    character(len=*), intent(in) :: name
    integer(4), intent(in) :: a(:, :)
    integer :: u
    call header(name, 'i4', shape(a))
    call open_bin(name, u); write(u) a; close(u)
  end subroutine put_i2

end module oracle_io
