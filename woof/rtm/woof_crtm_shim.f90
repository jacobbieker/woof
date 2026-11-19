! woof's C shim over JCSDA CRTM v2.4+ (CC0): clear-sky forward and K-matrix calls.
!
! This file is woof's own code (Apache-2.0 with the rest of woof); it is compiled only by
! the opt-in builder woof/rtm/crtm_build.py against a user-supplied CRTM tree.
!
! Arrays are C order seen from Python: a C array [nprof][nlay] is a Fortran (nlay, nprof).
! Layers run top of atmosphere (1) to surface (nlay); level_p has nlay+1 interfaces, top first.
! Units: pressure hPa, temperature K, water vapour g/kg (mass mixing ratio), ozone ppmv.
!
! sfc(nsfc, nprof), nsfc = 7: land_fraction, water_fraction, snow_fraction, ice_fraction,
!   skin_temperature_K, wind_speed_ms, wind_direction_deg
! geo(ngeo, nprof), ngeo = 8: sensor_zenith, sensor_scan, sensor_azimuth, solar_zenith,
!   solar_azimuth, latitude, longitude, surface_altitude_m (angles in degrees)
! ymd(3): year, month, day
!
! Outputs: bt(nchan, nprof); with want_k /= 0 also k_t(nlay, nchan, nprof),
! k_q(nlay, nchan, nprof) (K per g/kg), k_o3(nlay, nchan, nprof), k_tskin(nchan, nprof).
! Every routine returns 0 on success and a nonzero CRTM status otherwise.
module woof_crtm_shim
  use iso_c_binding
  use CRTM_Module
  implicit none
  private
  public :: woof_crtm_init, woof_crtm_channels, woof_crtm_destroy, woof_crtm_clear
  integer, parameter :: MAX_SENSORS = 16
  ! CRTM keeps its coefficients in module state: one CRTM_Init per process, for every
  ! sensor at once, and CRTM_Destroy before another.
  type(CRTM_ChannelInfo_type), allocatable, save :: chinfo(:)
  integer, save :: n_sensors = 0
  integer, parameter :: NSFC = 7, NGEO = 8

contains

  subroutine c_to_f_string(cstr, fstr)
    character(kind=c_char), intent(in) :: cstr(*)
    character(len=*), intent(out) :: fstr
    integer :: i
    fstr = ' '
    do i = 1, len(fstr)
      if (cstr(i) == c_null_char) exit
      fstr(i:i) = cstr(i)
    end do
  end subroutine

  ! sensor_ids: comma-separated CRTM sensor ids (e.g. "abi_g16,ahi_himawari9").
  integer(c_int) function woof_crtm_init(sensor_ids, coeff_path, count) bind(C, name="woof_crtm_init")
    character(kind=c_char), intent(in) :: sensor_ids(*), coeff_path(*)
    integer(c_int), intent(out) :: count
    character(len=4096) :: ids
    character(len=2048) :: path
    character(len=256) :: names(MAX_SENSORS)
    integer :: start, comma, n, err
    call c_to_f_string(sensor_ids, ids)
    call c_to_f_string(coeff_path, path)
    count = 0
    if (n_sensors > 0) then
      woof_crtm_init = -3
      return
    end if
    n = 0
    start = 1
    do
      comma = index(ids(start:), ',')
      n = n + 1
      if (n > MAX_SENSORS) then
        woof_crtm_init = -1
        return
      end if
      if (comma == 0) then
        names(n) = adjustl(ids(start:))
        exit
      end if
      names(n) = adjustl(ids(start:start + comma - 2))
      start = start + comma
    end do
    allocate(chinfo(n))
    err = CRTM_Init(names(1:n), chinfo, File_Path=trim(path)//'/', &
                    Load_CloudCoeff=.false., Load_AerosolCoeff=.false., Quiet=.true.)
    woof_crtm_init = err
    if (err /= SUCCESS) then
      deallocate(chinfo)
      return
    end if
    n_sensors = n
    count = n
  end function

  integer(c_int) function woof_crtm_channels(sensor, n_channels, channels, wmo_ids) bind(C, name="woof_crtm_channels")
    integer(c_int), value :: sensor
    integer(c_int), intent(out) :: n_channels, channels(*), wmo_ids(2)
    integer :: n
    woof_crtm_channels = -1
    n_channels = 0
    if (sensor < 1 .or. sensor > n_sensors) return
    n = CRTM_ChannelInfo_n_Channels(chinfo(sensor))
    n_channels = n
    channels(1:n) = chinfo(sensor)%Sensor_Channel(1:n)
    wmo_ids(1) = chinfo(sensor)%WMO_Satellite_ID
    wmo_ids(2) = chinfo(sensor)%WMO_Sensor_ID
    woof_crtm_channels = 0
  end function

  integer(c_int) function woof_crtm_destroy() bind(C, name="woof_crtm_destroy")
    woof_crtm_destroy = 0
    if (n_sensors == 0) return
    woof_crtm_destroy = CRTM_Destroy(chinfo)
    deallocate(chinfo)
    n_sensors = 0
  end function

  integer(c_int) function woof_crtm_clear(sensor, nprof, nlay, level_p, p, t, q, o3, sfc, geo, ymd, &
                                          want_k, bt, k_t, k_q, k_o3, k_tskin) bind(C, name="woof_crtm_clear")
    integer(c_int), value :: sensor, nprof, nlay, want_k
    real(c_double), intent(in) :: level_p(nlay + 1, nprof), p(nlay, nprof), t(nlay, nprof), &
                                  q(nlay, nprof), o3(nlay, nprof), sfc(NSFC, nprof), geo(NGEO, nprof)
    integer(c_int), intent(in) :: ymd(3)
    real(c_double), intent(out) :: bt(*), k_t(*), k_q(*), k_o3(*), k_tskin(*)
    type(CRTM_Atmosphere_type), allocatable :: atm(:), atm_k(:, :)
    type(CRTM_Surface_type), allocatable :: sfc_c(:), sfc_k(:, :)
    type(CRTM_Geometry_type), allocatable :: geo_c(:)
    type(CRTM_RTSolution_type), allocatable :: rts(:, :), rts_k(:, :)
    integer :: nchan, m, l, err, base
    woof_crtm_clear = -1
    if (sensor < 1 .or. sensor > n_sensors) return
    nchan = CRTM_ChannelInfo_n_Channels(chinfo(sensor))
    allocate(atm(nprof), sfc_c(nprof), geo_c(nprof), rts(nchan, nprof))
    call CRTM_Atmosphere_Create(atm, nlay, 2, 0, 0)
    if (any(.not. CRTM_Atmosphere_Associated(atm))) then
      woof_crtm_clear = -2
      return
    end if
    do m = 1, nprof
      atm(m)%Climatology = US_STANDARD_ATMOSPHERE
      atm(m)%Absorber_ID = (/ H2O_ID, O3_ID /)
      atm(m)%Absorber_Units = (/ MASS_MIXING_RATIO_UNITS, VOLUME_MIXING_RATIO_UNITS /)
      atm(m)%Level_Pressure(0:nlay) = level_p(:, m)
      atm(m)%Pressure = p(:, m)
      atm(m)%Temperature = t(:, m)
      atm(m)%Absorber(:, 1) = q(:, m)
      atm(m)%Absorber(:, 2) = o3(:, m)
      sfc_c(m)%Land_Coverage = sfc(1, m)
      sfc_c(m)%Water_Coverage = sfc(2, m)
      sfc_c(m)%Snow_Coverage = sfc(3, m)
      sfc_c(m)%Ice_Coverage = sfc(4, m)
      sfc_c(m)%Land_Temperature = sfc(5, m)
      sfc_c(m)%Water_Temperature = sfc(5, m)
      sfc_c(m)%Snow_Temperature = sfc(5, m)
      sfc_c(m)%Ice_Temperature = sfc(5, m)
      sfc_c(m)%Wind_Speed = sfc(6, m)
      sfc_c(m)%Wind_Direction = sfc(7, m)
      call CRTM_Geometry_SetValue(geo_c(m), Sensor_Zenith_Angle=geo(1, m), Sensor_Scan_Angle=geo(2, m), &
        Sensor_Azimuth_Angle=geo(3, m), Source_Zenith_Angle=geo(4, m), Source_Azimuth_Angle=geo(5, m), &
        Latitude=geo(6, m), Longitude=geo(7, m), Surface_Altitude=geo(8, m), &
        Year=int(ymd(1)), Month=int(ymd(2)), Day=int(ymd(3)))
    end do
    if (want_k == 0) then
      err = CRTM_Forward(atm, sfc_c, geo_c, chinfo(sensor:sensor), rts)
    else
      allocate(atm_k(nchan, nprof), sfc_k(nchan, nprof), rts_k(nchan, nprof))
      call CRTM_Atmosphere_Create(atm_k, nlay, 2, 0, 0)
      call CRTM_Atmosphere_Zero(atm_k)
      call CRTM_Surface_Zero(sfc_k)
      call CRTM_RTSolution_Zero(rts_k)
      rts_k%Brightness_Temperature = 1.0_fp
      rts_k%Radiance = 0.0_fp
      err = CRTM_K_Matrix(atm, sfc_c, rts_k, geo_c, chinfo(sensor:sensor), atm_k, sfc_k, rts)
    end if
    woof_crtm_clear = err
    if (err /= SUCCESS) return
    do m = 1, nprof
      do l = 1, nchan
        bt(l + (m - 1) * nchan) = rts(l, m)%Brightness_Temperature
        if (want_k /= 0) then
          base = ((m - 1) * nchan + (l - 1)) * nlay
          k_t(base + 1:base + nlay) = atm_k(l, m)%Temperature(1:nlay)
          k_q(base + 1:base + nlay) = atm_k(l, m)%Absorber(1:nlay, 1)
          k_o3(base + 1:base + nlay) = atm_k(l, m)%Absorber(1:nlay, 2)
          ! One skin temperature feeds every surface type, so its derivative is the sum.
          k_tskin(l + (m - 1) * nchan) = sfc_k(l, m)%Land_Temperature + sfc_k(l, m)%Water_Temperature &
            + sfc_k(l, m)%Snow_Temperature + sfc_k(l, m)%Ice_Temperature
        end if
      end do
    end do
    call CRTM_Atmosphere_Destroy(atm)
    if (want_k /= 0) call CRTM_Atmosphere_Destroy(atm_k)
  end function

end module woof_crtm_shim
