// Auroral oval on model columns (space-weather layer L3b), one thread per column.
//
// * Geomagnetic coordinates: the centred dipole of the IGRF-14 degree-1
//   coefficients at the run's date.  The host passes the dipole axis m (unit
//   vector toward the boreal geomagnetic pole) and two unit vectors ex, ey
//   that complete it to a right-handed frame, plus the magnetic longitude of
//   the subsolar point; MLT = 12 + (mlon - mlon_sun) / 15 (hours, mod 24).
//   Positions are geocentric unit vectors from the WGS84 geodetic latitude.
// * Oval boundaries: Starkov 1994 (Geomagn. Aeron. 34, 331), in co-latitude
//   degrees, for the poleward (ocb) and equatorward (eab) edges of the oval
//   and the equatorward edge of the diffuse aurora:
//       theta = A0 + A1 cos(15 (t + a1)) + A2 cos(15 (2t + a2))
//                  + A3 cos(15 (3t + a3)),       t = MLT hours, degrees
//   The host evaluates A0..A3, a1..a3 from AL (Starkov's cubic in log10|AL|);
//   coef holds them as [boundary][A0 A1 A2 A3 a1 a2 a3], boundaries in the
//   order ocb, eab, diffuse.  The same boundaries are used in both
//   hemispheres, on |geomagnetic latitude|, as Sigernes et al. 2011 do.
//
// Outputs: geomagnetic latitude (deg), MLT (h), the zone (0 equatorward of
// the diffuse aurora, 1 diffuse aurora, 2 auroral oval, 3 polar cap) and the
// distance in degrees poleward of the diffuse equatorward boundary (negative
// equatorward of it).  float throughout; shared by CUDA and Metal.

#define AUR_DEG2RAD 0.017453292519943295f
#define AUR_RAD2DEG 57.29577951308232f
#define AUR_WGS84_1ME2 0.9933056200098587f   // 1 - e^2 (WGS84)

__device__ static float aur_boundary(const float *a, float t) {
    return a[0]
        + a[1] * cosf(AUR_DEG2RAD * 15.0f * (t + a[4]))
        + a[2] * cosf(AUR_DEG2RAD * 15.0f * (2.0f * t + a[5]))
        + a[3] * cosf(AUR_DEG2RAD * 15.0f * (3.0f * t + a[6]));
}

extern "C" __global__ void spacewx_aurora(
    const float *lat_deg, const float *lon_deg,
    const float *frame,       // m[3], ex[3], ey[3]
    const float *coef,        // 3 boundaries x 7 Starkov coefficients
    const float mlon_sun_deg,
    float *mlat_out, float *mlt_out, float *zone_out, float *eqdist_out,
    const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const float lat = AUR_DEG2RAD * lat_deg[c];
    const float lon = AUR_DEG2RAD * lon_deg[c];
    // Geocentric latitude of the surface point.
    const float gc = atan2f(AUR_WGS84_1ME2 * sinf(lat), cosf(lat));
    const float cl = cosf(gc);
    const float px = cl * cosf(lon), py = cl * sinf(lon), pz = sinf(gc);
    float s = px * frame[0] + py * frame[1] + pz * frame[2];
    s = fminf(1.0f, fmaxf(-1.0f, s));
    const float mlat = AUR_RAD2DEG * asinf(s);
    const float x = px * frame[3] + py * frame[4] + pz * frame[5];
    const float y = px * frame[6] + py * frame[7] + pz * frame[8];
    const float mlon = AUR_RAD2DEG * atan2f(y, x);
    float mlt = 12.0f + (mlon - mlon_sun_deg) / 15.0f;
    mlt = fmodf(mlt, 24.0f);
    if (mlt < 0.0f) mlt += 24.0f;
    const float colat = 90.0f - fabsf(mlat);
    const float ocb = aur_boundary(coef, mlt);
    const float eab = aur_boundary(coef + 7, mlt);
    const float dif = aur_boundary(coef + 14, mlt);
    float zone = 0.0f;
    if (colat < ocb) zone = 3.0f;
    else if (colat <= eab) zone = 2.0f;
    else if (colat <= dif) zone = 1.0f;
    mlat_out[c] = mlat;
    mlt_out[c] = mlt;
    zone_out[c] = zone;
    eqdist_out[c] = dif - colat;
}
