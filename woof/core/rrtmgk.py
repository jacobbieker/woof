"""RRTMG-K radiation: WRF 4.7.1 ``ra_lw_physics = 14`` / ``ra_sw_physics = 14``.

RRTMG-K (Baek 2017; phys/module_ra_rrtmg_lwk.F, phys/module_ra_rrtmg_swk.F,
phys/module_ra_effective_radius.F, built with ``-DBUILD_RRTMK=1``) is the
legacy RRTMG with a handful of statements changed.  This module runs the
legacy engines (woof.core.rrtmg_lw / woof.core.rrtmg_sw) and replaces only
those statements, each a transcription of the K source:

Wrapper (rad_rrtmg_driver, one routine for both spectra; kernels/rrtmgk_prep.cu)
  * effective radii from effectRad (Thompson-style diagnostics of qc, nc,
    qi, qs, qg and rho), then x1e6 and the clamps [2.51, 50], [10.01, 125],
    [25, 999] um; snow radius above 130 um is capped with the mass factor
    (130/re)**2;
  * cloud paths in g m-2 from p8w differences and 1000/g, with the
    ``cldfra > 0 and qi+qc+qs < 1e-9`` thin-cloud floor;
  * nlay = kte + 1: one layer above the model top, play = plev(kte+1)/2,
    plev(kte+2) = 1e-5 hPa, no Cavallo buffer layers; layer pressures by
    ``* 1.e-2``; fixed trace gases (co2 379 ppmv, ...); beyond WRF, with
    ghg_input = 1 the CAMtr gases legacy RRTMG reads (see ``RRTMGKRadiation``);
  * the wrapper ozone (o3data) over the K levels, or the CAM o3rad under
    o3input = 2 with the same top-layer splice as the legacy wrapper;
  * ONE McICA call for both spectra (mcica_subcol, permuteseed 1): 28
    subcolumns walked top down (generate_stochastic_redu) and repeated
    five times to 140 g-points; the shortwave reads g-points 1..112;
  * heating rates hr/86400 with no Exner division, GSW, GLW, OLR.
Longwave (kernels/rrtmgk_lw.cu)
  * taugb7's lower-atmosphere fac*01/fac*11 block tests specparm, not
    specparm1, in its first arm; heating uses cp = 1.0046e3.
Shortwave (kernels/rrtmgk_sw.cu)
  * cldprmc stores per-phase optics; spcvmc runs without delta scaling
    and hands reftra per-layer two-stream coefficients (kmodts = 5); the
    liquid cloud tables are K's own (_K_LIQUID_TABLES).

Every other kernel is the legacy one, unchanged.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from woof.core.device_cache import cuda_cache
from woof.core import rrtmg_lw as _lw
from woof.core import rrtmg_sw as _sw

F = np.float32
_KDIR = Path(__file__).resolve().parent / "kernels"

#: nsubclw (= ngptlw) and ngptsw.
NGPTLW, NGPTSW = 140, 112
#: mcica_subcol's permuteseed (rad_rrtmg_driver) and the reduced
#: generator's distinct subcolumns.
PERMUTESEED = 1
NSUB28 = 28
#: rrtmg_lw_ini(cp): module_ra_rrtmg_lwk.F:12002 ``cp=1.0046e+3``.
LW_CPDAIR = F("1.0046e3")
#: rad_rrtmg_driver's fixed gases (module_ra_rrtmg_swk.F:10340-10372).
GASES = {"co2": F("379.e-6"), "o2": F("0.209488"), "ch4": F("1774.e-9"),
         "n2o": F("319.e-9"), "cfc11": F("0.251e-9"),
         "cfc12": F("0.538e-9"), "cfc22": F("0.169e-9"),
         "ccl4": F("0.093e-9")}
#: The gases ghg_input = 1 takes from the CAMtr table (read_CAMgases, as
#: legacy RRTMG_LWRAD/RRTMG_SWRAD read them); o2, cfc22 and ccl4 stay the
#: driver's fixed values, as they stay legacy RRTMG's.
from woof.core.camtr_gases import GAS_NAMES as GHG_INPUT_GASES  # noqa: E402


def camtr_gases(yr, julian):
    """ghg_input = 1: legacy RRTMG's CAMtr gases at ``yr`` / ``julian``,
    narrowed to the REAL(4) profile values (``co2vmr(ncol,k) = co2``)."""
    from woof.core.camtr_gases import camtr_trace_gases
    values = camtr_trace_gases(yr, julian)
    return {name: F(values[name]) for name in GHG_INPUT_GASES}


# --- BEGIN GENERATED K LIQUID TABLES (tools/rrtmg_k_column_oracle/gen_static.py) ---
# Source: rrsw_cld_k extliq1/ssaliq1/asyliq1, Fortran (58, 14), from the
# column oracle built on unmodified WRF 4.7.1 module_ra_rrtmg_swk.F.
_K_LIQUID_TABLES = {
    "extliq1": (
        "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"
    ),
    "ssaliq1": (
        "WHtQP9qeSD9pTEE/XQo7P1fnNT+P1TE/97suP5l8LD/g9yo/cw4qP1oAKz/hkCg/"
        "FosmP2vjJD9NjiM/N4AiP5WtIT/ZCiE/dYwgP9omID92zh8/u3cfPxwXHz8GoR4/"
        "6gkePzhGHT9gShw/0wobP0pfHD8rfhs/4rMaPwH/GT8MXhk/ks8YPxpSGD8x5Bc/"
        "X4QXPzIxFz8x6RY/6KoWP+B0Fj+nRRY/wxsWP8D1FT8q0hU/i68VP2qMFT9XZxU/"
        "1z4VP3oRFT/E3RQ/RqIUP4ddFD8QDhQ/bbITPylJEz/O0BI/7UcSP4YLUT8B2kw/"
        "+RxIPwpvQz8+Ej8//zI7P2f1Nz9OejU//uEzP4xNMz9bRzQ/JzMyPxx+MD+iHS8/"
        "FwcuP+IvLT9ljSw/ARUsPxy8Kz8ZeCs/WT4rP0UEKz88vyo/pWQqP+LpKT9aRCk/"
        "b2koP4dOJz/Yjyg/PsgnP2EUJz/3ciY/teIlP09iJT988CQ/9IskP2gzJD+S5SM/"
        "I6EjP9VkIz9aLyM/bP8iP7zTIj8BqyI/8IMiP0JdIj+pNSI/3QsiP5DeIT9+rCE/"
        "VnQhP9I0IT+n7CA/h5ogPyo9ID9H0x8/klsfP4DUHj+1cX0/MEV8P2fSej/uKnk/"
        "mWd3P6KhdT+d8XM/U29yPwUycT+jUHA/2zpvPzo3bj+HPG0/M0psP7Ffaz92fGo/"
        "8p9pP5fJaD/a+Gc/LC1nPwJmZj/NomU/AeNkPxAmZD9sa2M/irJiP936YT/VQ2E/"
        "uLNgPyj9Xz90Sl8/hJtePzvwXT+ASF0/NaRcP0ADXD+GZVs/7cpaP1czWj+qnlk/"
        "zQxZP6J9WD8N8Vc/9mZXPz7fVj/OWVY/h9ZVP1BVVT8M1lQ/olhUP/fcUz/sYlM/"
        "aepSP1FzUj+N/VE//YhRP4UVUT8vo1A/c/9/P2atfz8TJ38/KL1+P5pmfj91Gn4/"
        "1c99P919fT+/G30/pqB8P6kifD9vtXs/UkV7P6rSej/RXXo/HOd5P+FueT939Xg/"
        "Ont4P34AeD+ZhXc/5Qp3P7eQdj9pF3Y/UZ91P8YodT8ftHQ/s0F0P8TDcz8bT3M/"
        "ddpyP+plcj+O8XE/eX1xP7wJcT9wlnA/qCNwP4Cxbz8FQG8/U89uP3xfbj+Y8G0/"
        "vIJtP/wVbT9tqmw/KUBsP0PXaz/Pb2s/5QlrP5qlaj8EQ2o/N+JpP0uDaT9SJmk/"
        "ZstoP5pyaD8EHGg/pMdnP5t5fz/QFH8/abh+P2tjfj8+FH4/asl9P4SBfT8vO30/"
        "A/V8P5etfD8Qanw/2CR8Pwrgez+gm3s/lld7P+gTez+R0Ho/j416P9tKej9zCHo/"
        "UcZ5P3CEeT/OQnk/ZQF5PzPAeD8wf3g/Wj54P6z9dz8iwnc/6oB3PxlAdz+y/3Y/"
        "uL92PzKAdj8hQXY/iwJ2P3HEdT/YhnU/xUl1PzkNdT870XQ/zZV0P/NadD+yIHQ/"
        "DedzPwWucz+idXM/5z1zP9UGcz9y0HI/w5pyP8llcj+IMXI/B/5xP0fLcT9MmXE/"
        "G2hxP/Q3cT84tn8/ao5/P9dofz80RX8/PCN/P6oCfz83434/mMR+P4mmfj+/iH4/"
        "rGx+Px1Pfj/LMX4/sRR+P8j3fT8N230/d759PwSifT+shX0/a2l9PztNfT8ZMX0/"
        "/RR9P+H4fD/C3Hw/mcB8P2KkfD8YiHw/Em98P1xSfD/JNXw/WRl8PxD9ez/s4Hs/"
        "8sR7PyCpez96jXs/AXJ7P7VWez+aO3s/sCB7P/gFez9063o/JdF6Pwy3ej8tnXo/"
        "iIN6Px5qej/wUHo/Ajh6P1Ifej/jBno/uO55P9DWeT8sv3k/0ad5P7+QeT8lenk/"
        "ktt/PwHJfz+Tt38/IKd/P3+Xfz+EiH8/Bnp/P91rfz/aXX8/109/PwVCfz/eM38/"
        "0SV/P9wXfz/+CX8/Mvx+P3vufj/U4H4/PtN+P7bFfj88uH4/zap+P2adfj8JkH4/"
        "s4J+P2J1fj8WaH4/yVp+P/ROfj81QX4/ijN+P+slfj9cGH4/3wp+P3L9fT8W8H0/"
        "z+J9P5vVfT96yH0/brt9P3mufT+YoX0/z5R9PxyIfT+De30/Am99P5pifT9NVn0/"
        "G0p9PwU+fT8JMn0/LCZ9P20afT/LDn0/SQN9P+X3fD+l7Hw/xuF8P1r6fz+t938/"
        "HPV/P6ryfz9V8H8/HO5/P/nrfz/o6X8/5Od/P+Tlfz/B438/rOF/P5vffz+O3X8/"
        "hNt/P37Zfz97138/eNV/P3rTfz990X8/gc9/P4jNfz+Py38/mMl/P6DHfz+pxX8/"
        "s8N/P7zBfz/6v38/8b1/P+q7fz/kuX8/4rd/P+G1fz/js38/6LF/P++vfz/7rX8/"
        "B6x/Pxeqfz8rqH8/QaZ/P1ykfz95on8/maB/P7+efz/nnH8/E5t/P0SZfz94l38/"
        "sJV/P+2Tfz8ukn8/dZB/P7+Ofz8OjX8/Y4t/PwmKfz/n/38/3f9/P9H/fz/H/38/"
        "v/9/P7X/fz+s/38/ov9/P5j/fz+Q/38/hv9/P3v/fz9x/38/af9/P1//fz9X/38/"
        "TP9/P0T/fz86/38/Mv9/Pyj/fz8f/38/F/9/Pw3/fz8E/38/+v5/P/L+fz/o/n8/"
        "3/5/P9f+fz/N/n8/xf5/P7z+fz+y/n8/qv5/P6H+fz+Z/n8/kf5/P4j+fz9+/n8/"
        "dv5/P2/+fz9n/n8/Xv5/P1b+fz9N/n8/Rf5/Pz7+fz82/n8/Lv5/Pyf+fz8e/n8/"
        "GP5/PxH+fz8K/n8/BP5/P/v9fz8J/n8//v9/P/3/fz/7/38/+/9/P/v/fz/5/38/"
        "+f9/P/j/fz/4/38/9v9/P/b/fz/0/38/9P9/P/P/fz/z/38/8f9/P/H/fz/x/38/"
        "7/9/P+7/fz/u/38/7v9/P+z/fz/s/38/6v9/P+r/fz/p/38/6f9/P+f/fz/n/38/"
        "5f9/P+X/fz/l/38/4/9/P+P/fz/i/38/4v9/P+L/fz/g/38/3v9/P97/fz/e/38/"
        "3f9/P93/fz/b/38/2/9/P9n/fz/Z/38/2f9/P9j/fz/Y/38/1v9/P9b/fz/U/38/"
        "1P9/P9T/fz/T/38/AACAP/v/fz/4/38/9P9/P/P/fz/v/38/7v9/P+z/fz/q/38/"
        "6f9/P+f/fz/l/38/4/9/P+L/fz/g/38/3v9/P9v/fz/Z/38/2P9/P9b/fz/U/38/"
        "0/9/P9H/fz/O/38/zP9/P8r/fz/J/38/x/9/P8X/fz/E/38/wv9/P8D/fz+//38/"
        "u/9/P7r/fz+4/38/tv9/P7X/fz+z/38/sf9/P6//fz+u/38/qv9/P6r/fz+n/38/"
        "pf9/P6T/fz+i/38/oP9/P5//fz+d/38/m/9/P5r/fz+Y/38/lv9/P5X/fz+T/38/"
        "kf9/P1j/fz/l/38/2P9/P8z/fz/A/38/tv9/P6z/fz+k/38/m/9/P5P/fz+M/38/"
        "hP9/P3r/fz9w/38/Zv9/P13/fz9T/38/Sf9/P0H/fz83/38/Lf9/PyP/fz8a/38/"
        "EP9/Pwb/fz/+/n8/9P5/P+v+fz/h/n8/2f5/P8/+fz/F/n8/u/5/P7D+fz+o/n8/"
        "nv5/P5T+fz+M/n8/g/5/P3n+fz9v/n8/Z/5/P13+fz9U/n8/Sv5/P0L+fz85/n8/"
        "L/5/Pyf+fz8e/n8/Fv5/Pw7+fz8E/n8/+/1/P/P9fz/s/X8/5P1/P9v9fz8J/n8/"
        "4P9/P9P/fz/F/38/uv9/P67/fz+g/38/lv9/P4v/fz+A/38/dv9/P2z/fz9h/38/"
        "U/9/P0f/fz88/38/MP9/PyT/fz8Y/38/Df9/P//+fz/0/n8/6f5/P97+fz/Q/n8/"
        "xf5/P7n+fz+v/n8/of5/P5f+fz+M/n8/gP5/P3T+fz9o/n8/Xf5/P1H+fz9F/n8/"
        "Of5/Py7+fz8i/n8/GP5/Pwr+fz8A/n8/9f1/P+r9fz/f/X8/1f1/P8n9fz+9/X8/"
        "s/1/P6f9fz+d/X8/k/1/P4n9fz9//X8/df1/P2v9fz9h/X8/Yf1/PzO/WT/ktFg/"
        "WuhWP5X0Uz/6ok8/C0BKP5dXRD/Ekz4/n605P0tlNj/nnjM/t0QwP3A+LT+4hSo/"
        "KhQoP2rjJT8l7SM/EisiP/WWID+dKh8/8d8dP+KwHD9ylxs/t40aP9mNGT8Rkhg/"
        "rJQXPwuQFj8g0RY/DcoVP+naFD9VAhQ/8j4TP1mPEj8u8hE/DWYRP5XpED9mexA/"
        "HBoQP1nEDz+6eA8/4DUPP2b6Dj/uxA4/FJQOP3tmDj+9Og4/fg8OP1njDT/utA0/"
        "3oINP8VLDT9EDg0/+8gMP4V6DD+GIQw/mrwLPyNKCz8="
    ),
    "asyliq1": (
        "YDZQP+Q3UD8+QFE/qwZTP5BTVT8/+Fc/EstaP6OkXT8DXmA/1M9iP33cXD+y6GE/"
        "7c5lPzexaD9vsWo/XvFrP8iSbD90t2w/O4FsPwwSbD/4i2s/HRFrP7vDaj8pxmo/"
        "wjprP+dDbD/jA24/9ZxwP4uaaz/hEm0/6FFuP59bbz8BNHA/Bd9wP6VgcT/dvHE/"
        "p/dxPwMVcj/pGHI/WgdyP1PkcT/Ws3E/43lxP306cT+n+XA/ZbtwP76DcD+0VnA/"
        "UjhwP6EscD+iN3A/ZV1wP+uhcD9CCXE/a5dxP3BQcj9aOHM/51J0P0sjYT80xmE/"
        "iT5iP3moYj8NGWM/CadjPwxrZD/WfmU/vfxmP03/aD/RTWI/xk9nP/wfaz8e4m0/"
        "wrlvP2zKcD+mN3E/9yRxP+y1cD8mDnA/UFFvPyWjbj9oJ24/6AFuP4JWbj8GSW8/"
        "QP1wP+uWcz+tPW4/XqlvP7jacD/X1XE/2Z5yP9c5cz/sqnM/NfZzP9EfdD/VK3Q/"
        "Yh50P5L7cz+Hx3M/WoZzPy48cz8e7XI/S51yP9VQcj/bC3I/gdJxP+SocT8sk3E/"
        "cpVxP96zcT+N8nE/qFVyP0nhcj+TmXM/pYJ0P3ugdT/GDlk/w7FTPz3uUD/UR1A/"
        "aj5RP4JOUz/o8FU/G5tYPzPAWj9M0ls/8GNVP604Wj+A7V0/76NgP4R9Yj/Im2M/"
        "QyBkP3wsZD/64WM/RWJjP+HOYj9WSWI/JvNhP9rtYT/1WmI//FtjP3kSZT/on2c/"
        "XLZiP1kkZD80W2U/x15mP+4yZz+O22c/d1xoP4y5aD+j9mg/nxdpP1UgaT+kFGk/"
        "aPhoP3vPaD+7nWg/AmdoPysvaD8S+mc/kctnP4WnZz/KkWc/Po5nP7mgZz8UzWc/"
        "LhdoP+OCaD8PFGk/jM5pPzS2aj8tz2s/8lhSP4tcTD9sA0o/93hKP+zlTD/LclA/"
        "WkhUP6WPVz/rcVk/mRhZP6k8Uj9e6VY/DHlaPxwNXT8Dx14/L8hfPw4yYD8QJmA/"
        "pMVfPz4yXz9IjV4/NfhdP3OUXT92g10/peZdP3ffXj9aj2A/vBdjPwEqXj8ZjF8/"
        "DbhgP6CxYT+ofGI/8BxjP0SWYz9y7GM/RSNkP5A+ZD8bQmQ/tDFkPyoRZD9L5GM/"
        "465jP750Yz+pOWM/dwFjP+3PYj/cqGI/FpBiP2SJYj+RmGI/bMFiP8MHYz9lb2M/"
        "G/xjP7ixZD8GlGU/oKZmP61tTj9xN0s/D8lKPxlrTD/vZE8/6/9SP5mGVj/hQ1k/"
        "kIJaPwmNWT9fjlI/WC5XP7WvWj8mNF0/Xt1ePw/NXz/mJGA/lAZgP86TXz9D7l4/"
        "pDdeP6ORXT/xHV0/Pv5cPztUXT+aQV4/DuhfP0VpYj+2ZF0/ArxePxXdXz+7y2A/"
        "y4thPxchYj9vj2I/odpiP4YGYz/vFmM/rA9jP470Yj9oyWI/D5JiP1JSYj8DDmI/"
        "9chhP/qGYT/jS2E/ghthP7D5YD816mA/5vBgP5YRYT8XUGE/PrBhP9Y1Yj+35GI/"
        "tcBjP3XNZD8H4Uk/DTFLPy9UTT9C+U8/MNRSP8WZVT/t/lc/ebhZPy57Wj/D+1k/"
        "YOpSP698Vz/D7lo/kmJdPxv6Xj9b118/ThxgP/DqXz8+ZV8/M61eP8zkXT8HLl0/"
        "36pcP059XD9Ux1w/7KpdPxRKXz/IxmE/c61cP9X7XT/2E18/qPlfP8CwYD8ZPWE/"
        "g6JhP9bkYT/pB2I/kQ9iP6D/YT/x22E/WKhhP6doYT+6IGE/YtRgP3qHYD/QPWA/"
        "QPtfP5rDXz+6ml8/cIRfP5mEXz8An18/hddfP/kxYD8zsmA/B1xhP00zYj+4O2M/"
        "jflHP0TVSz9AQ08/sUBSP+rKVD8v31Y/ynpYPwGbWT8dPVo/aV5aP+88Uz+UyFc/"
        "8jJbPzeeXT+QLF8/IgBgPyA7YD+x/18//29fPzyuXj+N3F0/IR1dPyCSXD+5XVw/"
        "F6JcP2aBXT/RHV8/gJlhP0d1XD+gv10/q9NeP0C1Xz81aGA/X/BgP59RYT/Dj2E/"
        "qK5hPyayYT8XnmE/TXZhP6U+YT/w+mA/Ea9gP9VeYD8aDmA/tsBfP4J6Xz9WP18/"
        "CRNfP3D5Xj9q9l4/yQ1fP2VDXz8Wm18/vBhgPyXAYD8tlWE/0JtiP/FcST+DME8/"
        "+2RTPz1FVj+zGlg/Ii5ZPxvIWT8+MVo/UbJaP1WUWz+HMlQ/PrZYP3IWXD+rdV4/"
        "cvZfP1S7YD/V5mA/gZtgP9/7Xz96Kl8/3EleP4t8XT8T5Vw//KVcP87hXD8Qu10/"
        "UVRfPxjQYT+Bllw/DNxdPyHrXj+gx18/Y3VgP0j4YD8vVGE/9IxhP3KmYT+GpGE/"
        "EothP+5dYT/7IGE/FNhgPxeHYD/fMWA/TdxfPz2KXz+MP18/FwBfP7vPXj9Usl4/"
        "w6teP+S/Xj+Q8l4/qUdfPwrDXz+TaGA/HTxhP19BYj9clE4//4dTP7b+Vj8oQFk/"
        "CJRaP+ZBWz9FkVs/o8lbP3oyXD9JE10/4G9VP4nsWT8WQ10/dpZfP6YJYT+Uv2E/"
        "QdthP5p/YT+Yz2A/M+5fP2H+Xj8XI14/S39dP/c1XT8Kal0/gj5eP1LWXz9sVGI/"
        "4wRdPxFHXj+dUl8/ZCtgP03VYD87VGE/EaxhP7DgYT/+9WE/3e9hPzHSYT/doGE/"
        "wl9hP8gSYT/LvWA/tGRgP2QLYD+/tV8/qWdfPwMlXz+y8V4/l9FeP5jIXj+X2l4/"
        "dwtfPxlfXz9l2V8/OX5gP3xRYT8AV2I/gEFTPzeCVj8x4Fg/9YNaP8SVWz/SPVw/"
        "TKRcP1vxXD8qTV0/299dPz8SVj/iilo/CdxdP+UoYD+plGE/gkJiP6FVYj868WE/"
        "ezhhP5dOYD+/Vl8/HnReP+vJXT9We10/i6tdP8J9Xj8kFWA/6ZRiP7k5XT8Qel4/"
        "p4NfP2haYD82AmE/+35hP5rUYT/8BmI/BRpiP54RYj+r8WE/EL5hP7Z6YT+HK2E/"
        "ZNRgPzJ5YD/dHWA/SMZfP1d2Xz/2MV8/B/1eP3PbXj8b0V4/6uFeP8cRXz+VZF8/"
        "Pt5fP6KCYD+uVWE/lltiP3RJVj9cx1g/H5FaP4zHWz99i1w/yP1cP0c/XT/PcF0/"
        "O7NdP14nXj8PRFY/ybhaP2YFXj83TWA/g7NhP5tbYj/JaGI/WP5hP5g/YT/RT2A/"
        "VFJfP2lqXj9fu10/gGhdPxmVXT95ZF4/6/lfP7l4Yj/JFl0/B1VeP4RcXz8pMWA/"
        "3NZgP4hRYT8QpWE/XtVhP1nmYT/s22E/+rlhP2uEYT8nP2E/GO5gPyKVYD8tOGA/"
        "IdtfP+mBXz9nMF8/hepePyq0Xj9BkV4/q4VeP1WVXj8jxF4//hVfP82OXz96MmA/"
        "6ARhP1IKYj831Fc/h9tZP3ZSWz/GU1w/PvpcP5xgXT+aoV0/7dddP0keXj9jj14/"
        "VqNWPyMVWz/5XV4/QKFgP20CYj/npGI/JqxiP5I7Yj+cdmE/tYBgP0Z9Xz/Ej14/"
        "mttdPzmEXT8PrV0/iHlePxoNYD8qi2I/FSJdPwBfXj8iZV8/ZThgP7DcYD/xVWE/"
        "DahhP+/WYT+A5mE/rNphP1e3YT9sgGE/1TlhP37nYD9MjWA/KS9gP//QXz+3dl8/"
        "PSRfP3TdXj9Mpl4/qYJeP3V2Xj+fhV4/CbReP58FXz9Mfl8/9CFgP4f0YD/w+WE/"
        "42BYP14uWj/Wd1s/CldcP7flXD+fPV0/fnhdPxOwXT8e/l0/YHxePxegVj9VEls/"
        "mFteP0afYD/JAGI/h6NiP+6qYj9jOmI/TnVhPyB/YD85e18/BY1eP+7XXT9ef10/"
        "uqZdP2xxXj/fAmA/en5iP88VXT9oUl4/OlhfPy8rYD8uz2A/J0hhP/2ZYT+ZyGE/"
        "6NdhP9XLYT9DqGE/HXFhP08qYT/E12A/XX1gPwsfYD+1wF8/RGZfP6MTXz+1zF4/"
        "b5VeP69xXj9hZV4/d3ReP8uiXj9T9F4/9WxfP5YQYD8k42A/PuhhP8azRz/3IU8/"
        "IyFVP4W8WD8m/1k/epVZP5J6WD9D0lc/3tZYP6DPXD8PGVc/4m9cP2GuYD9v9GM/"
        "xmFmPwUWaD+2MGk/U9FpP1gXaj9GImo/qRFqPxgFaj88HGo/z3ZqP540az99dWw/"
        "UFluP/j/cD9rhWw/MRluP9Vzbz82mXA/NY1xP7FTcj+L8HI/oWdzP9e8cz8I9HM/"
        "GxF0P+8XdD9oDHQ/avJzP9fNcz+XonM/j3RzP6dHcz/IH3M/2QBzP8Tucj9y7XI/"
        "zwBzP8Uscz8+dXM/Jd5zP2NrdD/kIHU/kAJ2PxIUdz8="
    ),
}
# --- END GENERATED K LIQUID TABLES ---


def _k_liquid(name):
    buf = base64.b64decode("".join(_K_LIQUID_TABLES[name]))
    return np.frombuffer(buf, "<f4").reshape((58, 14), order="F").astype(F)


# ---------------------------------------------------------------------------
# host constants
# ---------------------------------------------------------------------------

def rgmma(x):
    """effectRad's rgmma (module_ra_effective_radius.F), FP32 with glibc expf."""
    from woof.core.rrtmg_lw import expf
    x = F(x)
    if x == F(1.0):
        return F(0.0)
    euler = F("0.577215664901532")
    r = F(x * expf(F(euler * x)))
    for i in range(1, 10001):
        y = F(i)
        q = F(x / y)
        r = F(F(r * F(F(1.0) + q)) * expf(F(-q)))
    return F(F(1.0) / r)


_CDM2 = None


def cdm2():
    """rgmma(5./3.), once per process."""
    global _CDM2
    if _CDM2 is None:
        _CDM2 = rgmma(F(F(5.0) / F(3.0)))
    return _CDM2


def check_taugb7_ratio(chi_mls):
    """Fail closed unless chi_mls(1,jp)/chi_mls(3,jp) never increases from
    jp to jp+1 over the lower atmosphere (jp 1..13).  K's taugb7 leaves
    fac201/fac211 unassigned when specparm >= 0.125 > specparm1, which
    needs rat_h2oo3_1 > rat_h2oo3 (kernels/rrtmgk_lw.cu); this proves the
    reference profile never takes that arm."""
    chi = np.asarray(chi_mls, np.float32)
    if chi.shape[0] != 7:
        chi = chi.reshape((7, -1), order="F")
    rat = (chi[0, :14] / chi[2, :14]).astype(np.float32)
    if not np.all(rat[1:] <= rat[:-1]):
        raise RuntimeError(
            "RRTMG-K taugb7: chi_mls water/ozone ratio increases inside the "
            "lower atmosphere; the stale fac201/fac211 arm would be reachable")


_LW_K_COEFFS = None


def lw_coefficients_k():
    """The batched LW coefficient dict with K's heating factor (cp 1.0046e3);
    every table is the legacy one (rrtmg_legacy._lw_coeffs)."""
    global _LW_K_COEFFS
    if _LW_K_COEFFS is None:
        from woof.core.rrtmg_legacy import _lw_static_tables
        from woof.ingest import rrtmg_coeffs as _coeffs
        loaded = _coeffs.load_rrtmg_lw_coefficients()
        C = _lw.build_lw_coefficients(loaded, LW_CPDAIR)
        for band in range(1, 17):
            for name, value in loaded[f"rrlw_kg{band:02d}"].items():
                C.setdefault(f"kg{band:02d}/{name}", value)
        C.update(_lw_static_tables())
        _lw.lw_batched_const_bytes(C)
        check_taugb7_ratio(C["ref/chi_mls"])
        _LW_K_COEFFS = C
    return _LW_K_COEFFS


_SW_K_TABLES = None


def sw_tables_k():
    """The legacy SW tables with K's liquid optics (rrsw_cld_k)."""
    global _SW_K_TABLES
    if _SW_K_TABLES is None:
        from woof.core.rrtmg_legacy import _sw_tables
        base = _sw_tables()
        t = _sw.SWTables()
        for slot in _sw.SWTables.__slots__:
            if hasattr(base, slot):
                setattr(t, slot, getattr(base, slot))
        for name in ("extliq1", "ssaliq1", "asyliq1"):
            setattr(t, name, _k_liquid(name))
        _SW_K_TABLES = t
    return _SW_K_TABLES


# ---------------------------------------------------------------------------
# compiled units
# ---------------------------------------------------------------------------

def _load(cp, source, name, key):
    from woof import nvrtc_ptx_cache as _cc
    from woof.certify.kernel_manifest import record_module
    options = ("-std=c++17", "--ftz=false")
    ptx, _mapping = _cc.compile_using_nvrtc(source, options, None, name)
    mod = cp.cuda.function.Module()
    mod.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(key, source=source, options=options, module=mod)
    return mod


def prep_source():
    return "".join((_KDIR / n).read_text(encoding="utf-8") for n in (
        "glibc_flt32.cuh", "rrtmg_mcica_wrf.cu", "rrtmgk_prep.cu"))


def lw_post_source():
    return "".join((_KDIR / n).read_text(encoding="utf-8") for n in (
        "rrtmg_lw.cu", "rrtmg_lw_taugb06_09.cu", "rrtmgk_lw.cu"))


def sw_source(defines=None):
    """K's SW unit: the packed-table defines (K's tables unless given),
    kernels/rrtmg_sw.cu, its spectral helpers and kernels/rrtmgk_sw.cu."""
    if defines is None:
        defines = _sw._pack_cuda_tables(sw_tables_k())[1]
    return (defines + (_KDIR / "rrtmg_sw.cu").read_text(encoding="ascii")
            + _sw.spectral_source()
            + (_KDIR / "rrtmgk_sw.cu").read_text(encoding="ascii"))


_MODULES = {}


@cuda_cache(maxsize=None)
def _module(which):
    import cupy as cp
    key = (int(cp.cuda.Device().id), which)
    if key not in _MODULES:
        if which == "prep":
            _MODULES[key] = _load(cp, prep_source(), "rrtmgk_prep.cu",
                                  "woof.core.rrtmgk:prep")
        else:
            _MODULES[key] = _load(cp, lw_post_source(), "rrtmgk_lw.cu",
                                  "woof.core.rrtmgk:lw")
    return _MODULES[key]


def prep_module():
    """The prep + McICA unit on this card (kernel pre-warm's route)."""
    return _module("prep")


def lw_module():
    """The LW taumol_post unit on this card (kernel pre-warm's route)."""
    return _module("lw")


def _kernel(which, name):
    return _module(which).get_function(name)


_PREFLIGHTED = set()


def gpu_preflight(force=False):
    """Prove subnormal survival, the libm transcriptions and gfortran MAX
    on the live device."""
    import cupy as cp
    from woof.core.rrtmg_lw import expf, powf
    device = int(cp.cuda.Device().id)
    if device in _PREFLIGHTED and not force:
        return
    x = np.array([1.e-30, 1.e-10, 0.6283185, 2.0, -3.7, np.nan], F)
    y = cp.zeros(4, F)
    _kernel("prep", "rkp_probe")((1,), (32,), (cp.asarray(x), y))
    want = np.array([x[0] * x[1], powf(x[2], x[3]), expf(x[4]), 1.0], F)
    if cp.asnumpy(y).view(np.uint32).tolist() != want.view(np.uint32).tolist():
        raise RuntimeError("RRTMG-K prep preflight failed: got %r want %r"
                           % (cp.asnumpy(y), want))
    _PREFLIGHTED.add(device)


class CudaSWK(_sw.CudaSW):
    """The legacy CUDA SW chain with K's cloud optics and two-stream solver.

    The unit is kernels/rrtmg_sw.cu + its spectral helpers +
    kernels/rrtmgk_sw.cu, packed with K's tables; rrtmg_sw_batched_device
    runs unchanged except that its two cloud kernels resolve to the K ones
    (``_k``), which carry the per-phase optics slab between them."""

    def __init__(self, tab):
        import cupy as cp
        self.cp = cp
        self.tab = tab
        self.spectral_output = "none"
        packed, defines = _sw._pack_cuda_tables(tab)
        self.module = _load(cp, sw_source(defines), "rrtmgk_sw.cu",
                            "woof.core.rrtmgk:sw")
        self.tab_gpu = cp.asarray(packed)
        self.ngb_gpu = cp.asarray(np.asarray(tab.ngb, dtype=np.int32))
        self._laysolfr_spec_gpu = cp.asarray(
            np.asarray(self._laysolfr_band_spec(), dtype=np.int32))
        self._tables_ready = cp.cuda.Event(disable_timing=True)
        self._tables_ready.record()
        self._table_streams = {}
        self.max_nlay = None
        self._scratch_by_stream = {}
        self._chunk_by_nlayers = {}
        # cldprmc -> spcvmc hand-off of K's optics slab and error word,
        # owned per (thread, stream) so interleaved calls never share one.
        self._handoff = {}

    def _owner(self, kw):
        import threading
        stream = kw.get("stream") or self.cp.cuda.get_current_stream()
        return (threading.get_ident(), int(getattr(stream, "ptr", 0) or 0))

    def drop_handoff(self):
        """Release this thread's pending optics slab: a call that raised
        between cldprmc and spcvmc must not pin it on the shared engine."""
        import threading
        thread = threading.get_ident()
        for owner in [owner for owner in self._handoff if owner[0] == thread]:
            self._handoff.pop(owner, None)

    def _k(self, name):
        fn = super()._k({"rsw_cldprmc_b": "rswk_cldprmc_b",
                         "rsw_spcvmc_gpt_b": "rswk_spcvmc_gpt_b"}.get(name, name))
        if name == "rsw_cldprmc_b":
            def cldprmc(grid, block, args, **kw):
                nc, nl = int(args[0]), int(args[1])
                dtwa = self.cp.empty((9, nc, nl, NGPTSW), np.float32)
                self._handoff[self._owner(kw)] = (dtwa, args[-1])
                return fn(grid, block, tuple(args) + (dtwa,), **kw)
            return cldprmc
        if name == "rsw_spcvmc_gpt_b":
            def spcvmc(grid, block, args, **kw):
                dtwa, err = self._handoff.pop(self._owner(kw))
                return fn(grid, block, tuple(args) + (dtwa, err), **kw)
            return spcvmc
        return fn


_ENGINES = {}


def _engines():
    """(LW coefficients, CudaSWK) on this card, built once."""
    import cupy as cp
    device = int(cp.cuda.Device().id)
    if device not in _ENGINES:
        C = lw_coefficients_k()
        _ENGINES[device] = (C, CudaSWK(sw_tables_k()))
    return _ENGINES[device]


def _taugb7_post(nc, nl, laytrop_d, fs_d, isv_d, wx_d, K, taug_d, fracs_d):
    total = nc * nl
    _kernel("lw", "rlwk_taugb7_lower")(
        ((total + 127) // 128,), (128,),
        (np.int32(nc), np.int32(nl), laytrop_d, fs_d, isv_d, wx_d,
         K["chi"], K["oneminus"], K["band"][7][1], taug_d, fracs_d))


def flags(f_qc, f_qi, f_qs):
    """rad_rrtmg_driver's (inflag, iceflag): 2/3, f_qc -> 3, f_qi -> 4/4,
    f_qs -> 5/5 (the same for both spectra; liqflag 1)."""
    inflg, iceflg = 2, 3
    if f_qc:
        inflg = 3
    if f_qi:
        inflg, iceflg = 4, 4
    if f_qs:
        inflg, iceflg = 5, 5
    return inflg, iceflg


def rrtmgk_vram_bytes(*, ncol, nz, column_chunk=None, resident_threads=None,
                      o3input=2):
    """Peak device bytes of ONE RRTMGKRadiation call (host-side pricing).

    An upper bound composed the way :func:`rrtmgk_columns` allocates: the
    adapter's whole-call column planes and results, then per column chunk
    the prep profiles and McICA slabs held through both spectra, plus the
    largest of the three phases (the McICA generator's cloud-fraction
    slab, the LW engine, the SW engine with its gathered day-column copies
    and K's per-phase optics slab), plus both engines' resident tables.
    The SW engine runs at the LW chunk width (``column_chunk``), so every
    day column of a chunk is priced as lit.
    """
    from woof.core.rrtmg_legacy import (
        _lw_coeffs, legacy_shortwave_constant_bytes)
    r = _lw._r512
    ncol, nz = int(ncol), int(nz)
    nl = nz + 1
    f = 4
    chunk = int(column_chunk) if column_chunk else _lw.batch_column_chunk(
        NGPTLW, _lw.LW_BATCH_COLUMN_CHUNK_CEILING,
        resident_threads=resident_threads)
    n = min(chunk, ncol)
    plane, level = r(ncol * nz * f), r(ncol * nl * f)
    # p3d t3d rho, five moist species, the zero plane, nc, cldfra, o33d
    # and its grid copy, the three (ncol, nz) results and the two
    # transposed tendencies + use_theta_m's total; p8w zw t8w; the
    # 21 per-column results, coszen and SWDOWN.
    whole = (18 * plane + 3 * level + 23 * r(ncol * f)
             + (plane if int(o3input) == 2 else 0))
    pn, pn1 = r(n * nl * f), r(n * (nl + 1) * f)
    held = (11 * pn + 2 * pn1 + 3 * r(3 * n * nz * f)      # P + radii
            + 4 * r(n * nl * NGPTLW * f)                  # McICA slabs
            + 8 * pn)                                     # well-mixed gases
    gen = r(n * NSUB28 * nl * f)
    lw = (r(n * _lw.NBNDLW * f) + r(n * nl * NGPTLW * f)
          + r(n * nl * _lw.NBNDLW * f)
          + _lw.lw_batched_vram_bytes(n, nl, mcica_layout="column"))
    sw_slab = r(n * nl * NGPTSW * f)
    sw = (4 * sw_slab + 4 * sw_slab                       # mc + zero/one
          + 9 * pn + 2 * pn1 + 9 * r(n * f) + r(n * 8)
          + r(9 * n * nl * NGPTSW * f)                    # K's dtwa slab
          + _sw.sw_batched_vram_bytes(n, nl, mcica_layout="column"))
    constants = (_lw.lw_batched_const_bytes(_lw_coeffs())
                 + legacy_shortwave_constant_bytes())
    return whole + held + max(gen, lw, sw) + constants


def rrtmgk_columns(*, p3d, p8w, t3d, t8w, rho, qv, qc, qi, qs, qg, nc,
                   cldfra, o33d, tsk, emiss, albedo, coszen, solcon,
                   f_qc, f_qi, f_qs, f_qnc, o3input, column_chunk=None,
                   radii=False, gases=None, band_factor=None):
    """rad_rrtmg_driver over a batch of columns, on the device.

    Profiles are C-contiguous float32 device arrays (ncol, nz) (p8w, t8w:
    (ncol, nz + 1)); surface fields (ncol,); ``o33d`` may be None unless
    o3input == 2.  Returns device arrays: rthratenlw, rthratensw (the
    wrapper's swhr/86400, zero on night columns) (ncol, nz); glw, olr, gsw
    and the eight lw*/sw* boundary fluxes (ncol,); the McICA-ready layer
    cldfrac (ncol, nz); with ``radii`` the clamped effective radii
    re_qc/re_qi/re_qs (ncol, nz) before the snow-mass cap.  ``gases``
    (name -> float32 mole fraction) replaces those of the driver's fixed
    ``GASES`` it names (ghg_input = 1, ``camtr_gases``); None keeps them.
    ``band_factor`` (14 floats, bands 16..29) scales the SW source per band
    (solar_ssi_opt); None leaves it bitwise unchanged."""
    import cupy as cp
    gpu_preflight()
    inflg, iceflg = flags(f_qc, f_qi, f_qs)
    if iceflg != 5:
        raise NotImplementedError(
            "RRTMG-K's shortwave cloud optics are ported for the snow-carrying "
            "microphysics (iceflag 5, f_qs); this scheme has no snow")
    if o3input not in (0, 2):
        raise ValueError(f"o3input must be 0 or 2, got {o3input!r}")
    ncol, nz = (int(n) for n in p3d.shape)
    nl = nz + 1
    if nz < 3:
        raise ValueError("kissvec seeding reads four layers (nz >= 3)")
    C, sw = _engines()
    chunk = int(column_chunk) if column_chunk else int(_lw.LW_BATCH_COLUMN_CHUNK)
    f32 = np.float32
    out = {k: cp.zeros((ncol, nz), f32)
           for k in ("rthratenlw", "rthratensw", "cldfrac")}
    for k in ("glw", "olr", "gsw", "lwupt", "lwuptc", "lwdnt", "lwdntc",
              "lwupb", "lwupbc", "lwdnb", "lwdnbc", "swupt", "swuptc",
              "swdnt", "swdntc", "swupb", "swupbc", "swdnb", "swdnbc"):
        out[k] = cp.zeros(ncol, f32)
    if radii:
        re = cp.empty((3, ncol, nz), f32)
    else:
        re = None
    from woof.core.rrtmg_legacy_device import _tables as _prep_tables
    tabs = _prep_tables(int(cp.cuda.Device().id))
    o3wrk, ppwrkh = tabs[3], tabs[4]
    zero = cp.zeros((1,), f32)
    err = cp.zeros(1, np.int32)
    cdm = cdm2()
    day_all = cp.asnumpy(coszen) > F(0.0)
    table = GASES if gases is None else {**GASES, **gases}
    for c0 in range(0, ncol, chunk):
        c1 = min(c0 + chunk, ncol)
        n = c1 - c0
        rows = slice(c0, c1)
        P = {k: cp.empty((n, nl), f32) for k in (
            "play", "tlay", "h2ovmr", "o3vmr", "cldfrac", "clwp", "ciwp",
            "cswp", "rel", "rei", "res")}
        P["plev"] = cp.empty((n, nl + 1), f32)
        P["tlev"] = cp.empty((n, nl + 1), f32)
        rd = cp.empty((3, n, nz), f32) if radii else None
        _kernel("prep", "rkp_prep")(
            ((n + 63) // 64,), (64,),
            (np.int32(n), np.int32(nz), np.int32(inflg), np.int32(iceflg),
             np.int32(o3input), np.int32(1 if f_qnc else 0), F(cdm),
             p3d[rows], p8w[rows], t3d[rows], t8w[rows], rho[rows],
             qv[rows], qc[rows], qi[rows], qs[rows], qg[rows], nc[rows],
             cldfra[rows], zero if o33d is None else o33d[rows],
             o3wrk, ppwrkh,
             P["play"], P["plev"], P["tlay"], P["tlev"], P["h2ovmr"],
             P["o3vmr"], P["cldfrac"], P["clwp"], P["ciwp"], P["cswp"],
             P["rel"], P["rei"], P["res"],
             np.uint64(0) if rd is None else rd))
        if rd is not None:
            re[:, rows] = rd
        out["cldfrac"][rows] = P["cldfrac"][:, :nz]
        M = {k: cp.empty((n, nl, NGPTLW), f32)
             for k in ("cldfmcl", "ciwpmcl", "clwpmcl", "cswpmcl")}
        cdf = cp.empty((n, NSUB28, nl), f32)
        _kernel("prep", "rkp_mcica")(
            ((n * NSUB28 + 127) // 128,), (128,),
            (np.int32(n), np.int32(nl), np.int32(PERMUTESEED),
             P["play"], P["cldfrac"], P["ciwp"], P["clwp"], P["cswp"], cdf,
             M["cldfmcl"], M["ciwpmcl"], M["clwpmcl"], M["cswpmcl"], err))
        del cdf
        gas = {k: cp.full((n, nl), v, f32) for k, v in table.items()}
        res = _lw.gpu_rrtmg_lw_batched_device(
            n, nl, 2, P["play"], P["plev"], P["tlay"], P["tlev"],
            tsk[rows], P["h2ovmr"], P["o3vmr"], gas["co2"], gas["ch4"],
            gas["n2o"], gas["o2"], gas["cfc11"], gas["cfc12"],
            gas["cfc22"], gas["ccl4"],
            cp.ascontiguousarray(cp.broadcast_to(emiss[rows][:, None],
                                                 (n, _lw.NBNDLW))),
            inflg, iceflg, 1, M["cldfmcl"],
            cp.zeros((n, nl, NGPTLW), f32), M["ciwpmcl"], M["clwpmcl"],
            M["cswpmcl"], P["rei"], P["rel"], P["res"],
            cp.zeros((n, nl, _lw.NBNDLW), f32), C, column_chunk=chunk,
            mcica_layout="column", taumol_post=_taugb7_post)
        _kernel("prep", "rkp_lw_out")(
            ((n * nz + 127) // 128,), (128,),
            (np.int32(n), np.int32(nz), np.int32(nl), np.int64(c0),
             res["hr"], res["uflx"], res["dflx"], out["rthratenlw"],
             out["glw"], out["olr"]))
        for suffix, key, level in (
                ("upt", "uflx", -1), ("uptc", "uflxc", -1),
                ("dnt", "dflx", -1), ("dntc", "dflxc", -1),
                ("upb", "uflx", 0), ("upbc", "uflxc", 0),
                ("dnb", "dflx", 0), ("dnbc", "dflxc", 0)):
            out["lw" + suffix][rows] = res[key][:, level]
        del res
        day = np.nonzero(day_all[c0:c1])[0]
        if day.size:
            d = cp.asarray(day.astype(np.int64))
            nd = int(day.size)
            take = (lambda a: cp.ascontiguousarray(a[d]))
            # One gather of the lit columns' first NGPTSW g-points (a
            # gather then a slice would copy the full 140-wide slab first).
            mc = {k: cp.ascontiguousarray(v[d, :, :NGPTSW])
                  for k, v in M.items()}
            shape = (nd, nl, NGPTSW)
            alb = take(albedo[rows])
            try:
                r = sw.rrtmg_sw_batched_device(
                    nd, nl, 2, take(P["play"]), take(P["plev"]), take(P["tlay"]),
                    take(P["tlev"]), take(tsk[rows]), take(P["h2ovmr"]),
                    take(P["o3vmr"]), gas["co2"][:nd], gas["ch4"][:nd],
                    gas["n2o"][:nd], gas["o2"][:nd], alb, alb, alb, alb,
                    take(coszen[rows]), cp.ones(nd, f32), 0,
                    cp.full(nd, F(solcon), f32), inflg, iceflg, 1,
                    mc["cldfmcl"], cp.zeros(shape, f32), cp.ones(shape, f32),
                    cp.zeros(shape, f32), cp.zeros(shape, f32),
                    mc["ciwpmcl"], mc["clwpmcl"], mc["cswpmcl"],
                    take(P["rei"]), take(P["rel"]), take(P["res"]),
                    aer_opt=0, column_chunk=chunk, mcica_layout="column",
                    band_factor=band_factor)
            except BaseException:
                sw.drop_handoff()
                raise
            sel = d + c0
            _kernel("prep", "rkp_sw_out")(
                ((nd * nz + 127) // 128,), (128,),
                (np.int32(nd), np.int32(nz), np.int32(nl), sel, r["swhr"],
                 r["swuflx"], r["swdflx"], out["rthratensw"], out["gsw"]))
            for suffix, key, level in (
                    ("upt", "swuflx", -1), ("uptc", "swuflxc", -1),
                    ("dnt", "swdflx", -1), ("dntc", "swdflxc", -1),
                    ("upb", "swuflx", 0), ("upbc", "swuflxc", 0),
                    ("dnb", "swdflx", 0), ("dnbc", "swdflxc", 0)):
                out["sw" + suffix][sel] = r[key][:, level]
            del r, mc
        del P, M, gas
    sw.release_scratch()

    def _verdict(e):
        if int(e[0]):
            raise ValueError("MCICA_SUBCOL: KISSVEC SEED GENERATOR REQUIRES "
                             "PMID FROM BOTTOM FOUR LAYERS (RRTMG-K)")
    from woof.core.deferred_device_checks import check
    check(err, _verdict)
    if radii:
        out["re_qc"], out["re_qi"], out["re_qs"] = re[0], re[1], re[2]
    return out



# ---------------------------------------------------------------------------
# forecast adapter
# ---------------------------------------------------------------------------

#: Restart identities: distinct from legacy RRTMG's, so a checkpoint written
#: under one 4/4 or 14/14 pair refuses to resume under the other.
from woof.checkpoint_identity import (  # noqa: E402
    RRTMGK_ABOVE_ATMOSPHERE_POLICY, RRTMGK_ALGORITHM_IDENTITY)


def qnc_active(mp_physics):
    """WRF's F_QNC for this microphysics: whether its Registry package
    carries ``qnc`` (woof.wrf_physics_inventory, the packaged
    Registry.EM_COMMON contracts).  rad_rrtmg_driver's effectRad reads the
    cloud droplet number only under F_QNC."""
    from woof.wrf_physics_inventory import stock_wrf_physics_inventory
    inv = stock_wrf_physics_inventory(int(mp_physics))
    return any(f.registry_name == "qnc" for f in inv.wrfinput_fields)


class RRTMGKRadiation:
    """WRF 4.7.1 ``ra_lw_physics = ra_sw_physics = 14`` on the forecast path.

    module_radiation_driver.F:2103-2129 calls rad_rrtmg_driver from the
    longwave case with RTHRATEN as its longwave output, and the RRTMK_SWSCHEME
    case only adds RTHRATENSW (:2701-2709); the driver then publishes SW as
    RTHRATEN - RTHRATENLW like every pair (the legacy adapter's
    rla_driver_sw_tendency) and SWDOWN = GSW/(1-ALBEDO).  The driver's own
    inputs are the legacy adapter's: phy_prep's t8w (rla_t8w), cal_cldfra1
    (icloud = 1), radconst/calc_coszen and the CAM ozone root chain under
    o3input = 2.

    Fail closed on anything rad_rrtmg_driver does not read the way this
    port does (aer_opt != 0, icloud != 1, the MYNN/KF/BMJ/
    shallow-cumulus cloud merges, spectral output, nests receiving parent
    ozone) and on microphysics without snow (iceflag 5 is the ported
    shortwave cloud-optics arm).

    ghg_input = 1 goes beyond WRF, whose rad_rrtmg_driver never reads it:
    each call takes CO2, N2O, CH4, CFC-11 and CFC-12 from the CAMtr table
    at the valid time, exactly as legacy RRTMG does (read_CAMgases,
    woof.core.camtr_gases), into both spectra; o2, cfc22 and ccl4 stay
    fixed.  ghg_input = 0 is rad_rrtmg_driver's fixed gases, unchanged."""

    publishes_olr = True
    publishes_coszen = True
    publishes_cldfra = True
    supplies_surface_direct = False
    glw_provenance = "scheme"
    #: solar_ssi_opt (woof.spacewx.solar_spectrum): the band factor on RRTMG
    #: SW's 14 bands, set by PhysicsDriver before each call; None is off.
    accepts_sw_band_factor = True
    sw_band_factor = None

    def __init__(self, start_time, latitude_deg, longitude_deg, *,
                 o3input=2, icloud=1, column_chunk=None, ozone_parent=None,
                 ozone_routing=None, p_top=None, ghg_input=0):
        from datetime import datetime
        from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
        if not isinstance(start_time, datetime):
            raise TypeError("radiation_start_time must be a datetime")
        self.start_time = start_time
        lat = np.asarray(RRTMGLegacyRadiation._host(latitude_deg), F)
        lon = np.asarray(RRTMGLegacyRadiation._host(longitude_deg), F)
        if lat.shape != lon.shape:
            raise ValueError("radiation latitude/longitude shapes must match")
        self.latitude_deg = np.ascontiguousarray(lat)
        self.longitude_deg = np.ascontiguousarray(lon)
        self.o3input = int(o3input)
        if self.o3input not in (0, 2):
            raise ValueError("RRTMG-K implements o3input=0 (wrapper O3DATA) "
                             f"and o3input=2 (CAM climatology), got {o3input!r}")
        if int(icloud) != 1:
            raise NotImplementedError(
                f"RRTMG-K is ported for icloud=1 (cal_cldfra1); got icloud={icloud!r}")
        if int(ghg_input) not in (0, 1):
            raise ValueError(f"ghg_input must be 0 or 1, got {ghg_input!r}")
        self.ghg_input = int(ghg_input)
        if ozone_parent is not None and not callable(ozone_parent):
            raise TypeError("ozone_parent must be a callable provider or None")
        if ozone_parent is not None and self.o3input != 2:
            raise ValueError("o3input=0 builds ozone inside rad_rrtmg_driver "
                             "and must not receive a parent o3rad provider")
        #: A nest radiates its parent's interpolated o3rad, as WRF's nests
        #: do (legacy RRTMG's child route, woof.core.cam_ozone).
        self._ozone_provider = ozone_parent
        if column_chunk is not None and int(column_chunk) < 1:
            raise ValueError("column_chunk must be positive")
        self.column_chunk = None if column_chunk is None else int(column_chunk)
        from woof.core.cam_ozone import resolve_ozone_routing
        self.ozone_routing = resolve_ozone_routing(
            ozone_routing, o3input=self.o3input,
            has_parent=ozone_parent is not None)
        self.p_top = None if p_top is None else float(p_top)
        self._o33d_grid = None
        self.update_count = 0
        if self.o3input == 2 and ozone_parent is None:
            try:
                from woof.ingest import wrf_ozone as _ozone
                self._ozone = _ozone
                self._ozone_climo = _ozone.load_ozone_climatology()
            except Exception as exc:
                raise RuntimeError(
                    "ra_lw_physics=ra_sw_physics=14 with o3input=2 needs the "
                    f"packaged CAM ozone climatology: {exc}") from exc
        else:
            self._ozone = self._ozone_climo = None
        try:
            _engines()
            gpu_preflight()
            _lw.gpu_preflight()
        except Exception as exc:
            raise RuntimeError(
                "RRTMG-K (ra_lw_physics=ra_sw_physics=14) cannot build its "
                f"tables or kernels on this installation: {exc}") from exc

    # The retained o33d field, the CAM ozone this adapter radiated through
    # (nz, ny, nx), with legacy RRTMG's device/host contract.
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation as _L
    _o33d_grid = _L._o33d_grid
    o33d_share = _L.o33d_share
    o33d_device = _L.o33d_device
    _host = staticmethod(_L._host)
    _vertical_weight_device = _L._vertical_weight_device
    _call_coszen = _L._call_coszen
    coszen_now = _L.coszen_now
    del _L

    def restart_identity(self):
        identity = {
            "algorithm": RRTMGK_ALGORITHM_IDENTITY,
            "algorithms": {"lw": RRTMGK_ALGORITHM_IDENTITY,
                           "sw": RRTMGK_ALGORITHM_IDENTITY},
            "above_atmosphere_policy": RRTMGK_ABOVE_ATMOSPHERE_POLICY,
            "permuteseed": PERMUTESEED,
            "o3input": self.o3input,
            "column_chunk": self.column_chunk,
            "ozone_routing": self.ozone_routing,
        }
        if self.ghg_input:
            # Absent at 0, so a fixed-gas checkpoint keeps its identity.
            from woof.core.camtr_gases import camtr_identity
            identity["ghg_input"] = self.ghg_input
            identity["ghg_source"] = camtr_identity()
        if self.o3input == 2:
            from woof.ingest.wrf_ozone import (OZONE_LAT_SHA256,
                                                OZONE_PLEV_SHA256,
                                                OZONE_SHA256)
            identity["ozone_assets"] = {
                "ozone.formatted": OZONE_SHA256,
                "ozone_lat.formatted": OZONE_LAT_SHA256,
                "ozone_plev.formatted": OZONE_PLEV_SHA256,
            }
        return identity

    def _check_pins(self, cfg, mp_physics):
        def refuse(what):
            raise NotImplementedError(
                f"RRTMG-K (ra_lw_physics=ra_sw_physics=14): {what}; nothing "
                "is substituted")
        if int(getattr(cfg, "icloud", 1)) != 1:
            refuse(f"icloud={cfg.icloud} is not ported (icloud=1 only)")
        if int(getattr(cfg, "o3input", self.o3input)) != self.o3input:
            raise ValueError(
                f"adapter was constructed for o3input={self.o3input} but the "
                f"run config requests o3input={cfg.o3input}")
        ghg_input = int(getattr(cfg, "ghg_input", 0) or 0)
        if ghg_input != self.ghg_input:
            raise ValueError(
                f"adapter was constructed for ghg_input={self.ghg_input} but "
                f"the run config requests ghg_input={ghg_input}")
        aer_opt = int(getattr(cfg, "aer_opt", 0) or 0)
        if aer_opt:
            refuse(f"aer_opt={aer_opt} is not ported (aer_opt=0 only: no aerosol)")
        if int(getattr(cfg, "shcu_physics", 0) or 0) == 5:
            refuse("the shallow-cumulus cloud merge (shcu_physics=5) is not ported")
        if str(getattr(cfg, "rrtmg_spectral_output", "none") or "none") != "none":
            refuse("rrtmg_spectral_output is a legacy RRTMG (4/4) diagnostic; "
                   "rad_rrtmg_driver has no spectral flux output")
        from woof.core.rrtmg_legacy import mynn_bl_cloud_active
        if mynn_bl_cloud_active(getattr(cfg, "bl_pbl_physics", 0),
                                getattr(cfg, "icloud_bl", 0)):
            refuse("the MYNN subgrid-cloud merge (icloud_bl=1) is not ported")
        # The same convective-merge refusals as config.validate_rrtmgk_config,
        # for routes that build the adapter without validate_run_config.
        cu = int(getattr(cfg, "cu_physics", 0) or 0)
        if cu == 10:
            refuse("cu_physics=10 (KF-CuP) radiation cloud feedback is not ported")
        for name in ("cu_rad_feedback", "bmj_rad_feedback"):
            if bool(getattr(cfg, name, False)):
                refuse(f"the convective cloud merge ({name}=.true.) is not ported")
        if mp_physics in (11, 95):
            refuse(f"mp_physics={mp_physics} supplies radiation clouds "
                   "through a path rad_rrtmg_driver is not ported for")

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp
        from datetime import timedelta
        from woof.core.physics import RadiationResult
        from woof.core import rrtmgp as _rrtmgp
        from woof.core.physics_inventory import (
            RADIATION_BOUNDARY_FLUXES, registered_ice_mass_for_radiation)
        from woof.core.rrtmg_legacy import (
            _adapter_kernel, _device_flat, _device_grid, _grid_launch,
            adapter_gpu_preflight, legacy_cloud_fraction_flags)

        mp_physics = int(getattr(cfg, "mp_physics", 0))
        self._check_pins(cfg, mp_physics)
        if atmosphere.get("cumulus_feedback") is not None:
            raise NotImplementedError(
                "RRTMG-K: cumulus radiation feedback is not ported")
        if "radiation_cldfra3" in fields:
            raise NotImplementedError(
                "RRTMG-K: a scheme-supplied radiation cloud fraction is not "
                "ported (cal_cldfra1 only)")
        pressure = atmosphere["pressure"]
        nz, ny, nx = pressure.shape
        ncol = ny * nx
        if self.latitude_deg.shape != (ny, nx):
            raise ValueError("radiation latitude/longitude must match state grid")
        adapter_gpu_preflight()
        f_qnc = qnc_active(mp_physics)

        def cols(a, nk):            # (nk, ny, nx) -> (ncol, nk)
            return cp.ascontiguousarray(_device_grid(a, nk).T)

        g_p3d = _device_grid(pressure, nz)
        p3d = cp.ascontiguousarray(g_p3d.T)
        p8w = cols(atmosphere["p_interface"], nz + 1)
        t3d = cols(atmosphere["temperature"], nz)
        zw = cols(atmosphere["z_interface"], nz + 1)
        rho = cols(atmosphere["rho"], nz)
        zero = cp.zeros((ncol, nz), F)
        moist, flags_on = {}, {}
        for name in ("qv", "qc", "qi", "qs", "qg"):
            value = (registered_ice_mass_for_radiation(state) if name == "qi"
                     else getattr(state, name, None))
            flags_on[name] = value is not None
            moist[name] = zero if value is None else cols(value, nz)
        if not flags_on["qs"]:
            raise NotImplementedError(
                f"RRTMG-K: mp_physics={mp_physics} carries no snow; the "
                "ported shortwave cloud optics are rad_rrtmg_driver's "
                "iceflag 5 arm (F_QS)")
        nc_state = getattr(state, "nc", None)
        if f_qnc and nc_state is None:
            raise ValueError(f"mp_physics={mp_physics} registers QNC but the "
                             "state carries no droplet number (state.nc)")
        nc = cols(nc_state, nz) if f_qnc else zero

        fnm_d = self._vertical_weight_device(state.fnm)
        fnp_d = self._vertical_weight_device(state.fnp)
        t8w = cp.empty((ncol, nz + 1), F)
        _adapter_kernel("rla_t8w")(
            *_grid_launch(ncol * (nz + 1)),
            (np.int32(ncol), np.int32(nz), t3d, zw, fnm_d, fnp_d, t8w))
        del zw
        f_qi, f_qs = legacy_cloud_fraction_flags(mp_physics)
        cldfra = _rrtmgp.cal_cldfra1(
            moist["qv"], moist["qc"], moist["qi"], moist["qs"], t3d, p3d,
            f_qc=True, f_qi=f_qi, f_qs=f_qs, mp_physics=mp_physics)

        valid_time = self.start_time + timedelta(seconds=float(state.elapsed_seconds))
        julday = int(valid_time.timetuple().tm_yday)
        hour = (valid_time.hour + valid_time.minute / 60.0
                + valid_time.second / 3600.0 + valid_time.microsecond / 3.6e9)
        julian = F((julday - 1) + hour / 24.0)
        lat_flat = self.latitude_deg.reshape(-1)
        lon_flat = self.longitude_deg.reshape(-1)
        coszen_d, _coszen, _declin, solcon = self._call_coszen(
            state, cfg, julian, lat_flat, lon_flat, cp.asarray(lat_flat))
        coszen_d = cp.ascontiguousarray(cp.asarray(coszen_d, dtype=F))

        o33d = None
        if self.o3input == 2 and self._ozone_provider is not None:
            driver = getattr(state, "physics", None)
            owner = getattr(driver, "cam_ozone", None)
            if owner is not None and owner.mode == "parent-interpolated":
                if driver.call_counts.get("cam_ozone", 0) == 0:
                    raise RuntimeError(
                        "child radiation ran before parent CAM ozone FORCE")
                source = driver.o3rad
                if not (isinstance(source, cp.ndarray) and source.dtype == F
                        and source.shape == (nz, ny, nx)):
                    source = cp.asarray(np.asarray(
                        self._host(source), F).reshape(nz, ny, nx))
                snapshot = source.copy(order="C")
            else:
                host = np.asarray(self._ozone_provider(), F)
                if host.shape != (ncol, nz):
                    raise ValueError(
                        f"ozone_parent provider returned shape {host.shape}; "
                        f"this child grid needs ({ncol}, {nz})")
                snapshot = cp.asarray(np.ascontiguousarray(
                    host.reshape(ny, nx, nz).transpose(2, 0, 1)))
            self.__dict__["_o33d_host"] = None
            self.__dict__["_o33d_device"] = snapshot
            o33d = cp.ascontiguousarray(snapshot.reshape(nz, ncol).T)
        elif self.o3input == 2:
            ozmixt = np.asarray(self._ozone.ozn_latitude_time_int(
                julday, julian, lat_flat, self._ozone_climo,
                out=self.__dict__.get("_ozmixt_buf")))
            self.__dict__["_ozmixt_buf"] = ozmixt
            pin = np.asarray(self._ozone_climo.plev)
            if pin.dtype != np.float32 or pin.ndim != 1:
                raise ValueError("pin must be float32 (levsiz,)")
            if not np.all(np.diff(pin) > 0):
                raise ValueError("OZN_P_INT: Bad ozone data: "
                                 "non-monotonicity suspected")
            if ozmixt.dtype != np.float32 or ozmixt.shape != (ncol, pin.size):
                raise ValueError("ozmixt must be float32 (ncol, levsiz)")
            grid_o3 = cp.empty((nz, ncol), dtype=F)
            _adapter_kernel("rla_ozn_p_int")(
                ((ncol + 127) // 128,), (128,),
                (np.int32(ncol), np.int32(nz), np.int32(pin.size),
                 g_p3d, np.int64(ncol), np.int64(1), cp.asarray(pin),
                 cp.asarray(np.ascontiguousarray(ozmixt)),
                 grid_o3, np.int64(ncol), np.int64(1)))
            self.__dict__["_o33d_host"] = None
            self.__dict__["_o33d_device"] = grid_o3.reshape(nz, ny, nx)
            o33d = cp.ascontiguousarray(grid_o3.T)
        else:
            self._o33d_grid = None
        del g_p3d

        out = rrtmgk_columns(
            p3d=p3d, p8w=p8w, t3d=t3d, t8w=t8w, rho=rho, qv=moist["qv"],
            qc=moist["qc"], qi=moist["qi"], qs=moist["qs"], qg=moist["qg"],
            nc=nc, cldfra=cldfra, o33d=o33d,
            tsk=_device_flat(fields["tsk"]), emiss=_device_flat(fields["emiss"]),
            albedo=_device_flat(fields["albedo"]), coszen=coszen_d,
            solcon=F(solcon), f_qc=flags_on["qc"], f_qi=flags_on["qi"],
            f_qs=flags_on["qs"], f_qnc=f_qnc, o3input=self.o3input,
            column_chunk=self.column_chunk,
            gases=(camtr_gases(valid_time.year, julian) if self.ghg_input
                   else None),
            band_factor=self.sw_band_factor)
        del p3d, p8w, t3d, t8w, rho, moist, nc, o33d

        rthratenlw = cp.ascontiguousarray(out["rthratenlw"].T)
        rthratensw = cp.ascontiguousarray(out["rthratensw"].T)
        rthraten_total = (rthratenlw + rthratensw
                          if getattr(cfg, "use_theta_m", 0) else None)
        _adapter_kernel("rla_driver_sw_tendency")(
            *_grid_launch(nz * ncol), (np.int64(nz * ncol), rthratenlw, rthratensw))
        albedo = _device_flat(fields["albedo"])
        swdown = cp.empty(ncol, dtype=F)
        _adapter_kernel("rla_swdown")(
            *_grid_launch(ncol), (np.int64(ncol), out["gsw"], albedo, swdown))
        self.update_count += 1
        return RadiationResult(
            rthraten_total=(None if rthraten_total is None
                            else rthraten_total.reshape(nz, ny, nx)),
            boundary_fluxes={name: out[name].reshape(ny, nx)
                             for name in RADIATION_BOUNDARY_FLUXES},
            rthratenlw=rthratenlw.reshape(nz, ny, nx),
            rthratensw=rthratensw.reshape(nz, ny, nx),
            swdown=swdown.reshape(ny, nx),
            glw=out["glw"].reshape(ny, nx),
            gsw=out["gsw"].reshape(ny, nx),
            coszen=coszen_d.reshape(ny, nx),
            olr=out["olr"].reshape(ny, nx),
            cldfra=cp.ascontiguousarray(cldfra.T).reshape(nz, ny, nx))
