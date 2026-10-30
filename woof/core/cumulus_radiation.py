"""Cumulus cloud handed to radiation: WRF's ``cu_rad_feedback`` and Deng's.

``module_radiation_driver.F`` (WRF v4.7.1) modifies the radiation's copy of
the hydrometeors and cloud fraction in three places before any radiation
scheme runs (and restores QC/QI from qc_save/qi_save afterwards):

* ICLOUD_CU = 1 (Grell-Freitas, Grell 3D; ``cu_rad_feedback``, with
  ``cu_diag = 1``): before the cloud fraction, ``QC = QC + QC_CU`` and
  ``QI = QI + QI_CU``, the time-averaged convective cloud water/ice that
  convtrans_prep forms (:1264-1281);
* ICLOUD_CU = 2 (Kain-Fritsch; ``cu_rad_feedback``), inside the ICLOUD = 1
  branch right after cal_cldfra1: ``CLDFRA_CU = CLDFRA_DP + CLDFRA_SH``,
  ``CLDFRA = MIN(1, (1-CLDFRA_CU)*CLDFRA + CLDFRA_CU)``,
  ``QC = QC + QC_CU*CLDFRA_CU``, ``QI = QI + QI_CU*CLDFRA_CU`` (:1337-1348);
* ``shcu_physics = 5`` (Deng), after every ICLOUD branch:
  ``CLDFRA = MAX(CLDFRA_SH, CLDFRA)``, ``QC = CW_RAD + QC`` (:1566-1575).

ICLOUD_CU is derived from ``cu_rad_feedback`` and the cumulus scheme
(module_physics_init.F:1103-1118).  These functions are the one
transcription both RRTMG adapters call; each statement is one binary32
operation in the Fortran's order (separate launches, so nothing is
contracted), MIN/MAX are gfortran's fminnm/fmaxnm (cp.fmin/cp.fmax), and
tests/test_cu_rad_feedback_oracle.py holds them, with cal_cldfra1 between
them, to the verbatim WRF lines compiled by tools/cu_rad_feedback_oracle.
"""

from __future__ import annotations

import numpy as np

DTYPE = np.float32

#: ICLOUD_CU by cumulus scheme under cu_rad_feedback
#: (module_physics_init.F:1105-1118): KF 2; GF, G3 (and GD) 1.
ICLOUD_CU_BY_SCHEME = {1: 2, 3: 1, 5: 1}


def icloud_cu(cfg) -> int:
    """WRF's derived ICLOUD_CU for this configuration."""
    if not bool(getattr(cfg, "cu_rad_feedback", False)):
        return 0
    return ICLOUD_CU_BY_SCHEME.get(int(cfg.cu_physics), 0)


def add_convective_condensate(qc, qi, qc_cu, qi_cu, *, f_qc: bool,
                              f_qi: bool):
    """ICLOUD_CU = 1, :1264-1281: QC = QC + QC_CU, QI = QI + QI_CU."""
    if f_qc:
        qc = qc + qc_cu
    if f_qi:
        qi = qi + qi_cu
    return qc, qi


def merge_kf_cloud(cldfra, qc, qi, *, cldfra_dp, cldfra_sh, qc_cu, qi_cu):
    """ICLOUD_CU = 2, :1337-1348, statement for statement."""
    import cupy as cp
    cldfra_cu = cldfra_dp + cldfra_sh
    cldfra = (DTYPE(1.0) - cldfra_cu) * cldfra
    cldfra = cldfra + cldfra_cu
    cldfra = cp.fmin(DTYPE(1.0), cldfra)
    qc = qc + qc_cu * cldfra_cu
    qi = qi + qi_cu * cldfra_cu
    return cldfra, qc, qi


def merge_deng_cloud(cldfra, qc, *, cldfra_sh, cw_rad):
    """``shcu_physics = 5``, :1566-1575."""
    import cupy as cp
    cldfra = cp.fmax(cldfra_sh, cldfra)
    qc = cw_rad + qc
    return cldfra, qc


#: The driver fields each merge reads (``fields`` names, WRF Registry names
#: lower-cased).
KF_FEEDBACK_FIELDS = ("cldfra_dp", "cldfra_sh", "qc_cu", "qi_cu")
GRELL_FEEDBACK_FIELDS = ("qc_cu", "qi_cu")
DENG_FIELDS = ("cldfra_sh", "cw_rad")


def radiation_cloud_fields(cfg) -> tuple[str, ...]:
    """Every driver field the radiation's cloud merge reads for ``cfg``."""
    names = []
    level = icloud_cu(cfg)
    if level == 2:
        names += KF_FEEDBACK_FIELDS
    elif level == 1:
        names += GRELL_FEEDBACK_FIELDS
    if int(getattr(cfg, "shcu_physics", 0)) == 5:
        names += DENG_FIELDS
    return tuple(dict.fromkeys(names))


__all__ = ["DENG_FIELDS", "GRELL_FEEDBACK_FIELDS", "ICLOUD_CU_BY_SCHEME",
           "KF_FEEDBACK_FIELDS", "add_convective_condensate", "icloud_cu",
           "merge_deng_cloud", "merge_kf_cloud", "radiation_cloud_fields"]
