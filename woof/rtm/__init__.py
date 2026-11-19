"""Opt-in radiative transfer for imager radiance assimilation.

``woof.rtm.crtm`` drives JCSDA CRTM v2.4+ (CC0) clear-sky infrared brightness
temperatures and their Jacobians through a small C shim; ``woof.rtm.profiles``
turns woof model columns into CRTM atmospheres.  Nothing here runs unless the
user builds CRTM (``python -m woof.rtm.crtm_build``) and names its library and
coefficient directory (``WOOF_CRTM_LIB``, ``WOOF_CRTM_COEFFS``).  The default
wheel carries no CRTM, no library and no coefficients
(docs/dev/WRFDA-3DVAR-PLAN.md section 11).
"""
