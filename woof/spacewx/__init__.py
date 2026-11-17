"""Space weather: opt-in solar and geomagnetic forcing beyond WRF/MPAS parity.

Nothing in this package runs unless a run turns it on (``solar_tsi_opt`` and the
later ``spacewx_*`` keys); the default WRF and hex paths never import it.  The
design, the physics ranking, data licences and the validation plan are in
docs/spacewx.md.
"""
