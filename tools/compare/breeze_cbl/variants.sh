# shellcheck shell=bash
# The variant table shared by run_timing.sh and run_e2e_small.sh: one name,
# one command, wherever it runs.  Source after setting WOOF (python driver
# prefix), BREEZE (julia driver prefix) and SUBSTEPS; fills the associative
# array CMD (declare -A CMD before sourcing).
#
# *_ci are the exact BreezeBenchmarks CI configurations (stock u* drag,
# implicit LES, 12 proportional substeps, forward weight 0.65); *_matched
# use woof's operator set (h5/v3 upwind, WRF km_opt=3 Smagorinsky, bulk
# drag, WRF substep distribution, epssm-equivalent forward weight 0.55).
CMD[woof_km3]="$WOOF --km-opt 3 --substeps $SUBSTEPS --label woof_km3"
CMD[woof_km2]="$WOOF --km-opt 2 --substeps $SUBSTEPS --label woof_km2"
CMD[breeze_anelastic_weno5_ci]="$BREEZE --dynamics anelastic --advection WENO5 --closure none --drag ustar --label breeze_anelastic_weno5_ci"
CMD[breeze_se_weno5_ci]="$BREEZE --dynamics compressible_splitexplicit --advection WENO5 --closure none --drag ustar --substeps 12 --distribution proportional --forward-weight 0.65 --label breeze_se_weno5_ci"
CMD[breeze_se_matched]="$BREEZE --dynamics compressible_splitexplicit --advection Upwind53 --closure Smagorinsky --drag bulk --substeps $SUBSTEPS --distribution wrf --forward-weight 0.55 --label breeze_se_matched"
CMD[breeze_se_weno5_smag]="$BREEZE --dynamics compressible_splitexplicit --advection WENO5 --closure Smagorinsky --drag bulk --substeps $SUBSTEPS --distribution wrf --forward-weight 0.55 --label breeze_se_weno5_smag"
CMD[breeze_anelastic_matched]="$BREEZE --dynamics anelastic --advection Upwind53 --closure Smagorinsky --drag bulk --label breeze_anelastic_matched"
