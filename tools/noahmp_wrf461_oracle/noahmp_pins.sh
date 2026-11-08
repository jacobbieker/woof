# Sourced (POSIX sh) by the build scripts in this directory: the two pinned
# Noah-MP trees an oracle may be built from.  noahmp_pins.py carries the same
# table for the Python tools; tests/test_noahmp_oracle_harness.py holds the two
# in step.
#
#   wrf-4.6.1  WRF d66e442fccc04111067e29274c9f9eaccc3cef28,
#              phys/noahmp at 848f54ad3d28c4303151fe5ad83724e232694422
#   wrf-4.7.1  WRF f52c197ed39d12e087d02c50f412d90d418f6186,
#              phys/noahmp at e5c0859874407859936739e8be8741f9aed369ee
#
# noahmp_pins <module_sf_noahmplsm.F> picks the tree whose LSM digest the file
# carries and sets the digests every script checks against that same tree:
#   NOAHMP_TREE        wrf-4.6.1 | wrf-4.7.1 (the label sums files carry)
#   NOAHMP_WRF_COMMIT  WRF commit
#   NOAHMP_COMMIT      phys/noahmp submodule commit
#   PRISTINE_LSM_SHA   module_sf_noahmplsm.F
#   PRISTINE_DRV_SHA   module_sf_noahmpdrv.F
#   PRISTINE_GLA_SHA   module_sf_noahmp_glacier.F
#   PATCHED_SHA        module_sf_noahmplsm.F after
#                      patches/noahmp-lsm-leaf-visibility.patch (public ::)
#   PADDED_SHA         module_sf_noahmplsm.F after visibility_patch_leaves.py
#                      (public  ::, padded)
# Any other file is refused (exit 3).  The groundwater module and the tables
# are byte-identical between the two trees.

noahmp_pins() {
  _np_got=$(sha256sum "$1" | cut -d' ' -f1)
  case "$_np_got" in
    bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282)
      NOAHMP_TREE=wrf-4.6.1
      NOAHMP_WRF_COMMIT=d66e442fccc04111067e29274c9f9eaccc3cef28
      NOAHMP_COMMIT=848f54ad3d28c4303151fe5ad83724e232694422
      PRISTINE_LSM_SHA=bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282
      PRISTINE_DRV_SHA=9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890
      PRISTINE_GLA_SHA=bf94f3522c3b9c2c9cfbb34fa7e485ff58519106db434520968793409a520579
      PATCHED_SHA=3cd3690d6455cfb8549cb41979b7e101e7436464c478f7f7973ab226488ac206
      PADDED_SHA=bfdc0f3632cd30b87208b26a309c533b12d9bc2a39d1a36e9165ecf90d0a12c3
      ;;
    74b475600ad1c999fe43299bcaf4d3e5eaf59b9f861c47fdbff4d69739e3a45b)
      NOAHMP_TREE=wrf-4.7.1
      NOAHMP_WRF_COMMIT=f52c197ed39d12e087d02c50f412d90d418f6186
      NOAHMP_COMMIT=e5c0859874407859936739e8be8741f9aed369ee
      PRISTINE_LSM_SHA=74b475600ad1c999fe43299bcaf4d3e5eaf59b9f861c47fdbff4d69739e3a45b
      PRISTINE_DRV_SHA=29f16622740d7388a17ff9ba6c2f4640487e44f3f26b8c4bdd47aad42e0a4572
      PRISTINE_GLA_SHA=a35d61ea29f660f4f4c6384bc019d4a57ddc8b0c2ca0f3ef7c6e621b79b4dece
      PATCHED_SHA=5e31fa7fbd6aadfdc2e63581b9dc400cf2ec74305859c29a35d9c578a8c7ad06
      PADDED_SHA=072309bc12d64a3e15cf13252ca74411d321e44d3ff9a88bbb8cfcb261ec5ed4
      ;;
    *)
      echo "$1 is neither pinned module_sf_noahmplsm.F (got $_np_got)" >&2
      echo "  wrf-4.6.1/848f54ad bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282" >&2
      echo "  wrf-4.7.1/e5c0859  74b475600ad1c999fe43299bcaf4d3e5eaf59b9f861c47fdbff4d69739e3a45b" >&2
      exit 3
      ;;
  esac
  echo "[pins] $NOAHMP_TREE: WRF $NOAHMP_WRF_COMMIT, phys/noahmp $NOAHMP_COMMIT"
}

# noahmp_label_sums <sums-file> <tree-root>: rewrite the tree's absolute path
# prefix to <$NOAHMP_TREE-tree>/ and append the submodule commit, so a sums
# file names the tree it was built from and nothing about the host.
noahmp_label_sums() {
  _np_real=$(realpath "$2")
  _np_given=${2%/}
  sed -e "s#$_np_real/#<$NOAHMP_TREE-tree>/#" \
      -e "s#$_np_given/#<$NOAHMP_TREE-tree>/#" "$1" > "$1.tmp" && mv "$1.tmp" "$1"
  echo "$NOAHMP_COMMIT  <$NOAHMP_TREE-tree>/phys/noahmp (submodule commit)" >> "$1"
}
