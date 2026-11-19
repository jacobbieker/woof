#!/bin/bash
# Run the WRFDA 3D-Var tutorial case (2008-02-05 12Z, 90x60x41, 60 km) inside
# the oracle container.  Layout under $W (default /d/scratch/wrfda):
#   src/<TREE>/   a built WRFV4.7.1 (TREE defaults to WRFV4.7.1; the dump
#                 copy made by apply_dump_patch.sh is WRFV4.7.1-dump)
#   data/         the unpacked WRFDAV3.9-testdata tarball
# usage: run_tutorial.sh NAME [extra wrfvar10 settings, e.g. test_transforms=true,]
set -ex
W=${W:-/d/scratch/wrfda}; S=$W/src/${TREE:-WRFV4.7.1}; D=$W/data; R=$W/run/$1
rm -rf "$R"; mkdir -p "$R"; cd "$R"
ln -sf "$S/var/build/da_wrfvar.exe" .
ln -sf "$S/run/LANDUSE.TBL" .
cp "$D/be/be.dat" be.dat
cp "$D/ob/2008020512/ob.ascii" ob.ascii
cp "$D/rc/2008020512/wrfinput_d01" fg
sed "s/^test_transforms=false,/test_transforms=false,${2}/" "$S/var/test/tutorial/namelist.input" \
  | sed "s/^start_year=2008,/force_use_old_data=.true.,\nstart_year=2008,/" > namelist.input
/usr/bin/time -v ./da_wrfvar.exe > wrfda.out 2> time.txt || true
grep -E "Elapsed|Maximum resident" time.txt
