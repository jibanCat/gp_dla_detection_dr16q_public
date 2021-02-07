#!/bin/bash

# downlaod_catalogs_dr16q.sh: download SDSS DR16Q catalogs

base_directory='..'
pushd $base_directory

# DR16Q
directory='dr16q'

mkdir -p $directory/spectra $directory/processed $directory/distfiles
pushd $directory/distfiles
filename='DR16Q_v4.fits'
wget https://data.sdss.org/sas/dr16/eboss/qso/DR16Q/$filename -O $filename
popd
