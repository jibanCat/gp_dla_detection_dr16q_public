#!/bin/bash

# download_spectra_dr16q.sh: downloads DR16Q spectra from SDSS

base_directory='..'
pushd $base_directory/dr16q/spectra

rsync --info=progress2 -h --no-motd --files-from=file_list rsync://data.sdss.org/dr16/eboss/spectro/redux/ . 2> /dev/null
