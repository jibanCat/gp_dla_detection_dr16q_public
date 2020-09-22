% run_catalog.m : build catalog

cd ..

addpath multi_dlas/
addpath dr16q/

set_parameters_multi;
build_catalogs_dr16q;
