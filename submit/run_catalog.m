% run_catalog.m : build catalog

cd ..

addpath multi_dlas/
addpath dr16q/

set_parameters_multi;
build_catalog_dr16q;
