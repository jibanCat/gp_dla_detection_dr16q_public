% run_preload.m Run preload and build catalog

cd ..

addpath multi_dlas/
addpath dr16q/

release = 'dr16q';

set_parameters_multi;
assert(loading_max_lambda > 1400)

file_loader = @(plate, mjd, fiber_id) ...
  (read_spec(sprintf('%s/%i/spec-%i-%i-%04i.fits', ...
    spectra_directory(release),                  ...
    plate,                                       ...
    plate,                                       ...
    mjd,                                         ...
    fiber_id)));

preload_qsos_dr16q;
