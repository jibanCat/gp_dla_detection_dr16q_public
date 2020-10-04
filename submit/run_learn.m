% run_learn.m : learn a GP null model from a subset of
% DR12Q in DR16Q spectra.

cd ..

addpath multi_dlas/
addpath dr16q/

set_parameters_multi;

cd minFunc_2012/
addpath(genpath(pwd));
mexAll;
cd ..

training_set_name = 'dr16q_minus_dr12q_gp';
learn_qso_model_meanflux_dr16q;
