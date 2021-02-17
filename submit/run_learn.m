% run_learn.m : learn a GP null model from a subset of
% DR12Q in DR16Q spectra.

cd ..

set_parameters;

cd minFunc_2012/
addpath(genpath(pwd));
mexAll;
cd ..

training_set_name = 'dr16q_minus_dr12q_gp';
learn_qso_model;
