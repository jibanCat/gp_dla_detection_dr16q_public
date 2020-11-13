% load_process_catalogs.m : pre-load the prior catalogs and preloaded_qsos
% before running the process_a_qso script.

addpath multi_dlas
set_parameters_multi;

% prior settings
% specify the learned quasar model to use
training_release  = 'dr16q';                % where the leanrned file is
training_set_name = 'dr16q_minus_dr12q_gp';
prior_release     = 'dr12q';                % where the prior catalog and sample files are

% specify the spectra to use for computing the DLA existence prior
% Note: here we are still using dr9 prior
dla_catalog_name  = 'dr9q_concordance';
prior_ind = ...
    [' prior_catalog.in_dr9 & '             ...
    '(prior_catalog.filter_flags == 0) & ' ...
    ' prior_catalog.los_inds(dla_catalog_name)'];

% specify the spectra to process
release       = 'dr16q';
test_set_name = 'dr16q';
test_ind      = '(catalog.filter_flags == 0)';

% set lls parameters
set_lls_parameters;

% multi-dlas parameters
max_dlas = 4;
min_z_separation = kms_to_z(3000);

% the mean values of Kim's effective optical depth
tau_0_mu    = 0.0023;
tau_0_sigma = 0.0007;
beta_mu     = 3.65;
beta_sigma  = 0.21;
% % Kamble 2019 values
% tau_0_mu    = 0.00554;
% tau_0_sigma = 0.00064;
% beta_mu     =   3.182;
% beta_sigma  =   0.074;
% % Becker 2013
% tau_0_mu    = 0.0097;
% tau_0_sigma = 0.0021;
% beta_mu     =   2.90;
% beta_sigma  =   0.12;


% load redshifts/DLA flags from training release
prior_catalog = ...
    load(sprintf('%s/catalog', processed_directory(prior_release)));

if (ischar(prior_ind))
  prior_ind = eval(prior_ind);
end

prior.z_qsos  = prior_catalog.z_qsos(prior_ind);
prior.dla_ind = prior_catalog.dla_inds(dla_catalog_name);
prior.dla_ind = prior.dla_ind(prior_ind);

% filter out DLAs from prior catalog corresponding to region of spectrum below
% Ly∞ QSO rest
prior.z_dlas = prior_catalog.z_dlas(dla_catalog_name);
prior.z_dlas = prior.z_dlas(prior_ind);

for i = find(prior.dla_ind)'
  if (observed_wavelengths(lya_wavelength, prior.z_dlas{i}) < ...
      observed_wavelengths(lyman_limit,    prior.z_qsos(i)))
    prior.dla_ind(i) = false;
  end
end

prior = rmfield(prior, 'z_dlas');

% load QSO model from training release
variables_to_load = {'rest_wavelengths', 'mu', 'M', 'log_omega', ...
                     'log_c_0', 'log_tau_0', 'log_beta'};
load(sprintf('%s/learned_qso_model_lyseries_variance_wmu_boss_%s_%d-%d', ...
             processed_directory(training_release),                 ...
             training_set_name,                                     ...
             int64(min_lambda), int64(max_lambda)),                 ...
     variables_to_load{:});

% load DLA samples from training release
variables_to_load = {'offset_samples', 'log_nhi_samples', 'nhi_samples'};
load(sprintf('%s/dla_samples_a03', processed_directory(prior_release)), ...
     variables_to_load{:});

% load redshifts from catalog to process
catalog = load(sprintf('%s/catalog', processed_directory(release)));

% load preprocessed QSOs
variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
                     'all_pixel_mask'};
load(sprintf('%s/preloaded_qsos', processed_directory(release)), ...
     variables_to_load{:});

% enable processing specific QSOs via setting to_test_ind
if (ischar(test_ind))
  test_ind = eval(test_ind);
end

fprintf_debug('Debug:real size of the full data set: %i\n', sum(test_ind))

all_wavelengths    =    all_wavelengths(test_ind);
all_flux           =           all_flux(test_ind);
all_noise_variance = all_noise_variance(test_ind);
all_pixel_mask     =     all_pixel_mask(test_ind);

z_qsos = catalog.z_qsos(test_ind);

num_quasars = 1e1;

% preprocess model interpolants
mu_interpolator = ...
    griddedInterpolant(rest_wavelengths,        mu,        'linear');
M_interpolator = ...
    griddedInterpolant({rest_wavelengths, 1:k}, M,         'linear');
log_omega_interpolator = ...
    griddedInterpolant(rest_wavelengths,        log_omega, 'linear');

% initialize results
min_z_dlas                 = nan(num_quasars, 1);
max_z_dlas                 = nan(num_quasars, 1);
log_priors_no_dla          = nan(num_quasars, 1);
log_priors_dla             = nan(num_quasars, max_dlas);
log_likelihoods_no_dla     = nan(num_quasars, 1);
sample_log_likelihoods_dla = nan(num_quasars, num_dla_samples, max_dlas);
base_sample_inds           = zeros(num_quasars, num_dla_samples, max_dlas - 1, 'uint32');
log_likelihoods_dla        = nan(num_quasars, max_dlas);
log_posteriors_no_dla      = nan(num_quasars, 1);
log_posteriors_dla         = nan(num_quasars, max_dlas);

% initialize lls results
log_likelihoods_lls        = nan(num_quasars, 1);
log_posteriors_lls         = nan(num_quasars, 1);
log_priors_lls             = nan(num_quasars, 1);
sample_log_likelihoods_lls = nan(num_quasars, num_dla_samples);

% save maps: add the initilizations of MAP values
% N * (1~k models) * (1~k MAP dlas)
MAP_z_dlas   = nan(num_quasars, max_dlas, max_dlas);
MAP_log_nhis = nan(num_quasars, max_dlas, max_dlas);
MAP_inds     = nan(num_quasars, max_dlas, max_dlas);


c_0   = exp(log_c_0);
tau_0 = exp(log_tau_0);
beta  = exp(log_beta);

% handle the inds of empty spectra
all_exceptions = nan(num_quasars, 1);
