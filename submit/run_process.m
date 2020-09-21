% run_process.m : run the processing script from a single file
% qso_start_ind  : the start index for the batch to run
% qso_end_ind    : the end index for all of the batches
% num_quasars    : the size of the saved array
% qsos_num_offset: the starting quasar_ind for the processed quasars

cd ..

addpath multi_dlas

for i = qso_start_ind:qso_end_ind
    set_parameters_multi;

    % change min z to lyb
    min_z_dla = @(wavelengths, z_qso) ...         % determines minimum z_DLA to search
        max(min(wavelengths) / lya_wavelength - 1,                          ...
            observed_wavelengths(lyb_wavelength, z_qso) / lya_wavelength - 1 + ...
            min_z_cut);


    % prior settings
    % specify the learned quasar model to use
    training_release  = 'dr12q';
    training_set_name = 'dr12q_minus_gp';

    % specify the spectra to use for computing the DLA existence prior
    dla_catalog_name  = 'dr9q_concordance';
    prior_ind = ...
        [' prior_catalog.in_dr9 & '             ...
        '(prior_catalog.filter_flags == 0) & ' ...
        ' prior_catalog.los_inds(dla_catalog_name)'];

    % specify the spectra to process
    release = 'dr14q';
    test_set_name = 'dr14q';
    test_ind = '(catalog.filter_flags == 0)';

    % set lls parameters
    set_lls_parameters;

    % start from the stopped ind last time
    qsos_num_offset = offset + (i - 1) * num_quasars;

    process_qsos_multiple_dlas_meanflux
end
