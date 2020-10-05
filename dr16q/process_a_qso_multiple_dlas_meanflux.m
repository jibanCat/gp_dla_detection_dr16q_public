% process_a_qso_multiple_dlas_meanflux: run DLA detection algorithm on a specified object
% while using lower lognhi range (defined in set_lls_parameters.m) as an alternative model;

thing_ids = catalog.thing_ids(test_ind);

% change min z to lyb
min_z_dla = @(wavelengths, z_qso) ...         % determines minimum z_DLA to search
    max(min(wavelengths) / lya_wavelength - 1,                          ...
        observed_wavelengths(lyb_wavelength, z_qso) / lya_wavelength - 1 + ...
        min_z_cut);

selected_thing_ids = [355787041]; % 23097883
[vals, selected_quasar_inds]= intersect(thing_ids, selected_thing_ids, 'stable');
selected_thing_ids = vals; % update the vals since the ordering would change

for i = 1:numel(selected_thing_ids)
  tic;
  rng('default');  % random number should be set for each qso run

  quasar_ind = selected_quasar_inds(i);

  % initialize an empty array for this sample log likelihood
  this_sample_log_likelihoods_dla = nan(num_dla_samples, max_dlas);

  z_qso = z_qsos(quasar_ind);

  fprintf('processing quasar %i/%i (z_QSO = %0.4f) ...', ...
        quasar_ind, num_quasars, z_qso);

  this_wavelengths    =    all_wavelengths{quasar_ind};
  this_flux           =           all_flux{quasar_ind};
  this_noise_variance = all_noise_variance{quasar_ind};
  this_pixel_mask     =     all_pixel_mask{quasar_ind};

  % saving index set to 1
  quasar_ind = 1;

  % convert to QSO rest frame
  this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qso);

  unmasked_ind = (this_rest_wavelengths >= min_lambda) & ...
                 (this_rest_wavelengths <= max_lambda);

  % keep complete copy of equally spaced wavelengths for absorption
  % computation
  this_unmasked_wavelengths = this_wavelengths(unmasked_ind);

  ind = unmasked_ind & (~this_pixel_mask);

  this_wavelengths      =      this_wavelengths(ind);
  this_rest_wavelengths = this_rest_wavelengths(ind);
  this_flux             =             this_flux(ind);
  this_noise_variance   =   this_noise_variance(ind);
      
  % DLA existence prior
  less_ind = (prior.z_qsos < (z_qso + prior_z_qso_increase));

  this_num_dlas    = nnz(prior.dla_ind(less_ind));
  this_num_quasars = nnz(less_ind);
  this_p_dlas      = (this_num_dlas / this_num_quasars).^(1:max_dlas);

  % the prior for having exactly k DLAs is (M / N)^k - (M / N)^(k + 1) 
  for i = 1:(max_dlas - 1)
    this_p_dlas(i) = this_p_dlas(i) - this_p_dlas(i + 1);
  end
  % make sure the sum of multi-DLA priors is equal to M / N, which is
  % the prior for at least one DLA.
  assert( abs(sum(this_p_dlas) - (this_num_dlas / this_num_quasars) ) < 1e-4 )

  log_priors_dla(quasar_ind, :) = log(this_p_dlas);

  % lls priors : assume extrapolated log_nhi prior for lls
  % lls prior = M / N * Z_lls / Z_dla
  log_priors_lls(quasar_ind) = ...
    log(this_num_dlas) - log(this_num_quasars) + ...
    log(Z_lls)         - log(Z_dla);

  % no dla priors : subtract from dla and lls priors
  % no dla prior = 1 - M / N - M / N * Z_lls / Z_dla
  log_priors_no_dla(quasar_ind) = ...
      log(this_num_quasars - this_num_dlas - Z_lls * this_num_dlas / Z_dla) ...
      - log(this_num_quasars);

  fprintf_debug('\n');
  for i = 1:max_dlas
    fprintf_debug(' ...     p(%i  DLAs | z_QSO)       : %0.3f\n', i, this_p_dlas(i));
  end
  fprintf_debug(' ...     p(no DLA  | z_QSO)       : %0.3f\n', exp(log_priors_no_dla(quasar_ind)) );
  fprintf_debug(' ...     p(sub DLA | z_QSO)       : %0.3f\n', exp(log_priors_lls(quasar_ind)) );

  % interpolate model onto given wavelengths
  % error will appear if the input spectrum is empty
  try
    this_mu = mu_interpolator( this_rest_wavelengths);
    this_M  =  M_interpolator({this_rest_wavelengths, 1:k});
  catch me
    if (strcmp(me.identifier, 'MATLAB:griddedInterpolant:NonVecCompVecErrId'))
      all_exceptions(quasar_ind, 1) = 1;
      fprintf(' took %0.3fs.\n', toc);      
      continue
    else
      rethrow(me)
    end
  end

  this_log_omega = log_omega_interpolator(this_rest_wavelengths);
  this_omega2 = exp(2 * this_log_omega);

  % set Lyseries absorber redshift for mean-flux suppression
  % apply the lya_absorption after the interpolation because NaN will appear in this_mu
  total_optical_depth = effective_optical_depth(this_wavelengths, ...
      beta_mu, tau_0_mu, z_qso, ...
      all_transition_wavelengths, all_oscillator_strengths, ...
      num_forest_lines);
  
  % total absorption effect of Lyseries absorption on the mean-flux
  lya_absorption = exp(- sum(total_optical_depth, 2) );  
  
  this_mu = this_mu .* lya_absorption;
  this_M  = this_M  .* lya_absorption;

  % set another Lysieres absorber redshift to use in coveriance
  lya_optical_depth = effective_optical_depth(this_wavelengths, ...
      beta, tau_0, z_qso, ...
      all_transition_wavelengths, all_oscillator_strengths, ...
      num_forest_lines);

  this_scaling_factor = 1 - exp( -sum(lya_optical_depth, 2) ) + c_0;

  % this is the omega included the Lyseries
  this_omega2 = this_omega2 .* this_scaling_factor.^2;

  % re-adjust (K + Ω) to the level of μ .* exp( -optical_depth ) = μ .* a_lya
  % now the null model likelihood is:
  % p(y | λ, zqso, v, ω, M_nodla) = N(y; μ .* a_lya, A_lya (K + Ω) A_lya + V)
  this_omega2 = this_omega2 .* lya_absorption.^2;

  % baseline: probability of no DLA model
  log_likelihoods_no_dla(quasar_ind) = ...
      log_mvnpdf_low_rank(this_flux, this_mu, this_M, ...
          this_omega2 + this_noise_variance);

  log_posteriors_no_dla(quasar_ind) = ...
      log_priors_no_dla(quasar_ind) + log_likelihoods_no_dla(quasar_ind);

  fprintf_debug(' ... log p(  D  | z_QSO, no DLA ) : %0.2f\n', ...
                log_likelihoods_no_dla(quasar_ind));

  min_z_dlas(quasar_ind) = min_z_dla(this_wavelengths, z_qso);
  max_z_dlas(quasar_ind) = max_z_dla(this_wavelengths, z_qso);

  sample_z_dlas = ...
       min_z_dlas(quasar_ind) +  ...
      (max_z_dlas(quasar_ind) - min_z_dlas(quasar_ind)) * offset_samples;

  this_base_sample_inds = zeros(max_dlas - 1, num_dla_samples, 'uint32');

  % save maps: make extract MAP values much easier
  % the only difference is including the 1st model inds
  this_MAP_base_sample_inds       = zeros(max_dlas, num_dla_samples, 'uint32');
  this_MAP_base_sample_inds(1, :) = 1:num_dla_samples;

  % ensure enough pixels are on either side for convolving with
  % instrument profile
  padded_wavelengths = ...
      [logspace(log10(min(this_unmasked_wavelengths)) - width * pixel_spacing, ...
                log10(min(this_unmasked_wavelengths)) - pixel_spacing,         ...
                width)';                                                       ...
       this_unmasked_wavelengths;                                              ...
       logspace(log10(max(this_unmasked_wavelengths)) + pixel_spacing,         ...
                log10(max(this_unmasked_wavelengths)) + width * pixel_spacing, ...
                width)'                                                        ...
      ];

  % to retain only unmasked pixels from computed absorption profile
  % this has to be done by using the unmasked_ind which has not yet
  % been applied this_pixel_mask.
  mask_ind = (~this_pixel_mask(unmasked_ind));

  for num_dlas = 1:max_dlas
    % compute probabilities under DLA model for each of the sampled
    % (normalized offset, log(N HI)) pairs
    parfor i = 1:num_dla_samples
      % absorption corresponding to this sample
      absorption = voigt(padded_wavelengths, sample_z_dlas(i), ...
                         nhi_samples(i), num_lines);

      % absorption corresponding to other DLAs in multiple DLA samples
      for j = 1:(num_dlas - 1)
        k = this_base_sample_inds(j, i);
        absorption = absorption .* ...
            voigt(padded_wavelengths, sample_z_dlas(k), ...
                  nhi_samples(k), num_lines);
      end

      absorption = absorption(mask_ind);

      dla_mu     = this_mu     .* absorption;
      dla_M      = this_M      .* absorption;
      dla_omega2 = this_omega2 .* absorption.^2;

      this_sample_log_likelihoods_dla(i, num_dlas) = ...
        log_mvnpdf_low_rank(this_flux, dla_mu, dla_M, ...
            dla_omega2 + this_noise_variance) - log(num_dla_samples);
            % additional occams razor

      % compute lls model
      if (num_dlas == 1)
        % absorption with lls column density
        absorption = voigt(padded_wavelengths, sample_z_dlas(i), ...
          lls_nhi_samples(i), num_lines);

        absorption = absorption(mask_ind);

        lls_mu     = this_mu     .* absorption;
        lls_M      = this_M      .* absorption;
        lls_omega2 = this_omega2 .* absorption.^2;

        sample_log_likelihoods_lls(quasar_ind, i) = ...
          log_mvnpdf_low_rank(this_flux, lls_mu, lls_M, ...
              lls_omega2 + this_noise_variance) - log(num_dla_samples);
              % additional occams razor
      end
    end

    % check if any pair of dlas in this sample is too close this has to
    % happen outside the parfor because "continue" slows things down
    % dramatically
    if (num_dlas > 1)
      ind = this_base_sample_inds(1:(num_dlas - 1), :);
      all_z_dlas   = [sample_z_dlas; sample_z_dlas(ind)];
      all_log_nhis = [log_nhi_samples; log_nhi_samples(ind)]; 

      ind = any(diff(sort(all_z_dlas)) < min_z_separation, 1);
      this_sample_log_likelihoods_dla(ind, num_dlas) = nan;

    elseif (num_dlas == 1)
      all_z_dlas   = sample_z_dlas;
      all_log_nhis = log_nhi_samples;

    end

    max_log_likelihood = ...
        nanmax(this_sample_log_likelihoods_dla(:, num_dlas));

    sample_probabilities = ...
        exp(this_sample_log_likelihoods_dla(:, num_dlas) - ...
            max_log_likelihood);

    log_likelihoods_dla(quasar_ind, num_dlas) = ...
        max_log_likelihood + log(nanmean(sample_probabilities)) ...
        - log( num_dla_samples ) * (num_dlas - 1); % occam's razor

    log_posteriors_dla(quasar_ind, num_dlas) = ...
        log_priors_dla(quasar_ind, num_dlas) + ...
        log_likelihoods_dla(quasar_ind, num_dlas);

    % compute lls log posterior          
    if (num_dlas == 1)
      max_log_likelihood_lls = ...
        nanmax(sample_log_likelihoods_lls(quasar_ind, :));
      
      sample_probabilities_lls = ...
        exp(sample_log_likelihoods_lls(quasar_ind, :) - ...
            max_log_likelihood_lls);
      
      log_likelihoods_lls(quasar_ind) = ...
        max_log_likelihood_lls + log(nanmean(sample_probabilities_lls)) ...
        - log( num_dla_samples ) * (num_dlas - 1); % occam's razor

      log_posteriors_lls(quasar_ind) = ...
          log_priors_lls(quasar_ind) + ...
          log_likelihoods_lls(quasar_ind);

      fprintf_debug(' ... log p(D | z_QSO, sub DLA) : %0.2f\n', ...
          log_likelihoods_lls(quasar_ind) );
      fprintf_debug(' ... log p(sub DLA | D, z_QSO) : %0.2f\n', ...
          log_posteriors_lls(quasar_ind) );    
    end
        
    % save map: extract MAP values of z_dla and log_nhi
    [~, maxidx] = nanmax(this_sample_log_likelihoods_dla(:, num_dlas), [], 1);

    MAP_inds(quasar_ind, num_dlas, 1:num_dlas)     = [maxidx; this_base_sample_inds(1:(num_dlas - 1), maxidx)];

    % save map: save the MAP values to the array
    MAP_z_dlas(quasar_ind, num_dlas, 1:num_dlas)   = all_z_dlas(:, maxidx);
    MAP_log_nhis(quasar_ind, num_dlas, 1:num_dlas) = all_log_nhis(:, maxidx);
    
    fprintf_debug(' ... log p(D | z_QSO, %i DLAs) : %0.2f\n', ...
                  num_dlas, log_likelihoods_dla(quasar_ind, num_dlas));
    fprintf_debug(' ... log p(%i DLAs | D, z_QSO) : %0.2f\n', ...
                  num_dlas, log_posteriors_dla(quasar_ind, num_dlas));

    if (num_dlas == max_dlas)
      break;
    end

    % if p(D | z_QSO, num_dlas DLA) is NaN, then
    % finish the loop. 
    % It's usually because p(D | z_QSO, no DLA) is very high, so
    % the higher order DLA model likelihoods already uderflowed
    if isnan(log_likelihoods_dla(quasar_ind, num_dlas))
      fprintf('Finish the loop earlier because NaN value in log p(D | z_QSO, %i DLAs) : %0.2f\n', ...
        num_dlas, log_likelihoods_dla(quasar_ind, num_dlas));
      break;
    end

    % avoid nan values in the randsample weights
    nanind = isnan(sample_probabilities);
    W = sample_probabilities;
    W(nanind) = double(0);

    this_base_sample_inds(num_dlas, :) = ...
        uint32(randsample(num_dla_samples, num_dla_samples, true, W)');
  end

  % exclude to save memory
  base_sample_inds(quasar_ind, :, :)           = this_base_sample_inds';
  sample_log_likelihoods_dla(quasar_ind, :, :) = this_sample_log_likelihoods_dla(:, :);

  fprintf(' took %0.3fs.\n', toc);
end

max_log_posteriors = ...
    max([log_posteriors_no_dla, log_posteriors_lls, log_posteriors_dla], [], 2);

model_posteriors = ...
    exp(bsxfun(@minus, ...
               [log_posteriors_no_dla, log_posteriors_lls, log_posteriors_dla], ...
               max_log_posteriors));

model_posteriors = ...
    bsxfun(@times, model_posteriors, 1 ./ sum(model_posteriors, 2));

p_no_dlas = model_posteriors(:, 1);
p_lls     = model_posteriors(:, 2);
p_dlas    = 1 - p_no_dlas - p_lls;
