% generate_dla_samples: generates DLA parameter samples from training
% catalog

% [train_dr12q] also load the processed file
training_release = 'dr12q';
processed = load(sprintf('%s/processed_qsos_multi_lyseries_a03_lyb_zwarn_occams_dr12q', ...
       processed_directory(training_release)));

% load training catalog
catalog = load(sprintf('%s/catalog', processed_directory(training_release)));

% generate quasirandom samples from p(normalized offset, log₁₀(N_HI))
rng('default');
sequence = scramble(haltonset(2), 'rr2');

% the first dimension can be used directly for the uniform prior over
% offsets
offset_samples  = sequence(1:num_dla_samples, 1)';

% we must transform the second dimension to have the correct marginal
% distribution for our chosen prior over column density, which is a
% mixture of a uniform distribution on log₁₀ N_HI and a distribution
% we fit to observed data

% uniform component of column density prior
u = makedist('uniform', ...
             'lower', uniform_min_log_nhi, ...
             'upper', uniform_max_log_nhi);

% extract observed log₁₀ N_HI samples from catalog
ind = (processed.p_dlas > 0.99);
fprintf('Training for logNHI prior %d', sum(ind));

% [train_dr12q] only get the first DLA out
log_nhis = processed.MAP_log_nhis(ind, 1, 1);

% make a quadratic fit to the estimated log p(log₁₀ N_HI) over the
% specified range
x = linspace(fit_min_log_nhi, fit_max_log_nhi, 1e3);
kde_pdf = ksdensity(log_nhis, x);
f = polyfit(x, log(kde_pdf), 2);

% convert this to a PDF and normalize
unnormalized_pdf = @(nhi) (exp(polyval(f, nhi)));
Z = integral(unnormalized_pdf, fit_min_log_nhi, 25.0);

% create the PDF of the mixture between the unifrom distribution and
% the distribution fit to the data
normalized_pdf = @(nhi) ...
          alpha  * (unnormalized_pdf(nhi) / Z) + ...
     (1 - alpha) * (pdf(u, nhi));

cdf = @(nhi) (integral(normalized_pdf, fit_min_log_nhi, nhi));

% use inverse transform sampling to convert the quasirandom samples on
% [0, 1] to appropriate values
log_nhi_samples = zeros(1, num_dla_samples);
for i = 1:num_dla_samples
  log_nhi_samples(i) = ...
      fzero(@(nhi) (cdf(nhi) - sequence(i, 2)), 20.5);
end

% precompute N_HI samples for convenience
nhi_samples = 10.^log_nhi_samples;

variables_to_save = {'uniform_min_log_nhi', 'uniform_max_log_nhi', ...
                     'fit_min_log_nhi', 'fit_max_log_nhi', 'alpha', ...
                     'offset_samples', 'log_nhi_samples', 'nhi_samples', 'f'};
save(sprintf('%s/dla_samples_dr12q', processed_directory(training_release)), ...
     variables_to_save{:}, '-v7.3');
