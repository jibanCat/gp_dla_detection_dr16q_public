% generate_optical_depth_samples_full_int.m : generate samples for (tau_0, beta)
% in a bivariate normal distribution with a diagonal covariance

% the mean values of Kim's effective optical depth
tau_0_mu    = 0.00554;
tau_0_sigma = 0.00064;
beta_mu     =   3.182;
beta_sigma  =   0.074;

% % an univariate normal
% pd_tau_0 = makedist('Normal', 'mu', tau_0_mu, 'sigma', tau_0_sigma);
% pd_beta  = makedist('Normal', 'mu', beta_mu,  'sigma', beta_sigma);

% generate quasirandom samples from p(tau_0)
rng('default');
sequence = scramble(haltonset(5), 'rr2');

tau_0_samples = zeros(1, num_optical_depth_samples);
beta_samples  = zeros(1, num_optical_depth_samples);

% for i = 1:num_optical_depth_samples
%     tau_0_samples(i) = fzero(@(tau_0) (cdf(pd_tau_0, tau_0) - sequence(i, 4)), tau_0_mu);
%     beta_samples(i)  = fzero(@(beta) (cdf(pd_beta, beta) - sequence(i, 5)), beta_mu);
% end

jitter = 1e-4;

for i = 1:num_optical_depth_samples
     tau_0_samples(i) = fzero(@(tau_0) (normcdf(tau_0, tau_0_mu, tau_0_sigma) - sequence(i + 1, 5)), tau_0_mu);
     beta_samples(i)  = fzero(@(beta) (normcdf(beta, beta_mu, beta_sigma) - sequence(i + 1, 4)), beta_mu);
end
 
% % uniform prior for tau_0 only
% min_tau_0 = 0.0001;
% max_tau_0 = 0.0100;

% offset_samples_qso  = sequence(1:num_optical_depth_samples, 1)';

% tau_0_samples = min_tau_0 + (max_tau_0 - min_tau_0) * offset_samples_qso;

% tau_0_samples = normrnd(tau_0_mu, tau_0_sigma, [1, num_optical_depth_samples]);
% beta_samples  = normrnd(beta_mu, beta_sigma, [1, num_optical_depth_samples]);

variables_to_save = {'tau_0_samples', 'tau_0_mu', 'tau_0_sigma', ...
                     'beta_samples',  'beta_mu',  'beta_sigma'};
save(sprintf('%s/tau_0_samples_%d', processed_directory(training_release), ...
     num_optical_depth_samples), ...
     variables_to_save{:}, '-v7.3');
