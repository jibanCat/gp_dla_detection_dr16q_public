% generate_optical_depth_samples.m : generate samples for (tau_0, beta)
% in a bivariate normal distribution with a diagonal covariance

num_optical_depth_samples = 10000;

% the mean values of Kim's effective optical depth
tau_0_mu    = 0.0023;
tau_0_sigma = 0.0007;
beta_mu     = 3.65;
beta_sigma  = 0.21;

% an univariate normal
pd_tau_0 = makedist('Normal', 'mu', tau_0_mu, 'sigma', tau_0_sigma);
pd_beta  = makedist('Normal', 'mu', beta_mu,  'sigma', beta_sigma);

% generate quasirandom samples from p(tau_0, beta)
rng('default');
sequence = scramble(haltonset(2), 'rr2');

tau_0_samples = zeros(1, num_optical_depth_samples);
beta_samples  = zeros(1, num_optical_depth_samples);

for i = 1:num_optical_depth_samples
    tau_0_samples(i) = fzero(@(tau_0) (cdf(pd_tau_0, tau_0) - sequence(i, 1)), tau_0_mu);
    beta_samples(i)  = fzero(@(beta) (cdf(pd_beta, beta) - sequence(i, 2)), beta_mu);
end
