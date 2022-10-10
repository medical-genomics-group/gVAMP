
function [y, X, beta_true, beta0] = VAMP_LR_gen(distr, corr0, gamw, M, N)

% INPUT ARGUMENTS:
%
%   [distr] - struct objet containing two column subvectors: 
%           eta - mixture variances
%           probs - mixture probabilities
%   [corr0] - estimation of corr(beta_true, beta0)
%   [gamw] - noise precision
%   [N] - number of individuals
%   [M] - number of markers

%
% OUTPUT ARGUMENTS:
%
%   [y] - phenotype vector
%   [X] - generated genotype matrix (normalized s.t. mean and sd of each columns are
%   0 and 1 / sqrt(N)
%   [beta_true] - true value of marker values
%   [beta0] - initial estimate of marker effects
%
%DEPENDENCIES:
%
%   None.
%

    if corr0 == 0
        gam0 = 1e-6;
    else
        gam0 = corr0^2 / ( 1 - corr0^2 ) / ( distr.eta' * distr.probs );
    end
    
    X = normrnd(0, 1 / sqrt(N), N, M);
    
    noise = normrnd(0, 1 / sqrt(gamw), N, 1);
    
    beta_true = zeros(M,1);
    
    t = unifrnd(0,1, M,1);
    K = length(distr.eta);
    c_probs = cumsum(distr.probs);

    where = ( (c_probs' >= t) == 1 );
    [~,inds] = max ( where, [], 2 );
    for i = 1:K
        indi_i = (inds == i);
        beta_true(indi_i) = normrnd( 0, sqrt( distr.eta(i) ), sum(indi_i), 1 );
    end

    y = X * beta_true + noise;

    beta0 = beta_true + normrnd(0,1,M,1) / sqrt( gam0 );
    
end