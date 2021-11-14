function [est] = f_LMMSE(y, X, eta, probs, sigma_noise)
    sigma_m = eta'*probs;
    est = (X*X'*sigma_m + sigma_noise) \ y;
    est = sigma_m * X' * est;
end