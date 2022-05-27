format long;


%% 1. LOADING .BED VIA PlinkRead_binary2

fileprefix = '/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000_clumped_01';
snps = PlinkRead_bim('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000_clumped_01')
%fileprefix = '/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000';
%snps = PlinkRead_bim('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000')
N = 44992;
N = 4000;
%genomat = PlinkRead_binary2(N,1:size(snps.snplist,1),fileprefix);
end_ind = min(9000, size(snps.snplist,1));
genomat = PlinkRead_binary2(N,1:end_ind,fileprefix);
I = find(genomat == -1);
genomat(I) = 0;
X = normalize(double(genomat));
M = end_ind;
X = X / ( sqrt( M ) );

delta = N / M;

beta0 = ones(M,1);

rng(1510)
eta_signal = [0.01 0.0001]';
probs_signal = [0.8 0.2]';
probs_zero = 0.85;
probs_final = [ probs_zero; (1-probs_zero) * probs_signal ];
eta_final =  [ 1e-20; eta_signal ];
b0 = 0;

%generating marker values
sigma_noise = 0.02^2;
noise = normrnd( 0, sqrt(sigma_noise), N, 1 ); %second argument is std dev

t = unifrnd(0,1, M,1);
beta_true1 = normrnd( 0, sqrt(eta_final(1)), M, 1 );
beta_true2 = normrnd( 0, sqrt(eta_final(2)), M, 1 );
beta_true3 = normrnd( 0, sqrt(eta_final(3)), M, 1 );
beta_true = (t < probs_final(1)) .* beta_true1 + (t >= probs_final(1) & t < probs_final(1) + probs_final(2)) .* beta_true2 + (t > probs_final(1) + probs_final(2)) .* beta_true3;

y = X * beta_true + noise;

'signal-to-noise-ratio:'

norm(X * beta_true ) / norm(noise)  


%%% CALCULATING STATE EVOLUTION PARAMETERS USING MCMC METHODS
numb_rep = 2000;
numb_iter_SE = 3;


Sigma0 = 1 / delta * [ eta_final'*probs_final 0; 0 1 ];
Sigmak = Sigma0
for k = 0:(numb_iter_SE-1)
    k

    %CALCULATING T_{k+1}
    %case k == 0
    if k == 0

        %calculating T_1
        gk = mvnrnd(zeros(1, 2), Sigmak, numb_rep);
        %X_sim = normrnd(0, 1/sqrt(M), numb_rep, M);
        
        t = unifrnd(0,1, M,1);
        beta_true1 = normrnd( 0, sqrt(eta_final(1)), M, 1 );
        beta_true2 = normrnd( 0, sqrt(eta_final(2)), M, 1 );
        beta_true3 = normrnd( 0, sqrt(eta_final(3)), M, 1 );
        c_probs = cumsum(probs_final);
        beta_true_sim = (t < c_probs(1)) .* beta_true1 + (t >= c_probs(1) & t < c_probs(2)) .* beta_true2 + (t >  c_probs(2)) .* beta_true3;

        noise_sim = normrnd( 0, sqrt(sigma_noise), numb_rep, 1 );
        y_sim = X(1:numb_rep, :) * beta_true_sim + noise_sim; 

        theta1 = ( y_sim - gk(:,2) );
        %a = mean( (X(1:numb_rep, :)* beta_true_sim) .^ 2);
        %b = mean( (X(1:numb_rep, :)* beta_true_sim) .* theta1 );
        c = mean ( theta1 .^ 2 );
        Tk = [c];
        
        mu = 1;

    else
        
        %calculations for Sigma_k (Sigma_k has dimensions (k+2) x (k+2))
        t = unifrnd(0,1, numb_rep,1);
        beta_true1 = normrnd( 0, sqrt(eta_final(1)), numb_rep, 1 );
        beta_true2 = normrnd( 0, sqrt(eta_final(2)), numb_rep, 1 );
        beta_true3 = normrnd( 0, sqrt(eta_final(3)), numb_rep, 1 );
        c_probs = cumsum(probs_final);
        beta_true_sim = (t < c_probs(1)) .* beta_true1 + (t >= c_probs(1) & t < c_probs(2)) .* beta_true2 + (t >  c_probs(2)) .* beta_true3;

        Gk_tau = mvnrnd(zeros(1, k), Tk, numb_rep);
        fk = zeros(numb_rep, k+2);
        %calculating fk_pseudoMem
        for i = 1:numb_rep
            fk(i,3:end) = fk_pseudoMem(k, Gk_tau(i,:)', inv(Tk), mu, probs_final, eta_final);
        end
        fk(:,1) = beta_true_sim;
        fk(:,2) = ones(numb_rep,1);
        Sigmak = 1.0 / delta * (fk' * fk) / numb_rep


        %calculations for T_k
        Gk_sigma = mvnrnd(zeros(1, k+1), Sigmak(2:end, 2:end), numb_rep);
        gk = zeros(numb_rep,k);

        t = unifrnd(0,1, M,1);
        beta_true1 = normrnd( 0, sqrt(eta_final(1)), M, 1 );
        beta_true2 = normrnd( 0, sqrt(eta_final(2)), M, 1 );
        beta_true3 = normrnd( 0, sqrt(eta_final(3)), M, 1 );
        c_probs = cumsum(probs_final);
        beta_true_sim = (t < c_probs(1)) .* beta_true1 + (t >= c_probs(1) & t < c_probs(2)) .* beta_true2 + (t >  c_probs(2)) .* beta_true3;

        noise_sim = normrnd( 0, sqrt(sigma_noise), numb_rep, 1 );
        y_sim = X(1:numb_rep, :)*beta_true_sim + noise_sim; 

        %calculating gk(Gk_sigma)
        Sigmak_ext = zeros(k+3, k+3); %jer nismo povecali k za 1
        Sigmak_ext(1:(k+2), 1:(k+2)) = Sigmak;
        Sigmak_ext(k+3, k+3) = delta * mean(y_sim .^ 2);  
        Sigmak_ext(1, k+3) = eta_final'*probs_final;
        Sigmak_ext(k+3, 1) = eta_final'*probs_final;
        Sigmak_ext(k+3, 2:(k+2)) = Sigmak_ext(1, 2:(k+2));
        Sigmak_ext(2:(k+2), k+3) = Sigmak_ext(2:(k+2), 1);

        val =  Sigmak_ext(1,2:(k+3)) * inv( Sigmak_ext( 2:(k+3), 2:(k+3) ) ) * [Gk_sigma y_sim]' - Sigmak_ext(1,2:(k+2)) * inv( Sigmak_ext( 2:(k+2), 2:(k+2) ) ) * Gk_sigma';
        val = val';
        val = [ theta1 val ];

        %calculation of Tk
        Tk = ( val' * val ) / numb_rep

        %calculation of muk
        gk_der_y = Sigmak_ext(1,2:(k+3)) * inv( Sigmak_ext( 2:(k+3), 2:(k+3) ) );
        gk_der_y = gk_der_y(end);
        mu = [ mu ; gk_der_y ];
    end
end
