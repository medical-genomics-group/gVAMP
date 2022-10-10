
%this scripts runs VAMP code on the real height dataset
rng(1510)
format long;

file_loc='/nfs/scistore13/robingrp/human_data/adepope_preprocessing/pheno_height';
lib_loc='/nfs/scistore13/robingrp/human_data/adepope_preprocessing/AMP_library';
addpath( lib_loc )   


%loading and preparing data
bed_name= 'ukb22828_UKB_EST_v3_ldp08_unrel_extrSNPs_10000top_5000bottom' 
%bed_name= strcat( 'ukb_imp_v3_UKB_EST_uncorrpeople' )
fileprefix = strcat( file_loc,'/', bed_name );
snps = PlinkRead_bim(fileprefix);
M = size(snps.snplist,1); 
M = 15000;

delta = 0.85;
N = ceil(delta*M);
Ntotal = 382390;
genomat = PlinkRead_binary2(Ntotal, 1:M, fileprefix);

people_indices = randsample(Ntotal, N);
genomat_filt = genomat(people_indices_tr, 1:M);
I = find(genomat_filt == -1);
genomat_filt(I) = 0;

X = normalize( double(genomat_filt) );


%SCENARIO A.) seeing how well horseshoe performs when signal comes from gaussian mixture
%simulating under the model  
distr = struct('eta', N * [0 1.131e-8 4.81e-5]', 'probs', [0.711 0.2644 0.0246]');
corr0 = sqrt(0.0);

%setting gamw such that SNR = 1
SNR = 1;
gamw = SNR * (M * distr.eta' * distr.probs);

[~, ~, beta_true, beta0] = VAMP_LR_gen(distr, corr0, gamw, M, N);
y =  X * beta_true + normrnd(0, sqrt(1/gamw), N, 1);


n_iter = 5;
[xhat_all, corrs, sig2_all] = RIAMP_fast(n_iter, distr, y, X, beta_true);
corrs

%VAMP:
iterNumb = 10;
[U,S,V] = svd(X);
distr_init = distr;
%beta0 = beta_true + normrnd(0, sqrt(0.1), M, 1);
[x_hat, gams, l2_signal_err, corrs, real_gams, l2_pred_err, R2, betas] = EM_VAMP_LR_fast(distr, corr0, gamw, y, X, iterNumb, beta0, beta_true, U, S, V);