format long


%% 1. LOADING .BED VIA PlinkRead_binary2

fileprefix = '/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000_clumped_01';
snps = PlinkRead_bim('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000_clumped_01')
fileprefix = '/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000';
snps = PlinkRead_bim('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000')
N = 44992;
N = 5000; %before it was 4000/ 1000
%genomat = PlinkRead_binary2(N,1:size(snps.snplist,1),fileprefix);
end_ind = min(9000, size(snps.snplist,1)); %before it was 9000 /1500
genomat = PlinkRead_binary2(N,1:end_ind,fileprefix);
I = find(genomat == -1);
genomat(I) = 0;
X = normalize(double(genomat));
M = end_ind;
X = X / ( sqrt( N ) );


rng(1510)
%% 2. GENERATING GAUSSIAN MATRIX
%M = 1000;
%delta = 2;
%N = ceil(delta * M);
%X = normrnd( 0, 1/sqrt( M ), N, M ); %second argument is std dev
%in the paper 'A unifying tutorial on approximate message passing' the assumption is that X_{ij} \sim N(0, 1/N) where X \in R^{N x M}
%X = normrnd( 0, 1 / sqrt( N ), N, M ); %second argument is std dev

eta_signal = [0.01 0.0001]';
probs_signal = [0.8 0.2]';
probs_zero = 0.85;
probs_final = [ probs_zero; (1-probs_zero) * probs_signal ];
%eta_final =  [ 1e-20; eta_signal ];
eta_final =  [ 0; eta_signal ];
b0 = 0;
beta0 = ones(M,1);

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

norm( X * beta_true ) / norm(noise)  

iterNumb = 6;

%pseudoMem AMP
[beta_out_pAMP, sigma_out_pAMP, muk_out_pAMP, ratio_measures_pAMP, l2_err_pred_pAMP, l2_err_signal_pAMP, corrs_pAMP, sigmas_pAMP] = f_infere_pseudoMemAMP(y,X,iterNumb, beta0, b0, N, M, eta_final, probs_final, @fk, @fkd, beta_true, sigma_noise);
f_ErrMes_print(beta_true, beta_out_pAMP(:,end), y, X, muk_out_pAMP(end), sigma_out_pAMP, eta_final)

%basic AMP
[beta_out_bAMP, sigma_out_bAMP, muk_out_bAMP, ratio_measures_bAMP, l2_err_pred_bAMP, l2_err_signal_bAMP, corrs_bAMP, muks_bAMP, sigmas_bAMP] = f_infere_AMP(y,X,iterNumb, beta0, b0, N, M, eta_final, probs_final, @fk, @fkd, beta_true, sigma_noise);
f_ErrMes_print(beta_true, beta_out_bAMP, y, X, muk_out_bAMP, sigma_out_bAMP, eta_final)

%LASSO
[beta_out_Lasso] = f_Lasso(X,y);
beta_out_Lasso = beta_out_Lasso(:, ceil( size(beta_out_Lasso , 2) / 2) );
f_ErrMes_print(beta_true, beta_out_Lasso, y, X)

%LMMSE
[beta_out_LMMSE] = f_LMMSE(y, X, eta_final, probs_final, sigma_noise);
f_ErrMes_print(beta_true, beta_out_LMMSE, y, X)

%Gibbs
burnin_iter = 10;
numb_iter = 20;
beta0 = ones(M,1);
[beta_out_Gibbs] =  f_Gibbs(y, X, beta0, sigma_noise, eta_final, probs_final, burnin_iter, numb_iter);
beta_out_Gibbs_final = mean(beta_out_Gibbs')';
f_ErrMes_print(beta_true, beta_out_Gibbs_final, y, X);


scaled_out = (beta_true - beta_out_pAMP) / (sqrt(sigma_out(end)) / muk_out);
[h,p_val] = kstest( scaled_out )
[h2,p_val2] = kstest( ( scaled_out - mean(scaled_out) ) / std(scaled_out) )

figure(1)
subplot(1,2,1)
histogram( (beta_true - beta_out_pAMP) / (sqrt(sigma_out(end)) / muk_out), 'Normalization','pdf', 'EdgeColor', 'blue');
hold on;
xline(1);
xline(-1);

xline(2, 'b');
xline(-2, 'b');

xline(3, 'r');
xline(-3, 'r');

text(2,0.3,[ 'p val: ', num2str(p_val)])
text(2,0.2,[ 'p val2: ', num2str(p_val2)])
hold off;
ax = gca;
exportgraphics(ax,'hist_pseudo_AMP_lowdelta_M25000_N10000.jpg')

subplot(1,2,2)
qqplot( (beta_true - beta_out_pAMP(:,end)) / (sqrt(sigma_out(end)) / muk_out) )
ax = gca;
exportgraphics(ax,'hist_pseudoMem_beta_out_qqplot.jpg')

corr_pseudoMem = [];
for i = 1:10
    corr_pseudoMem = [ corr_pseudoMem, beta_true' * beta_out_pAMP(:,i) / norm( beta_true ) / norm( beta_out_pAMP(:,i) )];
end
corr_pseudoMem



rng(1210)
nn = 0.005;
noised = beta_true + 0.005 * normrnd( 0, 1, M, 1 ); 
corr(noised, beta_true)
norm(noised - beta_true) / norm(beta_true)
fk_denoise = arrayfun(@(y) fk(y, nn^2, 1, probs_final, eta_final), noised);
corr(fk_denoise, beta_true)
norm(fk_denoise - beta_true) / norm(beta_true)
