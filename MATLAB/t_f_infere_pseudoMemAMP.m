format long




%% 1. LOADING .BED VIA .MAT FILE


%PlinkRead_bim('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000_clumped_01')
%load("/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000_clumped_0001_BEDmatrix.mat")
load("ukb_imp_v3_UKB_EST_uncorrpeople_N45000_clumped_0001_BEDmatrix.mat")

N = 44992;
X = normalize(sparse(double(Mat)));
corr_mat = (X'*X)/ N;
lambdas = eig(corr_mat);
sec_cumulant = mean(lambdas)
M = size(lambdas,1);
iterNumb = 3;

%calculation of the second free cumulant
sqrt(N)
sqrt(sec_cumulant)

%figure(1)
%histogram(lambdas, floor(M/4));
%xlim([250 2700])

%doing SVD truncating
%lambda_minus = (1 - sqrt(M/N))^2;
%lambda_plus = (1 + sqrt(M/N))^2;
%sum(lambdas <= lambda_minus)
%[U, S, V] = svd(full(X));
%diagS = diag(S);
%diagS(diagS.^2 <= lambda_minus * sqrt(N) ) = 0;
%diagS(diagS.^2 >= lambda_plus * sqrt(N) ) = 0;
%Strunc = diag(diagS);
%Xtrunc = U(:,1:M) * Strunc * V';
%Xtrunc = Xtrunc / mean(diagS.^2);
%Xtrunc = normalize(Xtrunc);


%X = Xtrunc;
%renormalizing genotype matrix
X = X / sqrt(sec_cumulant * N);
%X = X * mean(abs(beta_true));
corr(X);


%% 2. LOADING .BED VIA PlinkRead_binary2

fileprefix = '/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000_clumped_01';
snps = PlinkRead_bim('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000_clumped_01')
fileprefix = '/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000';
snps = PlinkRead_bim('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000')
N = 44992;
genomat = PlinkRead_binary2(N,1:size(snps.snplist,1),fileprefix);
end_ind = min(20000, size(snps.snplist,1));
genomat = PlinkRead_binary2(N,1:end_ind),fileprefix);
I = find(genomat == -1);
genomat(I) = 0;
X = normalize(double(genomat));
M = end_ind;
X = X / sqrt( max(M, N) );
%X = X / 2.4766; %multiplication correction for uk chr 22 genotype matrix


 
%% 3. SIMULATING GENOTYPE MATRIX


%setting up a correlation for genotype matrix
pair_corr_bound = 0.00; %expected correlation between each pair is pair_corr_bound / 2

rng(1) % for reproducibility
Rho = unifrnd(0,pair_corr_bound, M, M);
for i = 1:M
    Rho(i,i) = 1;
end
Rho = (Rho + Rho')/2;
U = copularnd('Gaussian', Rho, N);


%generating genotype matrix X
probs = [0.7 0.2 0.1]';
cumprobs = cumsum(probs);
vals = [0 1 2]';
X = zeros(N,M);
for i = 1:N
    for j = 1:M
        X(i,j) = vals ( min( find( (cumprobs >= U(i,j)) == 1) ) );
        %X(i,j) = normrnd( 0, 0.005 );
    end
end

means = mean(X);
stds = std(X);

for i = 1:N
    for j = 1:M
        X(i,j) = (X(i,j) - means(j)) / stds(j);
    end  
end

%calculation of the second free cumulant
[V,D] = eig(X'*X);
lambdas = diag(D);
sec_cumulant = mean(lambdas)
sqrt(N)
sqrt(sec_cumulant)


X = X / sqrt(sec_cumulant);


%generating Gaussian matrix X
probs = [0.7 0.2 0.1]';
cumprobs = cumsum(probs);
vals = [0 1 2]';
X = zeros(N,M);
for i = 1:N
    for j = 1:M
        X(i,j) = normrnd( 0, 1/sqrt(N) ); %second argument is std dev
    end
end
%oder:
X = normrnd( 0, 1/sqrt(M), N, M ); %second argument is std dev







%eta_signal = [0.01 0.0001]';
eta_signal = [0.01 0.0001]';
probs_signal = [0.5 0.5]';
probs_zero = 0.8;
probs_final = [ probs_zero; (1-probs_zero) * probs_signal ];
eta_final =  [ 1e-20; eta_signal ];
b0 = 0;
beta0 = zeros(M,1);
beta0 = ones(M,1);

%generating marker values
noise = normrnd( 0, 0.1, N, 1 ); %second argument is std dev


sigma_sig = zeros(1, M, size(eta_signal,1));
sigma_sig(1,:,:) = repmat(eta_signal, 1, M)';
gm = gmdistribution( zeros( size(eta_signal,1), M ), sigma_sig, probs_signal' );
beta_true = random(gm)';
beta_true = binornd(1,1-probs_zero, M, 1) .* beta_true;
beta_true_norm = norm(beta_true);

%generating marker values by hand
beta_true1 = normrnd( 0, sqrt(eta_signal(1)), M, 1 );
beta_true2 = normrnd( 0, sqrt(eta_signal(2)), M, 1 );
beta_true = binornd(1,1-probs_signal(1), M, 1);
beta_true = (beta_true == 0) .* beta_true1 + (beta_true == 1) .* beta_true2;
beta_true = binornd(1,1-probs_zero, M, 1) .* beta_true;
beta_true_norm = norm(beta_true);


t = unifrnd(0,1, M,1);
beta_true1 = normrnd( 0, sqrt(eta_final(1)), M, 1 );
beta_true2 = normrnd( 0, sqrt(eta_final(2)), M, 1 );
beta_true3 = normrnd( 0, sqrt(eta_final(3)), M, 1 );
beta_true = (t < probs_final(1)) .* beta_true1 + (t >= probs_final(1) & t < probs_final(1) + probs_final(2)) .* beta_true2 + (t > probs_final(1) + probs_final(2)) .* beta_true3;


beta_true = beta_true / beta_true_norm;
eta_signal = eta_signal / ( beta_true_norm ^ 2);


y = X * beta_true + noise;

beta0 = beta_true + normrnd( 0.1, 1, M, 1 );
beta0 = normrnd( 0.1, 10, M, 1 );
beta0 = ones(M,1);

iterNumb = 5;

[beta_out, sigma_out, muk_out] = f_infere_pseudoMemAMP(y,X,iterNumb, beta0, b0, N, M, [0; eta_signal], probs_final, @fk, @fkd, beta_true);

[beta_out, sigma_out, muk_out] = f_infere_AMP(y,X,iterNumb, beta0, b0, N, M, eta_final, probs_final, @fk, @fkd, beta_true);


'final l2 error:'
norm(y-X*beta_out) / norm(y-X*beta_true)

'final corr:'
beta_true' * beta_out / norm( beta_true ) / norm( beta_out )

'ratio measure:'
sigma_out(end)/(muk_out^2)

'sqrt ( sigma_out / max(eta_final) ): '
sqrt( sigma_out / max(eta_final) )

histogram( (beta_true - beta_out) / (sigma_out(end) / muk_out), 'Normalization','pdf', 'EdgeColor', 'blue');
ax = gca;
hold on;
xline(1);
xline(-1)

xline(2, 'b')
xline(-2, 'b')

xline(3, 'r')
xline(-3, 'r')
hold off;
exportgraphics(ax,'hist_beta_diff_basicAMP_gaussian.jpg')