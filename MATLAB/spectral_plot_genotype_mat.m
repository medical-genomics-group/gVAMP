
%%%%%%%%%% bad matrix example
rng(1510)
format long;

file_loc='/nfs/scistore13/robingrp/human_data/adepope_preprocessing/pheno_height';
lib_loc='/nfs/scistore13/robingrp/human_data/adepope_preprocessing/AMP_library';
addpath( lib_loc )   


%loading and preparing data
bed_name= 'ukb22828_UKB_EST_v3_ldp08_unrel_extrSNPs_10000top_5000bottom' 
bed_name= strcat( 'ukb_imp_v3_UKB_EST_uncorrpeople' )
fileprefix = strcat( file_loc,'/', bed_name );
snps = PlinkRead_bim(fileprefix);
M = size(snps.snplist,1); 
M = 15000;

delta = 0.85;
N = ceil(delta*M);
Ntotal = 382390;
%Ntotal = 366342;
genomat = PlinkRead_binary2(Ntotal, 1:2*M, fileprefix);
genomat = genomat(:, M+1:2*M);

people_indices = randsample(Ntotal, N);
genomat_filt = genomat(people_indices, 1:M);
I = find(genomat_filt == -1);
genomat_filt(I) = 0;

X = normalize( double(genomat_filt) );
X = X / sqrt(N);

lambda = eig(X*X') / M * N;
lambdas = sort(lambda);

figure(3)
clf;
delta = N / M;
lp = (1 + sqrt(delta))^2;
lm = (1 - sqrt(delta))^2;
yyaxis left
xlabel('x-axis zoomed-in', 'FontSize',14)
edges = [0:0.1:lambda_up lambda_up:0.35:max(lambda) max(lambda) + 10];
histogram(lambda, edges, 'normalization', 'pdf');
ylabel('pdf (estimate)')
hold on;
%plot([1 1]*lm, ylim, '-r')
xline(lm)
hold on;
fplot(@(x) (0.5 / pi * sqrt( (lp - x) .* (x - lm) ) / delta ./ x ), [lm, lp]);
%plot([1 1]*lp, ylim, '-r')
xline(lp)
yyaxis right
ylabel('absolute count of eigenvalues')
lambda_up = lambda(lambda > lp);
lambda_down = lambda(lambda < lm);
%histogram(lambda_up, 'Normalization', 'count')
histogram(lambda_up, edges, 'Normalization', 'count')
%set(gca,'xscale','log');
xlim([0, lp+5])
%xlim([0, max(lambda)+10])
%set(get(gca, 'Title'), 'String', ['M=', num2str(M), ', N=', num2str(N), ', eff kappa = ',  num2str( max(lambda) / min(lambda(lambda >0)) ) ]);
set(get(gca, 'Title'), 'String', ['eff kappa = ',  num2str( sqrt( max(lambda) / lambdas(2) ) ) ]);
%saveas(gcf, 'comp_X_MPdist_uncorr_people_sharp.jpg')
exportgraphics(gcf,'/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/testing08122021/spectral_bad_zoomed.jpg','Resolution',1200)



figure(4)
%lambda = eig(X*X') / M * N;
clf;
delta = N / M;
lp = (1 + sqrt(delta))^2;
lm = (1 - sqrt(delta))^2;
yyaxis left
xlabel('x-axis normal scale', 'FontSize', 14)
edges = [0:0.1:lambda_up lambda_up:5:max(lambda) max(lambda)];
histogram(lambda, edges, 'normalization', 'pdf');
hold on;
ylabel('pdf (estimate)')
%plot([1 1]*lm, ylim, '-r')
xline(lm)
hold on;
fplot(@(x) (0.5 / pi * sqrt( (lp - x) .* (x - lm) ) / delta ./ x ), [lm, lp]);
%plot([1 1]*lp, ylim, '-r')
xline(lp)
yyaxis right
ylabel('absolute count of eigenvalues')
lambda_up = lambda(lambda > lp);
lambda_down = lambda(lambda < lm);
%histogram(lambda_up, 'Normalization', 'count')
histogram(lambda_up, edges, 'Normalization', 'count')
%set(gca,'xscale','log');
%xlim([0, lp+5])
%xlim([0, max(lambda)+10])
%set(get(gca, 'Title'), 'String', ['M=', num2str(M), ', N=', num2str(N), ', eff kappa = ',  num2str( max(lambda) / min(lambda(lambda >0)) ) ]);
set(get(gca, 'Title'), 'String', ['eff kappa = ',  num2str( sqrt( max(lambda) / lambdas(2) ) ) ]);
%saveas(gcf, 'comp_X_MPdist_uncorr_people_sharp.jpg')
exportgraphics(gcf,'/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/testing08122021/spectral_bad_normal.jpg','Resolution',1200)





%%%%%%%%%%%%%%%% good matrix example
%loading and preparing data
bed_name= 'ukb22828_UKB_EST_v3_ldp08_unrel_extrSNPs_10000top_5000bottom' 
fileprefix = strcat( file_loc,'/', bed_name );
snps = PlinkRead_bim(fileprefix);
M = size(snps.snplist,1); 
M = 15000;

delta = 0.85;
N = ceil(delta*M);
Ntotal = 366342;
genomat = PlinkRead_binary2(Ntotal, 1:M, fileprefix);

people_indices = randsample(Ntotal, N);
genomat_filt = genomat(people_indices, 1:M);
I = find(genomat_filt == -1);
genomat_filt(I) = 0;

X = normalize( double(genomat_filt) );
X = X / sqrt(N);

lambda = eig(X*X') / M * N;
lambdas = sort(lambda);

figure(5)
clf;
delta = N / M;
lp = (1 + sqrt(delta))^2;
lm = (1 - sqrt(delta))^2;
yyaxis left
xlabel('x-axis log scale', 'FontSize',14)
edges = [0 exp(-10:log(lm)) max(lambda_down) (max(lambda_down) + lm)/2 lm 0.1:0.1:lambda_up lambda_up:0.5:max(lambda) max(lambda) + 10];
histogram(lambda, edges, 'normalization', 'pdf');
ylabel('pdf (estimate)')
hold on;
%plot([1 1]*lm, ylim, '-r')
xline(lm)
hold on;
fplot(@(x) (0.5 / pi * sqrt( (lp - x) .* (x - lm) ) / delta ./ x ), [lm, lp]);
%plot([1 1]*lp, ylim, '-r')
xline(lp)
yyaxis right
ylabel('absolute count of eigenvalues')
lambda_up = lambda(lambda > lp);
lambda_down = lambda(lambda < lm);
%histogram(lambda_up, 'Normalization', 'count')
histogram(lambda_up, edges, 'Normalization', 'count')
hold on;
histogram(lambda_down, edges, 'Normalization', 'count')
set(gca,'xscale','log');
%xlim([0, lp+5])
%xlim([0, max(lambda)+10])
%set(get(gca, 'Title'), 'String', ['M=', num2str(M), ', N=', num2str(N), ', eff kappa = ',  num2str( max(lambda) / min(lambda(lambda >0)) ) ]);
set(get(gca, 'Title'), 'String', ['eff kappa = ',  num2str( sqrt( max(lambda) / min(lambdas(2)) ) ) ]);
%saveas(gcf, 'comp_X_MPdist_uncorr_people_sharp.jpg')
exportgraphics(gcf,'/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/testing08122021/spectral_good_log_scale.jpg','Resolution',1200)



figure(6)
%lambda = eig(X*X') / M * N;
clf;
delta = N / M;
lp = (1 + sqrt(delta))^2;
lm = (1 - sqrt(delta))^2;
yyaxis left
xlabel('x-axis normal scale', 'FontSize', 14)
edges = [0:0.1:lambda_up lambda_up:0.25:max(lambda) max(lambda)];
edges = [0 exp(-10:log(lm)) max(lambda_down) (max(lambda_down) + lm)/2 lm 0.1:0.1:lambda_up lambda_up:0.5:max(lambda) max(lambda) ];
histogram(lambda, edges, 'normalization', 'pdf');
hold on;
ylabel('pdf (estimate)')
%plot([1 1]*lm, ylim, '-r')
xline(lm)
hold on;
fplot(@(x) (0.5 / pi * sqrt( (lp - x) .* (x - lm) ) / delta ./ x ), [lm, lp]);
%plot([1 1]*lp, ylim, '-r')
xline(lp)
yyaxis right
ylabel('absolute count of eigenvalues')
lambda_up = lambda(lambda > lp);
lambda_down = lambda(lambda < lm);
%histogram(lambda_up, 'Normalization', 'count')
histogram(lambda_up, edges, 'Normalization', 'count')
hold on;
histogram(lambda_down, edges,  'Normalization', 'count')
%set(gca,'xscale','log');
%xlim([0, lp+5])
%xlim([0, max(lambda)+10])
%set(get(gca, 'Title'), 'String', ['M=', num2str(M), ', N=', num2str(N), ', eff kappa = ',  num2str( max(lambda) / min(lambda(lambda >0)) ) ]);
set(get(gca, 'Title'), 'String', ['eff kappa = ',  num2str( sqrt( max(lambda) / lambdas(2) ) ) ]);
%saveas(gcf, 'comp_X_MPdist_uncorr_people_sharp.jpg')
exportgraphics(gcf,'/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/testing08122021/spectral_good_normal.jpg','Resolution',1200)
