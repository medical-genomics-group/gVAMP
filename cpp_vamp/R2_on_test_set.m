clear all;

% libraries for loading genomic data
lib_loc='/nfs/scistore13/robingrp/human_data/adepope_preprocessing/AMP_library';
addpath( lib_loc )  

nranks = 5

R2s = [];

for i1 = 1:10
        
    sig_est = [];

    for i0 = 0:(nranks-1)

        str =  strcat("/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/sig_estimates/x1_hat_height_gamwL_CG_25_PL_rho_09_13_9_22_it_",  num2str(i1), "_rank_", num2str(i0));

        val = table2array( readtable(str) );

        sig_est = [sig_est; val];

    end

    size(sig_est);

    % reading genomic data

    bed_name = '/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/ukb22828_UKB_EST_v3_ldp08_test_HT';

    N_test = 15000;

    N = 438361;

    Mt = 326165;

    genomat = PlinkRead_binary2(N_test, 1:Mt, bed_name);

    I = find(genomat == -1);

    genomat(I) = 0;

    X = normalize( double(genomat) );

    % reading test phenotypes

    phen_tmp = readtable("/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/ukb_test_HT.txt");

    phen =   table2array( phen_tmp(:,3) );

    pred = X * sig_est / sqrt(N);

    R2 = 1 - sum( (phen - pred).^2  ) / sum( (phen - mean(phen)).^2 )

    R2s = [R2s, R2];
end

R2s

%% calculating R2 on the test set based on Gibbs estimates

est_gibbs_tmp = table2array( readtable("/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/gmrm_height/ukb_ht_noNA_Gibbs_est.csv") );
est_gibbs = est_gibbs_tmp(:,2);

est_gibbs_VAMPprior_tmp = table2array( readtable("/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/gmrm_height_VAMPprior/ukb_ht_noNA_Gibbs_est.csv") );
est_gibbs_VAMPprior = est_gibbs_VAMPprior_tmp(:,2);

bed_name = '/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/ukb22828_UKB_EST_v3_ldp08_test_HT';

N_test = 15000;

N = 438361;

Mt = 326165;

genomat = PlinkRead_binary2(N_test, 1:Mt, bed_name);

I = find(genomat == -1);

genomat(I) = 0;

X = normalize( double(genomat) );

% reading test phenotypes

phen_tmp = readtable("/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/ukb_test_HT.txt");

phen =   table2array( phen_tmp(:,3) );

pred = X * est_gibbs;

pred_VAMPprior = X * est_gibbs_VAMPprior;

R2_gibbs = 1 - sum( (phen - pred).^2  ) / sum( (phen - mean(phen)).^2 )

R2_gibbs_VAMPprior = 1 - sum( (phen - pred_VAMPprior).^2  ) / sum( (phen - mean(phen)).^2 )

