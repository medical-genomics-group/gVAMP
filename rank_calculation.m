
%this scripts runs VAMP code on the real height dataset
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
%M = 15000;
Mtotal = 111429;
M = 40000


delta = 2;
%N = ceil(delta*M);
Ntotal = 382390;
N = Ntotal; %M max = 111429
%N = M;
N = 100000

r_vals = [];
start_vals = randi([1,Mtotal - M], 10);
for start = start_vals
    genomat = PlinkRead_binary2(Ntotal, 1:Mtotal, fileprefix);
    genomat = genomat(:, start:start+M-1);

    people_indices = randsample(Ntotal, N);
    genomat_filt = genomat(people_indices, 1:M);
    I = find(genomat_filt == -1);
    genomat_filt(I) = 0;

    X = normalize( double(genomat_filt) );
    X = X / sqrt(N);

    r = rank(X);
    r_vals = [ r_vals, r ]

end
save('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/testing08122021/VAMP/genotype_mat_rank_square.mat','r_vals');
r_vals