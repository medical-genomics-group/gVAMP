
M = 15000;
Nrange = 1500:1500:15000;
trial_range = 1:1;
gene_loc = '/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPvsGibbsvsLasso/genomes';
gout_loc = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPvsGibbsvsLasso/Gibbsout";
vout_loc = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPvsGibbsvsLasso/VAMPout";
vis_loc  = '/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPvsGibbsvsLasso/visualization';
addpath('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/AMP_library');
addpath('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/testing08122021/VAMP')

[corrs_r, corrs_g, corrs_v, corrs_lasso, l2_signal_errs_r,  l2_signal_errs_g, l2_signal_errs_v, l2_signal_errs_lasso, l2_pred_errs_r, l2_pred_errs_g, l2_pred_errs_v, l2_pred_errs_lasso, true_pred_err, true_pred_err_t, l2_pred_errs_r_t, l2_pred_errs_g_t, l2_pred_errs_v_t, l2_pred_errs_lasso_t] = deal( zeros( length(trial_range),  length(Nrange)) );
for Nind = 1:length(Nrange)
    Nind
    for trial_ind = 1 %1:1
        trial = trial_range(trial_ind);
        N = Nrange(Nind);
        fileprefix = strcat( gene_loc,'/', [num2str(N), '_', num2str(trial)]);
        genomat = PlinkRead_binary2(N, 1:M, fileprefix);
        X = normalize( double(genomat) );
        y_tr = readtable( strcat(gene_loc, '/', num2str(N), '_',  num2str(trial), '_y.txt') );
        y_tr = table2array( y_tr(:,end) );
        beta_true = load( strcat(gene_loc, '/', num2str(N), '_',  num2str(trial), '_beta_true.txt') );
        gibbs_est = readtable( strcat(gout_loc, '/', num2str(N), '_',  num2str(trial), '_Gibbs_est.csv') );
        gibbs_est = gibbs_est.Var2;
        vamp_est = load(strcat(vout_loc, '/betas_', num2str(N), '_', num2str(trial),'.mat'));
        vamp_est = vamp_est.x_hat;
        riamp_est = load(strcat(vout_loc, '/RIAMP_betas_', num2str(N), '_', num2str(trial),'.mat'));
        riamp_est = riamp_est.x_hat;
        vars = load( strcat(gene_loc, '/', [num2str(N), '_', num2str(trial)], '_distr_eta.txt') );
        %probs = load( strcat(gene_loc, '/', [num2str(N), '_', num2str(trial)], '_distr_probs.txt') );
        vars = vars(:);
        scale = 4.81e-5 / vars(end); 
        
        %test set
        SNR = 1;
        distr = struct('eta', [0 1.131e-8 4.81e-5]', 'probs', [0.711 0.2644 0.0246]');
        gamw = SNR / (M * distr.eta' * distr.probs);
        corr0 =0;
        fileprefix_te = strcat( gene_loc,'/', [num2str(N), '_', num2str(trial),'_test']);
        genomat_te = PlinkRead_binary2(N, 1:M, fileprefix_te);
        X_te = normalize( double(genomat_te) );
        [~, ~, ~, beta0] = VAMP_LR_gen(distr, corr0, gamw, M, N);
        y_te =  X_te * beta_true + normrnd(0, sqrt(1/gamw), N, 1);
        scale_te = var(y_te);
        y_te = y_te / sqrt(scale_te);

        corrs_g(trial_ind, Nind) = beta_true(:)' * gibbs_est(:) / norm(beta_true) / norm(gibbs_est);
        l2_signal_errs_g(trial_ind, Nind) = norm( beta_true - gibbs_est(:) * sqrt(scale) ) / norm(beta_true);
        l2_pred_errs_g(trial_ind, Nind) = norm( y_tr - X * gibbs_est(:) ) / norm(y_tr);
        l2_pred_errs_g_t(trial_ind, Nind) = norm( y_te - X_te * gibbs_est(:) ) / norm(y_te);
        
        corrs_v(trial_ind, Nind) = beta_true(:)' * vamp_est(:) / norm(beta_true) / norm(vamp_est);
        l2_signal_errs_v(trial_ind, Nind) = norm( beta_true - vamp_est(:) * sqrt(scale) ) / norm(beta_true);
        l2_pred_errs_v(trial_ind, Nind) = norm( y_tr - X * vamp_est(:) ) / norm(y_tr);
        l2_pred_errs_v_t(trial_ind, Nind) = norm( y_te - X_te * vamp_est(:) ) / norm(y_te);
        
        corrs_r(trial_ind, Nind) = beta_true(:)' * riamp_est(:) / norm(beta_true) / norm(riamp_est);
        l2_signal_errs_r(trial_ind, Nind) = norm( beta_true - riamp_est(:) * sqrt(scale) ) / norm(beta_true);
        l2_pred_errs_r(trial_ind, Nind) = norm( y_tr - X * riamp_est(:) ) / norm(y_tr);
        l2_pred_errs_r_t(trial_ind, Nind) = norm( y_te - X_te * riamp_est(:) ) / norm(y_te);

        [B,fitinfo] = lasso(X,y_tr,'CV', 5);
        lasso_est = B(:,fitinfo.IndexMinMSE);
        corrs_lasso(trial_ind, Nind) = beta_true(:)' * lasso_est(:) / norm(beta_true) / norm(lasso_est);
        l2_signal_errs_lasso(trial_ind, Nind) = norm( beta_true - lasso_est(:) * sqrt(scale) ) / norm(beta_true);
        l2_pred_errs_lasso(trial_ind, Nind) = norm( y_tr - X * lasso_est(:) ) / norm(y_tr);
        l2_pred_errs_lasso_t(trial_ind, Nind) = norm( y_te - X_te * lasso_est(:) ) / norm(y_te);
        
        true_pred_err(trial_ind, Nind) = norm( y_tr - X * beta_true(:) / sqrt(scale) ) / norm(y_tr);
        true_pred_err_t(trial_ind, Nind) = norm( y_te - X_te * beta_true(:) / sqrt(scale) ) / norm(y_te);
    end
end
corrs_rp = mean(corrs_r, 1);
corrs_gp = mean(corrs_g, 1);
corrs_lassop = mean(corrs_lasso, 1);
l2_signal_errs_rp = mean(l2_signal_errs_r, 1);
l2_signal_errs_gp = mean(l2_signal_errs_g, 1);
l2_signal_errs_lassop = mean(l2_signal_errs_lasso, 1);
corrs_vp = mean(corrs_v, 1);
l2_pred_errs_rp = mean(l2_pred_errs_r, 1);
l2_pred_errs_gp = mean(l2_pred_errs_g, 1);
l2_signal_errs_vp = mean(l2_signal_errs_v, 1);
l2_pred_errs_vp = mean(l2_pred_errs_v, 1);
l2_pred_errs_lassop = mean(l2_pred_errs_lasso, 1);
true_pred_errp = mean(true_pred_err, 1);
true_pred_err_tp = mean(true_pred_err_t, 1);
l2_pred_errs_r_tp = mean(l2_pred_errs_r_t, 1);
l2_pred_errs_g_tp = mean(l2_pred_errs_g_t, 1);
l2_pred_errs_v_tp = mean(l2_pred_errs_v_t, 1);
l2_pred_errs_lasso_tp = mean(l2_pred_errs_lasso_t, 1);
deltas = Nrange / M;

figure(1)
orange = '#EDB120';
lasso_col = sscanf(orange(2:end),'%2x%2x%2x',[1 3])/255;
clf
title(['COMPARISON GIBBS-VAMP: ', 'M=', num2str(M)]);
subplot(2,2,1)
plot(deltas, corrs_gp, 'ro-');
hold on;
plot(deltas, corrs_vp, 'bo-');
hold on;
plot(deltas, corrs_rp, 'mo-');
hold on;
lasso_plot = plot(deltas, corrs_lassop, 'o-');
lasso_plot.Color = lasso_col;
set(get(gca, 'YLabel'), 'String', 'corr');
xlabel('$\delta$', 'Interpreter', 'Latex');
legend('gmrm', 'VAMP', 'RIAMP', 'LASSO', 'Location', 'southeast');

subplot(2,2,2)
plot(deltas, l2_signal_errs_gp, 'ro-');
hold on;
plot(deltas, l2_signal_errs_vp, 'bo-');
hold on;
plot(deltas, l2_signal_errs_rp, 'mo-');
hold on;
lasso_plot = plot(deltas, l2_signal_errs_lassop, 'o-');
lasso_plot.Color = lasso_col;
ylabel('$\frac{||\beta - \hat{\beta} ||}{||\beta||}$', 'Interpreter', 'Latex');
xlabel('$\delta$', 'Interpreter', 'Latex');

subplot(2,2,3)
plot(deltas, l2_pred_errs_gp, 'ro-');
hold on;
plot(deltas, l2_pred_errs_vp, 'bo-');
hold on;
plot(deltas, l2_pred_errs_rp, 'mo-');
hold on;
lasso_plot = plot(deltas, l2_pred_errs_lassop, 'o-');
lasso_plot.Color = lasso_col;
plot(deltas, true_pred_err, 'kx');
ylabel('$\frac{||y - X\hat{\beta} ||}{||y||}$', 'Interpreter', 'Latex');
xlabel('$\delta$', 'Interpreter', 'Latex');

subplot(2,2,4)
plot(deltas, l2_pred_errs_g_tp, 'ro-');
hold on;
plot(deltas, l2_pred_errs_v_tp, 'bo-');
hold on;
plot(deltas, l2_pred_errs_r_tp, 'mo-');
hold on;
%plot(deltas, l2_pred_errs_lasso_tp, 'Color', lasso_col, 'o-');
lasso_plot = plot(deltas, l2_pred_errs_lasso_tp, 'o-');
lasso_plot.Color = lasso_col;
plot(deltas, true_pred_err_tp, 'kx');
%ylabel('$\frac{||y_{test}} - X_{test}\hat{\beta} ||}{||y_{test}||}$', 'Interpreter', 'Latex');
ylabel('$\frac{||yt - Xt\hat{\beta} ||}{||yt||}$', 'Interpreter', 'Latex');
xlabel('$\delta$', 'Interpreter', 'Latex');

%saveas(gcf,  [vis_loc, '/Gibbs_vs_VAMP_2_impr.jpg'])
exportgraphics(gcf,[vis_loc, '/RIAMP_vs_VAMP_vs_GMRM_vs_Lasso.jpg'],'Resolution',1200)


%[B,fitinfo] = lasso(X,y_tr,'CV', 5);
%B = lasso(X,y);
%UU = y - X* B;
%[val, I] = min(vecnorm(UU, 2))
%corr_Lasso = B(:,I)' * beta_true / norm(beta_true) / norm(B(:,I));