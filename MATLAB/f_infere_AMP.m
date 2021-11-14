function [beta_est, sigma, muk, ratio_measures, l2_err_pred, l2_err_signal, corrs, muks, sigmas] = f_infere_AMP(y,X,iterNumb, beta0, b0, N, M, eta, probs, fk, fkd, beta_true, sigma_noise)
    
    r_prev = zeros(N,1);
    r = r_prev;
    beta = beta0;
    beta_est = beta0;
    b = b0;

    ratio_measures = [];
    l2_err_pred = [];
    l2_err_signal = [];
    corrs = [];
    muks = [];
    sigmas = [];
    
    delta = N / M;
    [mus_SE, sigmas_SE] = f_basicAMP_SE(iterNumb-1, delta, M, eta, probs, sigma_noise, beta_true, beta0)

    for it = 0:(iterNumb - 1)       
        
        r = y - X * beta + b * r_prev;
        
        %estimating muk and sigma 
        
        sigma = sum(r.^2) / N;

        muk = sqrt( abs( sum(beta_est.^2) / M - sigma ) / ( eta'*probs ) );

        muk = 1;

        %sigma = sigmas_SE(it + 1);
        
        beta_est = X' * r + beta;

        %muk = mus_SE(it+1);

        %{
        if sum(beta_est.^2) / M - sigma < 0
            it
            sum(beta_est.^2) / M
            sigma
            abs( sum(beta_est.^2) / M - sigma ) / ( eta'*probs )
            error('problem occured while estimating muk!');
            %'problem occured while estimating muk!'
            %muk = 1;
        end
        %}
        
        
        if it == -1

            muk;
            mean(beta_est(:,end) - muk * beta_true)
            std( (beta_est(:,end) - muk * beta_true) )
            [h,p,ksstat,cv] = kstest( (beta_est(:,end) - muk * beta_true) / std( (beta_est(:,end) - muk * beta_true) ) )
            histogram(beta_est(:,end) - muk * beta_true);
            ax = gca;
            hold on;
            sqrt(sigma(end))
            xline(3* sqrt(sigma(end)))
            xline(- 3 * sqrt(sigma(end)))
            xline(1 * sqrt(sigma(end)))
            xline(- 1 * sqrt(sigma(end)))
            hold off;
            exportgraphics(ax, 'beta_est_hist_it0_basicAMP_gaussian.jpg');
            break;
            
        end


        %fk
        
        beta_est_fk = beta_est;
        
        for j = 1:M
            
            beta_est_fk(j) = fk(beta_est(j), sigma, muk, probs, eta);
            
        end
        
        beta = beta_est_fk;
        
        %fkd
        
        beta_est_fkd = beta_est;
        
        for j = 1:M
            
            beta_est_fkd(j) = fkd(beta_est(j), sigma, muk, probs, eta);
            
        end
        
        b = sum( beta_est_fkd ) / N;
        
        r_prev = r;


        i;
        muks = [muks, muk ];
        sigmas = [ sigmas, sigma ];
        l2_err_signal = [ l2_err_signal , norm(beta_true - beta_est/muk) / norm(beta_true) ];
        'l2 prediction error:';
        l2_err_pred = [ l2_err_pred , norm(y-X*beta_est/muk) / norm(y-X*beta_true) ];
        'correlation with the true signal';
        corrs = [ corrs, beta_true' * beta_est / norm( beta_true ) / norm( beta_est ) ];
        'ratio measure:';
        ratio_measures = [ ratio_measures, sigma(end)/(muk^2) ];
        'sqrt ( sigma_out / max(eta_final) ): ';
        sqrt( sigma / max(eta) );

        
    end
    
    beta_est = beta_est / muk;
    
end