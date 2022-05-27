function [beta_est, sigma, mu, ratio_measures, l2_err_pred, l2_err_signal, corrs, sigmas] = f_infere_pseudoMemAMP(y,X,iterNumb, beta0, b0, N, M, eta, probs, fk, fkd, beta_true, sigma_noise)
    
    btp1 = b0;
    sigma = [];

    r_prev = zeros(N,1);
    r = zeros(N,1);

    beta = [beta0]; %this will be a matrix of dimension (M x it) at the step it
    beta_est = [beta0]; %final estimate of the signal

    theta = []; %this will be a matrix of dimension (N x it) at the step it
    r_all = []; %this will be a matrix of dimension (N x it) at the step it
    mu = []; %this will be a vector of dimension (it x 1) at the step it

    ratio_measures = [];
    l2_err_pred = [];
    l2_err_signal = [];
    corrs = [];
    sigmas = [];

    numb_iter_SE = iterNumb;
    Gauss_ind = 1;

    %[mu_SE, sigmas_SE] = f_pseudoMemAMP_SE(Gauss_ind, X, N, M, eta, probs, sigma_noise, numb_iter_SE)

    for it = 1:(iterNumb)

       it
       
        theta = [theta, X * beta(:, end) - btp1 * r_prev];

        if it >= 2
            
            %calculating Gamma_t
            
            Gamma_t = zeros(it+1,it+1);

            for i = 2:(it)

                for j = 2:(it)

                    Gamma_t(i,j) = mean( beta(:,i) .* beta(:,j) );

                end

            end

            for j = 2: (it)

                Gamma_t(1,j) = mean( beta(:,j) .^ 2 );
                Gamma_t(j,1) = Gamma_t(1,j);
                Gamma_t(end,j) = Gamma_t(1,j);
                Gamma_t(j,end) = Gamma_t(1,j);

            end

            Gamma_t(1,1) = eta' * probs;
            Gamma_t(end, 1) = eta' * probs;
            Gamma_t(1,end) = eta' * probs;        

            Gamma_t(end, end) = mean(y .^ 2) * N / M;

            Gamma_t = Gamma_t * M / N


            
            %sigma_Z = Gamma_t(end-1, end-1) - Gamma_t(1, end-1)^2 / Gamma_t(1,1)

            %mu_Z = Gamma_t(1, end-1) / Gamma_t(1,1)

            covv = cov ( X * beta_true, theta(:,end) );

            covv_theta_y = cov ( X * beta_true, y );
    
            %covv(1,2) / var( X * beta_true )

            

        
            %calculation the inverse od certain submatrices of Gamma_t

            %for the 1. term

            tmp_Gamma_t = Gamma_t( 2:end, 2:end );
            
            tmp_Gamma_t = flip(tmp_Gamma_t, 2); %we are fliping columns od Gamma_t

            [Q,R] = qr(tmp_Gamma_t);

            tol=1e-8;

            diagR = diag(R);

            r_rank = find(abs(diagR) >= tol*diagR(1), 1, 'last');

            idx_lin_indep1 = (it + 1)*ones(1,r_rank) - [1:r_rank];

            idx_lin_indep1 = sort(idx_lin_indep1);

            tmp_Gamma_t = flip(tmp_Gamma_t, 2);

            iGamma_t1 = inv( tmp_Gamma_t(idx_lin_indep1, idx_lin_indep1) )


            
            %for the 2. term

            tmp_Gamma_t = Gamma_t( 2:(it), 2:(it) );
            
            tmp_Gamma_t = flip(tmp_Gamma_t, 2);

            [Q,R] = qr(tmp_Gamma_t);

            tol=1e-8;

            diagR = diag(R);

            r_rank = find(abs(diagR) >= tol*diagR(1), 1, 'last');

            idx_lin_indep2 = (it)*ones(1,r_rank) - [1:r_rank];

            idx_lin_indep2 = sort(idx_lin_indep2);

            tmp_Gamma_t = flip(tmp_Gamma_t, 2);

            iGamma_t2 = inv( tmp_Gamma_t(idx_lin_indep2, idx_lin_indep2) )

        end
        

        if it ~= 1 % && cond(iGamma_t1) < 1e4 && cond(iGamma_t2) < 1e4 

            for j = 1:N
                
                r(j) = gk_pseudoMem(it, theta(j,2:end)', y(j), Gamma_t, iGamma_t1, idx_lin_indep1, iGamma_t2, idx_lin_indep2, probs, eta);
                
            end

        else

            for j = 1:N
                
                r(j) = y(j) - theta(j, end);
                
            end

        end



        r_all = [r_all, r];


        if it ~= 1 

            ct = 0;

            for j = 1:N
            
                ct = ct + gkd_pseudoMem(it, theta(j,2:end)', y(j), Gamma_t, iGamma_t1, idx_lin_indep1, iGamma_t2, idx_lin_indep2, probs, eta);

            end

            ct = ct / N;

        else

            ct = -1;

        end

        
        beta_est = [ beta_est, X' * r - ct * beta(:,end) ];
                       
        
        %estimating muk and sigma 
        
        sigma = [sigma, sum(r.^2) / N]
        
        if it ~= 1
            mu = [mu; sqrt( abs( sum(beta_est(:,end).^2) / M - sigma(end) ) / ( eta'*probs ) )];
        else 
            mu = [mu; 1];
        end

        if sum(beta_est(:,end).^2) / M - sigma(end) < 0
            it;
            sum(beta_est.^2) / M;
            sigma;
            'problem occured while estimating muk!'
            error('problem occured while estimating muk!');
            %'problem occured while estimating muk!'
            %muk = 1;
        end


        if it == -1

            mu(end)
            mean(beta_est(:,end) - mu(end) * beta_true)
            std( (beta_est(:,end) - mu(end) * beta_true) )
            [h,p,ksstat,cv] = kstest( (beta_est(:,end) - mu(end) * beta_true) / std( (beta_est(:,end) - mu(end) * beta_true) ) )
            histogram(beta_est(:,end) - mu(end) * beta_true, floor(M/18));
            ax = gca;
            hold on;
            mu(end)
            sqrt(sigma(end))
            xline(3* sqrt(sigma(end)))
            xline(- 3 * sqrt(sigma(end)))
            xline(1 * sqrt(sigma(end)))
            xline(- 1 * sqrt(sigma(end)))
            hold off;
            exportgraphics(ax, 'beta_est_hist_it1_pseudoMem_gaussian.jpg');
            break;
            
        end


        %calculating Delta_t

        Delta = zeros(it,it);

        for i = 1:(it)

            for j = 1:(it)

                Delta(i,j) = mean( r_all(:,i) .* r_all(:,j) );

            end

        end

        tmp_Delta = Delta
        
        tmp_Delta = flip(tmp_Delta, 2); %we are fliping columns od Delta

        [Q,R] = qr(tmp_Delta);

        tol=1e-8;

        diagR = diag(R);

        r_rank = find( abs(diagR) >= tol * diagR(1), 1, 'last' );

        idx_lin_indep_Delta = (it + 1) * ones(1,r_rank) - [1:r_rank];

        idx_lin_indep_Delta = sort(idx_lin_indep_Delta);

        tmp_Delta = flip(tmp_Delta, 2);

        iDelta = inv( tmp_Delta(idx_lin_indep_Delta, idx_lin_indep_Delta) );



        %Delta = N / M * Delta;

        if 1>0 %1>0 || cond(iDelta) < 1e6 && norm(iDelta) / it^2 < 1e6

            beta_last = zeros(M,1);
            
            for j = 1:M
                
                beta_last(j, 1) = fk_pseudoMem(it, beta_est(j,1 + [idx_lin_indep_Delta])', iDelta, mu(idx_lin_indep_Delta), probs, eta);
                
            end


            btp1 = 0;

            for j = 1:M

                btp1 = btp1 + fkd_pseudoMem(it, beta_est(j,1 + [idx_lin_indep_Delta])', iDelta, mu(idx_lin_indep_Delta), probs, eta);

            end

            btp1 = btp1 / N;

            beta = [beta, beta_last];

        else

            beta_last = zeros(M,1);
            
            for j = 1:M

                beta_last(j, 1) = fk(beta_est(j,end), sigma(end), mu(end), probs, eta);
                
            end


            btp1 = 0;

            for j = 1:M

                btp1 = btp1 + fkd(beta_est(j,end), sigma(end), mu(end), probs, eta);

            end

            btp1 = btp1 / N;

            beta = [beta, beta_last];

        end
    

        
        r_prev = r;
        

        %printing error information for iteration it
        it;
        sigmas = [ sigmas, sigma(end) ];
        l2_err_signal = [ l2_err_signal , norm(beta_true - beta_est(:,end) / mu(end) ) / norm(beta_true) ];
        'l2 prediction error:';
        l2_err_pred = [ l2_err_pred , norm(y - X * beta_est(:,end) / mu(end) ) / norm(y - X * beta_true) ];
        'correlation with the true signal';
        corrs = [ corrs, beta_true' * beta_est(:,end) / norm( beta_true ) / norm( beta_est(:,end) ) ];
        'ratio measure:';
        ratio_measures = [ ratio_measures, sigmas(end)/(mu(end)^2) ];
        'sqrt ( sigma_out / max(eta_final) ): ';
        sqrt( sigma / max(eta) );

    end

    %beta_est = beta_est(:,end) / mu(end);
    beta_est = beta_est(:,2:end) ./ mu(end)';
    
end