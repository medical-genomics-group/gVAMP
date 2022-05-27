function [x1_hat, gams, l2_signal_err, corrs, real_gams, l2_pred_err, R2, betas] = EM_VAMP_LR_fast(distr, corr0, gamw, y, A, iterNumb, beta0, beta_true, U, S, V)

    %INPUT ARGUMENTS:
    %
    %   [distr] - struct object containing two column subvectors: 
    %           eta - mixture variances
    %           probs - mixture probabilities
    %   [corr0] - estimation of corr(beta_true, beta0)
    %   [gamw] - noise precision
    %   [y] - input phenotype vector
    %   [A] - genotype matrix (normalized s.t. mean and sd of each columns are
    %         0 and 1 / sqrt(N) (not hard-coded in the code)
    %   [iterNumb] - number of VAMP iterations
    %   [beta0] - initial estimate of marker effects
    %   [beta_true] - true value of marker values
    %   [U,S,V] - SVD of genotype matrix
    %
    %OUTPUT ARGUMENTS:
    %
    %   [x1_hat] - estimate of marker values
    %   [gams] - precision of the noise in the measurement of r throught
    %            iterations
    %   [l2_signal_err] - vector of signal errors
    %   [corrs] - vector of correlations throughout iterations
    %   [real_gams] - observed precision of the noise in the measurement of r throught
    %            iterations
    %   [l2_pred_err] - vectors of prediction error
    %
    %DEPENDENCIES:
    %
    %   function val = g1(y, gam1, distr)
    %   function val = g1d(y, gam1, distr)
    %
    
        %recovering values of M and N
        [N, M] = size(A);
        delta = N/M;

        [corrs, l2_signal_err, l2_pred_err, R2] = deal( zeros(iterNumb,1) );    
        if corr0 == 0
            gam0 = 1e-6;
        else
            %Let sigma = E[beta^2]
            %rho = E[beta * beta0] / sqrt( E[beta] * E[beta0])
            %take into account that beta0 = beta + N(0, 1/gam0)
            %gam0 = rho^2/ (1-rho^2) / E[beta^2]
            gam0 = corr0^2 / (1 - corr0^2 ) / ( distr.eta' * distr.probs );
        end
        tol = 1e-11;
           
        if nargin <= 8
            %caluclating SVD on matrix A
            [U,S,V] = svd(A);
        end
        s = diag(S);
        R = length(S);
                
        [r1, x1_hat] = deal(beta0);
        [gam1, gam2] = deal(gam0);

        if nargout >= 3
            [gams, real_gams] = deal([]);
        end
        if nargout >= 8
            betas = zeros(M, iterNumb);
        end

        rho = 0.97;
        %start iterations
        for it = 1:iterNumb

            it
            %Input denoising
            x1_hat_prev = x1_hat;
            x1_hat = rho * g1(r1, gam1, distr) + (1-rho) * x1_hat;
            %x1_hat = g1_SURE(r1, gam1);
            %x1_hat = g1_dhyp2(r1, gam1, delta);
            eta1 = gam1 / mean( g1d(r1, gam1, distr) );
            %eta1 = gam1 / mean( g1d_SURE(r1, gam1) );
            %eta1 = gam1 / mean( g1d_dhyp2(r1, gam1, delta) );
            gam2 = min( max( eta1 - gam1, 1e-11 ) , 1e11 );
            r2 = ( eta1 * x1_hat - gam1 * r1 ) / gam2;
            if nargout >= 2
                gams = [ gams, gam1 ];
            end
            
            %Input parameter update
            max_iter = 50;
            err_thr = 1e-6;
            %distr.probs = infereMixtures(distr.probs, r1, gam1, distr.eta, max_iter, err_thr);
            %[distr.probs, distr.eta] = infereMixtures_with_lambda(distr.probs, r1, gam1, distr.eta, max_iter, err_thr);
            distr.probs;
            distr.eta;
      
            %Output estimation
            dinvQ = zeros(M,1);
            dinvQ(1:min(M,N)) = gamw * s.^2;
            dinvQ = 1 ./ ( dinvQ + ones(M,1) * gam2 );
            invQ = V * diag( dinvQ ) * V';
            %invQ = inv( gamw * A'*A + gam2 * eye(M) );
            x2_hat = invQ * ( gamw * A' * y + gam2 * r2 );
            eta2 = M / sum( dinvQ );
            %eta2 = M / trace( invQ );
            gam1 = rho * min( max( eta2 - gam2, 1e-11 ), 1e11 ) + (1 - rho) * gam1; % gamma \in [ gamma_min, gamma_max ]
            r1_prev = r1;
            r1 = (eta2 * x2_hat - gam2 * r2) / gam1;

            %Output parameter update
            %gamw = M / ( norm( y - A*x2_hat )^2 + sum( s.^2 ./ ( gamw * s.^2 + gam2 ) ) )  %originally this is theta2
           
            %saving statistics    
            l2_pred_err(it) = norm(y - A * x1_hat) / norm(y);
            if  nargin >= 8
                l2_signal_err(it) = norm(beta_true - x1_hat) / norm (beta_true);
                corrs(it) = beta_true'*x1_hat / norm(beta_true) / norm(x1_hat);
                
                if  corrs(it) > 0.999
                    break; 
                end
            end

            R2(it) = 1 - norm(y - A * x1_hat)^2 / norm(y - mean(y))^2;
            if ( it > 1 & R2(it) < R2(it-1) )
                x1_hat = x1_hat_prev;
                break;
            end

            if norm(r1 - r1_prev) / norm(r1) < 1e-4
                break;
            end

            if nargout >= 8
               betas(:,it) = x1_hat; 
            end
        end
    end
    
    
    %denoiser function g1
    function val = g1(y, gam1, distr)
    % INPUT ARGUMENTS:
    %
    %   [y] - values to be denoised, y = beta + N(0, 1/gam1)
    %   [gam1] - precision of noise
    %   [distr] - distribution of prior on marker values
    %
    % OUTPUT ARGUMENTS:
    %
    %   [val] - value after denoising
    %
    % DEPENDENCIES:
    %
    %   None.
    %
    
        sigma = 1 / gam1;
        probs = distr.probs;
        eta = distr.eta;
        eta_max = max(eta);
        
        [ pk, pkd ] = deal(0);
        
        if (sigma == 0)
            val = y;
            return;
        end
        
        for i=1:size(probs,1)
    
            expe_sup = - 0.5 * y.^2 * ( eta_max - eta(i) ) / ( eta(i) + sigma ) / ( eta_max + sigma );
    
            z = probs(i) / sqrt( eta(i) + sigma ) * exp( expe_sup );

            pk = pk + z;
            
            z = z / ( eta(i) + sigma ) .* y;
            
            pkd = pkd - z;        
        end
    
        val = (y + sigma * pkd ./ pk );
        
        if (isnan(val) == 1)
            distr.eta
            distr.probs
           error("problem inside denoiser g1!")
        end
       
    end
    
    
    %derivative of denoiser function g1'
    function val = g1d(y, gam1, distr)
    % INPUT ARGUMENTS:
    %
    %   [y] - values to be denoised, y = beta + N(0, 1/gam1)
    %   [gam1] - precision of noise
    %   [distr] - distribution of prior on marker values
    %
    % OUTPUT ARGUMENTS:
    %
    %   [val] - derivative of denoiser function eval at y
    %
    % DEPENDENCIES:
    %
    %   None.
    %

        sigma = 1 / gam1;
        probs = distr.probs;
        eta = distr.eta;
        eta_max = max(eta);
        
        [pk, pkd, pkdd] = deal(0);
        
        if (sigma == 0)
            val = 1.0;
            return;
        end
            
        for i=1:size(probs,1)
    
            expe_sup = - 0.5 * y.^2 * ( eta_max - eta(i) ) / ( eta(i) + sigma ) / ( eta_max + sigma );
            
            z = probs(i) / sqrt( eta(i) + sigma ) * exp( expe_sup );
            
            pk = pk + z;
            
            z = z / ( eta(i) + sigma ) .* y;
            
            pkd = pkd - z;
            
            z2 = z / ( eta(i) + sigma ) .* y;
            
            pkdd = pkdd - probs(i) / ( eta(i) + sigma )^(3/2) * exp( expe_sup ) + z2;
            
        end
        
        val = (1.0 + sigma * ( pkdd ./ pk - ( pkd ./ pk ).^2 ) );
        
        if (isnan(val) == 1)
           error("problem inside denoised derivation g1d!")
        end   

    end


    
    %function for infeering mixtures of Gaussians
    function pis = infereMixtures(pi_prev, r1, gam1, mix_var, max_iter, err_thr)
        % INPUT ARGUMENTS:
        %
        %   [pi_prev] - previous estimate od mixture probabilities vector
        %   [gam1] - precision of noise in compound decision problem
        %   [mix_var] - array of mixture variances
        %   [max_iter] - maximal allowed number of iterations
        %   [err_thr]  - error threshold
        %
        % OUTPUT ARGUMENTS:
        %
        %   [pi] - an updated estimate of mixture probabilities
        %
        % DEPENDENCIES:
        %
        %   None.
        %
    
        max_sigma = max(mix_var);
        noise_var = 1/ gam1;
        pis = pi_prev;

        for it = 1:max_iter
            
            beta = exp( -r1.^2 / 2 * (max_sigma - mix_var') ./ (mix_var' + noise_var) / (max_sigma + noise_var) ) ./ sqrt(mix_var' + noise_var) .* pis';
            beta = beta ./ sum(beta,2);

            pis_n = sum(beta)' / sum(sum(beta));

            if norm(pis - pis_n)  / norm(pis) < err_thr
                break;
            end

            pis = pis_n;

        end
        'infereMixtures'
        it
        
    end

    
    %function for infeering mixtures of Gaussians
    function [pis, etas] = infereMixtures_with_lambda(pi_prev, r1, gam1, mix_var, max_iter, err_thr)
        % INPUT ARGUMENTS:
        %
        %   [pi_prev] - previous estimate od mixture probabilities vector
        %   [gam1] - precision of noise in compound decision problem
        %   [mix_var] - array of mixture variances
        %   [max_iter] - maximal allowed number of iterations
        %   [err_thr]  - error threshold
        %
        % OUTPUT ARGUMENTS:
        %
        %   [pi] - an updated estimate of mixture probabilities
        %   [etas] - an updated estimate of variances of mixture components
        %
        % DEPENDENCIES:
        %
        %   None.
        %
    
        max_sigma = max(mix_var(2:end));
        noise_var = 1/ gam1;
        lambda = 1-pi_prev(1);
        omegas = pi_prev(2:end) / lambda;

        for it = 1:max_iter

            %lambda = 0.01;
            
            pis_pr = [1- lambda; lambda * omegas];

            %beta = lambda * normpdf(r1, 0, mix_var(2:end)' + noise_var) .* omegas';
            beta = lambda * exp( -r1.^2 / 2 * (max_sigma - mix_var(2:end)') ./ (mix_var(2:end)' + noise_var) / (max_sigma + noise_var) ) ./ sqrt(mix_var(2:end)' + noise_var) / sqrt(2*pi) .* omegas';
            betat = beta ./ sum(beta,2);

            pin = 1 ./ ( 1 + (1-lambda) / sqrt(2 * pi * noise_var) * exp( -r1.^2 / 2 * max_sigma / noise_var / (noise_var + max_sigma) ) ./ sum(beta,2)  );

            gammas = gam1 * r1 ./ ( 1./mix_var(2:end)' + gam1 );
            v = 1 ./ ( 1./mix_var(2:end)' + gam1 );
            etas = ( (betat .* (gammas.^2 + v) )' * pin) ./ (betat' * pin);

            lambda = mean(pin);
            %lambda = 0.01;

            omegas = betat' * pin / sum(pin);

            pis = [1- lambda; lambda * omegas];

            if norm(pis - pis_pr)  / norm(pis_pr) < err_thr 
                break;
            end

        end
        'infereMixtures'
        it

        etas = [ 0; etas ];
        
    end


    %hiddenly sparse multiplication of A and x
    function out = Ax(x, r0, c0, r1, c1, r2, c2, mus, sds, N)
        % INPUT ARGUMENTS:
        % 
        %   [x] - a vector to be multiplied by the genotype matrix
        %   [r0, c0, r1, c1, r2, c2] - row and column indices of 0s, 1s and 2s in genotype matrix
        %   [mus] - vector of column means
        %   [sds] - vector of column sds
        %   [N] - number of individuals
        %
        % OUTPUT ARGUMENTS:
        %
        %   [out] - vector Ax
        %
        % DEPENDENCIES:
        %
        %   None.
        %   
        val0 = cell2mat( arrayfun(@(z) sum( x( c0( r0==z ) ) * (-means(r0==z)) ./ sds(r0==z) ), 1:N, 'UniformOutput', false) )';
        val1 = cell2mat( arrayfun(@(z) sum( x( c1( r1==z ) ) * (1-means(r1==z)) ./ sds(r1==z) ), 1:N, 'UniformOutput', false) )';
        val2 = cell2mat( arrayfun(@(z) sum( x( c2( r2==z ) ) * (2-means(r1==z)) ./ sds(r2==z) ), 1:N, 'UniformOutput', false) )';
        out = val0 + val1 + val2;
    end


    %hiddenly sparse multiplication of A^T and y
    function out = Aty(y, r0, c0, r1, c1, r2, c2, mus, sds)
        % INPUT ARGUMENTS:
        % 
        %   [y] - a vector to be multiplied by the transpose of genotype matrix
        %   [r0, c0, r1, c1, r2, c2] - row and column indices of 0s, 1s and 2s in genotype matrix
        %   [mus] - vector of column means
        %   [sds] - vector of column sds
        %
        % OUTPUT ARGUMENTS:
        %
        %   [out] - vector A^Ty
        %
        % DEPENDENCIES:
        %
        %   None.
        %   
        val0 = cell2mat( arrayfun(@(z) sum( y( r0( c0==z ) ) ) * (-means(z)) / sds(z) , 1:M, 'UniformOutput', false) )';
        val1 = cell2mat( arrayfun(@(z) sum( y( r1( c1==z ) ) ) * (1-means(z)) / sds(z), 1:M, 'UniformOutput', false) )';
        val2 = cell2mat( arrayfun(@(z) sum( y( r2( c2==z ) ) )* (2-means(z)) / sds(z), 1:M, 'UniformOutput', false) )';
        out = val0 + val1 + val2;
    end

    %function handle to provide to svds
    function out = Afun(x,r0, c0, r1, c1, r2, c2, mus, sds)
        if strcmp(tflag,'notransp')
            out = Ax(x, r0, c0, r1, c1, r2, c2, mus, sds, N);
        else
            out = Aty(x, r0, c0, r1, c1, r2, c2, mus, sds);
        end
    end

    function out = ekf1(r, c)
        out = r;
    end

    function out = dekf1(r, c)
        out = ones(size(r(:),1), 1);
    end

    function out = ekf2(r, c)
        T = 6 * sqrt(c);
        out = r .* exp( - r.^2 / 2 / T^2 );
    end

    function out = dekf2(r, c)
        T = 6 * sqrt(c);
        out = (1 - r.^2 / T^2) .* exp( - r.^2 / 2 / T^2 );
    end

    function out = splkf1(r, c)
        
        beta1 = 1 / ( 1 + 6 * sqrt(c) );
        out = r;
        out( r <= - beta1 ) = -1;
        out(- beta1 < r & r < beta1) = r(- beta1 < r & r < beta1) / beta1;
        out(r >= beta1) = 1;
    end

    function out = dsplkf1(r, c)
        beta1 = 1 / ( 1 + 6 * sqrt(c) );
        out = r;
        out( r <= - beta1 ) = 0;
        out(- beta1 < r & r < beta1) = 1 / beta1;
        out(r >= beta1) = 0;
    end

    function out = splkf2(r, c)
        beta1 = 1 / ( 1 + 6 * sqrt(c) );
        beta2 = 1 / ( 1 + 2 * sqrt(c) );
        out = r;

        out(r <= -beta2) = -1;
        out(- beta2 < r & r < - beta1) = (r(- beta2 < r & r < - beta1) + beta1) / (beta2 - beta1);
        out(- beta1 <= r & r <= beta1) = 0;
        out(beta1 < r & r < beta2) = (r(beta1 < r & r < beta2) - beta1) / (beta2 - beta1);
        out(r >= beta2) = 1;
    end

    function out = dsplkf2(r, c)
        beta1 = 1 / ( 1 + 6 * sqrt(c) );
        beta2 = 1 / ( 1 + 2 * sqrt(c) );
        out = r;

        out(r <= -beta2) = 0;
        out(- beta2 < r & r < - beta1) = 1 / (beta2 - beta1);
        out(- beta1 <= r & r <= beta1) = 0;
        out(beta1 < r & r < beta2) = 1 / (beta2 - beta1);
        out(r >= beta2) = 01;
    end

    
    %denoiser function g1
    function val = g1_SURE(y, gam1)
        % INPUT ARGUMENTS:
        %
        %   [y] - values to be denoised, y = beta + N(0, 1/gam1)
        %   [gam1] - precision of noise
        %
        % OUTPUT ARGUMENTS:
        %
        %   [val] - value after denoising
        %
        % DEPENDENCIES:
        %
        %   None.
        %
        
        %{
            sigma = 1 / gam1;
            F = [ mean( splkf1(y, sigma) .^2 ) mean(splkf1(y, sigma) .* splkf2(y, sigma)); mean(splkf1(y, sigma) .* splkf2(y, sigma)) mean( splkf2(y, sigma) .^2 ) ];
            d = [ mean( dsplkf1(y, sigma) ); mean( dsplkf2(y, sigma) ) ];
            a = linsolve(F,-sigma*d);

            val = [splkf1(y, sigma) splkf2(y, sigma)] * a;         
        %}

            sigma = 1 / gam1;
            F = [ mean( ekf1(y, sigma) .^2 ) mean(ekf1(y, sigma) .* ekf2(y, sigma)); mean(ekf1(y, sigma) .* ekf2(y, sigma)) mean( ekf2(y, sigma) .^2 ) ];
            d = [ mean( dekf1(y, sigma) ); mean( dekf2(y, sigma) ) ];
            a = linsolve(F,-sigma*d);

            val = [ekf1(y, sigma) ekf2(y, sigma)] * a;           
    end

    
    %denoiser derivative
    function val = g1d_SURE(y, gam1)
        % INPUT ARGUMENTS:
        %
        %   [y] - values to be denoised, y = beta + N(0, 1/gam1)
        %   [gam1] - precision of noise
        %
        % OUTPUT ARGUMENTS:
        %
        %   [val] - value after denoising
        %
        % DEPENDENCIES:
        %
        %   None.
        %
        
        %{

            sigma = 1 / gam1;
            F = [ mean( splkf1(y, sigma) .^2 ) mean(splkf1(y, sigma) .* splkf2(y, sigma)); mean(splkf1(y, sigma) .* splkf2(y, sigma)) mean( splkf2(y, sigma) .^2 ) ];
            d = [ mean( dsplkf1(y, sigma) ); mean( dsplkf2(y, sigma) ) ];
            a = linsolve(F,-sigma*d);

            val = [dsplkf1(y, sigma) dsplkf2(y, sigma)] * a;      
        %}

        sigma = 1 / gam1;
        F = [ mean( ekf1(y, sigma) .^2 ) mean(ekf1(y, sigma) .* ekf2(y, sigma)); mean(ekf1(y, sigma) .* ekf2(y, sigma)) mean( ekf2(y, sigma) .^2 ) ];
        d = [ mean( dekf1(y, sigma) ); mean( dekf2(y, sigma) ) ];
        a = linsolve(F,-sigma*d);

        val = [dekf1(y, sigma) dekf2(y, sigma)] * a;  

        sum( dekf2(y, sigma) < 0)
        sum ( dekf2(y, sigma) > 1)
        a

    end

     %denoiser Finnish Horseshoe
     function val = g1_dhyp2(y, gam1, delta)
        % INPUT ARGUMENTS:
        %
        %   [y] - values to be denoised, y = beta + N(0, 1/gam1)
        %   [gam1] - precision of noise
        %
        % OUTPUT ARGUMENTS:
        %
        %   [val] - value after denoising
        %
        % DEPENDENCIES:
        %
        %   None.
        %
        sigma = 1 / gam1;
        tau = 0.7 / delta;
        %tau = 2;
        val = zeros(length(y),1);
        for i0 = 1: length(y)
            val(i0) = y(i0) * ( 1 - 2* dhyp2(0.5, 1, 2.5, y(i0).^2 / 2 / sigma, 1- 1/(tau^2) ) / 3 / dhyp2(0.5,1,1.5, y(i0).^2 /2/sigma, 1 - 1/(tau^2)) );
            if isnan(val(i0)) == 1
                val(i0) = y(i0);
                %error('nan in g1_dhyp2');
            end
        end
        %val = y .* ( 1 - 2* dhyp2(0.5, 1, 2.5, y.^2 / 2 / sigma, 1- 1/(tau^2) ) / 3 / dhyp2(0.5,1,1.5, y.^2 /2/sigma, 1 - 1/(tau^2)) );
    end

    %derivative of Finnish Horseshoe denoiser
    function val = g1d_dhyp2(y, gam1, delta)
        % INPUT ARGUMENTS:
        %
        %   [y] - values to be denoised, y = beta + N(0, 1/gam1)
        %   [gam1] - precision of noise
        %
        % OUTPUT ARGUMENTS:
        %
        %   [val] - value after denoising
        %
        % DEPENDENCIES:
        %
        %   None.
        %

        sigma = 1 / gam1;
        tau = 0.7 / delta;
        %tau = 2;
        val = zeros(length(y),1);
        for i0 = 1:length(y)
            T = g1_dhyp2(y(i0), gam1, delta);
            val = gam1 * ( sigma ./ y(i0) .* T - (T - y(i0)).^2 + y(i0).^2 * (8 * dhyp2(0.5,1,3.5,y(i0).^2/2/sigma, 1-1/(tau^2)) / 15 / dhyp2(0.5,1,1.5,y(i0).^2 /2/sigma, 1-1/(tau^2)) ) );
            if isnan(val) == 1
                val = 1;
                %error('nan in g1_dhyp2');
            end
        end
    end

    %degenerate hypergeometric function of two variables
    function val = dhyp2(a,b,c,y,x)
        %csvwrite('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/testing08122021/VAMP/in.csv',[a,b,c,x,y]);
        %[ status, cmdout ] = system('module load R; R CMD BATCH /nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/testing08122021/VAMP/script.R');
        %!R CMD BATCH /nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/testing08122021/VAMP/script.r
        %val = csvread('/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/testing08122021/VAMP/out.csv',1,1);
        func=@(t)(t.^(a-1)).*((1-t).^(c-a-1)).*((1-x.*t).^(-b)).*exp(y.*t);
        val = integral(func,0,1) / beta(a,c-a);
        %val = quadgk(func,0,1,'AbsTol',1e-4, 'MaxIntervalCount',1e6) * beta(a,b); %/ beta(a, c-a);
    end
