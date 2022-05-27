function [all_beta] =  f_Gibbs(y, X, beta, sigma_noise, eta, probs, burnin_iter, numb_iter)
    M = size(X,2);
    K = size(eta,1);

    %eta = [ 1 0.1 0001];
    %eta(1) = 0;
    all_beta = zeros(M, numb_iter);

    for iter = 1:(burnin_iter + numb_iter)
        iter
        
        for j = 1:M
            if mod(j,100) == 0
               j
            end
            
            ytilda = y - X(:, 1:end ~= j ) * beta(1:end ~= j);

            %calculating group probabilities
            mix_probs = [];
            
            logl = [];

            for l = 1:K
                part1 = 0.5 * log ( sigma_noise / ( X(:,j)' * X(:,j) * eta(l) + sigma_noise ) );
                part2 = eta(l) * (ytilda' * X(:,j) )^2 / 2 / sigma_noise / ( X(:,j)' * X(:,j) * eta(l) + sigma_noise );
                part3 = log ( probs(l) );

                logl = [ logl, part1 + part2 + part3 ];
                
                %{
                for kk = 1:K
                        logkk = 0.5 * log ( sigma_noise / (X(:,j)' * X(:,j) * eta(kk) + sigma_noise ) ) + eta(kk) * (ytilda' * X(:,j) )^2 / 2 ...
                        / sigma_noise / ( X(:,j)' * X(:,j) * eta(kk) + sigma_noise ) + log ( probs(kk) );
                        kk
                        l
                        val = val + exp( logl - logkk )
                end
                
                if isinf(val)
                    'start'
                    logl
                    logkk
                    exp( logl - logkk )
                    eta
                    l
                    log ( sigma_noise / (X(:,j)' * X(:,j) * eta(kk) + sigma_noise ) )
                    eta(kk) * (ytilda' * X(:,j) )^2 / 2
                    (X(:,j)' * X(:,j) * eta(kk) + sigma_noise )
                    error(1)
                end
                %}
                
            end

            logl = logl - max(logl);

            mix_probs = exp(logl) / sum(exp(logl));

            %sum(mix_probs)

            %if iter == 1
            %    error(2)
            %end
         
            chosen_ind = discretesample(mix_probs, 1);

            %sampling from posterior given the group beta_j belongs to
            if eta(chosen_ind) ~= 0
                par_mu = ( X(:,j)' * X(:,j) + sigma_noise / eta(chosen_ind) )^(-1) * X(:,j)' * ytilda;
                par_sigma = sigma_noise * ( X(:,j)' * X(:,j) + sigma_noise / eta(chosen_ind) )^(-1);
                beta(j) = normrnd(par_mu, sqrt(par_sigma) );
            else 
                beta(j) = 0;
            end

        end

        if iter > burnin_iter
            all_beta(:,iter-burnin_iter) = beta;
        end
    
    end

end