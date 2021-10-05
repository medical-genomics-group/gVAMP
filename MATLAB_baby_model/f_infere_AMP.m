function [beta_est, sigma, muk] = f_infere_AMP(y,X,iterNumb, beta0, b0, N, M, eta, probs, fk, fkd, beta_true)
    
    r_prev = zeros(N,1);
    r = r_prev;
    beta = beta0;
    beta_est = beta0;
    b = b0;
    muk = 1;
    sigma = -1;
    
    for it = 0:(iterNumb - 1)
        
        it
        muk
        sigma
        'l2 error:'
        norm(y-X*beta_est/muk) / norm(y-X*beta_true)
        'correlation with the true signal'
        beta_true' * beta_est / norm( beta_true ) / norm( beta_est )
        
        
        r = y - X*beta + b*r_prev;
        
        
        %estimating muk and sigma 
        
        sigma = sum(r.^2)/N;
        
        beta_est = X'*r + beta;
        
        muk = sqrt( abs( sum(beta_est.^2) / M - sigma ) / ( eta'*probs ) );
        
        
        
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
        
    end
    
    beta_est = beta_est / muk;
    
end