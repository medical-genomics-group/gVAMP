function [val]  = gk_pseudoMem(it, theta, y_val, Gamma_t, iGamma_t1, idx_lin_indep1, iGamma_t2, idx_lin_indep2, probs, eta)

    vec = [theta; y_val];
    
    %if it >= 2
    %    'not okay'
    %    inv( Gamma_t( 2:(it+1), 2:(it+1) ) )
    %    'okay'
    %end
    
    %val =  Gamma_t(1, 2:(it+2)) * inv( Gamma_t( 2:(it+2), 2:(it+2) ) ) * vec - Gamma_t(1, 2:(it+1)) * iGamma_t * theta ;

    val =  Gamma_t(1, 1 + [idx_lin_indep1] ) * iGamma_t1 * vec(idx_lin_indep1) - Gamma_t(1, 1 + [idx_lin_indep2]) * iGamma_t2 * theta(idx_lin_indep2) ;

end