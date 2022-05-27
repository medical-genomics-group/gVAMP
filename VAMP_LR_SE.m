function [gams_SE, eta1] = VAMP_LR_SE( gamw, corr0, n_iter, distr, S, M, N )
    %key thing to remember: gam1 = tau1^(-1) and gam2 = tau2^(-1)

    gams_SE = [];
    if corr0 == 0
        gam0 = 1e-6;
    else
        gam0 = corr0^2 / (1 - corr0^2 ) / ( distr.eta' * distr.probs );
    end
    gam1 = gam0;

    for i = 1:n_iter
      gams_SE = [gams_SE, gam1];
      eta1 = gam1 / E1(gam1, distr);
      gam2 = min( [eta1 - gam1, 1e8] );

      eta2 = 1 / E2(gam2, S, gamw, M, N);
      gam1  = eta2 - gam2;
      
      if ( gam1 > min( max(40, 0.9*gamw), 600) )
         gam1 / (1 + gam1);
         %break;
      end
    end
end





%derivative of denoiser function g1'
function val = g1d(y, gam1, distr)
    sigma = 1 / gam1;
    probs = distr.probs;
    eta = distr.eta;
    muk = 1;
    
    pk = 0;
    pkd = 0;
    pkdd = 0;
    
    if (sigma == 0)
        val = 1.0/muk;
        return;
    end
        
    for i=1:size(probs,1)
        expe_sup = - 0.5 * y.^2 * muk^2 * ( max(eta) - eta(i) ) / ( muk^2 *  eta(i) + sigma ) / ( muk^2 *  max(eta) + sigma );        
        z = probs(i) / sqrt( muk^2 * eta(i) + sigma ) * exp( expe_sup );        
        pk = pk + z;
        z = z / ( muk^2 * eta(i) + sigma ) .* y;        
        pkd = pkd - z;        
        z2 = z / ( muk^2 * eta(i) + sigma ) .* y;        
        pkdd = pkdd - probs(i) / ( muk^2 * eta(i) + sigma )^(3/2) * exp( expe_sup ) + z2;       
    end
    
    val = (1.0/muk + sigma / muk * ( pkdd ./ pk - ( pkd ./ pk ).^2 ) );
    
    if (isnan(val) == 1)
       error("problem inside g1d!")
    end  
    
    %val = distr.eta / ( distr.eta + 1/gam1 );
end


%derivative of the denoiser function from [Plu-And-Play Learned Gaussian-mixture Approximate Message Passing]
function  out = g1d_2(y, gam1, distr)

    y = y(:);
    L = size(distr.probs,1);

    beta_t_n = distr.probs .* exp( -0.5 * (y.^2)' .* (max(distr.eta) - distr.eta) ./ ( 1/gam1 + distr.eta ) / ( 1/gam1 + max(distr.eta) ) ) ./ sqrt(distr.eta + 1/gam1);
    beta_t_n = beta_t_n ./ sum(beta_t_n);
    
    gamma_n = kron( y' , distr.eta ./ (distr.eta + 1/gam1) );

    out1 = ( distr.eta ./ (distr.eta + 1/gam1) )' * beta_t_n;
    out2 = zeros( L, size(y,1) );
    for y_ind = 1:size(y,1)
        for l = 1:L
            out2(l, y_ind) = 0;
            for lp = 1:L
                out2(l, y_ind) = out2(l, y_ind) + y(y_ind) * beta_t_n(l, y_ind) * beta_t_n(lp, y_ind) * (distr.eta(l) - distr.eta(lp)) / (distr.eta(l) + 1/gam1) / (distr.eta(lp) + 1/gam1);
            end
        end
    end
    out2 = diag ( gamma_n' * out2 );

    out = ( out1' + out2 )';

end


function [result] = E1(gam1, distr)
    fun = @(y)  g1d_2(y, gam1, distr) .* (  ( distr.probs ./ sqrt( distr.eta + 1/gam1 ) )' * exp( - y.^2 ./ ( distr.eta + 1/gam1 ) / 2 ) / sqrt(2 * pi)  );   
    result = integral(fun, -Inf, Inf);
    %result = distr.eta / ( distr.eta + 1/gam1 );
end

function [result] = E2(gam2, S, gamw, M, N)
    Sextend = zeros( max(M, N), 1 );
    Sextend( 1:min(M,N) ) = S;
    result = mean ( 1./ ( gamw * Sextend.^2 + gam2 ) );   
end