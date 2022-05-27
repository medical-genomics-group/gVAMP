function val = fkd(y, sigma, muk, probs, eta)

    pk = 0;
    pkd = 0;
    pkdd = 0;
    
    if (sigma == 0)
        val = 1.0/muk;
        return;
    end
        
    for i=1:size(probs,1)

        expe_sup = - 0.5 * y^2 * muk^2 * ( max(eta) - eta(i) ) / ( muk^2 *  eta(i) + sigma ) / ( muk^2 *  max(eta) + sigma );
        
        %z = probs(i) / sqrt( muk^2 * eta(i) + sigma ) * exp( -y^2 / ( muk^2 * eta(i) + sigma ) /2 );

        z = probs(i) / sqrt( muk^2 * eta(i) + sigma ) * exp( expe_sup );
        
        pk = pk + z;
        
        z = z / ( muk^2 * eta(i) + sigma ) * y;
        
        pkd = pkd - z;
        
        z2 = z / ( muk^2 * eta(i) + sigma ) * y;
        
        pkdd = pkdd - z / y + z2;
        
    end

    val = (1.0/muk + sigma / muk * ( pkdd / pk - ( pkd / pk )^2 ) );
    
    if (isnan(val) == 1)
       val = 1.0/ muk;
       error("problem inside fkd!")
    end
    
    %if (~isempty(ind))
    %    val = ( 1- probs(ind) ) * val;
    %end
    
end