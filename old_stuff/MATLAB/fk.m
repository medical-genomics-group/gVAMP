function val = fk(y, sigma, muk, probs, eta)

    pk = 0;
    pkd = 0;
    
    if (sigma == 0)
        val = y/muk;
        return;
    end
    
    for i=1:size(probs,1)

        expe_sup = - 0.5 * y^2 * muk^2 * ( max(eta) - eta(i) ) / ( muk^2 *  eta(i) + sigma ) / ( muk^2 *  max(eta) + sigma );
        
        %z = probs(i) / sqrt( muk^2 * eta(i) + sigma ) * exp( -y^2 / ( muk^2 * eta(i) + sigma ) /2 );

        z = probs(i) / sqrt( muk^2 * eta(i) + sigma ) * exp( expe_sup );
        
        pk = pk + z;
        
        z = z / ( muk^2 * eta(i) + sigma ) * y;
        
        pkd = pkd - z;        
    end

    val = (y / muk + sigma / muk * pkd / pk );
    
    if (isnan(val) == 1)
       y^2
       muk^2 * eta(i) + sigma
       pk
       pkd
       val
       val = y / muk;
       error("problem inside fk!")
       return;
    end
    
    %if (~isempty(ind))
    %    val = ( 1- probs(ind) ) * val;
    %end
    
end