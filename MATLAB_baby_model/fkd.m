function val = fkd(y, sigma, muk, probs, eta)

    pk = 0;
    pkd = 0;
    pkdd = 0;
    
    if (sigma == 0)
        val = 1.0/muk;
        return;
    end
    
    for i=1:size(probs,1)
        
        z = probs(i) / sqrt( muk^2 * eta(i) + sigma ) * exp( -y^2 / ( muk^2 * eta(i) + sigma ) /2 );
        
        pk = pk + z;
        
        z = z / ( muk^2 * eta(i) + sigma ) * y;
        
        pkd = pkd - z;
        
        z = z / ( muk^2 * eta(i) + sigma ) * y;
        
        pkdd = pkdd + z;
        
    end

    val = (1.0/muk + sigma / muk * ( pkdd / pk - ( pkd / pk )^2 ) );
    
    if (isnan(val) == 1)
       val = 1.0/ muk;
       return;
    end
   
end