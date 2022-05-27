function val = fk(y, sigma, muk, probs, eta)

    pk = 0;
    pkd = 0;
    
    if (sigma == 0)
        val = y/muk;
        return;
    end
    
    for i=1:size(probs,1)
        
        z = probs(i) / sqrt( muk^2 * eta(i) + sigma ) * exp( -y^2 / ( muk^2 * eta(i) + sigma ) /2 );
        
        pk = pk + z;
        
        z = z / ( muk^2 * eta(i) + sigma ) * y;
        
        pkd = pkd - z;        
    end

    val = (y / muk + sigma / muk * pkd / pk );
    
    if (isnan(val) == 1)
       val = y / muk;
       return;
    end
   
end