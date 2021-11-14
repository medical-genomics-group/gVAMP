function [val] = fkd_pseudoMem(it, beta_val, iDelta, mu, probs, eta)

    K = size(probs,1);

    A = 0;
    B = 0;
    Ap = 0;
    Bp = 0;

    max_eta = max(eta);

    for i = 1:K

        term = probs(i)  /  (mu' * iDelta * mu * eta(i) + 1 )^(1/2) * exp( (mu' * iDelta * beta_val)^2 / 2  * ( eta(i) - max_eta ) / (mu' * iDelta * mu * eta(i) + 1) / (mu' * iDelta * mu * max_eta + 1 ) );

        A = A + term / (mu' * iDelta * mu * eta(i) + 1 ) * (mu' * iDelta * beta_val * eta(i) );
    
        B = B + term;

        expe =  exp( (mu' * iDelta * beta_val)^2 / 2  * ( eta(i) - max_eta ) / (mu' * iDelta * mu * eta(i) + 1) / (mu' * iDelta * mu * max_eta + 1 ) );
        
        jth_row = mu'*iDelta;
        jth_row = jth_row(1,end);

        Ap = Ap + probs(i) * expe * ( eta(i)^2 * (mu' * iDelta * beta_val)^2 / (mu' * iDelta * mu * eta(i) + 1 )^(5/2) * jth_row + jth_row * eta(i) / (mu' * iDelta * mu * eta(i) + 1 )^(3/2) );
    
        Bp = Bp + probs(i) * expe * ( eta(i) * mu' * iDelta * beta_val ) / (mu' * iDelta * mu * eta(i) + 1 )^(3/2) * jth_row;
    
    end

    val = (Ap * B - A * Bp) / B^2;
    
    %{
    K = size(probs,1);

    %iDelta = inv(Delta);

    A = 0;
    B = 0;
    Ap = 0;
    Bp = 0;

    average_shift_mat = [];

    for i = 1:K
        average_shift_mat = [average_shift_mat,- 0.5 * beta_val' * iDelta * beta_val + (mu' * iDelta * beta_val)^2 / 2 / (mu' * iDelta * mu + 1/eta(i) ) ];
    end
    
    average_shift = mean(average_shift_mat);

    for i = 1:K

        term = probs(i)  /  (mu' * iDelta * mu + 1/eta(i) )^(1/2) * exp(- 0.5 * beta_val' * iDelta * beta_val + (mu' * iDelta * beta_val)^2 / 2 / (mu' * iDelta * mu + 1/eta(i) ) - average_shift );

        A = A + term / (mu' * iDelta * mu + 1/eta(i) ) * (mu' * iDelta * beta_val);
    
        B = B + term;

        expe =  exp(- 0.5 * beta_val' * iDelta * beta_val + (mu' * iDelta * beta_val)^2 / 2 / (mu' * iDelta * mu + 1/eta(i) ) - average_shift );
        
        jth_row = mu'*iDelta;
        jth_row = jth_row(1,end);

        Ap = Ap + probs(i) * expe * ( (mu'*iDelta*beta_val)^2 / (mu' * iDelta * mu + 1/eta(i) )^(5/2) * jth_row + jth_row / (mu' * iDelta * mu + 1/eta(i) )^(3/2) );
    
        Bp = Bp + probs(i) * expe * (mu'*iDelta*beta_val) / (mu' * iDelta * mu + 1/eta(i) )^(3/2) * jth_row;
    end
    
    val = (Ap * B - A * Bp) / B^2;
    %}

end