function [val] = fk_pseudoMem(it, beta_val, iDelta, mu, probs, eta)

    K = size(probs,1);

    %iDelta = inv(Delta);

    A = 0;
    B = 0;

    %average_shift_mat = [];

    %for i = 1:K
    %    average_shift_mat = [average_shift_mat, -0.5 * beta_val' * iDelta * beta_val + (mu' * iDelta * beta_val)^2 / 2 / (mu' * iDelta * mu + 1/eta(i) ) ];
    %end
    
    %average_shift = mean(average_shift_mat);

    max_eta = max(eta);

    for i = 1:K

        term = probs(i)  /  (mu' * iDelta * mu * eta(i) + 1 )^(1/2) * exp( (mu' * iDelta * beta_val)^2 / 2  * ( eta(i) - max_eta ) / (mu' * iDelta * mu * eta(i) + 1) / (mu' * iDelta * mu * max_eta + 1 ) );

        A = A + term / (mu' * iDelta * mu * eta(i) + 1 ) * (mu' * iDelta * beta_val * eta(i) );
    
        B = B + term;

        if isnan(term)  || term > 1e50
            (mu' * iDelta * beta_val)^2 / 2  * ( eta(i) - min_eta ) / (mu' * iDelta * mu * eta(i) + 1) / (mu' * iDelta * mu * min_eta + 1 )
            eta(i)
            mu
            iDelta
            beta_val
            term
            A
            B
            error(1)
        end
    
    end

    val = A / B;
    
    
    %{ 
        
    K = size(probs,1);

    %iDelta = inv(Delta);

    A = 0;
    B = 0;

    average_shift_mat = [];

    for i = 1:K
        average_shift_mat = [average_shift_mat, -0.5 * beta_val' * iDelta * beta_val + (mu' * iDelta * beta_val)^2 / 2 / (mu' * iDelta * mu + 1/eta(i) ) ];
    end
    
    average_shift = mean(average_shift_mat);

    for i = 1:K

        term = probs(i)  /  (mu' * iDelta * mu + 1/eta(i) )^(1/2) * exp( - 0.5 * beta_val' * iDelta * beta_val + (mu' * iDelta * beta_val)^2 / 2 / (mu' * iDelta * mu + 1/eta(i) ) - average_shift );

        if isinf(term) == -1 
            'isinf(term) == 1:'
            (mu' * iDelta * mu + 1/eta(i) )^(1/2)
            term
            iDelta
            (mu' * iDelta * beta_val)^2
            mu
            beta_val
        end

        A = A + term / (mu' * iDelta * mu + 1/eta(i) ) * (mu' * iDelta * beta_val);
    
        B = B + term;
    
    end

    val = A / B;

    %}
    
end