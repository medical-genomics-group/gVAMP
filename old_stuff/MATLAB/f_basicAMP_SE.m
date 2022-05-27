function [mus, sigmas] = f_basicAMP_SE(numb_iter_SE, delta, M, eta_final, probs_final, sigma_noise, beta_true, beta0)

    numb_rep = 10000;
    %numb_iter_SE = 3;
    mus = [1];
    sigmas = [];


    Sigma0 =  [ eta_final'*probs_final 0; 0 1  ] / delta ;

    zz0 = mvnrnd(zeros(1, 2), Sigma0, numb_rep);

    sigma1 = sigma_noise + mean ( (zz0(:,1) - zz0(:,2) ).^2 );

    %sigma1 = norm(beta_true - beta0)^2 / ( delta * M);

    sigmas = [ sigma1 ];
    
    for k = 0 : (numb_iter_SE - 1)

        t = unifrnd(0,1, numb_rep, 1);
        beta_true1 = normrnd( 0, sqrt(eta_final(1)), numb_rep, 1 );
        beta_true2 = normrnd( 0, sqrt(eta_final(2)), numb_rep, 1 );
        beta_true3 = normrnd( 0, sqrt(eta_final(3)), numb_rep, 1 );
        c_probs = cumsum(probs_final);
        beta_true_sim = (t < c_probs(1)) .* beta_true1 + (t >= c_probs(1) & t < c_probs(2)) .* beta_true2 + (t >  c_probs(2)) .* beta_true3;

        val = beta_true_sim + sqrt(sigmas(end)) * normrnd( 0, 1, numb_rep, 1); %second argument is std dev

        for i = 1:numb_rep
            val(i) = fk(val(i), sigmas(end), mus(end), probs_final, eta_final);
        end
        sigmas = [sigmas, sigma_noise + mean( (beta_true_sim - val).^2 ) / delta ];
       
        mus = [ mus, 1 ];

    end

end