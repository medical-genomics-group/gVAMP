function [xhat_all, corr, sig2_all, freecum] = RIAMP_fast(n_iter, distr, y, A, true_x)
    
    % A in R^{n x d}
    % doing Bayesian linear regression for y = Ax + w
    % INPUT:
    %   [n_iter] - number of iterations of RI-AMP to do
    %   [distr] - gaussian mixture prior distribution params 
    %   [y] - phenotype vector
    %   [A] - genotype matrix
    %   [true_x] - true value of marker effect sizes
    %
    % OUTPUT:
    %   [xhat_all] - RI-AMP estimate of marker effect sizes
    %   [corr] - correlation throughout iterations
    %   [sig2_all] - variances of noise around true value of the signal throughtout iterations
    %
    % DEPENDENCIES:
    %   [cumulants] = compute_cum(A, numbC)
    %   [val] = fk_pseudoMem(beta_val, iDelta, mu, distr)  
    %   [val] = fkd_pseudoMem(beta_val, iDelta, mu, distr) 
    %

    rng(1);
    [n,d] = size(A);
    delta = n / d;
    freecum = compute_cum(A, 2*(n_iter + 1)) %to be upscaled in the future
    freecum(1)
    freecum(2)
    
    [x_all, xhat_all] = deal( zeros(n_iter, d) );
    [z_all] = zeros(n_iter, n);
    [zhat_all] = zeros(n_iter + 1, n); %one extra because here we also have s1

    [mu_all, sig2_all] = deal( zeros(n_iter, 1) );
    [ sigma_mat, psi_mat, gamma_mat ] = deal ( zeros(n_iter + 1, n_iter + 1) );
    [ phi_mat, delta_mat ] = deal( zeros(n_iter + 2, n_iter + 2) );
    omega_mat = zeros(n_iter, n_iter);

    s1 = y; %we start with s1 = y
    delta_mat(2,2) = mean( s1.^2 );
    gamma_mat(1,1) = distr.eta' * distr.probs;
    phi_mat(2,1) = 1; %since we have s1 = y and <nabla_g s^1> = 1

    psitmp = psi_mat(1:2, 1:2);
    phitmp = phi_mat(1:2, 1:2);
    deltatmp = delta_mat(1:2,1:2);
    s_mat = psitmp * phitmp;

    %DOING FIRST ITERATION
    i_iter = 1
    % we start with s1 = y and x1 = A^T s1;
    zhat_all(1,:) = s1; % original: y
    x_all(1,:) = A' * zhat_all(1,:)';
    mu_all(1) = 0;
    for i_iter0 = 1:2
        mattmp = phitmp * s_mat^(i_iter0-1);
        mu_all(1) = mu_all(1) + delta * freecum(i_iter0) * mattmp(2,1); % this is calculation of mu_1, equiv to (3.8)
    end
    %mu_all(1)
    sig2_all(1) = sum( x_all(1,:) .^2 ) / d - mu_all(1) ^ 2 * (distr.eta' * distr.probs) %since we know that Xt = mu* X + Wt, and X and Wt are indep
    omega_mat(1,1) = sig2_all(1);
    'sig2_all new:'
    omega_mat(1,1) = delta * freecum(1) * mean(y.^2) + delta * freecum(2) * distr.eta' * distr.probs
    sigma_mat(1,1) = freecum(1) * distr.eta' * distr.probs; %(3.8)
    [partials] = fkd_pseudoMem(x_all(1,:), inv( omega_mat(1,1) ), mu_all(1), distr);
    psi_mat(2,2) = mean(partials); % <nabla_1 hat(x)^1>
    xhat_all(1,:) = fk_pseudoMem( x_all(1,:), inv( omega_mat(1,1) ), mu_all(1), distr ); %use the denoiser ft(...)
    %freecum(1) * psi_mat(2,2)
    z_all(1,:) = A * xhat_all(1,:)' - freecum(1) * psi_mat(2,2) * zhat_all(1,:)'; % with t = 1, M_(t+1)^alpha = [ 0 0 ; 0 <nabla_1 hat(x)^1> ]
    
    %'b:'
    %freecum(1) * psi_mat(2,2)
    %norm( xhat_all(1,:) )
    %'norm( z_all(1,:) )'
    %norm( z_all(1,:) )

    %updating parameters of Gamma since now we have hat(x)^t
    gamma_mat(1,2) = sum( xhat_all(1,:) .^ 2 ) / d;
    gamma_mat(2,1) = sum( xhat_all(1,:) .^ 2 ) / d;
    gamma_mat(2,2) = sum ( xhat_all(1,:) .^ 2 ) / d; % since ft is conditional expectation and we can use towe property
    gammatmp = gamma_mat(1:2, 1:2);
    psi_tmp = psi_mat(1:2, 1:2);
    s_mat = psitmp * phitmp;
    sigmatmp = zeros(2,2);

    %calculating Theta_(t+2) since we have Gamma
    for t1 = 0:3
        pow  = zeros(2,2);
        for t2 = 0:t1
            pow = pow + ( s_mat ^ t2 ) * gammatmp * (s_mat')^(t1-t2); % (3.15) calculating E, first summand
        end
        pow2 = zeros(2,2);
        for t2 = 0:(t1-1)
            pow2  = pow2 + ( s_mat^t2 ) * psitmp * deltatmp * psitmp' * (s_mat')^(t1-t2-1); % (3.15) calculating E, second summand
        end
        sigmatmp = sigmatmp + freecum(t1+1) * (pow + pow2);
    end
    sigma_mat(1:2, 1:2) = sigmatmp;
    %cov( A*true_x, z_all(1,:)' )

    %with t = 1, denoising r^t so that we obtain s^(t+1)
    %sighat = sigma_mat(2,2) - sigma_mat(1,2) * sigma_mat(2,1) / sigma_mat(1,1); % (3.23) noise around R_(t-1)
    zhat_all(2,:) = gk_pseudoMem( z_all(1,:), y, sigma_mat(1:2, 1:2), distr ); % h2(r^1, y)

    %updating matrix Phi ( nabla_i hat(x)^j )
    phi_mat(3,1) = gkd_pseudoMem( z_all(1,:), y, sigma_mat(1:2, 1:2), distr, 2 );
    phi_mat(3,2) = gkd_pseudoMem( z_all(1,:), y, sigma_mat(1:2, 1:2), distr, 1 );
    %updating Delta ( E[S_i S_j] )
    delta_mat(3,3) = mean( zhat_all(2, :) .^ 2 );
    delta_mat(2,3) = mean( zhat_all(2,:) .* zhat_all(1,:) );
    delta_mat(3,2) = delta_mat(2,3);

    corr = [ ( xhat_all(1,:) * true_x ) / norm( xhat_all(1,:) ) / norm( true_x ) ];
    %norm( xhat_all(1,:)' - true_x(:) ) / norm(true_x)

    %STARTING WITH ITERATION t>=2
    for i_iter = 2:n_iter
        i_iter
        psitmp = psi_mat( 1:(i_iter+1), 1:(i_iter+1) );
        phitmp = phi_mat( 1:(i_iter+1), 1:(i_iter+1) );
        deltatmp = delta_mat( 1:(i_iter+1), 1:(i_iter+1) );
        gammatmp = gamma_mat( 1:(i_iter+1), 1:(i_iter+1) );
        s_mat = psitmp * phitmp;
        mattmp = zeros( i_iter+1, i_iter+1 );

        for j2 = 0:i_iter
            mattmp = mattmp + delta * freecum(j2+1) * phitmp * (s_mat^j2); %(3.5)/(3.19) M_(t+1)^beta
        end

        %updating mu, so that we can apply ft(x^1,..., x^t)
        mu_all(i_iter) = mattmp(i_iter+1,1);
        %getting parameters beta_(t,i)
        beta = mattmp(i_iter+1, 2:i_iter);
        %calculating x^t
        x_all(i_iter, :) = A' * zhat_all(i_iter, :)' - ( beta * xhat_all(1:(i_iter-1), :) )';
        q_mat = phitmp * psitmp;
        omegatmp = zeros(i_iter+1, i_iter+1);

        %calculating Omega_(t+1) which we need for denoising of x^t
        for t1 = 0:(2*i_iter) %because I have only (W1, ... Wt) -> 2*t
            pow = zeros(i_iter+1, i_iter+1);
            for t2 = 0:t1
                pow = pow + q_mat^t2 * deltatmp * (q_mat')^(t1-t2); 
            end

            pow2 = zeros(i_iter+1, i_iter+1);
            for t2 = 0:(t1-1)
                pow2 = pow2 + (q_mat^t2) * phitmp * gammatmp * phitmp' * (q_mat')^(t1-t2-1);
            end
            omegatmp = omegatmp + freecum(t1+1) * (pow + pow2);
        end
        omegatmp = delta * omegatmp;
        omega_mat(1:i_iter, 1:i_iter) = omegatmp( 2:(i_iter+1), 2:(i_iter+1) );
        sig2_all(i_iter) = omega_mat(i_iter, i_iter);
        
        if ( min( eig( omega_mat(1:i_iter, 1:i_iter) ) ) < 1e-9 )
            'omega smallest eig too small!'
            break;
        end

        %denoising x^t and obtaining hat(x)^t
        temp = mu_all(1:i_iter);
        xhat_all(i_iter, :) = fk_pseudoMem( x_all(1:i_iter,:), inv( omega_mat(1:i_iter, 1:i_iter) ), temp, distr );
        %updating matrix Psi since now we have new hat(x)^t
        partials = fkd_pseudoMem( x_all(1:i_iter,:), inv( omega_mat(1:i_iter, 1:i_iter) ), temp, distr );
        psi_mat(i_iter+1, 2:(i_iter+1)) = mean(partials, 2); %<nabla 1/2/.../t ft>
        psitmp = psi_mat(1:i_iter+1, 1:i_iter+1);

        corr = [ corr, ( xhat_all(i_iter,:) * true_x ) / norm( xhat_all(i_iter,:) ) / norm( true_x ) ];

        %calculating alphas, so that we can calculate r^t
        q_mat = phitmp * psitmp;
        mattmp2 = zeros(i_iter+1, i_iter+1);

        for j2=0:(i_iter+1)
            mattmp2 = mattmp2 + freecum(j2+1) * psitmp * (q_mat^j2);
        end
        alpha = mattmp2(i_iter+1, 2:(i_iter+1));
        
        %finding r^t
        z_all(i_iter, :) = A * xhat_all(i_iter, :)' - ( alpha * zhat_all(1:i_iter, :) )';
        
        
        %updating Gamma, which we need for updating Omega ( covariance matrix for problem of denoising r^1,..., r^t )
        gamma_mat(1,i_iter+1) = mean( xhat_all(i_iter, :) .^ 2 ); %because ft is the conditional expe and we can use tower property
        gamma_mat(i_iter+1, 1) = gamma_mat(1, i_iter+1);
        gamma_mat(i_iter+1, i_iter+1) = gamma_mat(1,i_iter+1);
        for j2 = 2:(i_iter+1)
            gamma_mat(j2, i_iter+1) = mean( xhat_all(i_iter, :) .* xhat_all(j2-1, :) );
            gamma_mat(i_iter+1, j2) = gamma_mat(j2, i_iter+1);
        end
        gammatmp = gamma_mat(1:(i_iter+1), 1:(i_iter+1));
        psitmp = psi_mat(1:(i_iter+1), 1:(i_iter+1));

        %updating Sigma
        s_mat = psitmp * phitmp;
        sigmatmp = zeros(i_iter+1, i_iter+1);
        for t1 = 0:(2*i_iter+1)
            pow = zeros(i_iter+1, i_iter+1);
            for t2 = 0:t1
                pow = pow + (s_mat^t2) * gammatmp * (s_mat')^(t1-t2);
            end

            pow2 = zeros(i_iter+1, i_iter+1);
            for t2 = 0:(t1-1)
                pow2 = pow2 + (s_mat^t2) * psitmp * deltatmp * psitmp' * (s_mat^(t1-t2-1));
            end
            sigmatmp = sigmatmp + freecum(t1+1) * (pow + pow2);
        end
        sigma_mat( 1:(i_iter+1), 1:(i_iter+1) ) = sigmatmp(:,:);
        %cov( [ A*true_x, z_all(1:i_iter, :)'] )
        if ( min( sigma_mat( 2:(i_iter+1) , 2:(i_iter+1) ) ) < 1e-9 )
            'sigma smallest eig too small!'
            break;
        end

        %sighat = sigma_mat(1,1) - sigma_mat(1, 2:(i_iter+1)) * inv( sigma_mat(2:(i_iter+1), 2:(i_iter+1)) * sigma_mat(2:(i_iter+1), 1) ); %noise variance arounf r^(t-1)
        %if sighat < 0
        %    'sighat < 0!'
        %    break;
        %end

        %calculating s^(t+1)
        zhat_all(i_iter+1, :) = gk_pseudoMem( z_all(1:(i_iter), :), y, sigma_mat( 1:(i_iter+1), 1:(i_iter+1) ) , distr ); %because we started by storing s1 in zhat_all
        phi_mat(i_iter+2, 1)= gkd_pseudoMem( z_all(1:(i_iter), :), y, sigma_mat( 1:(i_iter+1), 1:(i_iter+1) ), distr, i_iter+1 ); % <nabla_g s^t> ... everything shifted by 2
        for ind = 2:(i_iter+1)
            phi_mat(i_iter+2, ind) = gkd_pseudoMem( z_all(1:(i_iter), :), y, sigma_mat( 1:(i_iter+1), 1:(i_iter+1) ), distr, ind-1 );
        end

        delta_mat(i_iter+2, i_iter+2) = mean( zhat_all(i_iter+1, :) .^ 2 );
        for j2 = 1:i_iter
            delta_mat(j2+1, i_iter+2) = mean( zhat_all(i_iter+1,:) .* zhat_all(j2, :) );
            delta_mat(i_iter+2, j2+1) = delta_mat(j2+1, i_iter+2);
        end 
    end
end

function [cumulants] = compute_cum(A, numbC)

    %returns free cumulants indeced by even numbers
    [N,M] = size(A);
    delta = N / M;
    %lambdas = eig(A*A'); %to be further optimised in the future
    %cov_mat = A*A';
    %AAT = eye(N);
    [mk, mk1, mk_fast, cumulants] = deal([]);
    L = 100;
    s = normrnd( 0, 1, M, L );
    s = A*s;

    for k = 1 : numbC
        %mk = [ mean( lambdas .^ k ), mk ];
        %AAT = AAT * cov_mat;
        %mk = [ trace(AAT) / N , mk]
        mk = [ mean( sum(s .^ 2) / N ), mk]
        if mod(k,2) == 1
            s = A' * s;
        else
            s = A * s;
        end
        value = mk(1);
        poly = poly2sym( [0] );
        Mpoly = poly2sym( [mk 0] );
        for j = 1 : (k - 1)
            poly = poly + cumulants(j) * ( poly2sym( [1 0] ) * ( delta * Mpoly + 1 ) * ( Mpoly + 1 ) )^(j);
        end
        poly_coeff = flip(sym2poly(poly));
        if k ~= 1
            value = value - poly_coeff(k+1);
        end
        cumulants = [ cumulants, value ];
    end
    cumulants = double(cumulants);
end
    

function [val] = fk_pseudoMem(beta_val, iDelta, mu, distr)
    % dim(mu) = 1 x i_iter
    % dim(beta_val) = i_iter x d
    % dim(iDelta) = i_iter x i_iter

    mu = mu(:);
    probs = distr.probs;
    eta = distr.eta;
    K = size(probs,1);
    [A, B] = deal(0);
    max_eta = max(eta);

    for i = 1:K
        expe = exp( (mu' * iDelta * beta_val).^2 / 2  * ( eta(i) - max_eta ) / (mu' * iDelta * mu * eta(i) + 1) / (mu' * iDelta * mu * max_eta + 1 ) );
        term = probs(i)  /  sqrt( mu' * iDelta * mu * eta(i) + 1 ) * expe;
        A = A + term / ( mu' * iDelta * mu * eta(i) + 1 ) .* ( mu' * iDelta * beta_val * eta(i) );
        B = B + term;

        if sum(isnan(term))  || sum(term > 1e50)
            (mu' * iDelta * beta_val)^2 / 2  * ( eta(i) - max_eta ) / (mu' * iDelta * mu * eta(i) + 1) / (mu' * iDelta * mu * max_eta + 1 )
            error(1)
        end
    end
    val = A ./ B;

    %dim(val) = 1 x d
end


function [val] = fkd_pseudoMem(beta_val, iDelta, mu, distr)
    % val is a vectors of partial derivatives
    % dim(mu) = 1 x i_iter
    % dim(beta_val) = i_iter x d
    % dim(iDelta) = i_iter x i_iter

    mu = mu(:);    
    probs = distr.probs;
    eta = distr.eta;
    K = size(probs,1);
    [A, B, Ap, Bp] = deal(0);
    max_eta = max(eta);

    for i = 1:K
        expe =  exp( (mu' * iDelta * beta_val).^2 / 2  * ( eta(i) - max_eta ) / (mu' * iDelta * mu * eta(i) + 1) / (mu' * iDelta * mu * max_eta + 1 ) );
        term = probs(i)  / sqrt( mu' * iDelta * mu * eta(i) + 1 ) * expe;
        A = A + term / ( mu' * iDelta * mu * eta(i) + 1 ) .* (mu' * iDelta * beta_val * eta(i) );
        B = B + term;
        jth_row = mu'*iDelta;
        %jth_row = jth_row(1,end);
        Ap = Ap + probs(i) * expe .* ( eta(i)^2 * (mu' * iDelta * beta_val).^2 / (mu' * iDelta * mu * eta(i) + 1 )^(5/2) .* jth_row' + jth_row' * eta(i) / (mu' * iDelta * mu * eta(i) + 1 )^(3/2) );
        Bp = Bp + probs(i) * expe .* ( eta(i) * mu' * iDelta * beta_val ) / (mu' * iDelta * mu * eta(i) + 1 )^(3/2) .* jth_row';
    end
    val = (Ap .* B - A .* Bp) ./ (B.^2);

end

function [val]  = gk_pseudoMem(r, y, Sigma, distr)

    % r should have dim = i_iter x n
    % dim(Sigma) = i_iter x i_iter
    % dim(y) = n x 1

    %size(Sigma)
    %Sigma
    %size(y)
    %size(r)

    t = size(Sigma, 1) - 1;
    probs = distr.probs;
    eta = distr.eta;
   
    S = zeros(t+2, t+2);
    S(1:(t+1), 1:(t+1)) = Sigma;
    S(1:(t+1), t+2) = Sigma(1:(t+1), 1);
    S(t+2, 1:(t+1)) = Sigma(1, 1:(t+1));
    S(t+2, t+2) = mean( y .^ 2 );
    vec = [r ; y(:)'];

    val =  S(1, 2:(t+2)) * inv( S( 2:(t+2), 2:(t+2) ) ) * vec - Sigma(1, 2:(t+1)) * inv( Sigma(2:(t+1), 2:(t+1)) ) * r ;
    %d = S(1, 2:(t+2)) * inv( S( 2:(t+2), 2:(t+2) ) );
    %d= d(1);
    %c = 1 / (1 - d); 
    %val = val * c;
    %val should be of dim = 1 x d
    %val = (y - r(end,:)')';

end


function [val]  = gkd_pseudoMem(r, y, Sigma, distr, ith)

    %r should have dim = i_iter x n
    % dim(Sigma) = i_iter x i_iter
    % dim(y) = n x 1

    %size(Sigma)
    %Sigma
    %size(y)
    %size(r)

    t = size(Sigma, 1) - 1;
    probs = distr.probs;
    eta = distr.eta;

    S = zeros(t+2, t+2);
    S(1:(t+1), 1:(t+1)) = Sigma;
    S(1:(t+1), t+2) = Sigma(1:(t+1), 1);
    S(t+2, 1:(t+1)) = Sigma(1, 1:(t+1));
    S(t+2, t+2) = mean(y .^ 2);

    if ith == (t+1) % dim(Sigma) = (t+1) x (t+1)
        vec = zeros(t+1,1);
        vec(t+1,1) = 1;
        val = S(1, 2:(t+2)) * inv( S(2:(t+2), 2:(t+2)) ) * vec;
        %val = 1;
    else
        vec = zeros(t,1);
        vec(ith,1) = 1;
        %ith
        %size(S(1, 2:(t+2)))
        %size(inv( S(2:(t+2), 2:(t+2)) ))
        %size([vec; 0])
        val = S(1, 2:(t+2)) * inv( S(2:(t+2), 2:(t+2)) ) * [vec; 0] - Sigma(1, 2:(t+1)) * inv( Sigma(2:(t+1), 2:(t+1)) ) * vec;
        %{
        if ith == t
            val = -1;
        else
            val = 0;
        end
        %}
    end
    %val should have dim = 1 x 1

end
