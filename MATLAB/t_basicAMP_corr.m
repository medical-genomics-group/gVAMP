format long

M = 551;
delta = 3;
N = ceil(delta * M);
iterNumb = 10;

%setting up a correlation for genotype matrix
pair_corr_bound = 0.00; %expected correlation between each pair is pair_corr_bound / 2

rng(1) % for reproducibility
Rho = unifrnd(0,pair_corr_bound, M, M);
for i = 1:M
    Rho(i,i) = 1;
end
Rho = (Rho + Rho')/2;
U = copularnd('Gaussian', Rho, N);

%alternative model for corrlation - two variables at distance d have correlation at
%most 1/d

%Rho = unifrnd(0, pair_corr_bound, M, M);
for i = 1:M
    for j = 1:M
        if i ~= j
            div = abs( i - j );
            %Rho(i,j) = unifrnd(0, pair_corr_bound / div , 1, 1);
        end
    end
end
for i = 1:M
    Rho(i,i) = 1;
end
Rho = (Rho + Rho')/2;
U = copularnd('Gaussian', Rho, N);




%generating genotype matrix X
probs = [0.85 0.08 0.07]';
cumprobs = cumsum(probs);
vals = [0 1 2]';
X = zeros(N,M);
for i = 1:N
    for j = 1:M
        X(i,j) = vals ( min( find( (cumprobs >= U(i,j)) == 1) ) );
        %X(i,j) = normrnd( 0, 0.005 );
    end
end

means = mean(X);
stds = std(X);

for i = 1:N
    for j = 1:M
        X(i,j) = (X(i,j) - means(j)) / stds(j);
    end  
end

%calculation of the second free cumulant
[V,D] = eig(X'*X);
lambdas = diag(D);
sec_cumulant = mean(lambdas)
sqrt(N)
sqrt(sec_cumulant)

figure(1)
histogram(lambdas, floor(M/4));
%xlim([250 2700])

%renormalizing genotype matrix
X = X / sqrt(sec_cumulant);
%X = X * mean(abs(beta_true));
corr(X);



%eta_signal = [0.01 0.0001]';
eta_signal = [0.1 1]';
probs_signal = [0.1 0.9]';
probs_zero = 0.6;
probs_final = [ probs_zero; (1-probs_zero) * probs_signal ];
eta_final =  [ 0; eta_signal ];
b0 = 0;
beta0 = zeros(M,1);

%generating marker values
noise = normrnd( 0, 0.05, N, 1 ); %second argument is std dev
sigma_sig = zeros(1, M, size(eta_signal,1));
sigma_sig(1,:,:) = repmat(eta_signal, 1, M)';
gm = gmdistribution( zeros( size(eta_signal,1), M ), sigma_sig, probs_signal' );
beta_true = random(gm)';
beta_true = binornd(1,1-probs_zero, M, 1) .* beta_true;
y = X * beta_true + noise;


[beta_out, sigma_out, muk_out] = f_infere_AMP(y,X,iterNumb, beta0, b0, N, M, eta_final, probs_final, @fk, @fkd, beta_true);

'final l2 error:'
norm(y-X*beta_out) / norm(y-X*beta_true)

'final corr:'
beta_true' * beta_out / norm( beta_true ) / norm( beta_out )

(sigma_out/muk_out)^2

max(lambdas) - min(lambdas)

max(lambdas) / min(lambdas)
