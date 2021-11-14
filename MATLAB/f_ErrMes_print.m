function  [] = f_ErrMes_print(beta_true, beta_out, y, X, muk_out, sigma_out, eta_final)

'final l2 signal error:'
norm(beta_out - beta_true) / norm(beta_true)

'final l2 prediction error:'
norm(y-X*beta_out) / norm(y-X*beta_true)

'final corr:'
beta_true' * beta_out / norm( beta_true ) / norm( beta_out )

if nargin > 4
    'ratio measure:'
    sigma_out(end)/(muk_out^2)

    'sqrt ( sigma_out / max(eta_final) ): '
    sqrt( sigma_out / max(eta_final) )
end

end