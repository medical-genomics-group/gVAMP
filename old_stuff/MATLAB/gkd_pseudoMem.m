function [val]  = gkd_pseudoMem(it, theta, y_val, Gamma_t, iGamma_t1, idx_lin_indep1, iGamma_t2, idx_lin_indep2, probs, eta)

    val1 =  Gamma_t(1, 1 + [idx_lin_indep1]) * iGamma_t1;
    val2 =  Gamma_t(1, 1 + [idx_lin_indep2]) * iGamma_t2;

    val = val1(end-1) - val2(end);

end