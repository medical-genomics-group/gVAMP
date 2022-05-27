function [cumulants, mk] = free_cumulants_calc(X, numbC)

    N = size(X,1);
    M = size(X,2);
    delta = N / M;

    lambdas = eig(X*X'); 

    mk = [];
    cumulants = [];

    for k = 1 : numbC

        mk = [ mean( lambdas .^ k ), mk ];

        value = mk(1);

        poly = poly2sym( [0] );

        Mpoly = poly2sym( [mk 0] );

        for j = 1 : (k - 1)

            poly = poly + cumulants(j) * ( poly2sym( [1 0] ) * ( delta * Mpoly + 1 ) * ( Mpoly + 1 ) )^(j);

        end

        %poly = poly2sym(double(sym2poly(poly)))

        %poly_c = sym2poly(poly);

        %double(poly_c(end));

        poly_coeff = flip(sym2poly(poly));

        if k ~= 1

            value = value - poly_coeff(k+1);
            
        end

        cumulants = [ cumulants, value ];

    end

    cumulants = double(cumulants);

    t = tiledlayout(1,1);
    nexttile
    title(t, ['cumulants of X(', 'M = ', num2str(M), ', N = ', num2str(N), ')']);
    semilogy( 2*( 1:size(cumulants,2) ), cumulants, 'bo' );
    hold on;
    yline(1 / delta, 'r')
    exportgraphics(t, 'free_cumulants_X.jpg')

end

%{
    mk(2) - (1 + delta) * mk(1)^2

%}