#include "denoiser.hpp"

static inline int isnan_real(float f)
{
    union { float f; uint32_t x; } u = { f };
    return (u.x << 1) > 0xff000000u;
}


// defining function fk from the AMP algorithm
double fk ( double y, double sigma, double muk, int K_groups, VectorXd probs, VectorXd eta ) 
{
    double pk = 0, pkd = 0;

    if (muk == 0)

        return 0;

    if (sigma == 0)

        return y/muk;

    for (int i = 0; i < K_groups; i++)
    {
        double z = probs(i) / sqrt(muk * muk * eta(i) + sigma) * exp( - y*y / (muk * muk * eta(i) + sigma) / 2 );
        
        pk += z;
        
        z = z / (muk * muk * eta(i) + sigma) * y;
        
        pkd -= z;
    }

    double tmp = (y + sigma*pkd/pk) / muk;

    if (isnan_real(tmp)!=0)
    {
        return y/muk;
    }
 
    return tmp;
}




//derivative of fk (fkd) -> this can be further improved in a sense that we dont need to calculate pk and pkd separately for fk and fkd, TODO
double fkd ( double y,  double sigma, double muk, int K_groups, VectorXd probs, VectorXd eta ) 
{
    double pk = 0, pkd = 0, pkdd = 0;

    if (muk == 0)

        return 0;

    if (sigma == 0)

        return 1.0/muk;

    for (int i = 0; i < K_groups; i++)
    {
        double z = probs(i) / sqrt( muk * muk * eta(i) + sigma) * exp( - y*y / ( muk * muk * eta(i) + sigma ) / 2 );

        pk += z;

        z = z / (muk * muk * eta(i) + sigma) * y;

        pkd -= z;

        z = z / (muk * muk * eta(i) + sigma) * y;

        pkdd += z;
    }

    double tmp = ( 1.0 + sigma * ( pkdd/pk - pow(pkd/pk,2) ) ) / muk;
    
    if (isnan_real(tmp)!=0)

        return 1.0 / muk;

    return tmp;
    
}