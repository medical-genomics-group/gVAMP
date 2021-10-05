#include "KPM_Chebyshev.hpp"

VectorXd KPM(VectorXd points, int num_points, int M_deg, size_t *N1S, size_t *N1L, size_t *N2S, size_t *N2L, double *mstd, double *mave, uint *I1, uint *I2, int Ntot, int M, double scaling)
{
    VectorXd xi = VectorXd::Zero(M_deg);
    VectorXd muk = VectorXd::Zero(M_deg);
    int n_vec = 25;

    std::default_random_engine generator;
    std::normal_distribution<double> gauss01(0.0,1.0); //2nd parameter is stddev
    for (int l=0; l<n_vec; l++)
    {
        VectorXd v0  = VectorXd::Zero(M);
        VectorXd vk_prev  = VectorXd::Zero(M);
        for (int i=0; i<M; i++)
        {
            v0(i) = gauss01(generator);
        }
        VectorXd vk  = v0;

        for (int k=0; k<M_deg; k++)
        {
            // cout << "hej_1" << endl;
            xi(k) = xi(k) +  v0.dot(vk);
            if (k!=0)
            {
                VectorXd vk_tmp = vk;
                VectorXd tmpNtot = VectorXd::Zero(Ntot);
                tmpNtot = xbeta_mult(vk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scaling);
                vk = 2* xtr_mult(tmpNtot, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scaling) - vk_prev;
                vk_prev = vk_tmp;
            }
            else
            {
                vk_prev = vk;
                VectorXd tmpNtot = VectorXd::Zero(Ntot);
                tmpNtot = xbeta_mult(vk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scaling);
                vk = xtr_mult(tmpNtot, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scaling);
            }
        }
        
    }

    xi = xi / n_vec;
    for (int k=0; k<M_deg; k++)
    {
        int deltak0 = 0;
        if (k == 0)
            deltak0 = 1;
        muk(k) = xi(k) * (2 - deltak0) / M / PI;  
    }


    VectorXd values = VectorXd::Zero(num_points);
    for (int point_ind=0; point_ind<num_points; point_ind++)
    {
        double Tkt;
        double Tkt_prev;
        for (int k=0; k<M_deg; k++)
        {
            if (k==0)
            {
                Tkt = 1.0;
            }
            else if (k==1)
            {
                Tkt = points(point_ind);
                Tkt_prev = 1.0;
            }
            else
            {
                double Tkt_tmp = Tkt;
                Tkt = 2*Tkt*points(point_ind) - Tkt_prev;
                Tkt_prev = Tkt_tmp;
            }
            double alphaM = PI / (M_deg + 2);
            double g_kM = (1.0 - k/(M+2))* sin(alphaM) + 1.0/(M_deg+2)*cos(alphaM)*sin(alphaM);
            g_kM /= sin(alphaM);
            values(point_ind) = values(point_ind) + muk(k)*g_kM*Tkt;
        }      
        values(point_ind) = values(point_ind) / sqrt(1-points(point_ind)*points(point_ind));
    }
    return values;
}