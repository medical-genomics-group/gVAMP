#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

int iterNumb = 11;
struct prior_info{
    int K;
    VectorXd pi;
    VectorXd eta;
};
const int delta = 3;
const int p = 100;
const int n = delta*p;
double sigma_noise = 0.005;
std::default_random_engine generator;

//const int n = 200;

// defining mixture groups 
    int K = 3;
    VectorXd eta =VectorXd::Zero(K);
    VectorXd pi =VectorXd::Zero(K);
    struct prior_info prior;
    
double sigma;
double muk;

// defining function fk from the AMP algorithm
double fk(double y) 
{
    double pk = 0, pkd = 0;
    if (sigma != 0)
    {
        for (int i = 0; i < K; i++)
        {
            double z = exp( - y*y / (eta(i) + sigma) / 2 );
            pk += pi(i) / sqrt(eta(i) + sigma) * z;
            pkd -= pi(i) * pow(eta(i) + sigma,-3/2) * z * y;
        }
        //cout << "pk: "<< pk << ", pkd: " << pkd <<", sigma: " << sigma << endl;
        double tmp = y + sigma*pkd/pk;
        if (isnan(tmp)==0)
        {
            return tmp/muk;
        }
        else
        {
            return y/muk;
        }
    }
    else
    {
        return y/muk;
    }
}

//derivative of fk
double fkd(double y) 
{
    double pk = 0, pkd = 0, pkdd = 0;
    if (sigma != 0)
    {
        for (int i = 0; i < K; i++)
        {
            double z = pi(i) / sqrt(eta(i) + sigma) * exp( - y*y / (eta(i) + sigma) / 2 );
            pk += z;
            z = z / (eta(i) + sigma) * y;
            pkd -= z;
            z = z / (eta(i) + sigma) * y;
            pkdd += z;
        }
        double tmp = 1 + sigma*( pkdd/pk - pow(pkd/pk,2) );
        if (isnan(tmp)==0)
        {
            return tmp/muk;
        }
        else
        {
            return 1/muk;
        }
    }
    else
    {
        return 1/muk;
    }
}


double generating_mixture_gaussians(int K_grp, VectorXd eta_gen, VectorXd pi_gen)
{
    std::uniform_real_distribution<double> unif(0.0,1.0);
    double u = unif(generator);
    double c_sum = 0;
    double out_val = 0;
    for (int j=0; j<K_grp; j++)
    {   
        c_sum += pi_gen(j);
        if (u <= c_sum)
        {
            if (eta(j) != 0)
            {
                std::normal_distribution<double> gauss_beta_gen(0.0,sqrt(eta_gen(j))); //2nd parameter is stddev
                out_val = gauss_beta_gen(generator);
            }
            else
            {
                out_val = 0;
            }
            break;
        }
    }
    return out_val;
}


VectorXd KPM(VectorXd points, int num_points, int M_deg,  int Ntot, int M, MatrixXd X, double scaling)
{
    VectorXd xi = VectorXd::Zero(M_deg);
    VectorXd muk = VectorXd::Zero(M_deg);
    int n_vec = 150;

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
        cout << "hej" << endl;
        for (int k=0; k<M_deg; k++)
        {
            cout << "hej_1" << endl;
            xi(k) = xi(k) +  v0.dot(vk);
            if (k!=0)
            {
                VectorXd vk_tmp = vk;
                vk = 2*(X.transpose())*X*vk;
                vk = vk / scaling / scaling - vk_prev;
                vk_prev = vk_tmp;
            }
            else
            {
                vk_prev = vk;
                vk = (X.transpose())*X*vk;
                vk = vk / scaling/scaling;
            }
        }
        
    }
    cout << "hej2 " << endl;
    xi = xi / n_vec;
    for (int k=0; k<M_deg; k++)
    {
        int deltak0 = 0;
        if (k == 0)
            deltak0 = 1;
        muk(k) = xi(k) * (2 - deltak0) / M / 3.14159265359;  
    }


    // cout<< "muk: " << muk << endl;
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
            values(point_ind) = values(point_ind) + muk(k)*Tkt;
        }      
        values(point_ind) = values(point_ind) / sqrt(1-points(point_ind)*points(point_ind));
    }
    return values;
}

// inference procedure
VectorXd infere_AMP(VectorXd y, MatrixXd X, int iterNumb, VectorXd beta0, double b0) 
{
    VectorXd r_prev =VectorXd::Zero(n);
    VectorXd r = r_prev;
    VectorXd beta = beta0;
    VectorXd beta_tmp = beta0;
    VectorXd beta_out = beta0;
    double sigma_analytic = -1;
    double b = b0;
    sigma_analytic = pow(sigma_noise,2) + eta.dot(pi)/delta;
    for (int it = 0; it < iterNumb; it++)
    {
        //estimation of sigma
        //cout << "sigma "<< sigma;
        //iteration step
        r = y - X*beta + b*r_prev;
        sigma = r.squaredNorm()/n;
        //sigma = pow(r.lpNorm<2>(),2)/n;

        // cout << "(beta_v.squaredNorm() / M: "<< beta.squaredNorm() / p << endl;
        // cout << "pi.dot(eta): "<< pi.dot(eta) << endl;
        
        // cout<< "sigma: "<<sigma<<endl;

        if (it>=1)
        {
            muk = sqrt( abs(beta_tmp.squaredNorm() / p - sigma) /  pi.dot(eta) );
        }
        else
            muk = 1;

        cout << "sigma: "<<sigma << ", sigma_analytic: "<< sigma_analytic << endl;
        cout << "muk: " << muk << endl;
        //muk = 1;
        //cout<< "r: "<<r<<endl;
        beta += X.transpose()*r; 
        beta_tmp = beta;
        //cout << "curr beta: " << endl << beta<< endl;
        //updating analytic state evolution parameters
        if (K==1)
        {
            //cout << "update sigma_analytic"<< endl;
            // sigma_analytic = pow(sigma_noise, 2) + sigma_analytic / delta / pow(eta(0) + sigma_analytic,2) * (eta(0)*eta(0) + pow(sigma_noise,2) * sigma_analytic + 4*eta(0)*sigma_analytic + 4*pow(sigma_analytic,2)  ); //sigma_noise is stddev
            // sigma_analytic = pow(sigma_noise, 2) + sigma_analytic * (pow(eta(0),2) + pow(sigma_noise,2)*sigma_analytic) / pow(eta(0)+ sigma_analytic,2);
            sigma_analytic = pow(sigma_noise,2) + sigma_analytic * eta(0) / (eta(0) + sigma_analytic)/delta;
        }
        else
        {
            //we perform MCMC sampling to calculate E[(beta - fk(beta + sigma_k * G))]
            int MCMCNoSamples = 12000;
            std::normal_distribution<double> standard_normal(0.0,1);
            VectorXd evolution_sigma_samples(MCMCNoSamples);
            for (int j = 0; j < MCMCNoSamples; j++)
            {
                    double g = standard_normal(generator);
                    double beta_tilde = generating_mixture_gaussians(K, eta, pi);
                    evolution_sigma_samples(j) = pow( beta_tilde - fk(beta_tilde + sqrt(sigma_analytic)*g), 2 );
            }
            sigma_analytic = evolution_sigma_samples.mean()/delta + pow(sigma_noise,2);
        }

        beta_out = beta;
        beta = beta.unaryExpr(&fk);
        //cout << "curr beta2: " << endl << beta<< endl;
        b = beta_tmp.unaryExpr(&fkd).mean() * p / n;  
        //cout<< "b: "<<b<<endl; 
        r_prev = r;      
    }
    return beta_out;
}

int main() { 

    eta(0) = 0.01;
    eta(1) = 0.001;
    eta(2) = 0.1;
    pi(0) = 1;
    pi(1) = 1;
    pi(2) = 0.1;

    pi = pi/pi.sum();
    
    prior.K = K;  
    prior.pi = pi;
    prior.eta = eta;


    cout << "eta: "<< eta << endl;
    cout << "pi: " << pi << endl;
    //generating observations
    MatrixXd X(n,p);
    MatrixXd Xd(n,p);

    //generating X
    std::default_random_engine generator;
    // std::normal_distribution<double> gauss(0.0,1.0/sqrt(n));
    std::normal_distribution<double> gauss(0.0,1.0);
    for (int i=0; i<n; i++) 
    {
        for (int j=0; j<p; j++)
        { 
            X(i,j) = gauss(generator);
        }
    }

    //generating discrete X
     std::random_device rd;
     std::mt19937 gen(rd());
     std::discrete_distribution<> d({90, 8, 2});
     for (int i=0; i<n; i++) 
    {
        for (int j=0; j<p; j++)
        { 
            Xd(i,j) = d(gen);
        }
    }

    //cout << Xd << endl;

    RowVectorXd mean = Xd.colwise().mean();
    RowVectorXd var = (Xd.rowwise() - mean).array().square().colwise().mean();
    Xd = (Xd.rowwise() - mean).array().rowwise() / sqrt(var.array());
    //Xd = Xd / sqrt(n);
    // Xd = Xd/ sqrt(n); //normalization of Xd

    //cout << Xd<< endl;
    //VectorXd means(p);
    //VectorXd sds(p);
    //for (int j=0; j<p; j++)
    //{
    //    means(j) = Xd.col(j).mean();
    //    //sds(j) = Xd.col(j).stddev();
    //}
    //cout << means << endl;
    //cout << Xd.colwise() - means;

    int num_points = 100;
    VectorXd points = VectorXd::Zero(num_points);
    for (int point_ind=0; point_ind<num_points; point_ind++)
    {
        points(point_ind) = 0 + 1.0 / num_points * point_ind;
    }
    int M_deg = 40;
    double scaling = 10000.0;
    scaling = sqrt(n)*2;
    VectorXd DOS = VectorXd::Zero(num_points);
    // DOS = KPM(points, num_points, M_deg,  n, p, X, scaling); //DOS= Density Of States
    cout << DOS << endl;

    

    EigenSolver<MatrixXd> es(Xd.transpose()*Xd);
    VectorXd z = es.eigenvalues().real();
    cout << z.transpose() << endl;
    cout << "second cumulant est:" << z.mean() << endl;

    X = Xd / sqrt(z.mean());


    

    //X = X / sqrt(z.mean());

    //generating true beta
    std::uniform_real_distribution<double> unif(0.0,1.0);
    VectorXd beta_true(p);
    for (int i=0; i<p; i++)
    {
        double u = unif(generator);
        double c_sum = 0;
        for (int j=0; j<K; j++)
        {   
            c_sum += pi(j);
            if (u <= c_sum)
            {
                if ( eta(j) != 0)
                {
                    std::normal_distribution<double> gauss_beta(0.0,sqrt(eta(j))); //2nd parameter is stddev
                    beta_true(i) = gauss_beta(generator);
                }
                else
                {
                    beta_true(i) = 0;
                }
                break;
            }
        }
    }
    //cout<< "true beta: "<< endl << beta_true << endl;
    //beta_true = beta_true.array() -100;

    //generating noise
    VectorXd noise(n);
    std::normal_distribution<double> gauss_noise(0.0,sigma_noise);
    for (int i=0; i<n; i++) 
    {
       noise(i) = gauss_noise(generator);
    }
    //noise = VectorXd::Zero(n);
    //cout<<noise<<endl;
    
    //generating y
    VectorXd y(n);
    y = X*beta_true + noise;

    //initialization of the algorithm
    VectorXd beta0(p);
    for (int i = 0; i<p; i++) 
        beta0(i) = 0; 

    //spectral initialization
    // MatrixXd D = y.asDiagonal();
    //cout<< D<< endl;
    // Eigen::EigenSolver<MatrixXd> es(p);
    // es.compute(X.transpose()*D*X);
    // beta0= es.eigenvectors().col(0).real();

    VectorXd noise_beta0(p);
    std::normal_distribution<double> gauss_noise_beta0(0.0,0.001);
    for (int i=0; i<p; i++) 
    {
       noise_beta0(i) = gauss_noise_beta0(generator);
    }

    //beta0 = beta_true + noise_beta0;
    //cout << "evect" << es.eigenvectors().col(0).real() << endl;
    //cout<< "beta0: " << beta0<< endl;

    cout << "corr(beta_true, beta_v): " << beta_true.dot(beta0) / beta_true.lpNorm<2>() / beta0.lpNorm<2>() << endl; 

    //beta estimator
    VectorXd beta_est;
    beta_est = infere_AMP(y,X,iterNumb,beta0,0);
    //cout << "beta estimator: "<< endl << beta_est << endl;
    cout << "sigma: " << sigma << endl;
    

    //least squres solution
    //VectorXd beta_LS = X.colPivHouseholderQr().solve(y);

    //cout << "l2 error: "<< (beta_true - beta_est).lpNorm<2>() << endl;
    cout << "l2 prediction error: "<< (y - X*beta_est).lpNorm<2>() << endl;
    cout << "l2 prediction error normalized (beta_true): "<< (y - X*beta_est).lpNorm<2>() / (y - X*beta_true).lpNorm<2>() << endl;
    //cout << "l2 prediction error normalized (beta_LS): "<< (y - X*beta_est).lpNorm<2>() / (y - X*beta_LS).lpNorm<2>() << endl;
    //cout << "l2 beta_est norm 2: "<< beta_est.lpNorm<2>() << endl;
    cout << "y_phen abs mean: "  << y.array().abs().mean() << endl;
    cout << "X*beta_true abs mean: "  << (X*beta_true).array().abs().mean() << endl;
    cout << "l2 prediction error / sqrt(Ntot1) : "<< (y - X*beta_est).lpNorm<2>() / sqrt(n) << endl;
    cout << "||beta-muk*beta_true||/ ||beta_true||: "<< (muk*beta_true - beta_est).lpNorm<2>() / beta_true.lpNorm<2>() << endl;
    //cout << "||beta-beta_LS||/ ||beta_LS||: "<< (beta_true - beta_LS).lpNorm<2>() / beta_LS.lpNorm<2>() << endl;
    cout << "||beta-muk*beta_true||/ sqrt(p): "<< (muk*beta_true - beta_est).lpNorm<2>() / sqrt(p) << endl;
    VectorXd y_mean = VectorXd::Ones(n); 
    y_mean = y_mean * y.mean();
    cout << "R2: " << 1 - (y - X*beta_est).array().square().sum()/(y - y_mean).array().square().sum() << endl;

    // std::vector<double> w = y_mean.data();
    // cout << "w: "<< w<< endl;
}

