#include "BayesRRm.h"

static inline int isnan_real(float f);

double fkd ( double y,  double sigma, double muk, int K_groups, VectorXd probs, VectorXd eta );

double fk ( double y, double sigma, double muk, int K_groups, VectorXd probs, VectorXd eta );