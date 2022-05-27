#ifndef SRC_BAYESW_ARMS_H_
#define SRC_BAYESW_ARMS_H_

#include "distributions_boost.hpp"

int arms (double *xinit,
          int    ninit,
          double *xl,
          double *xr,
          double (*myfunc)(double x, void *mydata),
          void   *mydata,
          double *convex,
          int    npoint,
          int    dometrop,
          double *xprev,
          double *xsamp,
          int    nsamp,
          double *qcent,
          double *xcent,
          int    ncent,
          int    *neval,
          Distributions_boost &dist);

double expshift(double y, double y0);

//#define YCEIL 50.                /* maximum y avoiding overflow in exp(y) */

const double YCEIL = 50.0;

#endif /* SRC_BAYESW_ARMS_H_ */
