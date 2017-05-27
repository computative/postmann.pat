#include <iostream>
#include <string>
#include <armadillo>
#include <random>
#include <mpi.h>
#include "basis.h"
#include "maths.h"

using namespace std;
using namespace arma;

// definere globale variable fordi jeg ønsker det.
int n = 20;
int iterations = pow(2,12);
double w = 0.1;


basis B = basis(5);
mat c = ones<mat>(n,n);

random_device rd;mt19937 gen(rd()); uniform_int_distribution<int> rand_particle(0, n-1);
uniform_int_distribution<int> rand_bool(0, 1);uniform_real_distribution<double> rand_double(0.0,1.0);

mat D(mat r, int spin, double a, double w);
double psiC(mat r, double b, mat c);
double logpsiC(mat r, double b, mat c);


int main(int nargs, char *args[])
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j< n; j++) {
             if (i % 2 == j % 2)
                 c(i,j) = 1/3.;
        }
    }
    double a=0.8, b=0.2;
    if (w == 1) {
        switch(n) {
            case 6:
               a = 1.03741; b = 0.472513; break;
            case 12:
               a = 1.10364; b = 0.468861; break;
        case 20:
           a = 1.06019; b = 0.474467; break;
            default:
               a = 1.0; b = 0.4;
        }
    } else if(w == 0.1)  {
        switch(n) {
            case 6:
               a = 0.831104; b = 0.211443; break;
            case 12:
               a = 0.84105; b = 0.208143;  break;
        case 20:
           a = 0.856981; b = 0.200372; break;
            default:
               a = 0.952833; b = 0.354292;
        }
    } else if(w == 0.01)  {
        switch(n) {
            default:
               a = 0.911692; b = 0.203919;
        }
    }

    mat r = randn<mat>(2,n)*sqrt(1./w);
    mat Dplus = D(r, 0, a, w);
    mat Dminus = D(r, 1, a, w);
    mat invDplus  = inv(Dplus);
    mat invDminus = inv(Dminus);

    mat rpp = r;
    mat Dpluspp = Dplus;
    mat Dminuspp = Dminus;
    mat invDpluspp  = invDplus;
    mat invDminuspp = invDminus;

    double wf = det(Dplus)*det(Dminus)*psiC(r,b,c);
    double E = 0;
    int u = 0; int v = 0; int k;
    double wfpp = wf;
    double result = 0;
    double logdet, logdetpp;

    MPI_Init(&nargs, &args); int numprocs; MPI_Comm_size(MPI_COMM_WORLD, &numprocs); int my_rank; MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double time_start = MPI_Wtime();
    if (my_rank == 0) {
        cout << "Numprocs = " << numprocs << endl;
        cout << "a = " << a << ", b = " << b << endl;
    }
    for (u = 0; u < iterations; u++) {
        k = rand_particle(gen);
        r = rpp;
        r.col(k) = rpp.col(k) + randn<vec>(2)*sqrt(1./w);
        Dplus  = D(r, 0, a, w);
        Dminus = D(r, 1, a, w);
        invDplus = inv(Dplus);
        invDminus = inv(Dminus);
        wf = det(Dplus)*det(Dminus)*psiC(r,b,c);
        logdet = log( abs(det(Dplus) *det(Dminus) ) );
        if ( logdet - logdetpp + logpsiC(r,b,c) - logpsiC(rpp,b,c) > log(rand_double(gen))/2. ) {
            rpp = r; wfpp = wf; Dpluspp = Dplus; Dminuspp = Dminus; invDpluspp = invDplus; invDminuspp = invDminus; logdetpp = logdet;
            v++;
        }
    }
    MPI_Reduce(&E, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); MPI_Finalize();
    if (my_rank == 0)
        cout << "Acceptance = " << (double) v/iterations << ", elapsed: " << MPI_Wtime() - time_start << endl;
    return 0;
}


mat D(mat r, int spin, double a, double w) {
    mat D = ones<mat>(n/2,n/2);
    for (int i = 0; i < n/2; i++) {
        for (int j = 0; j< n/2; j++) {
            int nx = B.get_state(2*j+1+spin)[0];
            int ny = B.get_state(2*j+1+spin)[1];
            D(i,j) = B.psi(nx,ny,sqrt(a)*r(0,2*i+spin),sqrt(a)*r(1,2*i+spin),w);
        }
    }
    return D;
}

double psiC(mat r, double b, mat c) {
    double jastrow = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            double rij = norm(r.col(i) - r.col(j));
            jastrow += c(i,j)*rij/(1+b*rij);
        }
    }
    return exp(jastrow);
}

double logpsiC(mat r, double b, mat c) {
    double jastrow = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            double rij = norm(r.col(i) - r.col(j));
            jastrow += c(i,j)*rij/(1+b*rij);
        }
    }
    return jastrow;
}
