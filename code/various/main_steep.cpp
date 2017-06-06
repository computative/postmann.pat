#include <iostream>
#include <string>
#include <armadillo>
#include <random>
#include <mpi.h>
#include "basis.h"
#include "maths.h"
#include <fstream>

using namespace std;
using namespace arma;

basis B = basis(5);
int n;

random_device rd;
mt19937 gen(rd());

double laplacePsi(int i, mat r, mat invDplus, mat invDminus, double a, double b, mat c, double w);
double laplaceD(mat invDplus, mat invDminus, int i, int spin, mat r, double a, double w);
double laplaceJastrow(int k, mat r, double b, mat c);
vec delPsi(int i, mat r, mat invDplus, mat invDminus, double a, double b, mat c, double w);
vec delD(mat invDplus, mat invDminus, int i, int spin, mat r, double a, double w);
vec delJastrow(int k, mat r, double b, mat c);
mat D(mat r, int spin, double a, double w);
double logpsiC(mat r, double b, mat c);
void observational(double * observables , mat rpp, mat invDpluspp, mat invDminuspp, double a, double b, mat c, double w);
void decent(double * observables, mat rpp, mat Dpluspp, mat Dminuspp, mat invDpluspp, mat invDminuspp, double a, double b, mat c, double w);
void vmc(double * observables, void (*sample)(double * ,mat, mat , mat, mat , mat, double, double, mat, double),
         double a,double b, mat c, double w, double dt, int iterations);
double dpsida(mat r, mat D, mat invD,double a,double w, int spin);
double dpsidb(mat r, double b, mat c);

int main(int nargs, char *args[])
{
    int numprocs, my_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    n = atoi(args[1]);
    mat c = ones<mat>(n,n);
    vec x = ones<vec>(2);
    vec xp = ones<vec>(2);
    vec xpp = ones<vec>(2);

    x[0] = atof(args[2]);
    x[1] = atof(args[3]);
    xp = x + 0.001*randn<vec>(2);
    xpp = xp + 0.001*randn<vec>(2);
    double w = atof(args[4]);
    double dt = atof(args[5]);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j< n; j++) {
             if (i % 2 == j % 2)
                 c(i,j) = 1/3.;
        }
    }
    // steepest decent
    int iterations = pow(2,16);

    vec delFp = zeros<vec>(2);
    vec delFpp = 0.01*randn<vec>(2);
    int s = 0;
    while(true) {
        double * observables = new double [5];
        for (int j = 0; j < 5; j++)
            observables[j] = 0;
        vmc(observables, decent, x[0], x[1], c, w, dt, iterations);
        double * globalsum = new double [5];
        MPI_Allreduce(observables, globalsum, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delFp[0] = 2*( globalsum[3]/(numprocs*iterations) - globalsum[0]*globalsum[1]/pow(numprocs*iterations,2) );
        delFp[1] = 2*( globalsum[4]/(numprocs*iterations) - globalsum[0]*globalsum[2]/pow(numprocs*iterations,2) );
        double g = dot(xp - xpp,delFp - delFpp)/dot(delFp - delFpp,delFp - delFpp);
        x = xp - delFp*g;
        xpp = xp;
        xp = x;
        if (my_rank == 0) {
            cout.precision(7);
            x.raw_print();
            cout << "E = " << globalsum[0]/(numprocs*iterations) << endl;
        }
        if (norm(g*delFp) < 1e-7 and s > 2) {
            break;
        }
        delFpp = delFp;
        s++;
    }
    MPI_Finalize();
    return 0;
}

void vmc(double * observables, void (*sample)(double *, mat, mat , mat, mat , mat, double, double, mat, double), double a,double b, mat c, double w, double dt, int iterations) {
    uniform_int_distribution<int> rand_particle(0, n-1);
    uniform_int_distribution<int> rand_bool(0, 1);
    uniform_real_distribution<double> rand_double(0.0,1.0);

    mat rpp = randn<mat>(2,n);
    mat Dpluspp = randn<mat>(n/2,n/2);
    mat Dminuspp = randn<mat>(n/2,n/2);
    mat invDpluspp = inv(Dpluspp);
    mat invDminuspp = inv(Dminuspp);

    mat Dplus = randn<mat>(n/2,n/2);
    mat invDplus = randn<mat>(n/2,n/2);
    mat Dminus = randn<mat>(n/2,n/2);
    mat invDminus = randn<mat>(n/2,n/2);

    int accept = 0;
    double logdetpp, logJastrowpp;


    // variational monte-carlo loop
    double time_start = MPI_Wtime();
    for (int u = 0; u < iterations; u++) {
        // importance sampling
        int s = rand_particle(gen);
        vec Fpp = 2*delPsi(s,rpp,invDpluspp,invDminuspp,a,b,c,w);
        mat r = rpp;
        r.col(s) = rpp.col(s) + 0.5*Fpp*dt + randn<vec>(2)*sqrt(dt);
        if (s % 2 == 0) {
            Dplus  = D(r, 0, a, w);
            if( abs( det(Dplus)) < 1e-40) {
                cout << det(Dplus) << endl;
                continue;
            }
            invDplus = inv(Dplus);
        } else {
            Dminus = D(r, 1, a, w);
            if( abs( det(Dminus)) < 1e-40) {
                cout << det(Dminus) << endl;
                continue;
            }
            invDminus = inv(Dminus);
        }
        vec F = 2*delPsi(s,r,invDplus,invDminus,a,b,c,w);
        vec p = rpp.col(s) - r.col(s) - 0.5*dt*F;
        vec q = r.col(s) - rpp.col(s) - 0.5*dt*Fpp;
        double logG = abs((dot(q,q) - dot(p,p))/(2*dt));
        double logdet = log( abs(det(Dplus) *det(Dminus) ) );
        double logJastrow = logpsiC(r,b,c);

        // Hastings-Metropolis style test
        if ( logG + 2*(logdet - logdetpp + logJastrow - logJastrowpp) > log(rand_double(gen)) ) {
            rpp = r; Dpluspp = Dplus; Dminuspp = Dminus; invDpluspp = invDplus;
            invDminuspp = invDminus; logdetpp = logdet; logJastrowpp = logJastrow;
            accept++;
        }
        sample(observables, rpp, Dpluspp, Dminuspp,invDpluspp, invDminuspp, a, b, c, w);
    }
}

void decent(double * observables, mat rpp, mat Dpluspp, mat Dminuspp, mat invDpluspp, mat invDminuspp, double a, double b, mat c, double w) {
    double e = 0;
    for (int i = 0; i < n; i++) {
        e += -0.5*laplacePsi(i, rpp, invDpluspp, invDminuspp, a, b, c, w) + 0.5*w*w*dot(rpp.col(i),rpp.col(i));
        for (int j = i+1; j < n; j++) {
            double rij = norm(rpp.col(i) - rpp.col(j));
            e += 1./rij;
        }
    }
    double tmpa = dpsida(rpp, Dpluspp, invDpluspp, a, w, 0) + dpsida(rpp, Dminuspp, invDminuspp, a, w, 1);
    double tmpb = dpsidb(rpp, b, c);
    observables[0] += e;
    observables[1] += tmpa;
    observables[2] += tmpb;
    observables[3] += e*tmpa;
    observables[4] += e*tmpb;
}

void observational(double * observables, mat rpp, mat invDpluspp, mat invDminuspp, double a, double b, mat c, double w) {
    double ki = 0; double vi = 0; double e = 0; double ri = 0;
    for (int i = 0; i < n; i++) {
        ki += -0.5*laplacePsi(i, rpp, invDpluspp, invDminuspp, a, b, c, w);
        vi += 0.5*w*w*dot(rpp.col(i),rpp.col(i));
        e += -0.5*laplacePsi(i, rpp, invDpluspp, invDminuspp, a, b, c, w) + 0.5*w*w*dot(rpp.col(i),rpp.col(i));

        for (int j = i+1; j < n; j++) {
            double rij = norm(rpp.col(i) - rpp.col(j));
            ri += rij;
            vi += 1./rij;
            e += 1./rij;
        }
    }
    observables[0] += e;
    observables[1] += ki;
    observables[2] += vi;
    observables[3] += (double) 2.*ri/(n*(n-1));
}

double laplaceJastrow(int k, mat r, double b, mat c) {
    double x = 0; double y = 0; double xy = 0; double rkj;
    for (int j = 0; j < n; j++) {
        if (j == k)
            continue;
        rkj = norm(r.col(k) - r.col(j));
        x += c(k,j)*(r(0,k) - r(0,j))/( pow(1 + b*rkj,2)*rkj);
        y += c(k,j)*(r(1,k) - r(1,j))/( pow(1 + b*rkj,2)*rkj);
        xy += c(k,j)*(1 - b*rkj)/(pow(1 + b*rkj,3)*rkj);
    }
    return x*x + y*y + xy;
}

vec delJastrow(int k, mat r, double b, mat c) {
    vec sum = zeros<vec>(2);
    for (int j = 0; j <n; j++) {
        if (k == j)
            continue;
        double rjk = norm(r.col(j) - r.col(k));
        double tmp = c(j,k)/(pow(1 + b*rjk,2)*rjk);
        sum += tmp*(r.col(k) - r.col(j));
    }
    return sum;
}

double dpsida(mat r, mat D, mat invD,double a,double w, int spin) {
    double sum = 0;
    for (int i = 0; i < n/2; i++) {
        for (int j = 0; j < n/2; j++) {
            double x = r(0,2*j + spin);
            double y = r(1,2*j + spin);
            int nx = B.get_state(2*i + 1 + spin)[0];
            int ny = B.get_state(2*i + 1 + spin)[1];
            sum += invD(i,j)*( x*nx*sqrt(w/a)*B.psi(nx-1,ny,sqrt(a)*x,sqrt(a)*y,w) + y*ny*sqrt(w/a)*B.psi(nx,ny-1,sqrt(a)*x,sqrt(a)*y,w)
                               - 0.5*w*dot(r.col(2*j+spin),r.col(2*j+spin))*B.psi(nx, ny, sqrt(a)*x, sqrt(a)*y,w) );
        }
    }
    return sum;
}

double dpsidb(mat r, double b, mat c) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            double rij = norm(r.col(i) - r.col(j));
            sum += c(i,j)*pow( rij/(1+b*rij),2);
        }
    }
    return -sum;
}

vec delPsi(int i, mat r, mat invDplus, mat invDminus, double a, double b, mat c, double w) {
    return delD(invDplus,invDminus, i, i % 2, r, a, w) + delJastrow(i, r, b, c);
}

double laplaceD(mat invDplus, mat invDminus, int i, int spin, mat r, double a,double w) {
    double sum = 0;
    mat invD;
    if (spin == 1) {
        invD = invDminus;
    } else {
        invD = invDplus;
    }
    for (int j = 0; j < n/2; j ++ ) { // sjekk indekser her
        int nx = B.get_state(2*j + 1 + spin)[0];
        int ny = B.get_state(2*j + 1 + spin)[1];
        double x = r(0,i);
        double y = r(1,i);
        sum += invD(j,i/2)*(4*a*w*(nx*(nx-1)*B.psi(nx-2,ny,sqrt(a)*x,sqrt(a)*y,w) + ny*(ny-1)*B.psi(nx,ny-2,sqrt(a)*x,sqrt(a)*y,w))
               - 4*pow(a*w,1.5)*(x*nx*B.psi(nx-1,ny,sqrt(a)*x,sqrt(a)*y,w) + y*ny*B.psi(nx,ny-1,sqrt(a)*x,sqrt(a)*y,w) )
               + a*w*B.psi(nx,ny,sqrt(a)*x,sqrt(a)*y,w)*(a*w*dot(r.col(i),r.col(i)) - 2));
    }
    return sum;
}

vec delD(mat invDplus, mat invDminus, int i, int spin, mat r, double a, double w) {
    vec sum = zeros<vec>(2);
    mat invD;
    if (spin == 1) {
        invD = invDminus;
    } else {
        invD = invDplus;
    }

    for (int j = 0; j <n/2; j++) {
        int nx = B.get_state(2*j + 1 + spin)[0];
        int ny = B.get_state(2*j + 1 + spin)[1];
        double x = r(0,i);
        double y = r(1,i);
        sum[0] += invD(j,i/2)*(2*nx*sqrt(a*w)*B.psi(nx-1,ny,sqrt(a)*x,sqrt(a)*y,w) - a*w*x*B.psi(nx,ny,sqrt(a)*x,sqrt(a)*y,w));
        sum[1] += invD(j,i/2)*(2*ny*sqrt(a*w)*B.psi(nx,ny-1,sqrt(a)*x,sqrt(a)*y,w) - a*w*y*B.psi(nx,ny,sqrt(a)*x,sqrt(a)*y,w));
    }
    return sum;
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

double laplacePsi(int i, mat r, mat invDplus, mat invDminus, double a, double b, mat c, double w) {
    double sum = 0;
    int spin = i % 2; // 0 impliserer spin opp
    sum += laplaceD(invDplus, invDminus, i,spin, r,a,w);
    sum += laplaceJastrow(i, r, b, c);
    sum += 2*dot(delD(invDplus, invDminus, i, spin, r, a, w), delJastrow(i, r, b, c));
    return sum;
}
