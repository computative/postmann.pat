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
int n = 2;

basis B = basis(5);
mat c = ones<mat>(n,n);

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<int> rand_particle(0, n-1);
uniform_int_distribution<int> rand_bool(0, 1);
uniform_real_distribution<double> rand_double(0.0,1.0);

double laplacePsi(int i, mat r, mat invDplus, mat invDminus, double a, double b, mat c, double w);
double laplaceD(mat invDplus, mat invDminus, int i, int spin, mat r, double a, double w);
double laplaceJastrow(int k, mat r, double b, mat c);
vec delPsi(int i, mat r, mat invDplus, mat invDminus, double a, double b, mat c, double w);
vec delD(mat invDplus, mat invDminus, int i, int spin, mat r, double a, double w);
vec delJastrow(int k, mat r, double b, mat c);
mat D(mat r, int spin, double a, double w);
double psiC(mat r, double b, mat c);


double psi(vec r1, vec r2,double a, double b, double c, double w) {
    double r12 = norm(r1 - r2);
    return exp(-0.5*a*w*( dot(r1,r1) + dot(r2,r2) ) + c*r12/(1 + b*r12) );
}


int main(int nargs, char *args[])
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j< n; j++) {
             if (i % 2 == j % 2)
                 c(i,j) = 1/3.;
        }
    }
    double w = 1;
    double a, b;
    switch(n) {
        case 6:
           a = 1.04; b = 0.47;
           break;
        case 12:
           a = 1.1; b = 0.47;
           break;
        default:
           a = 1.0; b = 0.4;
    }
    double dt = 0.005; double d = 0.5;
    mat r = randn<mat>(2,n);
    mat Dplus = D(r, 0, a, w);
    mat Dminus = D(r, 1, a, w);
    mat invDplus  = inv(Dplus);
    mat invDminus = inv(Dminus);

    mat rpp = r;
    mat Dpluspp = Dplus;
    mat Dminuspp = Dminus;
    mat invDpluspp  = invDplus;
    mat invDminuspp = invDminus;
    vec F = zeros<vec>(2);
    vec Fpp = zeros<vec>(2);
    vec p,q;

//    for (int s = 0; s < n; s++)
//        F += 2*delPsi(s, r, invDplus, invDminus, a, b, c, w);
//    Fpp = F;

    int iterations = pow(2,18);
    double wf = det(Dplus)*det(Dminus)*psiC(r,b,c);
    double e; double E = 0;
    int u = 0; int v = 0; int k;
    double wfpp = wf;
    double result = 0;

    MPI_Init(&nargs, &args);
    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double time_start = MPI_Wtime();
    if (my_rank == 0) {
        cout << "Numprocs = " << numprocs << endl;
        cout << "a = " << a << ", b= " << b << endl;
    }
    for (u = 0; u < iterations; u++) {
        /*
        k = rand_particle(gen);
        int not_k = (k+1) % 2;
        r = rpp;
        r.col(k) = rpp.col(k) + d*Fpp*dt + randn<vec>(2)*sqrt(dt);
//        r.col(k) = r.col(k) + randu<vec>(2) - 0.5;
        Dplus  = D(r, 0, a, w);
        Dminus = D(r, 1, a, w);
        invDplus = inv(Dplus);
        invDminus = inv(Dminus);
        F = 2*delPsi(k,r,invDplus,invDminus,a,b,c,w);
        p = rpp.col(k) - r.col(k) - d*dt*F;
        q = r.col(k) - rpp.col(k) - d*dt*Fpp;
        double G = exp((dot(q,q) - dot(p,p))/(4*d*dt));
        double Gyx = exp(- dot(p,p)/(4*d*dt));
        double Gxy = exp(- dot(q,q)/(4*d*dt));
        wf = psi(r.col(0),r.col(1), a,b,1,w);
//        wf = det(Dplus)*det(Dminus)*psiC(r,b,c);
*/
        // hasting-metropolis test
        if ( wf*wf*Gyx/(wfpp*wfpp*Gxy ) > rand_double(gen) ) {
            rpp = r; wfpp = wf; Dpluspp = Dplus; Dminuspp = Dminus; invDpluspp = invDplus; invDminuspp = invDminus; Fpp = F;
            v++;
        }
        /*
        // sample energy
        e = 0;
        for (int i = 0; i < n; i++) {
            e += -0.5*laplacePsi(i, rpp, invDpluspp, invDminuspp, a, b, c, w) + 0.5*w*w*dot(rpp.col(i),rpp.col(i));
            for (int j = i+1; j < n; j++) {
                e += 1./norm(rpp.col(i) - rpp.col(j));
            }
        }
        E += e/iterations;
        */
    }
    MPI_Reduce(&E, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    if (my_rank == 0)
        cout << "<E> = " << result/numprocs << ", r = " << (double) v/iterations << ", elapsed: " << MPI_Wtime() - time_start << endl;
    return 0;
}

double laplaceJastrow(int k, mat r, double b, mat c) {
    //return 0;
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
    //return zeros<vec>(2);
    vec sum = zeros<vec>(2);
    for (int j = 0; j <n; j++) {
        if (k == j)
            continue;
        double rjk = norm(r.col(j) - r.col(k));
        double tmp = c(j,k)/(pow(1 + b*rjk,2)*rjk);
        sum[0] += tmp*(r(0,k) - r(0,j));
        sum[1] += tmp*(r(1,k) - r(1,j));
    }
    return sum;
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


double laplacePsi(int i, mat r, mat invDplus, mat invDminus, double a, double b, mat c, double w) {
    double sum = 0;
    int spin = i % 2; // 0 impliserer spin opp
    // compute laplace D+/- avhengig av spin
    sum += laplaceD(invDplus, invDminus, i,spin, r,a,w);
    // compute laplace jastrow
    sum += laplaceJastrow(i, r, b, c);
    // compute del D+/- dot del jastrow
    sum += 2*dot(delD(invDplus, invDminus, i, spin, r, a, w), delJastrow(i, r, b, c));
    // return sum of terms
    return sum;
}


/*
    // for 2 partikkel, w = 1 regne ut produktet delJastrow*delpsi0
    cout << Dplus(0,0)*Dminus(0,0)*psiC(r, b, c) << endl;
    cout << psi(r.col(0), r.col(1), a, b, 1, w) << endl;

    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += dot( delD(invDplus,invDminus, i, i % 2, r, a, w), delJastrow(i, r, b, c) );
    }
    double exact = -a*1*w*norm(r.col(0) - r.col(1))/pow(1 + b*norm(r.col(0) - r.col(1)),2);
//    cout << "Exact answer: " << exact << endl;
    cout << "Numerical answer: " << sum << endl;

//    cout << "LaplaceD_1 numerics: " << laplaceD(invDplus, invDminus, 1,1, r,a,w) << endl;
//    cout << laplaceD(invDplus, invDminus, 0,0, r,a,w) << endl;
//    cout << "LaplaceD_1 exact: " << a*a*w*w*dot(r.col(1),r.col(1)) - 2*a*w << " " << a*a*w*w*dot(r.col(0),r.col(0)) - 2*a*w << endl;

    cout << "Laplace jastrow numerics: " << laplaceJastrow(0,r,a,b,c) << endl;
    double r12 = norm(r.col(0) - r.col(1) );
    cout << "Laplace jastrow exact: " << (1 + r12 - b*b*r12*r12 )/(r12*pow(1 + b*r12,4))  << endl;
    */
