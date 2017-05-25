#include <iostream>
#include <string>
#include <armadillo>
#include <random>
#include "basis.h"
#include "maths.h"

using namespace std;
using namespace arma;

// definere globale variable fordi jeg ønsker det.
int n = 2;
basis B = basis(n);
mat c = ones<mat>(n,n);


double laplacePsi(int i, mat r, double a, double b, mat c, double w);
double laplaceD(int i, int spin, mat r,double a,double b,mat c,double w);
double laplaceJastrow(int k,mat r, double a, double b, mat c);
vec delD(mat invDplus, mat invDminus, int i, int spin, mat r, double a, double b, mat c, double w);
vec delJastrow(int k, int spin, mat r, double a, double b, mat c);
mat D(mat r, int spin, double a, double w);
double psiC(mat r, double a, double b, mat c);


int main(int argc, char *argv[])
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j< n; j++) {
             if (i % 2 == j % 2)
                 c(i,j) = 1/3.;
        }
    }
    double E = 0;
    double w = 1;
    double a = 1;
    double b = 0.4;
    mat r = randn<mat>(2,n);
    mat Dplus = D(r, 0, a, w);
    mat Dminus = D(r, 0, a, w);
    mat invDplus  = inv(Dplus);
    mat invDminus = inv(Dminus);

    double sum = 0;
    for (int i = 0; i < n; i++) {
        vec U = delD(invDplus,invDminus, i, i % 2, r, a,b,c,w);
        //U.print();
        vec V = delJastrow(i,i%2,r,a,b,c);
        V.print();
        //sum += dot(delD(invDplus,invDminus, i, i % 2, r, a,b,c,w), delJastrow(i,i%2,r,a,b,c) );
    }
    double exact = -a*1*w*norm(r.col(0) - r.col(1))/pow(1 + b*norm(r.col(0) - r.col(1)),2);
    cout << "Exact answer: " << exact << endl;
    cout << "Numerical answer: " << sum << endl;
    return 0;

    for (int i = 0; i < n; i++) {
        //E += -0.5*laplacePsi(i, r, invDplus, invDminus, a, b, c, w) + 0.5*w*w*dot(r.col(i),r.col(i));
        for (int j = i+1; j < n; j++) {
            //E += 1/norm(r.col(i) - r.col(j));
        }
    }
    return 0;
}

vec delD(mat invDplus, mat invDminus, int i, int spin, mat r, double a, double b, mat c, double w) {
    vec sum = zeros<vec>(2);
    mat invD;
    if (spin) {
        invD = invDminus;
    } else {
        invD = invDplus;
    }
    for (int j = 0; j <n/2; j++) {
        int nx = B.get_state(2*j + 1 + spin)[1];
        int ny = B.get_state(2*j + 1 + spin)[1];
        double x = r(0,i);
        double y = r(1,i);
        sum[0] += invD(j,floor(i/2))*(2*nx*sqrt(a*w)*B.psi(nx-1,ny,x,y,a,w) - a*w*x*B.psi(nx,ny,x,y,a,w));
        sum[1] += invD(j,floor(i/2))*(2*ny*sqrt(a*w)*B.psi(nx,ny-1,x,y,a,w) - a*w*y*B.psi(nx,ny,x,y,a,w));
    }
    return sum;
}

vec delJastrow(int k, int spin, mat r, double a, double b, mat c) {
    vec sum = zeros<vec>(2);
    for (int j = spin; j <n; j+=2) {
        if (k == j)
            continue;
        cout << "hi" << endl;
        double rjk = norm(r.col(j) - r.col(k));
        double tmp = c(j,k)/((1 + b*rjk)*rjk);
        sum[0] += tmp*(r(0,k) - r(0,j));
        sum[1] += tmp*(r(1,k) - r(1,j));
    }
    return sum;
}

double psiC(mat r, double a, double b, mat c) {
    double jastrow = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double rij = norm(r.col(i) - r.col(j));
            jastrow += c(i,j)*rij/(1+b*rij);
        }
    }
    return jastrow;
}

mat D(mat r, int spin, double a, double w) {
    mat D = ones<mat>(n/2,n/2);
    for (int i = 0; i < n/2; i++) {
        for (int j = 0; j< n/2; j++) {
            int nx = B.get_state(2*i+1+spin)[1];
            int ny = B.get_state(2*i+1+spin)[0];
            D(i,j) = B.psi(nx,ny,r(0,2*j+spin),r(1,2*j+spin),a,w);
        }
    }
    return D;
}

double laplacePsi(int i, mat r, mat invDplus, mat invDminus, double a, double b, mat c, double w) {
    double sum = 0;
    int spin = i % 2; // 0 impliserer spin opp
    // compute laplace D+/- avhengig av spin
    sum += laplaceD(i, spin, r, a, b, c, w);
    // compute laplace jastrow
    sum += laplaceJastrow(i, r, a, b, c);
    // compute del D+/- dot del jastrow
    sum += dot(delD(invDplus, invDminus, i, spin, r, a, b, c, w), delJastrow(i, spin, r, a, b, c));
    // return sum of terms
    return sum;
}

double laplaceD(int i, int spin, mat r, double a,double b, mat c,double w) {
    double sum = 0;
    for (int j = spin; j < n; j += 2 ) { // sjekk indekser her
        int nx = B.get_state(j+1)[1];
        int ny = B.get_state(j+1)[0];
        double x = r(0,j);
        double y = r(1,j);
        sum += 4*a*w*(nx*(nx-1)*B.psi(nx-2,ny,x,y,a,w) + ny*(ny-1)*B.psi(nx,ny-2,x,y,a,w)) - 4*pow(a*w,1.5)*(x*nx*B.psi(nx-1,ny,x,y,a,w)
               + y*ny*B.psi(nx,ny-1,x,y,a,w) ) + a*w*B.psi(nx,ny,x,y,a,w)*(a*w*dot(r.col(j),r.col(j)) -2);
    }
    return sum;
}

double laplaceJastrow(int k, mat r,double a, double b, mat c) {
    double sum = 0;
    double s = 0;
    double rkj;
    for (int j = 0; j < n; j++) {
        if (j == k)
            continue;
        rkj = norm(r.col(k) - r.col(j));
        s += c(k,j)*(r(0,k) - r(0,j))/((1 + b*rkj)*rkj);
    }
    sum += s*s;
    s = 0;
    for (int j = 0; j < n; j++) {
        if (j == k)
            continue;
        rkj = norm(r.col(k) - r.col(j));
        s += c(k,j)*(r(1,k) - r(1,j))/((1 + b*rkj)*rkj);
    }
    sum += s*s;
    s = 0;
    for (int j = 0; j < n; j++) {
        if (j == k)
            continue;
        rkj = norm(r.col(k) - r.col(j));
        s += c(k,j)*(1 - b*rkj)/(pow(1 + b*rkj,3)*rkj);
    }
    sum += s;
    return sum;
}

