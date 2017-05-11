#include <iostream>
#include <fstream>
#include <string>
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

double psi(vec r1, vec r2,double a, double b, double c, double w);

double n = 2; // particle number. Starts counting from 1.

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<int> rand_particle(0, n-1);
uniform_int_distribution<int> rand_bool(0, 1);
uniform_real_distribution<double> rand_double(0.0,1.0);

int main(int argc, char *argv[])
{
    mat r = randn<mat>(2,2); mat rpp = randn<mat>(2,2);
    vec Fpp = randn<vec>(2); vec F = randn<vec>(2);
    vec p; vec q;
    int iterations = pow(2,20);
    int i = 0;
    double a = 1; double b = 0.40061; double c = 1; double w = 1;
    double dt = 0.005; double D = 0.5;
    double rij = norm(rpp.col(0) - rpp.col(1)); double rijpp = norm(rpp.col(0) - rpp.col(1));
    double wf; double wfpp = psi(rpp.col(0),rpp.col(1), a,b,c,w);
    double e; double E = 0;  double E2 = 0;
    double Gyx; double Gxy;
    int j = 0; int k; int not_k;
    while (i < iterations ) {
        k = rand_particle(gen);
        not_k = (k+1) % 2;

        rijpp = norm(rpp.col(0)-rpp.col(1));
        Fpp = -2*a*w*rpp.col(k) + 2*c*(rpp.col(k) - rpp.col(not_k))/( (1 + b*rijpp)*(1 + b*rijpp)*rijpp );

        r.col(k) = rpp.col(k) + D*Fpp*dt + randn<vec>(2)*sqrt(dt);
        rij = norm(r.col(0)-r.col(1));
        F    = -2*a*w*r.col(k) + 2*c*(r.col(k) - r.col(not_k))/( (1 + b*rij)*(1 + b*rij)*rij );
        p = rpp.col(k) - r.col(k) - D*dt*F;
        q = r.col(k) - rpp.col(k) - D*dt*Fpp;
        Gyx = exp(- dot(p,p)/(4*D*dt));
        Gxy = exp(- dot(q,q)/(4*D*dt));
        wf = psi(r.col(0),r.col(1), a,b,c,w);

        if ( wf*wf*Gyx/(wfpp*wfpp*Gxy ) > rand_double(gen) ) {
            rpp.col(0) = r.col(0);  rpp.col(1) = r.col(1); wfpp = wf; rij = norm(rpp.col(0) - r.col(1) );
            j++;
        }
        e = 1/rij + 0.5*w*w*(1-a*a)*( dot(rpp.col(0),rpp.col(0)) + dot(rpp.col(1),rpp.col(1)) )
                  + 2*a*w + a*w*c*rij/pow(1 + rij*b,2) - c*(1+rij*c-b*b*rij*rij)/( rij*pow(1 + rij*b,4) );
        E += e;
        E2 += e*e;
        i++;
    }
    cout << E/iterations << " " << (E2/iterations - (E/iterations)*(E/iterations)) << " " << (double) j/iterations << endl;
    return 0;
}

double psi(vec r1, vec r2,double a, double b, double c, double w) {
    double r12 = norm(r1 - r2);
    return exp(-0.5*a*w*( dot(r1,r1) + dot(r2,r2) ) + c*r12/(1 + b*r12) );
}

