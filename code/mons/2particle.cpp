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
    int iterations;// = pow(2,20);   3.6465
    vec delF = zeros<vec>(2);
    //vec x {0.9886,0.40061};
    vec x {0.9886,0.3985};
    vec xpp = x; // {1.05,0.395};
    double a; double b; double c = 1; double w = 1;
    double dt = 0.005; double D = 0.5;
    int k; int not_k;
    double Gyx; double Gxy; double g;
    double wf;
    double e; double E; double E2;

    for ( int d = 0; d < 1000000; d++ ) {
        iterations = 1000000;
        double apsi = 0;
        double bpsi = 0;
        double eapsi = 0;
        double ebpsi = 0;
        vec p; vec q;
        mat r = randn<mat>(2,2); mat rpp = randn<mat>(2,2);
        vec Fpp = randn<vec>(2); vec F = randn<vec>(2);
        int i = 0;
        int j;

        //double app = a - a/10; bpp = b - b/10;
        double rij = norm(rpp.col(0) - rpp.col(1)); double rijpp = norm(rpp.col(0) - rpp.col(1));
        double wfpp = psi(rpp.col(0),rpp.col(1), x[0],x[1],c,w);
        E = 0;  E2 = 0;

        j = 0;
        while (i < iterations ) {
            //a = 1; b = 0.40061;
            a = xpp[0]; b = xpp[1];
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
                rpp = r; wfpp = wf; rij = norm(r.col(0) - r.col(1) );
                j++;
            }
            e = 1/rij + 0.5*w*w*(1-a*a)*( dot(rpp.col(0),rpp.col(0)) + dot(rpp.col(1),rpp.col(1)) )
                    + 2*a*w + a*w*c*rij/pow(1 + rij*b,2) - c*(1+rij*c-b*b*rij*rij)/( rij*pow(1 + rij*b,4) );
            double tmpa = -(w/2.0)*(dot(r.col(0),r.col(0)) + dot(r.col(1),r.col(1)) );
            double tmpb = (-c*rij*rij)/pow(1 + b*rij,2);
            apsi += tmpa;
            //cout << apsi << endl;
            bpsi += tmpb;
            eapsi += e*tmpa;
            ebpsi += e*tmpb;
            E += e/iterations;
            E2 += e*e/iterations;
            i++;
        }

        //xpp.print();

        /*
        cout << (eapsi/iterations) << endl;
        cout << apsi/iterations << endl;
        */
        delF[0] = 2*( (eapsi/iterations) - E*(apsi/iterations) );
        delF[1] = 2*( (ebpsi/iterations) - E*(bpsi/iterations) );
        g = 0.1;
        x = xpp - g*delF;

        //delF.print();
        //cout << endl;
        //delF.print();
        if (i % 100000 == 0) {
            x.print();
            cout << E << endl;
            cout << "*******************************" << endl;
        }
        xpp = x;
    }

    /*
        iterations = 100000;
        double apsi = 0;
        double bpsi = 0;
        double eapsi = 0;
        double ebpsi = 0;
        vec p; vec q;
        mat r = randn<mat>(2,2); mat rpp = randn<mat>(2,2);
        vec Fpp = randn<vec>(2); vec F = randn<vec>(2);
        int i = 0;
        int j;

        //double app = a - a/10; bpp = b - b/10;
        double rij = norm(rpp.col(0) - rpp.col(1)); double rijpp = norm(rpp.col(0) - rpp.col(1));
        double wfpp = psi(rpp.col(0),rpp.col(1), x[0],x[1],c,w);
        E = 0;  E2 = 0;

        j = 0;
        while (i < iterations ) {
            //a = 1; b = 0.40061;
            a = 1; b = 0.40061;
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
                rpp = r; wfpp = wf; rij = norm(r.col(0) - r.col(1) );
                j++;
            }
            e = 1/rij + 0.5*w*w*(1-a*a)*( dot(rpp.col(0),rpp.col(0)) + dot(rpp.col(1),rpp.col(1)) )
                      + 2*a*w + a*w*c*rij/pow(1 + rij*b,2) - c*(1+rij*c-b*b*rij*rij)/( rij*pow(1 + rij*b,4) );

            apsi += -(w/2.0)*(dot(r.col(0),r.col(0)) + dot(r.col(1),r.col(1)) );
            bpsi += (-c*rij*rij)/pow(1 + b*rij,2);
            eapsi += e*apsi;
            ebpsi += e*bpsi;
            E += e/iterations;
            E2 += e*e/iterations;
            i++;
        }
    cout << E << " " << E2 - E*E << " " << (double) j/iterations << endl;
    */
    return 0;
}

double psi(vec r1, vec r2,double a, double b, double c, double w) {
    double r12 = norm(r1 - r2);
    return exp(-0.5*a*w*( dot(r1,r1) + dot(r2,r2) ) + c*r12/(1 + b*r12) );
}

