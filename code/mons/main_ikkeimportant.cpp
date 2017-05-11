#include <iostream>
#include <fstream>
#include <string>
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

double psi_T(vec r1, vec r2,double a, double b, double c, double w);
double G(vec r1,vec r2,vec r1_old,vec r2_old, double a, double b, double c, double w);
vec gradPsi(vec r1,vec r2, double a, double b, double c, double w);

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<int> rand_int(0, 1);
uniform_real_distribution<double> rand_double(0.0,1.0);

int main(int argc, char *argv[])
{
    /*
     * 1 Lag en ny bølgefunksjon
     * 2 Test kvotienten av ny og gammel bølgefunksjon mot hverandre
     */
    int iterations = pow(2,20);
    int i = 0;
    string filename = "/home/marius/Dokumenter/fys4411/postmann.pat/resources/data.txt";
    double a = 1;
    double b = 0.40061;
    double c = 1;
    double dt = 0.001;
    double w = 1;
    vec r1 = randn<vec>(2);
    vec r2 = randn<vec>(2);
    vec r1_old = randn<vec>(2);
    vec r2_old = randn<vec>(2);
    double wf;
    double wf_old = psi_T(r1_old,r2_old, a,b,c,w);
    double E = 0;
    double e;
    double rij = norm(r1_old - r2_old);
    int k = 0;
    vec F_old = randn<vec>(2);
    vec F = randn<vec>(2);

    while (i < iterations ) {

        r1 = r1_old + randu<vec>(2) - 0.5;
        r2 = r2_old + randu<vec>(2) - 0.5;
        //F = 2*gradPsi(r1,r2,a,b,c,w);
        wf = psi_T(r1,r2, a,b,c,w);
        if ( wf*wf/(wf_old*wf_old) > rand_double(gen) ) {
            r1_old = r1;  r2_old = r2; wf_old = wf; rij = norm(r1_old - r2_old);
            k++;
        }
        e = 1/rij + 0.5*w*w*(1-a*a)*( dot(r1_old,r1_old) + dot(r2_old,r2_old) ) + 2*a*w + a*w*c*rij/pow(1 + rij*b,2) - c*(1+rij*c-b*b*rij*rij)/( rij*pow(1 + rij*b,4) );
        E += e;
        //E += 2*w*a + 0.5*w*w*(1-a*a)*( dot(r1_old,r1_old) + dot(r2_old,r2_old) ) ;
        i++;
    }
    cout << E/iterations << " " << (double) k/iterations << endl;
    return 0;
}

double psi_T(vec r1, vec r2,double a, double b, double c, double w) {
    double r12 = norm(r1 - r2);
    //return exp(-0.5*a*w*( dot(r1,r1) + dot(r2,r2) ));
    return exp(-0.5*a*w*( dot(r1,r1) + dot(r2,r2) ) + c*r12/(1 + b*r12) );
}
