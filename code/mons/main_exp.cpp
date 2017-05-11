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
    double b = 0.4;
    double c = 1;

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
    ofstream myfile;
    myfile.open (filename);
    int k = 0;
    while (i < iterations ) {
        r1 = r1_old + 0.005*2*0.5*gradPsi(r1, r2, a, b, c, w) + randn<vec>(2)*sqrt(0.005);
        r2 = r2_old + 0.005*2*0.5*gradPsi(r1, r2, a, b, c, w) + randn<vec>(2)*sqrt(0.005);
//        r1 = r1_old + randu<vec>(2) - 0.5;
  //      r2 = r2_old + randu<vec>(2) - 0.5;
        wf = psi_T(r1,r2, a,b,c,w);
        if ( wf*wf*G(r1_old, r2_old,r1,r2,a,b,c,w)/(wf_old*wf_old*G(r1,r2,r1_old,r2_old,a,b,c,w)) > rand_double(gen) ) {
            r1_old = r1;  r2_old = r2; wf_old = wf; rij = norm(r1_old - r2_old);
            k++;
        }
        e = 1/rij + 0.5*w*w*(1-a*a)*( dot(r1_old,r1_old) + dot(r2_old,r2_old) ) + 2*a*w + a*w*c*rij/pow(1 + rij*b,2) - c*(1+rij*c-b*b*rij*rij)/( rij*pow(1 + rij*b,4) );
        E += e;
        myfile << e << endl;
        //E += 2*w*a + 0.5*w*w*(1-a*a)*( dot(r1_old,r1_old) + dot(r2_old,r2_old) ) ;
        i++;
    }
    myfile.close();
    cout << E/iterations << " " << (double) k/iterations << endl;
    return 0;
}

double psi_T(vec r1, vec r2,double a, double b, double c, double w) {
    double r12 = norm(r1 - r2);
    //return exp(-0.5*a*w*( dot(r1,r1) + dot(r2,r2) ));
    return exp(-0.5*a*w*( dot(r1,r1) + dot(r2,r2) ) + c*r12/(1 + b*r12) );
}

double G(vec r1, vec r2, vec r1_old, vec r2_old, double a, double b, double c, double w) {
    double D = 0.5;
    double dt = 0.005;
    vec f1 = r1 - r1_old - 2*D*dt*gradPsi(r1, r2, a, b, c, w);
    vec f2 = r2 - r2_old - 2*D*dt*gradPsi(r1, r2, a, b, c, w);
    return exp(- (dot(f1,f1) + dot(f2,f2))/(4*D*dt));
}

vec gradPsi(vec r1,vec r2, double a, double b, double c, double w) {
    vec A = zeros<vec>(2);
    double rij = norm(r1 - r2);
    double num = c*(r1[0] - r2[0] + r1[1] - r2[1] )/( (1 + b*rij)*(1 + b*rij)*rij );
    A[0] = -a*w*(r1[0] + r2[0]) + num;
    A[1] = -a*w*(r1[1] + r2[1]) + num;
    return A;
}
