#include <iostream>
#include <string>
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

double laplacePsi(int i) {
    int spin = i % 2 // 0 impliserer spin opp
    // compute laplace D+/- avhengig av spin

    // compute laplace jastrow

    // compute del D+/-

    // compute del jastrow

    // return sum of terms
}

double laplaceD(int i, int spin) {
    double sum = 0;
    for (int j = spin; j < n; j += 2 ) { // sjekk indekser her
        s += /* analytisk utrykk for laplace phi_j i NY posisjon r_i  */ * a;
    }
}

double laplaceJastrow() {

}

double delD(int i) {

}

double delJastrow() {

}


int main(int argc, char *argv[])
{
    double E = 0;
    double n = 2;
    double w = 1;
    for (int i = 0; i < n; i++) {
        E += -0.5*laplacePsi(i) + 0.5*w*w*dot(r.col(i),r.col(i));
        for (int j = i+1; j < n; j++) {
            E += 1/norm(r.col(i) - r.col(j));
        }
    }
    return 0;
}

