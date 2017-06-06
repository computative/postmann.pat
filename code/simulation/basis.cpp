#include <iostream>
#include <fstream>
#include "math.h"
#include "basis.h"
#include "maths.h"
#include "math.h"

using namespace std;

const double hbar  = 1.0;
const double m  = 1.0;

basis::basis(int n) // number of shells is n
{
    numStates = n*(n+1);                                // create table of numStates
    state = new int * [numStates];
    for (int i = 0; i < n; i++) {                       // for each shell
        for (int j = 0; j < 2*(i+1); j++) {             // for each state in shell
            state[i*(i+1) + j] = new int[4];
            state[i*(i+1) + j][0] = i - (j - j%2)/2;    // set nx
            state[i*(i+1) + j][1] = (j - j%2)/2;        // set ny
            state[i*(i+1) + j][2] = j % 2;              // set spinprojection
            state[i*(i+1) + j][3] = i + 1;              // set energy/omega
        }
    }
}

int basis::get_index(int * config)
{
    int i = config[3] - 1;              // get shell - 1
    int nx = config[0];                 // get nx
    int sigma = config[2];              // get spinprojection
    return i*(i+1) + 2*nx + sigma+1;    // return statenumber = 1,2,3,...
}

int * basis::get_state(int ni)
{
    return state[ni-1];
}

void basis::print(){
    int E = state[0][3];                // the lowest energy level
    for(int i = 0; i < numStates; i++){ // for each state
        if (state[i][3] != E) {         // if higher energy level
            cout << endl;               // then print linebreak to table
            E = state[i][3];
        }
        cout << '(' << state[i][0] <<  ',' << state[i][1] <<  ',' << state[i][2] <<  ',' << state[i][3] << "),";
    }
    cout << endl << "Legend: (nx,ny,sigma,E). sigma = 1 => spin = +1/2" << endl;
}

void basis::to_file(string filename){
    int E = state[0][3];
    ofstream myfile;
    myfile.open (filename);
    for(int i = 0; i < numStates; i++){
        if (state[i][3] != E) {         // if higher energy level
            myfile << endl;             //
            E = state[i][3];            // then print linebreak to table
        }
        myfile << '(' << state[i][0] <<  ',' << state[i][1] <<  ',' << state[i][2] <<  ',' << state[i][3] << "),";
    }
    myfile.close();
}

double basis::hermite(int n, double x)
{
    int N = floor(n/2);
    double S = 0;
    for (int m = 0; m<=N; m++) {
        S += pow(-1,m)*pow(2* (x), n-2*m)/((double)(factorial(m)*factorial(n-2*m)));
    }
    return S*factorial(n);
}

double basis::psi_n(int n, double x, double w ) {
//    return pow(m*w/(M_PI*hbar),0.25)*hermite(n,x*sqrt(w))*exp(-0.5*w*x*x)/sqrt(pow(2,n)*factorial(n));
    return hermite(n,x*sqrt(w))*exp(-0.5*w*x*x);
//    return hermite(n,x*sqrt(w))*exp(-0.5*w*x*x);
}

double basis::psi(int nx,int ny, double x, double y, double w){
    if (nx < 0 or ny <0)
        return 0;
    return psi_n(nx, x, w) * psi_n(ny, y, w);
}
