// g++ src/dis_generator.cpp -o build/dis_generator -I/usr/local/include -L/usr/local/lib -lLHAPDF -O3
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <LHAPDF/LHAPDF.h>
#include<chrono>

using namespace std;
using namespace std::chrono;

//constants
const double alpha = 1.0 / 137.0; //fine structure constant
const double Mp = 0.938; //proton mass in GeV

// PDF initialization
LHAPDF::PDF *pdf = LHAPDF::mkPDF("CT18NNLO", 0);

// random number generator
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0.01, 1.0);

// function to generate an event
void generate_event(double &x, double &Q2, double &weight){
    x = dis(gen); // sample x from uniform distribution
    Q2 = dis(gen) * 100.0; // sample Q2 up to 100 GeV^2

    double f_q = pdf->xfxQ(2, x, sqrt(Q2)); // u-quark distribution
    weight = (2 * M_PI * alpha * alpha) / (x * Q2 * Q2) * f_q; // DIS cross-section formula
}

// main function
int main(){
    // get starting timepoint
    auto start = high_resolution_clock::now();

    int N_events = 1000000;
    ofstream output("/home/ubuntu/DIS_EventGenerator/data/events.dat");

    for (int i = 0; i < N_events; i++){
        double x, Q2, weight;
        generate_event(x, Q2, weight);
        output << x << " " << Q2 << " " << weight << endl;
    }

    output.close();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Generated " << N_events << "events!" << endl;
    cout << "Time taken: " << duration.count() / 1000.0 << "seconds" << endl;
    return 0;
}