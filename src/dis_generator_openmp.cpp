// #include <iostream>
// #include <fstream>
// #include <random>
// #include <cmath>
// #include <LHAPDF/LHAPDF.h>
// #include<chrono>
// #include <omp.h>

// using namespace std;
// using namespace std::chrono;

// //constants
// const double alpha = 1.0 / 137.0; //fine structure constant
// const double Mp = 0.938; //proton mass in GeV

// // PDF initialization
// LHAPDF::PDF *pdf = LHAPDF::mkPDF("CT18NNLO", 0);

// // random number generator
// random_device rd;
// mt19937 gen(rd());
// uniform_real_distribution<> dis(0.01, 1.0);

// // function to generate an event
// void generate_event(double &x, double &Q2, double &weight){
//     x = dis(gen); // sample x from uniform distribution
//     Q2 = dis(gen) * 100.0; // sample Q2 up to 100 GeV^2

//     double f_q = pdf->xfxQ(2, x, sqrt(Q2)); // u-quark distribution
//     weight = (2 * M_PI * alpha * alpha) / (x * Q2 * Q2) * f_q; // DIS cross-section formula
// }

// // main function
// int main(){
//     // get starting timepoint
//     auto start = high_resolution_clock::now();

//     int N_events = 1000000;
//     ofstream output("events_openmp.dat");

//     #pragma omp parallel for
//     for (int i = 0; i < N_events; i++){
//         double x, Q2, weight;
//         generate_event(x, Q2, weight);

//         #pragma omp critical
//         output << x << " " << Q2 << " " << weight << endl;
//     }

//     output.close();

//     auto stop = high_resolution_clock::now();
//     auto duration = duration_cast<seconds>(stop - start);

//     cout << "Generated " << N_events << "events!" << endl;
//     cout << "Time taken: " << duration.count() << "seconds" << endl;
//     return 0;
// }


// g++ -fopenmp -O3 src/dis_generator_openmp.cpp -o build/dis_generator_openmp -I/usr/local/include -L/usr/local/lib -lLHAPDF

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <LHAPDF/LHAPDF.h>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Constants
const double alpha = 1.0 / 137.0;  // fine structure constant
const double Mp = 0.938;           // proton mass in GeV

// Random number generator
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0.01, 1.0);

// Function to generate an event
void generate_event(LHAPDF::PDF *pdf, double &x, double &Q2, double &weight) {
    x = dis(gen);            // Sample x from uniform distribution
    Q2 = dis(gen) * 100.0;   // Sample Q2 up to 100 GeV^2

    double f_q = pdf->xfxQ(2, x, sqrt(Q2)); // u-quark distribution
    weight = (2 * M_PI * alpha * alpha) / (x * Q2 * Q2) * f_q; // DIS cross-section formula
}

// Main function
//
int main() {
    int N_events = 10000000;
    vector<string> thread_outputs(omp_get_max_threads());  // Buffers for each thread

    auto start = high_resolution_clock::now();

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        LHAPDF::PDF *pdf = LHAPDF::mkPDF("CT18NNLO", 0);  // Local PDF instance per thread
        ostringstream local_buffer; // Thread-local buffer

        #pragma omp for
        for (int i = 0; i < N_events; i++) {
            double x, Q2, weight;
            generate_event(pdf, x, Q2, weight);
            local_buffer << x << " " << Q2 << " " << weight << "\n";
        }

        thread_outputs[thread_id] = local_buffer.str(); // Store results
        delete pdf;  // Clean up PDF instance
    }

    // Write everything to file in a single I/O operation
    ofstream output("/home/ubuntu/DIS_EventGenerator/data/events_openmp.dat");
    for (const auto &data : thread_outputs) {
        output << data;
    }
    output.close();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Generated " << N_events << " events!" << endl;
    cout << "Time taken: " << duration.count() / 1000.0 << " seconds" << endl;

    return 0;
}
