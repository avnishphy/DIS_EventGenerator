#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <LHAPDF/LHAPDF.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Define LHAPDF globally (runs on CPU)
LHAPDF::PDF* pdf = LHAPDF::mkPDF("CT18NNLO", 0); // Load PDF

// GPU Kernel
__global__ void generate_events_GPU(double *x, double *Q2, double *weights, double *pdf_values, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState state;
    for (int i = idx; i < N; i += stride) {
        curand_init(1234, i, 0, &state);
        x[i] = 0.01 + (curand_uniform(&state) * 0.99);  // x ∈ (0.01,1.0)
        Q2[i] = curand_uniform(&state) * 100.0;  // Q2 ∈ (0,100)
        
        // Interpolate f_q(x, Q2) from the precomputed table
        int x_idx = min((int)(x[i] * 100), 99); // Convert x to an index
        int Q2_idx = min((int)(Q2[i] / 1.0), 99); // Convert Q2 to an index
        double f_q = pdf_values[x_idx * 100 + Q2_idx];

        weights[i] = (2 * M_PI * 1/137.0 * 1/137.0) / (x[i] * Q2[i] * Q2[i]) * f_q;
    }
}

int main(){
    auto start = high_resolution_clock::now();

    int N_events = 10000000;
    double *x, *Q2, *weights;
    cudaMallocManaged(&x, N_events * sizeof(double));
    cudaMallocManaged(&Q2, N_events * sizeof(double));
    cudaMallocManaged(&weights, N_events * sizeof(double));

    // Step 1: Precompute LHAPDF values on CPU
    double pdf_values[100 * 100];  // Store PDF values for interpolation
    for (int i = 0; i < 100; i++) {
        double x_val = 0.01 + i * 0.01;
        for (int j = 0; j < 100; j++) {
            double Q2_val = j * 1.0;
            pdf_values[i * 100 + j] = pdf->xfxQ(2, x_val, sqrt(Q2_val));  // Get u-quark PDF
        }
    }

    // Step 2: Copy PDF values to GPU
    double *d_pdf_values;
    cudaMalloc(&d_pdf_values, 100 * 100 * sizeof(double));
    cudaMemcpy(d_pdf_values, pdf_values, 100 * 100 * sizeof(double), cudaMemcpyHostToDevice);

    // Step 3: Run GPU kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N_events + threadsPerBlock - 1) / threadsPerBlock;
    generate_events_GPU<<<blocksPerGrid, threadsPerBlock>>>(x, Q2, weights, d_pdf_values, N_events);
    cudaDeviceSynchronize();

    // Step 4: Write results to file
    ofstream output("/home/ubuntu/DIS_EventGenerator/data/events_gpu.dat");
    for (int i = 0; i < N_events; i++){
        output << x[i] << " " << Q2[i] << " " << weights[i] << endl;
    }
    output.close();

    // Cleanup
    cudaFree(x); cudaFree(Q2); cudaFree(weights); cudaFree(d_pdf_values);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Generated " << N_events << " events on GPU!" << endl;
    cout << "Time taken: " << duration.count() / 1000.0 << " seconds" << endl;

    return 0;
}

// /usr/local/cuda/bin/nvcc src/dis_generator_gpu.cu -o build/dis_generator_gpu -O3     -I/usr/local/include/LHAPDF -L/usr/local/lib -lLHAPDF     -Xcompiler -fopenmp