#include <iostream>
#include <fstream>
#include <cuComplex.h>
#include <cuda_runtime.h>

#define N 1024
#define DX 0.1f
#define DT 0.001f
#define NSTEPS 500

__global__ void evolve(cuFloatComplex* psi_new, const cuFloatComplex* psi_old, const float* V, float dx, float dt) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j <= 0 || j >= N - 1) return; // skip boundaries

    cuFloatComplex laplacian = cuCsubf(psi_old[j + 1], cuCaddf(cuCmulf(make_cuFloatComplex(2.0f, 0.0f), psi_old[j]), psi_old[j - 1]));
    laplacian = cuCdivf(laplacian, make_cuFloatComplex(dx * dx, 0.0f));

    cuFloatComplex potential = cuCmulf(make_cuFloatComplex(V[j], 0.0f), psi_old[j]);

    cuFloatComplex rhs = cuCsubf(laplacian, potential);
    rhs = cuCmulf(make_cuFloatComplex(0.0f, -1.0f), rhs);

    psi_new[j] = cuCaddf(psi_old[j], cuCmulf(make_cuFloatComplex(dt, 0.0f), rhs));
}

void write_density(const cuFloatComplex* h_psi, int step) {
    std::ofstream file("psi_" + std::to_string(step) + ".pgm");
    file << "P2\n" << N << " 1\n255\n";
    for (int j = 0; j < N; ++j) {
        float prob = cuCabsf(h_psi[j]);
        int intensity = (int)(255 * prob);
        file << intensity << " ";
    }
    file << "\n";
    file.close();
}

int main() {
    cuFloatComplex* d_psi1;
    cuFloatComplex* d_psi2;
    float* d_V;

    cuFloatComplex* h_psi = new cuFloatComplex[N];
    float* h_V = new float[N];

    for (int j = 0; j < N; ++j) {
        float x = (j - N / 2) * DX;
        float sigma = 1.0f;
        float k0 = 5.0f;
        float envelope = expf(-x * x / (2 * sigma * sigma));
        h_psi[j] = make_cuFloatComplex(envelope * cosf(k0 * x), envelope * sinf(k0 * x));
        h_V[j] = 0.0f;
    }

    cudaMalloc(&d_psi1, N * sizeof(cuFloatComplex));
    cudaMalloc(&d_psi2, N * sizeof(cuFloatComplex));
    cudaMalloc(&d_V, N * sizeof(float));

    cudaMemcpy(d_psi1, h_psi, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    for (int step = 0; step < NSTEPS; ++step) {
        evolve<<<numBlocks, threadsPerBlock>>>(d_psi2, d_psi1, d_V, DX, DT);
        std::swap(d_psi1, d_psi2);

        if (step % 50 == 0) {
            cudaMemcpy(h_psi, d_psi1, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
            write_density(h_psi, step);
            std::cout << "Saved psi_" << step << ".pgm\n";
        }
    }

    cudaFree(d_psi1);
    cudaFree(d_psi2);
    cudaFree(d_V);
    delete[] h_psi;
    delete[] h_V;

    std::cout << "Simulation completed.\n";
    return 0;
}