#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#define WIDTH 1024
#define HEIGHT 1024
#define MAX_ITER 1000

__global__ void mandelbrot_kernel(unsigned char* output, int width, int height, float xmin, float xmax, float ymin, float ymax, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float real = xmin + x * (xmax - xmin) / width;
    float imag = ymin + y * (ymax - ymin) / height;

    float z_real = 0.0f;
    float z_imag = 0.0f;
    int iter = 0;

    while (z_real * z_real + z_imag * z_imag <= 4.0f && iter < max_iter) {
        float temp = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2.0f * z_real * z_imag + imag;
        z_real = temp;
        iter++;
    }

    int idx = y * width + x;
    output[idx] = (unsigned char)(255 * iter / max_iter);
}

int main() {
    size_t image_size = WIDTH * HEIGHT;
    unsigned char* d_output;
    unsigned char* h_output = new unsigned char[image_size];

    cudaMalloc(&d_output, image_size);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(d_output, WIDTH, HEIGHT, -2.0f, 1.0f, -1.5f, 1.5f, MAX_ITER);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    // Write output to PGM file
    std::ofstream file("mandelbrot.pgm");
    file << "P2\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int i = 0; i < image_size; ++i) {
        file << (int)h_output[i] << " ";
        if ((i + 1) % WIDTH == 0) file << "\n";
    }
    file.close();

    cudaFree(d_output);
    delete[] h_output;

    std::cout << "Mandelbrot image written to mandelbrot.pgm\n";
    return 0;
}