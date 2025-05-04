#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#define WIDTH 1024
#define HEIGHT 1024

__device__ float3 operator*(float s, const float3& v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(const float3& v) {
    float inv_len = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

struct Sphere {
    float3 center;
    float radius;
};

__device__ float hit_sphere(const float3& ray_origin, const float3& ray_dir, const Sphere& sphere) {
    float3 oc = ray_origin - sphere.center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0f * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return -1.0f;
    return (-b - sqrtf(discriminant)) / (2.0f * a);
}

__global__ void raytrace_kernel(unsigned char* output, int width, int height, Sphere sphere, float3 light_dir) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float u = (2.0f * x / width - 1.0f) * (float)width / height;
    float v = 1.0f - 2.0f * y / height;

    float3 ray_origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 ray_dir = normalize(make_float3(u, v, -1.0f));

    float t = hit_sphere(ray_origin, ray_dir, sphere);

    unsigned char color = 0;
    if (t > 0.0f) {
        float3 hit_point = ray_origin + t * ray_dir;
        float3 normal = normalize(hit_point - sphere.center);
        float intensity = fmaxf(0.0f, dot(normal, light_dir));
        color = static_cast<unsigned char>(255 * intensity);
    }

    output[y * width + x] = color;
}

float3 normalize_host(const float3& v) {
    float inv_len = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

int main() {
    size_t image_size = WIDTH * HEIGHT;
    unsigned char* d_output;
    unsigned char* h_output = new unsigned char[image_size];

    cudaMalloc(&d_output, image_size);

    Sphere sphere;
    sphere.center = make_float3(0.0f, 0.0f, -3.0f);
    sphere.radius = 1.0f;

    float3 light_dir = normalize_host(make_float3(-1.0f, -1.0f, -1.0f));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    raytrace_kernel<<<numBlocks, threadsPerBlock>>>(d_output, WIDTH, HEIGHT, sphere, light_dir);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    std::ofstream file("raytracer.pgm");
    file << "P2\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int i = 0; i < image_size; ++i) {
        file << (int)h_output[i] << " ";
        if ((i + 1) % WIDTH == 0) file << "\n";
    }
    file.close();

    cudaFree(d_output);
    delete[] h_output;

    std::cout << "Raytraced image written to raytracer.pgm\n";
    return 0;
}