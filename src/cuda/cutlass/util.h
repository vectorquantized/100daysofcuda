#ifndef CUTLASS_UTIL_H
#define CUTLASS_UTIL_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

template <typename T> struct TransposeParams {
  T *__restrict__ input;
  T *__restrict__ output;

  const int M;
  const int N;

  TransposeParams(T *__restrict__ input_, T *__restrict__ output_, int M_, int N_)
      : input(input_), output(output_), M(M_), N(N_) {}
};

template<typename Element>
struct ElementConverter {
  static __device__ Element from_float(float f) {
    return static_cast<Element>(f);
  }

  static __device__ float to_float(Element e) {
    return static_cast<float>(e);
  }
};

template<>
struct ElementConverter<cutlass::half_t> {
  static __device__ cutlass::half_t from_float(float f) {
    return cutlass::half_t(f);
  }
  static __device__ float to_float(cutlass::half_t h) {
    return float(h);
  }
};

template<typename Element>
__global__ void uniform_random_fill(
  Element* input,
  int size,
  uint64_t seed,
  Element low,
  Element high,
  int bits_less_than_one
) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
    float rnd = curand_uniform(&state);
    float flow = ElementConverter<Element>::to_float(low);
    float fhigh = ElementConverter<Element>::to_float(high);
    float frange = high - low;
    float fvalue = flow + frange * rnd;
    if (bits_less_than_one == 0) {
      fvalue = static_cast<float>(static_cast<int>(fvalue));
    }
    input[idx] = ElementConverter<Element>::from_float(fvalue);
  }

}

//template <typename T> int benchmark(void (*transpose)(int M, int N, T* input, T* output), int M, int N, int iterations=10, bool verify=true) {
// template <typename T, bool isTranspose = true, bool isFMA = false> int benchmark(void (*transpose)(TransposeParams<T> params), int M, int N, int iterations=10, bool verify=true) {
//   using namespace cute;

//   auto tensor_shape_S = make_shape(M, N);
//   auto tensor_shape_D = (isTranspose) ? make_shape(N, M) : make_shape(M, N);

//   // Allocate and initialize
//   thrust::host_vector<T> h_S(size(tensor_shape_S));       // (M, N)
//   thrust::host_vector<T> h_D(size(tensor_shape_D)); // (N, M)

//   for (size_t i = 0; i < h_S.size(); ++i)
//     h_S[i] = static_cast<T>(i);

//   thrust::device_vector<T> d_S = h_S;
//   thrust::device_vector<T> d_D = h_D;

//   TransposeParams<T> params(thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D.data()), M, N);

  
//   for (int i = 0; i < iterations; i++) {
//     auto t1 = std::chrono::high_resolution_clock::now();
//     transpose(params);
//     cudaError result = cudaDeviceSynchronize();
//     auto t2 = std::chrono::high_resolution_clock::now();
//     if (result != cudaSuccess) {
//       std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
//                 << std::endl;
//       return -1;
//     }
//     std::chrono::duration<double, std::milli> tDiff = t2 - t1;
//     double time_ms = tDiff.count();
//     int numThreads = 256;
//     // int numThreads = 128;
//     size_t bytes = !isFMA ? 2 * M * N * sizeof(T) : (M * N + M * numThreads) * sizeof(T);
    
//     std::cout << "Trial " << i << " Completed in " << time_ms << "ms ("
//               << 1e-6 * bytes / time_ms << " GB/s)"
//               << std::endl;

//     if(isFMA) {
//       uint64_t flops = numThreads * 2 * uint64_t(M) * uint64_t(N);
//       double AI = double(flops) / double(bytes);
//       std::cout << "Arithmetic Intensity = " << AI << std::endl;
//       std::cout << "TFLOPs/s = " << 1e-9 * flops / time_ms << std::endl;

//     }
//   }

//   if(verify) {
//     h_D = d_D;
  
//     int bad = 0;
//     if constexpr (isTranspose) {
//       auto transpose_function = make_layout(tensor_shape_S, LayoutRight{});
//       for (size_t i = 0; i < h_D.size(); ++i) 
//         if (h_D[i] != h_S[transpose_function(i)])
//           bad++;
//     } else {
//       for (size_t i = 0; i < h_D.size(); ++i) 
//         if (h_D[i] != h_S[i])
//           bad++;
//     }
  
//     if (bad > 0) {
//       std::cout << "Validation failed. Correct values: " << h_D.size()-bad << ". Incorrect values: " << bad << std::endl;
//     } else {
//       std::cout << "Validation success." << std::endl;
//     }
//   }
//   return 0;
// }

#endif // CUTLASS_UTIL_H