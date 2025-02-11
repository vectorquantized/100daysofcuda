#ifndef CONVOLUTION_H
#define CONVOLUTION_H

template<typename T>
__global__ void conv2D(const T* __restrict__ input, const T* __restrict__ kernel, 
                       T* __restrict__ output, int batch_size, int in_channels, int rows, int cols, int filter_radius) {

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_idx = blockIdx.z;

    int filter_size = 2 * filter_radius + 1;
    int out_rows = rows - 2 * filter_radius;
    int out_cols = cols - 2 * filter_radius;

    T p_value = static_cast<T>(0);
    if(row < out_rows && col < out_cols) {
        for(int c = 0; c < in_channels; ++c) {
            const T* input_for_channel = input + batch_idx * in_channels * rows * cols + c * rows * cols;
            const T* kernel_channel = kernel + c * filter_size * filter_size;
            for(int i = 0; i < filter_size; ++i) {
                for(int j = 0; j < filter_size; ++j) {
                    int in_row = row - filter_radius + i;
                    int in_col = col - filter_radius + j;
                    if (in_row >= 0 && in_row < rows && in_col >= 0 && in_col < cols) {
                        p_value += kernel_channel[i * filter_size + j] * input_for_channel[in_row * cols + in_col];
                    }
                }
            }
        }
        output[batch_idx * out_rows * out_cols + row * out_cols + col] = p_value;
    }
}

#endif // CONVOLUTION_H