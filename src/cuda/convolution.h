#ifndef CONVOLUTION_H
#define CONVOLUTION_H

template<typename T>
__global__ void conv2D(const T* __restrict__ input,
                       const T* __restrict__ conv_mask,
                       T* __restrict__ output,
                       int filter_radius, int batch_size, 
                       int channels, int height, int width ) {
    int row_out = blockIdx.y * blockDim.y + threadIdx.y;
    int col_out = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.z;

    int filter_size = 2 * filter_radius + 1;
    int out_height = height - 2 * filter_radius;
    int out_width = width - 2 * filter_radius;

    if(row_out < out_height && col_out < out_width) {
        T p_value = static_cast<T>(0);
        // For each channel, accumulate its contribution.
        for (int c = 0; c < channels; c++) {
            // Compute pointers for channel c of the current batch.
            const T* input_channel = input + batch_idx * channels * width * height + c * width * height;
            const T* mask_channel = conv_mask + c * filter_size * filter_size;
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    int row_in = row_out  + i;
                    int col_in = col_out  + j;
                    if (row_in >= 0 && row_in < height && col_in >= 0 && col_in < width) {
                        p_value += mask_channel[i * filter_size + j] * input_channel[row_in * width + col_in];
                    }
                }
            }
        }
        output[batch_idx * out_height * out_width + row_out * out_width + col_out] = p_value;
    }
}

#endif // CONVOLUTION_H