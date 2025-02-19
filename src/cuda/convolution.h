#ifndef CONVOLUTION_H
#define CONVOLUTION_H

template<typename T>
__global__ void conv2D(const T* __restrict__ input,
                       const T* __restrict__ conv_mask,
                       T* __restrict__ output,
                       int filter_radius, int batch_size, 
                       int in_channels, int out_channels, 
                       int height, int width ) {
    int row_out = blockIdx.y * blockDim.y + threadIdx.y;
    int col_out = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.z;
    int filter_size = 2 * filter_radius + 1;
    int out_height = height - 2 * filter_radius;
    int out_width = width - 2 * filter_radius;

    if (row_out < out_height && col_out < out_width) {
        for (int out_c = 0; out_c < out_channels; ++out_c) {
            T p_value = static_cast<T>(0);

            for (int in_c = 0; in_c < in_channels; in_c++) {
                const T* input_channel = input + batch_idx * in_channels * width * height + in_c * width * height;
                const T* mask_channel = conv_mask + (out_c * in_channels + in_c) * filter_size * filter_size;

                for (int i = 0; i < filter_size; ++i) {
                    for (int j = 0; j < filter_size; ++j) {
                        int row_in = row_out + i;
                        int col_in = col_out + j;
                        if (row_in >= 0 && row_in < height && col_in >= 0 && col_in < width) {
                            p_value += mask_channel[i * filter_size + j] * input_channel[row_in * width + col_in];
                        }
                    }
                }
            }

            // Correct output indexing
            output[((batch_idx * out_channels + out_c) * out_height + row_out) * out_width + col_out] = p_value;
        }
    }
}

#endif // CONVOLUTION_H