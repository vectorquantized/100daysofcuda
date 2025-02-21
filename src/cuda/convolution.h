#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#define TILE_WIDTH 16
#define OUT_TILE_WIDTH 12
#define SHARED_INDEX(row, col, in_tile_width) ((row) * in_tile_width + (col))


template<typename T>
__global__ void conv2d_tiled(const T* __restrict__ input, const T* __restrict__ kernel,
                            T* __restrict__ output, int filter_size, int filter_radius, 
                            int batch_size, int in_channels, int height, int width, 
                            int out_height, int out_width) {

    // Calculate output position
    int row_out = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y;
    int col_out = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x;
    int batch_idx = blockIdx.z;

    // Calculate input position with filter offset
    int row_in = row_out; // - filter_radius;
    int col_in = col_out;// - filter_radius;

    // Shared memory setup
    int in_tile_width = blockDim.x;
    extern __shared__ T m_shared[];

    // Calculate batch offsets
    const T* input_batch = input + batch_idx * in_channels * height * width;
    T* output_batch = output + batch_idx * out_height * out_width;

    // Initialize accumulator
    T p_value = static_cast<T>(0);

    int tile_origin_row = blockIdx.y * OUT_TILE_WIDTH;
    int tile_origin_col = blockIdx.x * OUT_TILE_WIDTH;
    int global_row = tile_origin_row + threadIdx.y;
    int global_col = tile_origin_col + threadIdx.x;

    // Process each input channel
    for(int c = 0; c < in_channels; ++c) {
        // Calculate channel offset in shared memory
        int shared_offset = c * in_tile_width * in_tile_width;

        // Load input to shared memory
        if (global_row < height && global_col < width) {
            m_shared[shared_offset + SHARED_INDEX(threadIdx.y, threadIdx.x, in_tile_width)] =
                input_batch[c * height * width + global_row * width + global_col];
        }
        else {
            m_shared[shared_offset + SHARED_INDEX(threadIdx.y, threadIdx.x, in_tile_width)] = static_cast<T>(0);
        }
        
        __syncthreads();

        // Compute convolution if thread is responsible for output pixel
        if(threadIdx.y < OUT_TILE_WIDTH && threadIdx.x < OUT_TILE_WIDTH) {
            int kernel_offset = c * filter_size * filter_size;
            
            #pragma unroll
            for(int i = 0; i < filter_size; ++i) {
                #pragma unroll
                for(int j = 0; j < filter_size; ++j) {
                    p_value += kernel[kernel_offset + i * filter_size + j] * 
                              m_shared[shared_offset + SHARED_INDEX(threadIdx.y + i, threadIdx.x + j, in_tile_width)];
                }
            }
        }
        
        __syncthreads();
    }

    // Write output
    if (row_out < out_height && col_out < out_width) {
        output_batch[row_out * out_width + col_out] = p_value;
    }
}

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
            output[((batch_idx * out_channels + out_c) * out_height + row_out) * out_width + col_out] = p_value;
        }
    }
}


#endif // CONVOLUTION_H