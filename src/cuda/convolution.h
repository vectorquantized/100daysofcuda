#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#define TILE_WIDTH 16
#define OUT_TILE_WIDTH 12
#define SHARED_INDEX(row, col, in_tile_width) ((row) * in_tile_width + (col))

template<typename T>
__global__ void conv2d_tiled_channel(const T* __restrict__ input, const T* __restrict__ kernel,
                            T* __restrict__ output, int filter_size, int filter_radius, 
                            int batch_size, int in_channels, int height, int width, 
                            int out_height, int out_width) {

    // Calculate output positions
    const int row_out = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y;
    const int col_out = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x;
    const int batch_idx = blockIdx.z;

    extern __shared__ T m_shared[];
    // we could just use in_tile_wdith = blockDim.x;
    const int in_tile_width = OUT_TILE_WIDTH + filter_size - 1;

    const int row_in = row_out;
    const int col_in = col_out;

    // Calculate batch offsets
    const T* input_batch = input + batch_idx * in_channels * height * width;
    T* output_batch = output + batch_idx * out_height * out_width;

    // Initialize accumulator
    T p_value = static_cast<T>(0);

    // Process each input channel
    // Idea is to load one channel, calculate the result
    // accumulate the subsequent convs on p_value until all channels are done.
    for(int c = 0; c < in_channels; ++c) {
        // we need this as we load the values per channel 
        const int shared_offset = c * in_tile_width * in_tile_width;
        
        // Load input to shared memory
        if (row_in < height && col_in < width) {
            m_shared[shared_offset + SHARED_INDEX(threadIdx.y, threadIdx.x, in_tile_width)] =
                input_batch[c * height * width + row_in * width + col_in];
        } else {
            m_shared[shared_offset + SHARED_INDEX(threadIdx.y, threadIdx.x, in_tile_width)] = static_cast<T>(0);
        }
        
        __syncthreads();

        // current thread that is producing the output, should stay within bounds.
        // also, we want the row_out and col_out to be smaller than output size.
        if(threadIdx.y < OUT_TILE_WIDTH && threadIdx.x < OUT_TILE_WIDTH && 
            row_out < out_height && col_out < out_width) {
            // kernel is also laid out per channel.
            const int kernel_offset = c * filter_size * filter_size;
            // all threads have stored the corresponding tiles inside shared memory
            // we can make each thread calculate the output for one output cell by 
            // iterating over kernel values.
            // TODO: still need to ensure if we need #pragma here
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

    // Write output for valid cells.
    if (threadIdx.y < OUT_TILE_WIDTH && threadIdx.x < OUT_TILE_WIDTH && 
        row_out < out_height && col_out < out_width) {
        output_batch[row_out * out_width + col_out] = p_value;
    }
}

template<typename T>
__global__ void conv2d_tiled(const T* __restrict__ input, const T* __restrict__ kernel,
                             T* __restrict__ output, int filter_size, int filter_radius, int batch_size,
                             int width, int height, int out_width, int out_height) {

    int row_out = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y;
    int col_out = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x;

    int batch_idx = blockIdx.z;

    int row_in = row_out;
    int col_in = col_out;

    int in_tile_width = blockDim.x;
    extern __shared__ T m_shared[];

    const T* input_batch = input + batch_idx * height * width;
    T* output_batch = output + batch_idx * out_height * out_width;

    int row_shared = threadIdx.y;
    int col_shared = threadIdx.x;

    if (row_in >= 0 && row_in < height &&
        col_in >= 0  && col_in < width) {
            m_shared[SHARED_INDEX(row_shared, col_shared, in_tile_width)] = input_batch[row_in * width + col_in];
    } else {
            m_shared[SHARED_INDEX(row_shared, col_shared, in_tile_width)] = static_cast<T>(0);
    }
    
    __syncthreads();

    if(row_shared < OUT_TILE_WIDTH && col_shared < OUT_TILE_WIDTH) {
        T p_value = static_cast<T>(0);
        for(int i = 0; i < filter_size; ++i) {
            for(int j = 0; j < filter_size; ++j) {
                p_value += kernel[i * filter_size + j] * m_shared[SHARED_INDEX(row_shared + i, col_shared + j, in_tile_width)];
            }
        }
        if (row_out < out_height && col_out < out_width) {
            output_batch[row_out * out_width + col_out] = p_value;
        }
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