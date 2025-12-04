#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

// Kernel configuration
constexpr int TILE_DIM = 16; // blockDim.x = 16, blockDim.y = 16

// Device kernel: shared memory tile approach
// Each thread computes one output pixel inside the block (excluding 1-pixel halo handled by loading)
__global__ void sobel_shared_kernel(const uint8_t* __restrict__ d_in, int width, int height, float* d_out) {
    // block origin (top-left of tile) in global coordinates (pixel indices)
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Shared tile with halo of 1 pixel around
    __shared__ uint8_t tile[TILE_DIM + 2][TILE_DIM + 2];

    // global coordinate this thread will load (cover bigger area to fill halo)
    int gx = bx + tx - 1; // allow loading left halo
    int gy = by + ty - 1;

    // clamp coords
    int cx = gx;
    int cy = gy;
    if (cx < 0) cx = 0;
    if (cx >= width) cx = width - 1;
    if (cy < 0) cy = 0;
    if (cy >= height) cy = height - 1;

    // Each thread loads one element into shared mem (tile[ty][tx])
    tile[ty][tx] = d_in[cy * width + cx];

    // Need to ensure the whole tile (including halo rows / cols) is loaded.
    // Because our thread block is TILE_DIM x TILE_DIM but shared tile is (TILE_DIM+2)x(TILE_DIM+2),
    // we do additional loads: threads at edges load the extra halo elements.
    // Load right/ bottom halos by threads where tx/ty are within bounds.
    // We do 4 extra loads conditionally.

    // Load the extra column at right (tx == TILE_DIM-1) to cover tx+1 and tx+2 positions
    if (tx == TILE_DIM - 1) {
        int gx2 = bx + tx + 1 - 1; // tx+1 minus halo
        int cx2 = gx2;
        if (cx2 < 0) cx2 = 0;
        if (cx2 >= width) cx2 = width - 1;
        tile[ty][tx + 1] = d_in[cy * width + cx2];
    }

    // Load the extra row at bottom
    if (ty == TILE_DIM - 1) {
        int gy2 = by + ty + 1 - 1;
        int cy2 = gy2;
        if (cy2 < 0) cy2 = 0;
        if (cy2 >= height) cy2 = height - 1;
        tile[ty + 1][tx] = d_in[cy2 * width + cx];
    }

    // Corner bottom-right
    if (tx == TILE_DIM - 1 && ty == TILE_DIM - 1) {
        int gx2 = bx + tx + 1 - 1;
        int gy2 = by + ty + 1 - 1;
        int cx2 = gx2; if (cx2 < 0) cx2 = 0; if (cx2 >= width) cx2 = width - 1;
        int cy2 = gy2; if (cy2 < 0) cy2 = 0; if (cy2 >= height) cy2 = height - 1;
        tile[ty + 1][tx + 1] = d_in[cy2 * width + cx2];
    }

    __syncthreads();

    // compute position of the output pixel this thread corresponds to
    int out_x = bx + tx;
    int out_y = by + ty;

    if (out_x >= 1 && out_x < width - 1 && out_y >= 1 && out_y < height - 1) {
        // locate center in shared tile (shift by +1 because shared tile includes halo)
        int sx = tx + 1;
        int sy = ty + 1;

        int gx_val = 0;
        int gy_val = 0;

        // read neighbor values from shared memory
        int p00 = tile[sy - 1][sx - 1];
        int p01 = tile[sy - 1][sx];
        int p02 = tile[sy - 1][sx + 1];
        int p10 = tile[sy][sx - 1];
        int p12 = tile[sy][sx + 1];
        int p20 = tile[sy + 1][sx - 1];
        int p21 = tile[sy + 1][sx];
        int p22 = tile[sy + 1][sx + 1];

        gx_val = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
        gy_val = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;

        float mag = sqrtf((float)(gx_val * gx_val + gy_val * gy_val));
        d_out[out_y * width + out_x] = mag;
    }
    else {
        // border: write zero
        if (out_x < width && out_y < height)
            d_out[out_y * width + out_x] = 0.0f;
    }
}

// Host wrapper: copies data, launches kernel, copies back.
// d_input_gray_host: pointer to host grayscale data (uint8_t*), width, height
// d_output_magnitude_host: host pointer to float* pre-allocated width*height
extern "C" bool sobel_gpu(const uint8_t * h_input_gray, int width, int height, float* h_output_magnitude, float& kernel_ms_out) {
    if (!h_input_gray || !h_output_magnitude) return false;
    size_t num_pixels = size_t(width) * size_t(height);

    uint8_t* d_in = nullptr;
    float* d_out = nullptr;
    cudaError_t err;

    // allocate device
    err = cudaMalloc((void**)&d_in, num_pixels * sizeof(uint8_t));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_in failed: " << cudaGetErrorString(err) << "\n"; return false; }
    err = cudaMalloc((void**)&d_out, num_pixels * sizeof(float));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_out failed: " << cudaGetErrorString(err) << "\n"; cudaFree(d_in); return false; }

    // copy host -> device
    err = cudaMemcpy(d_in, h_input_gray, num_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << "\n"; cudaFree(d_in); cudaFree(d_out); return false; }

    // grid / block
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sobel_shared_kernel << <grid, block >> > (d_in, width, height, d_out);
    cudaEventRecord(stop);

    // wait
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    kernel_ms_out = ms; // milliseconds measured by events

    // copy back
    err = cudaMemcpy(h_output_magnitude, d_out, num_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n"; cudaFree(d_in); cudaFree(d_out); return false; }

    // cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // reset device optional
    cudaDeviceSynchronize();

    return true;
}
