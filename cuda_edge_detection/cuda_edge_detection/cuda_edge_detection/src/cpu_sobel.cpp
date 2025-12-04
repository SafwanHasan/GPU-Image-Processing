#include "utils.h"
#include <cstring>
#include <cmath>

void sobel_cpu(const uint8_t* gray_in, int width, int height, float* out_magnitude) {
    // Sobel kernels
    // Gx = [-1 0 1; -2 0 2; -1 0 1]
    // Gy = [-1 -2 -1; 0 0 0; 1 2 1]
    std::memset(out_magnitude, 0, sizeof(float) * width * height);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            int gx = 0, gy = 0;

            // row y-1
            gx += -gray_in[(y - 1) * width + (x - 1)];
            gx += 0;
            gx += gray_in[(y - 1) * width + (x + 1)];
            gy += -gray_in[(y - 1) * width + (x - 1)];
            gy += -2 * gray_in[(y - 1) * width + x];
            gy += -gray_in[(y - 1) * width + (x + 1)];

            // row y
            gx += -2 * gray_in[y * width + (x - 1)];
            gx += 0;
            gx += 2 * gray_in[y * width + (x + 1)];
            // gy row middle adds 0

            // row y+1
            gx += -gray_in[(y + 1) * width + (x - 1)];
            gx += 0;
            gx += gray_in[(y + 1) * width + (x + 1)];
            gy += gray_in[(y + 1) * width + (x - 1)];
            gy += 2 * gray_in[(y + 1) * width + x];
            gy += gray_in[(y + 1) * width + (x + 1)];

            float mag = std::sqrt(float(gx * gx + gy * gy));
            out_magnitude[idx] = mag;
        }
    }

    // borders = 0 already
}
