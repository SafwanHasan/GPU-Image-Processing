#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include "stb_image.h"
#include "stb_image_write.h"

// CUDA Sobel wrapper
extern "C" bool sobel_gpu(const uint8_t * h_input_gray, int width, int height, float* h_output_magnitude, float& kernel_ms_out);

namespace fs = std::filesystem;

// Simple CPU Sobel
void sobel_cpu(const uint8_t* img, float* out, int width, int height) {
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gx = -img[(y - 1) * width + (x - 1)] - 2 * img[y * width + (x - 1)] - img[(y + 1) * width + (x - 1)]
                + img[(y - 1) * width + (x + 1)] + 2 * img[y * width + (x + 1)] + img[(y + 1) * width + (x + 1)];
            int gy = -img[(y - 1) * width + (x - 1)] - 2 * img[(y - 1) * width + x] - img[(y - 1) * width + (x + 1)]
                + img[(y + 1) * width + (x - 1)] + 2 * img[(y + 1) * width + x] + img[(y + 1) * width + (x + 1)];
            out[y * width + x] = sqrtf(float(gx * gx + gy * gy));
        }
    }
}

// Save float array as 8-bit PNG
void save_float_image(const float* data, int width, int height, const std::string& path) {
    std::vector<uint8_t> buf(width * height);
    for (int i = 0; i < width * height; ++i) {
        float val = data[i];
        if (val > 255.f) val = 255.f;
        buf[i] = static_cast<uint8_t>(val);
    }
    stbi_write_png(path.c_str(), width, height, 1, buf.data(), width);
}

int main() {
    // Absolute paths
    std::string img_folder = "D:/personal/gpu/cuda_edge_detection/cuda_edge_detection/images/";
    std::string out_folder = "D:/personal/gpu/cuda_edge_detection/cuda_edge_detection/outputs/";

    // create outputs folder if missing
    fs::create_directories(out_folder);

    // Iterate over all PNG files in images folder
    for (const auto& entry : fs::directory_iterator(img_folder)) {
        if (entry.path().extension() != ".png") continue;

        std::string fname = entry.path().filename().string();
        int w, h, ch;
        uint8_t* img = stbi_load(entry.path().string().c_str(), &w, &h, &ch, 1); // grayscale
        if (!img) { std::cerr << "Failed to load " << fname << std::endl; continue; }

        std::cout << "Loaded " << fname << " (" << w << "x" << h << ")" << std::endl;

        // CPU Sobel
        std::vector<float> cpu_out(w * h, 0.f);
        auto t1 = std::chrono::high_resolution_clock::now();
        sobel_cpu(img, cpu_out.data(), w, h);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        // GPU Sobel
        std::vector<float> gpu_out(w * h, 0.f);
        float kernel_ms;
        bool ok = sobel_gpu(img, w, h, gpu_out.data(), kernel_ms);
        if (!ok) { std::cerr << "GPU Sobel failed for " << fname << std::endl; stbi_image_free(img); continue; }

        // Save outputs
        save_float_image(cpu_out.data(), w, h, out_folder + "cpu_" + fname);
        save_float_image(gpu_out.data(), w, h, out_folder + "gpu_" + fname);

        std::cout << fname << " | CPU: " << cpu_ms << " ms | GPU kernel: " << kernel_ms << " ms" << std::endl;

        stbi_image_free(img);
    }

    std::cout << "\nAll images processed! Results are in " << out_folder << std::endl;
    return 0;
}
