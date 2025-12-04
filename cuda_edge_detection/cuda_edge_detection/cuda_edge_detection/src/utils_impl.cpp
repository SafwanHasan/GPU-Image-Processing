#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "utils.h"
#include <iostream>
#include <cstring>

// load image and convert to grayscale
bool load_image_grayscale(const std::string& path, std::vector<uint8_t>& out_gray, int& width, int& height) {
    int channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "stbi_load failed: " << stbi_failure_reason() << "\n";
        return false;
    }
    if (channels < 3) {
        // treat as grayscale already
        out_gray.resize(width * height);
        if (channels == 1) {
            std::memcpy(out_gray.data(), data, width * height);
        }
        else {
            // unexpected
            for (int i = 0; i < width * height; ++i) out_gray[i] = data[i];
        }
        stbi_image_free(data);
        return true;
    }
    else {
        rgb_to_grayscale(data, width, height, channels, out_gray);
        stbi_image_free(data);
        return true;
    }
}

void rgb_to_grayscale(const uint8_t* rgb, int width, int height, int channels, std::vector<uint8_t>& out_gray) {
    out_gray.resize(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x);
            const uint8_t* pix = rgb + idx * channels;
            // standard luma
            float r = pix[0], g = pix[1], b = pix[2];
            uint8_t gval = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
            out_gray[idx] = gval;
        }
    }
}

bool save_image_grayscale(const std::string& path, const std::vector<uint8_t>& gray, int width, int height) {
    // stbi_write_png wants row stride
    int stride = width;
    int ok = stbi_write_png(path.c_str(), width, height, 1, gray.data(), stride);
    return ok != 0;
}
