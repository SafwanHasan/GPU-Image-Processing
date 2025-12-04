#pragma once
#pragma once
#include <string>
#include <vector>
#include <cstdint>

// Load an image (RGB or RGBA) and convert to grayscale (8-bit).
// Returns grayscale pixels as uint8_t vector and sets width/height.
// Channels read from disk are in 'channels' (3 or 4).
bool load_image_grayscale(const std::string& path, std::vector<uint8_t>& out_gray, int& width, int& height);

// Save an 8-bit grayscale image as PNG
bool save_image_grayscale(const std::string& path, const std::vector<uint8_t>& gray, int width, int height);

// Convert RGB(A) buffer to grayscale in-place (helper if needed)
void rgb_to_grayscale(const uint8_t* rgb, int width, int height, int channels, std::vector<uint8_t>& out_gray);

// CPU reference: sobel on grayscale (8-bit in, float out magnitude)
void sobel_cpu(const uint8_t* gray_in, int width, int height, float* out_magnitude);

// Utility to clamp values
inline int clampi(int v, int a, int b) { return (v < a ? a : (v > b ? b : v)); }
