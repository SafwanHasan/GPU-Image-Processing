# CUDA-Accelerated Sobel Edge Detection

## Project Overview
This project implements a **CUDA-accelerated Sobel edge detection pipeline** for processing large images in real-time. The pipeline includes both CPU and GPU implementations, allowing benchmarking and performance comparison. The GPU implementation leverages **shared memory**, **thread blocks**, and **grid configuration tuning** to optimize throughput and reduce memory latency.  

## Features
- **CPU and GPU Sobel filters** for grayscale images  
- **Parallelized GPU kernel** using shared memory tiles and block/grid layout  
- **Performance benchmarking**: logs CPU vs GPU runtime for each image  
- Supports batch processing of images in a folder  

## Setup & Usage
1. **Requirements**:
   - Windows 10/11  
   - Visual Studio 2022  
   - CUDA Toolkit (compatible with your GPU)  
   - C++17 standard enabled in project settings  

2. **Folder structure**:
cuda_edge_detection/\
├─ src/\
│ ├─ main.cpp\
│ ├─ kernel.cu\
│ ├─ utils.h\
│ ├─ utils_impl.cpp\
│ ├─ stb_image.h\
│ └─ stb_image_write.h\
├─ images/ ← Input images (PNG)\
└─ outputs/ ← Processed CPU and GPU edge maps


3. **Build & Run**:
   - Open the Visual Studio solution  
   - Ensure **C++17** and **CUDA build customizations** are enabled  
   - Press **F5** to build and run  
   - Outputs are saved in the `outputs/` folder as `cpu_*.png` and `gpu_*.png`  

## Performance
Example runtimes on 1024×1024 aerial images:  
| Image       | CPU Time (ms) | GPU Kernel (ms) | Speedup |
|------------|---------------|----------------|---------|
| 2.2.20.png | 15.16         | 3.41           | ~4.5×   |
| 2.2.21.png | 15.13         | 3.41           | ~4.4×   |
| 2.2.22.png | 15.16         | 4.53           | ~3.3×   |
| wash-ir.png (2250×2250) | 75.21 | 23.11 | ~3.3× |

> Note: GPU speedup varies depending on image size and memory usage.  

## Notes
- Outputs are **grayscale edge maps**.  
- CPU and GPU outputs should look visually similar.  
- The pipeline is reproducible and can process any set of PNG grayscale images.  
