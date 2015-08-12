# CMakeLists.txt for GPU Image Toolbox project.
project(ImageToolbox)

# Required CMake version.
cmake_minimum_required(VERSION 2.8)

# Packages.
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# NVCC Flags.
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)

# Compile all .cu files and main file.
file(GLOB cu *.cu)
cuda_add_executable(ImageToolbox main.cu ${cu})

# Get OpenCV's libraries.
target_link_libraries(ImageToolbox ${OpenCV_LIBS})
