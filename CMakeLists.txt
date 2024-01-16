cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(GpuCommunicationSimulate LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_COMPILER hipcc)

# Add definitions for HIP
add_definitions(-D__HIP_PLATFORM_AMD__)

# Set ROCM_PATH environment variable
set(ENV{ROCM_PATH} "/opt/rocm")

# Set the CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};/opt/rocm/lib/cmake/hip")

# Find HIP package
find_package(HIP REQUIRED)

# Include directories
include_directories(/opt/rocm/include)

# Define the executable
add_executable(GpuCommunicationSimulate main.cpp)
