cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(GpuCommunicationSimulate LANGUAGES CXX)

# Find HIP package
find_package(HIP REQUIRED)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the executable
add_executable(GpuCommunicationSimulate main.cpp)

# Link against HIP
target_link_libraries(GpuCommunicationSimulate PRIVATE hip::device)
