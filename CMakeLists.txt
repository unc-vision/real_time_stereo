cmake_minimum_required(VERSION 3.1)

project(RealTimeStereo)

set (CMAKE_CXX_STANDARD 11)


################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

if(EXISTS ${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
    include(${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
endif()


################################################################################
# Find packages
################################################################################

find_package(CUDA REQUIRED)
find_package(Glog REQUIRED)
find_package(FlyCapture2 REQUIRED)
find_package(OpenCV REQUIRED)


################################################################################
# Compiler specific configuration
################################################################################

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \
  -std=c++11 -O3 --expt-extended-lambda --expt-relaxed-constexpr")


################################################################################
# Add sources
################################################################################

add_subdirectory(src)


