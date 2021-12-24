cmake_minimum_required(VERSION 3.9)
project(CUDAPathTracer)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-11.5/include)

link_directories(${CMAKE_SOURCE_DIR}/libs)

aux_source_directory(${CMAKE_SOURCE_DIR}/cppSrc CPP_SRCS)
aux_source_directory(${CMAKE_SOURCE_DIR}/cuSrc CUDA_SRCS)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CUDA_PT_LIB_NAME cudaPathTracer)
add_library(${CUDA_PT_LIB_NAME} ${CUDA_SRCS})
set_target_properties( ${CUDA_PT_LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${CUDA_PT_LIB_NAME} PROPERTY CUDA_ARCHITECTURES 70-real 72-virtual)

add_executable(${PROJECT_NAME} ${CPP_SRCS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70-real 72-virtual)

set(LIBS avformat avcodec avutil swscale swresample opencv_core opencv_imgproc opencv_photo x264)
target_link_libraries(${PROJECT_NAME} ${LIBS})
target_link_libraries(${PROJECT_NAME} ${CUDA_PT_LIB_NAME})
