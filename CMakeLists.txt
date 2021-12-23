cmake_minimum_required(VERSION 3.9)
project(MiniPathTracer CXX CUDA)

SET(CMAKE_CXX_COMPILER "/usr/bin/c++")
SET(CMAKE_C_COMPILER "/usr/bin/cc")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-11.5/include)

link_directories(${CMAKE_SOURCE_DIR}/libs)

aux_source_directory(${CMAKE_SOURCE_DIR}/src PATH_TRACER_SRCS)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(${PROJECT_NAME} ${PATH_TRACER_SRCS})

set(LIBS avformat avcodec avutil swscale swresample opencv_core opencv_imgproc x264)
target_link_libraries(${PROJECT_NAME} ${LIBS})
