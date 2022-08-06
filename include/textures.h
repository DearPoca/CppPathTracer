#pragma once

#include <string>

#include "ray_tracing_common.h"

#define MAX_TEXTURES_AMOUNT 8

cudaTextureObject_t AddTexByFile(std::string file_path);

__device__ float4 GetTexture2D(cudaTextureObject_t tex_obj, float u, float v);


