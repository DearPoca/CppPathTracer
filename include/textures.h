#pragma once

#include <string>

#include "ray_tracing_common.h"

#define MAX_TEXTURES_AMOUNT 8

cudaTextureObject_t AddTexByFile(std::string file_path, cudaTextureAddressMode addr_mode = cudaAddressModeMirror, cudaTextureFilterMode filter_mode = cudaFilterModeLinear);

__device__ float4 GetTexture2D(cudaTextureObject_t tex_obj, float u, float v);


