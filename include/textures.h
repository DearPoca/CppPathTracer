#pragma once

#include <string>

#include "ray_tracing_common.h"

class PocaTextureUtils {
public:
	static cudaTextureObject_t AddTexByFile(std::string file_path,
		cudaTextureAddressMode addr_mode = cudaAddressModeMirror,
		cudaTextureFilterMode filter_mode = cudaFilterModeLinear);

	static __device__ float4 GetTexture2D(cudaTextureObject_t tex_obj, float u, float v);

	static void DestroyTexture(cudaTextureObject_t tex);
};

