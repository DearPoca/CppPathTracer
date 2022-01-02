#ifndef TEXTURES_H_479237525
#define TEXTURES_H_479237525

#include <string>

#include "path_tracing_common.h"

#define MAX_TEXTURES_AMOUNT 8

namespace poca_mus {
    cudaTextureObject_t AddTexByFile(std::string file_path);

    __device__ Float4 GetTex2D(cudaTextureObject_t tex_obj, float u, float v);
}  // namespace poca_mus

#endif  // TEXTURES_H_479237525