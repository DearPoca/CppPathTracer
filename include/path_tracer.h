#ifndef RAY_TRACER_H_3252363
#define RAY_TRACER_H_3252363

#include <vector>

#include "bvh.h"
#include "motional_camera.h"
#include "object.h"
#include "path_tracing_common.h"

#define MAX_RECURSION_DEPTH 32

class PathTracer {
private:
    int width_;
    int height_;
    uint spp_ = 5;
    uint8_t max_recursion_depth_ = 5;
    // Float4* render_target_gpu_handle_;
    uint8_t* output_buffer_gpu_handle_;

    MotionalCamera* camera_;
    MotionalCamera* camera_gpu_handle_;

    std::vector<Material*> materials_;
    // Material** materials_gpu_handle_;
    std::map<void*, void*> materials_cpu_handle_to_gpu_handle_;

    std::vector<Object*> scene_;
    std::vector<Object*> scene_tmp_buffer_;
    Object** scene_gpu_handle_;
    std::map<void*, void*> object_cpu_handle_to_gpu_handle_;

    curandState* d_rng_states_;

    BVHNode* bvh_root_;

public:
    void AddMeterial(Material* material);

    void AddObject(Object* obj);

    void AllocateGpuMemory();

    void DispatchRay(uint8_t* buf, int size, int64_t t);

    void ReSize(int width, int height);

    void SetCamera(MotionalCamera* camera);

    void SetSamplePerPixel(uint spp);
};

#endif  // RAY_TRACER_H_3252363