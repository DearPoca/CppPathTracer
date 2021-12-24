#ifndef RAY_TRACER_H_3252363
#define RAY_TRACER_H_3252363

#include <vector>

#include "object.h"
#include "path_tracing_common.h"

#define MAX_RECURSION_DEPTH 5

class MotionalCamera {
private:
    int width_;
    int height_;

    Float4 origin_;
    Float4 look_at_;
    float view_fov_ = 20;        //视角
    float dist_to_focus_ = 10;   //焦距
    float lens_radius_ = 0.05f;  //孔半径

    Float4 u_, v_, w_;
    Float4 top_left_corner_;
    Float4 horizontal_;
    Float4 vertical_;

public:
    MotionalCamera();
    MotionalCamera(int width, int height);

    ~MotionalCamera();

    void Resize(int width, int height);
    void SetOrigin(Float4 ori);
    void SetOrigin(float x, float y, float z);

    void SetLookAt(Float4 lookAt);
    void SetLookAt(float x, float y, float z);

    void SetViewFov(float fov);

    void Updata();

    __device__ Ray RayGen(int x, int y, curandState* state);
};

class PathTracer {
private:
    int width_;
    int height_;
    uint spp_ = 5;
    uint8_t max_recursion_depth_ = MAX_RECURSION_DEPTH;
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