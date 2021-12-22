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

    float4 origin_;
    float4 look_at_;
    float view_fov_ = 20;        //视角
    float dist_to_focus_ = 10;   //焦距
    float lens_radius_ = 0.05f;  //孔半径

    float4 u_, v_, w_;
    float4 top_left_corner_;
    float4 horizontal_;
    float4 vertical_;

public:
    MotionalCamera();
    MotionalCamera(int width, int height);

    ~MotionalCamera();

    void Resize(int width, int height);
    void SetOrigin(float4 ori);
    void SetOrigin(float x, float y, float z);

    void SetLookAt(float4 lookAt);
    void SetLookAt(float x, float y, float z);

    void SetViewFov(float fov);

    void Updata();

    Ray RayGen(int x, int y);
};

class Scene {
private:
    void MissShader(Ray& ray, RayPayload& payload);

    std::vector<Object*> objs_;

public:
    Scene();
    ~Scene();

    void TraceRay(Ray& ray, RayPayload& payload);

    void AddObject(Object* obj);
};

class PathTracer {
private:
    int width_;
    int height_;
    uint8_t max_recursion_depth_ = MAX_RECURSION_DEPTH;
    float4* render_target_;

    MotionalCamera* camera_;
    Scene* scene_;

    float4 SamplePixel(int x, int y);

public:
    float (*time_to_ori_x)(int64_t);
    float (*time_to_ori_y)(int64_t);
    float (*time_to_ori_z)(int64_t);

    float (*time_to_look_at_x)(int64_t);
    float (*time_to_look_at_y)(int64_t);
    float (*time_to_look_at_z)(int64_t);

    void DispatchRay(uint8_t* buf, int size, int64_t t);
    void ReSize(int width, int height);

    void SetCamera(MotionalCamera* camera);
    void SetScene(Scene* scene);
};

#endif  // RAY_TRACER_H_3252363