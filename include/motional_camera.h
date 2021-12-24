#ifndef MOTIONAL_CAMERA_H
#define MOTIONAL_CAMERA_H

#include "path_tracing_common.h"

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

#endif