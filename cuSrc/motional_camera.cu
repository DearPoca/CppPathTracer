
#include "motional_camera.h"

MotionalCamera::MotionalCamera() {}
MotionalCamera::MotionalCamera(int width, int height) : width_(width), height_(height) {}

MotionalCamera::~MotionalCamera() {}

void MotionalCamera::SetViewFov(float fov) { view_fov_ = fov; }

void MotionalCamera::Resize(int width, int height) {
    width_ = width;
    height_ = height;
}
void MotionalCamera::SetOrigin(Float4 ori) { origin_ = ori; }
void MotionalCamera::SetOrigin(float x, float y, float z) {
    origin_.x = x;
    origin_.y = y;
    origin_.z = z;
}

void MotionalCamera::SetLookAt(Float4 look_at) { look_at_ = look_at; }

void MotionalCamera::SetLookAt(float x, float y, float z) {
    look_at_.x = x;
    look_at_.y = y;
    look_at_.z = z;
}

void MotionalCamera::Updata() {
    float theta = view_fov_ * M_PI / 180;
    float aspectRatio = float(width_) / float(height_);
    float half_height = tan(theta / 2);
    float half_width = aspectRatio * half_height;
    Float4 vup(0.0f, 1.0f, 0.0f);

    w_ = poca_mus::GetNormalizeVec(origin_ - look_at_);
    u_ = poca_mus::GetNormalizeVec(poca_mus::Cross(vup, w_));
    v_ = poca_mus::Cross(w_, u_);

    dist_to_focus_ = poca_mus::Length(origin_ - look_at_);

    top_left_corner_ =
        origin_ - half_width * dist_to_focus_ * u_ + half_height * dist_to_focus_ * v_ - dist_to_focus_ * w_;
    horizontal_ = 2 * half_width * dist_to_focus_ * u_;
    vertical_ = -2 * half_height * dist_to_focus_ * v_;
}

__device__ Ray MotionalCamera::RayGen(int x, int y, curandState* state) {
    Float4 rd = lens_radius_ * poca_mus::GpuCreateRandomFloat4(state);
    Float4 offset = u_ * rd.x + v_ * rd.y;
    Ray ray;
    float dx = float(x) / float(width_);
    float dy = float(y) / float(height_);
    ray.origin = origin_ + offset;
    ray.dir = top_left_corner_ + dx * horizontal_ + dy * vertical_ - origin_ - offset;
    ray.tmin = 0.f;
    ray.tmax = DEFAULT_RAY_TMAX;
    return ray;
}
