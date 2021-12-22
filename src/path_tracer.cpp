#include "path_tracer.h"

#include <utility>

// ================================================================================================================
// ================================================ MotionalCamera ================================================
// ================================================================================================================

MotionalCamera::MotionalCamera() {}
MotionalCamera::MotionalCamera(int width, int height) : width_(width), height_(height) {}

MotionalCamera::~MotionalCamera() {}

void MotionalCamera::SetViewFov(float fov) { view_fov_ = fov; }

void MotionalCamera::Resize(int width, int height) {
    width_ = width;
    height_ = height;
}
void MotionalCamera::SetOrigin(float4 ori) { origin_ = ori; }
void MotionalCamera::SetOrigin(float x, float y, float z) {
    origin_.x = x;
    origin_.y = y;
    origin_.z = z;
}

void MotionalCamera::SetLookAt(float4 look_at) { look_at_ = look_at; }

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
    float4 vup(0.0f, 1.0f, 0.0f);

    w_ = poca_mus::GetNormalizeVec(origin_ - look_at_);
    u_ = poca_mus::GetNormalizeVec(poca_mus::Cross(vup, w_));
    v_ = poca_mus::Cross(w_, u_);

    top_left_corner_ =
        origin_ - half_width * dist_to_focus_ * u_ + half_height * dist_to_focus_ * v_ - dist_to_focus_ * w_;
    horizontal_ = 2 * half_width * dist_to_focus_ * u_;
    vertical_ = -2 * half_height * dist_to_focus_ * v_;
}

Ray MotionalCamera::RayGen(int x, int y) {
    float4 rd = lens_radius_ * poca_mus::CreateRandomFloat4();
    float4 offset = u_ * rd.x + v_ * rd.y;
    Ray ray;
    float dx = float(x) / float(width_);
    float dy = float(y) / float(height_);
    ray.origin = origin_ + offset;
    ray.dir = top_left_corner_ + dx * horizontal_ + dy * vertical_ - origin_ - offset;
    ray.tmin = 0.f;
    ray.tmax = 100000.f;

    return ray;
}

// ================================================================================================================
// ================================================ MotionalCamera ================================================
// ================================================================================================================

// ================================================================================================================
// ==================================================== Scene =====================================================
// ================================================================================================================

Scene::Scene() {}
Scene::~Scene() {}

void Scene::MissShader(Ray& ray, RayPayload& payload) {
    float t = poca_mus::Frac(0.5 * (poca_mus::GetNormalizeVec(ray.dir).y + 1.0));
    float4 color1 = float4(1.0, 1.0, 1.0);
    float4 color2 = float4(0.5, 0.7, 1.0);
    payload.radiance = (1.0 - t) * color1 + t * color2;
    payload.recursion_depth = MAX_RECURSION_DEPTH;
}

void Scene::AddObject(Object* obj) { objs_.push_back(obj); }

void Scene::TraceRay(Ray& ray, RayPayload& payload) {
    poca_mus::Normalize(ray.dir);
    Object* closet_hit_obj = nullptr;
    ProceduralPrimitiveAttributes attr;
    for (auto obj : objs_) {
        if (obj->IntersectionTest(ray, attr)) closet_hit_obj = obj;
    }
    if (closet_hit_obj != nullptr) {
        closet_hit_obj->ClosetHit(ray, payload, attr);
    } else {
        MissShader(ray, payload);
    }
}

// ================================================================================================================
// ==================================================== Scene =====================================================
// ================================================================================================================

// ================================================================================================================
// ================================================== PathTracer ==================================================
// ================================================================================================================

void PathTracer::SetCamera(MotionalCamera* camera) { this->camera_ = camera; }
void PathTracer::SetScene(Scene* scene) { this->scene_ = scene; }

void PathTracer::ReSize(int width, int height) {
    this->width_ = width;
    this->height_ = height;
    this->render_target_ = new float4[width * height];
}

float4 PathTracer::SamplePixel(int x, int y) {
    Ray ray = camera_->RayGen(x, y);
    RayPayload payload;
    float4 radiance = 0.0f;
    float4 attenuation = 1.0f;
    payload.recursion_depth = 0;

    while (payload.recursion_depth < max_recursion_depth_) {
        scene_->TraceRay(ray, payload);

        radiance += attenuation * payload.radiance;
        attenuation *= payload.attenuation;

        ray.origin = payload.hit_pos;
        ray.dir = payload.bounce_dir;
        ray.tmin = 0.f;
        ray.tmax = 100000.f;
        payload.recursion_depth++;
    }

    return radiance;
}

void PathTracer::DispatchRay(uint8_t* buf, int size, int64_t t) {
    camera_->SetOrigin(time_to_ori_x(t), time_to_ori_y(t), time_to_ori_z(t));
    camera_->SetLookAt(time_to_look_at_x(t), time_to_look_at_y(t), time_to_look_at_z(t));
    camera_->Updata();
    for (int i = 0; i < width_ * height_; ++i) {
        float4 cur(0.f);
        int spp = 5;
        for (int j = 0; j < spp; ++j) cur += SamplePixel(i % width_, i / width_);
        cur /= float(spp);
        buf[i * 3 + 0] = uint8_t((cur[0]) * 255.99f);
        buf[i * 3 + 1] = uint8_t((cur[1]) * 255.99f);
        buf[i * 3 + 2] = uint8_t((cur[2]) * 255.99f);
    }
}

// ================================================================================================================
// ================================================== PathTracer ==================================================
// ================================================================================================================