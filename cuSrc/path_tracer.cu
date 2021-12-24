

#include <iostream>
#include <utility>

#include "object.h"
#include "path_tracer.h"

struct PathTracerparams {
    uint width, height;

    Object** scene;
    uint scene_size;

    uint spp;
    uint max_recursion_depth;

    MotionalCamera* camera;
    uint8_t* output_buffer_gpu_handle;

    curandState* d_rng_states;
};

__device__ Float4 GpuCreateRandomFloat4(curandState* state) {
    return Float4(curand_uniform(state), curand_uniform(state), curand_uniform(state), curand_uniform(state));
}

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
    Float4 rd = lens_radius_ * GpuCreateRandomFloat4(state);
    Float4 offset = u_ * rd.x + v_ * rd.y;
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

// ****************************************************************************************************************

// ================================================================================================================
// ================================================== PathTracer ==================================================
// ================================================================================================================

__device__ void MissShader(Ray& ray, RayPayload& payload) {
    float t = poca_mus::Frac(0.5 * (poca_mus::GetNormalizeVec(ray.dir).y + 1.0));
    Float4 color1 = Float4(1.0, 1.0, 1.0);
    Float4 color2 = Float4(0.5, 0.7, 1.0);
    payload.radiance = (1.0 - t) * color1 + t * color2;
    payload.recursion_depth = MAX_RECURSION_DEPTH;
}

void PathTracer::AddObject(Object* obj) { scene_.push_back(obj); }

__device__ void TraceRay(PathTracerparams& params, Ray& ray, RayPayload& payload) {
    poca_mus::Normalize(ray.dir);
    Object* closet_hit_obj = nullptr;
    ProceduralPrimitiveAttributes attr;
    for (int i = 0; i < params.scene_size; ++i) {
        if (params.scene[i]->IntersectionTest(*params.scene[i], ray, attr)) closet_hit_obj = params.scene[i];
    }
    if (closet_hit_obj != nullptr) {
        closet_hit_obj->ClosetHit(*closet_hit_obj, ray, payload, attr);
    } else {
        MissShader(ray, payload);
    }
}

void PathTracer::AddMeterial(Material* material) { materials_.push_back(material); }

void PathTracer::SetCamera(MotionalCamera* camera) { this->camera_ = camera; }

void PathTracer::ReSize(int width, int height) {
    this->width_ = width;
    this->height_ = height;
}
void PathTracer::SetSamplePerPixel(uint spp) { spp_ = spp; }

__global__ void SamplePixel(PathTracerparams params) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint offset = y * params.width + x;
    Float4 radiance = 0.0f;
    curand_init(offset, 0, 0, &(params.d_rng_states[offset]));
    for (uint i = 0; i < params.spp; ++i) {
        Ray ray = params.camera->RayGen(x, y, &(params.d_rng_states[offset]));
        RayPayload payload;
        Float4 attenuation = 1.0f;
        payload.recursion_depth = 0;
        payload.d_rng_states = &params.d_rng_states[offset];

        while (payload.recursion_depth < params.max_recursion_depth) {
            TraceRay(params, ray, payload);

            radiance += attenuation * payload.radiance;
            attenuation *= payload.attenuation;

            ray.origin = payload.hit_pos;
            ray.dir = payload.bounce_dir;
            ray.tmin = 0.00002f;
            ray.tmax = 100000.f;
            payload.recursion_depth++;
        }
    }
    params.output_buffer_gpu_handle[offset * 3 + 0] = int(radiance.x / float(params.spp) * 255.99);
    params.output_buffer_gpu_handle[offset * 3 + 1] = int(radiance.y / float(params.spp) * 255.99);
    params.output_buffer_gpu_handle[offset * 3 + 2] = int(radiance.z / float(params.spp) * 255.99);
}

void PathTracer::AllocateGpuMemory() {
    // cudaMalloc((void**)&render_target_gpu_handle_, width_ * height_ * sizeof(Float4));
    cudaMalloc((void**)&output_buffer_gpu_handle_, width_ * height_ * 3);
    cudaMalloc((void**)&camera_gpu_handle_, sizeof(MotionalCamera));
    // cudaMalloc((void**)&materials_gpu_handle_, sizeof(Material*) * materials_.size());
    cudaMalloc((void**)&scene_gpu_handle_, sizeof(Object*) * scene_.size());
    for (int i = 0; i < materials_.size(); ++i) {
        Material* cur;
        cudaMalloc((void**)&cur, sizeof(Material));
        materials_cpu_handle_to_gpu_handle_[materials_[i]] = cur;
        MaterialMemCpyToGpu(materials_[i], cur);
    }
    for (int i = 0; i < scene_.size(); ++i) {
        Object* cur;
        cudaMalloc((void**)&cur, sizeof(Object));
        object_cpu_handle_to_gpu_handle_[scene_[i]] = cur;
        ObjectMemCpyToGpu(scene_[i], cur, materials_cpu_handle_to_gpu_handle_);
        cudaMemcpy(&(scene_gpu_handle_[i]), &cur, sizeof(Object*), cudaMemcpyHostToDevice);
    }
    cudaMalloc(reinterpret_cast<void**>(&d_rng_states_), height_ * width_ * sizeof(curandState));
}

void PathTracer::DispatchRay(uint8_t* buf, int size, int64_t t) {
    camera_->Updata();
    cudaMemcpy(camera_gpu_handle_, camera_, sizeof(MotionalCamera), cudaMemcpyHostToDevice);

    for (auto ptr_pair : materials_cpu_handle_to_gpu_handle_) {
        MaterialMemCpyToGpu((Material*)ptr_pair.first, (Material*)ptr_pair.second);
    }
    for (auto ptr_pair : object_cpu_handle_to_gpu_handle_) {
        ObjectMemCpyToGpu((Object*)ptr_pair.first, (Object*)ptr_pair.second, materials_cpu_handle_to_gpu_handle_);
    }

    PathTracerparams params;
    params.width = width_;
    params.height = height_;

    params.scene = scene_gpu_handle_;
    params.scene_size = scene_.size();

    params.spp = spp_;
    params.max_recursion_depth = max_recursion_depth_;

    params.camera = camera_gpu_handle_;
    params.output_buffer_gpu_handle = output_buffer_gpu_handle_;

    params.d_rng_states = d_rng_states_;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width_ / threadsPerBlock.x, height_ / threadsPerBlock.y);
    SamplePixel<<<numBlocks, threadsPerBlock>>>(params);
    cudaDeviceSynchronize();
    cudaMemcpy(buf, output_buffer_gpu_handle_, width_ * height_ * 3, cudaMemcpyDeviceToHost);
    const auto error = cudaGetLastError();
    if (error != 0) printf("[ERROR]Cuda Error %s\n", cudaGetErrorString(error));
}

// ================================================================================================================
// ================================================== PathTracer ==================================================
// ================================================================================================================