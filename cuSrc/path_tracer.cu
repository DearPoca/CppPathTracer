

#include <iostream>
#include <utility>

#include "bvh.h"
#include "object.h"
#include "path_tracer.h"

struct PathTracerparams {
    uint width, height;

    // Object** scene;
    // uint scene_size;
    BVHNode* bvh_root;

    uint spp;
    uint max_recursion_depth;

    MotionalCamera* camera;
    uint8_t* output_buffer_gpu_handle;

    curandState* d_rng_states;
};

__device__ void MissShader(Ray& ray, RayPayload& payload) {
    float t = poca_mus::Frac(0.5 * (poca_mus::GetNormalizeVec(ray.dir).y + 1.0));
    Float4 color1 = Float4(1.0, 1.0, 1.0);
    Float4 color2 = Float4(0.5, 0.7, 1.0);
    payload.radiance = (1.0 - t) * color1 + t * color2;
    payload.recursion_depth = MAX_RECURSION_DEPTH;
}

void PathTracer::AddObject(Object* obj) { objs_.push_back(obj); }

__device__ void TraceRay(PathTracerparams& params, Ray& ray, RayPayload& payload) {
    poca_mus::Normalize(ray.dir);
    ProceduralPrimitiveAttributes attr;
    Object* closet_hit_obj = poca_mus::TraceRay(params.bvh_root, ray, attr);
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
            poca_mus::Normalize(ray.dir);
            ray.tmin = 2e-6;
            ray.tmax = 100000.f;
            payload.recursion_depth++;
        }
    }
    params.output_buffer_gpu_handle[offset * 3 + 0] = int(radiance.x / float(params.spp) * 255.99);
    params.output_buffer_gpu_handle[offset * 3 + 1] = int(radiance.y / float(params.spp) * 255.99);
    params.output_buffer_gpu_handle[offset * 3 + 2] = int(radiance.z / float(params.spp) * 255.99);
}

void PathTracer::AllocateGpuMemory() {
    bvh_root_ = poca_mus::BuildBVH(objs_);
    // cudaMalloc((void**)&render_target_gpu_handle_, width_ * height_ * sizeof(Float4));
    cudaMalloc((void**)&output_buffer_gpu_handle_, width_ * height_ * 3);
    cudaMalloc((void**)&camera_gpu_handle_, sizeof(MotionalCamera));
    // cudaMalloc((void**)&materials_gpu_handle_, sizeof(Material*) * materials_.size());
    // cudaMalloc((void**)&scene_gpu_handle_, sizeof(Object*) * scene_.size());
    // for (int i = 0; i < materials_.size(); ++i) {
    //     Material* cur;
    //     cudaMalloc((void**)&cur, sizeof(Material));
    //     materials_cpu_handle_to_gpu_handle_[materials_[i]] = cur;
    //     MaterialMemCpyToGpu(materials_[i], cur);
    // }
    // for (int i = 0; i < scene_.size(); ++i) {
    //     Object* cur;
    //     cudaMalloc((void**)&cur, sizeof(Object));
    //     object_cpu_handle_to_gpu_handle_[scene_[i]] = cur;
    //     ObjectMemCpyToGpu(scene_[i], cur, materials_cpu_handle_to_gpu_handle_);
    //     cudaMemcpy(&(scene_gpu_handle_[i]), &cur, sizeof(Object*), cudaMemcpyHostToDevice);
    // }
    cudaMalloc(reinterpret_cast<void**>(&d_rng_states_), height_ * width_ * sizeof(curandState));
}

void PathTracer::DispatchRay(uint8_t* buf, int size, int64_t t) {
    camera_->Updata();
    cudaMemcpy(camera_gpu_handle_, camera_, sizeof(MotionalCamera), cudaMemcpyHostToDevice);

    // for (auto ptr_pair : materials_cpu_handle_to_gpu_handle_) {
    //     MaterialMemCpyToGpu((Material*)ptr_pair.first, (Material*)ptr_pair.second);
    // }
    // for (auto ptr_pair : object_cpu_handle_to_gpu_handle_) {
    //     ObjectMemCpyToGpu((Object*)ptr_pair.first, (Object*)ptr_pair.second, materials_cpu_handle_to_gpu_handle_);
    // }

    PathTracerparams params;
    params.width = width_;
    params.height = height_;

    // params.scene = scene_gpu_handle_;
    // params.scene_size = scene_.size();
    params.bvh_root = bvh_root_;

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
