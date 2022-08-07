#include "path_tracer.h"

#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>

#include "ray_tracing_math.hpp"
#include "ray_tracing_common.h"
#include "textures.h"
#include "logger.hpp"

#define ARGB_CHANNELS 4

struct PathTracerParams {
	int width, height;
	cudaTextureObject_t sky_tex_obj;

	SceneBVHGPUHandle bvh_root;

	uint max_recursion_depth;

	MotionalCamera camera;
	float* depth_info_buffer;
	float3* normal_info_buffer;
	float3* render_target;

	curandState* d_rng_states;
};

void PathTracer::AddObject(Object* obj) {
	if (scene_bvh_ == nullptr) {
		scene_bvh_.reset(new SceneBVH);
	}
	scene_bvh_->AddObject(obj);
}

__global__ void InitCuRand(curandState* d_rng_states, unsigned long long clock, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	for (int y = 0; y < height; ++y) {
		int offset = y * width + x;
		d_rng_states[offset] = init_rand_state(clock, x, y);
	}
}

void PathTracer::InitBuffers(MotionalCamera& cam) {
	cudaError_t err;
	if (!(cam.width_ == width_ && cam.height_ == height_)) {
		sky_tex_obj_ = AddTexByFile("textures\\sky.png");
		width_ = cam.width_;
		height_ = cam.height_;

		{ // output buffer
			output_buffer_.reset(new uint8_t[width_ * height_ * ARGB_CHANNELS]);
			err = cudaMalloc((void**)&output_buffer_gpu_handle_, height_ * width_ * sizeof(uint8_t) * ARGB_CHANNELS);
			if (err != cudaSuccess)
			{
				log_error("Error occur with InitGpuState: %s", cudaGetErrorString(err));
			}
		}

		{ // denoising info buffers
			err = cudaMalloc((void**)&depth_info_buffer_, height_ * width_ * sizeof(float));
			if (err != cudaSuccess)
			{
				log_error("Error occur with InitGpuState: %s", cudaGetErrorString(err));
			}

			err = cudaMalloc((void**)&normal_info_buffer_, height_ * width_ * sizeof(float3));
			if (err != cudaSuccess)
			{
				log_error("Error occur with InitGpuState: %s", cudaGetErrorString(err));
			}
		}

		{ // render target
			err = cudaMalloc((void**)(&render_target_gpu_handle_), height_ * width_ * sizeof(float3));
			if (err != cudaSuccess)
			{
				log_error("Error occur with InitGpuState: %s", cudaGetErrorString(err));
			}
		}

		{ // mixer
			err = cudaMalloc((void**)(&mix_buffer_gpu_handle_), height_ * width_ * sizeof(float3));
			if (err != cudaSuccess)
			{
				log_error("Error occur with InitGpuState: %s", cudaGetErrorString(err));
			}
		}

		{ // random states
			err = cudaMalloc((void**)(&d_rng_states_), height_ * width_ * sizeof(curandState));
			if (err != cudaSuccess)
			{
				log_error("Error occur with InitGpuState: %s", cudaGetErrorString(err));
			}

			dim3 threadsPerBlock(16);
			dim3 numBlocks(width_ / threadsPerBlock.x);
			InitCuRand << < numBlocks, threadsPerBlock >> > (d_rng_states_, clock(), width_, height_);
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess)
			{
				log_error("Error occur with InitGpuState: %s", cudaGetErrorString(err));
			}
		}
	}
}

__device__ void Miss(RayPayload& payload) {
	float3 d = normalize(payload.ray.dir);
	float v = asin(d.z) / M_PI + 0.5, u = atan(d.y / d.x) / 2 / M_PI;
	payload.radiance = make_float3(GetTexture2D(payload.sky_tex_obj, u, v));
	payload.radiance = make_float3(ABS(d.x), ABS(d.y), ABS(d.z)) / 3;
	payload.recursion_depth = MAX_RECURSION_DEPTH_SET;
}

__global__ void SamplePixel(PathTracerParams params) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint offset = y * params.width + x;
	float3 radiance = make_float3(0.f);
	float3 normals = make_float3(0.f);
	float depth = 0.0f;
	// curand_init(offset, 0, 0, &(params.d_rng_states[offset]));
	RayPayload payload;
	payload.ray = params.camera.RayGen(x, y, params.d_rng_states[offset]);
	float3 attenuation = make_float3(1.f);
	payload.recursion_depth = 0;
	payload.d_rng_states = &params.d_rng_states[offset];
	payload.sky_tex_obj = params.sky_tex_obj;
	bool first_recursion = false;

	while (payload.recursion_depth < params.max_recursion_depth) {
		IntersectionAttributes attr;
		Object closet_hit_obj;
		bool ret = params.bvh_root->TraceRay(payload.ray, attr, closet_hit_obj);

		if (ret) {
			closet_hit_obj.ClosetHit(payload, attr);
			//printf("[%f, %f, %f]\n", payload.attenuation.x, payload.attenuation.y, payload.attenuation.z);
		}
		else {
			attr.hit_pos = payload.ray.origin + DEFAULT_RAY_TMAX * payload.ray.dir;
			attr.normal = -payload.ray.dir;
			Miss(payload);
		}

		radiance += attenuation * payload.radiance;
		attenuation *= payload.attenuation;

		if (!first_recursion) {
			normals += attr.normal;
			depth += payload.ray.tmax;
			first_recursion = true;
		}

		payload.ray.origin = payload.hit_pos;
		payload.ray.dir = normalize(payload.bounce_dir);
		payload.ray.tmin = BOUNCE_RAY_TMIN;
		payload.ray.tmax = DEFAULT_RAY_TMAX;
		payload.recursion_depth++;
	}

	params.render_target[offset] = radiance;
	params.depth_info_buffer[offset] = depth;
	params.normal_info_buffer[offset] = normals;
}

__global__ void Mix(
	float3* src_gpu_handle_,
	float3* mix_buffer_gpu_handle_,
	uint8_t* dst_gpu_handle_,
	int cur_sample_idx,
	int width,
	int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	for (int y = 0; y < height; ++y) {
		int offset = y * width + x;
		mix_buffer_gpu_handle_[offset] = lerp(mix_buffer_gpu_handle_[offset], clamp(src_gpu_handle_[offset], 0.f, 1.f), 1.f / float(cur_sample_idx));
		dst_gpu_handle_[offset * 4 + 0] = 255.99f * mix_buffer_gpu_handle_[offset].z;
		dst_gpu_handle_[offset * 4 + 1] = 255.99f * mix_buffer_gpu_handle_[offset].y;
		dst_gpu_handle_[offset * 4 + 2] = 255.99f * mix_buffer_gpu_handle_[offset].x;
	}
}

void PathTracer::PipelineLoop() {
	cudaError_t err;
	while (running_.load()) {
		sem.Wait();
		DispatchRayArgs task = tasks_queue_.front();
		tasks_queue_.pop_front();
		MotionalCamera cma = camera_->GetCopy();
		InitBuffers(cma);
		PathTracerParams params;
		params.width = width_;
		params.height = height_;
		params.sky_tex_obj = sky_tex_obj_;
		params.bvh_root = scene_bvh_gpu_handle_;
		params.max_recursion_depth = max_recursion_depth_;
		params.camera = cma;
		params.depth_info_buffer = depth_info_buffer_;
		params.normal_info_buffer = normal_info_buffer_;
		params.render_target = render_target_gpu_handle_;
		params.d_rng_states = d_rng_states_;
		log_debug("params:[width: %d, height: %d,sky_tex_obj: %ld]", params.width, params.height, params.sky_tex_obj);
		log_debug("camera:[width: %d, height: %d]", cma.width_, cma.height_);

		dim3 threads_per_block_sample(16, 16);
		dim3 num_blocks_sample(width_ / threads_per_block_sample.x, height_ / threads_per_block_sample.y);
		SamplePixel << < num_blocks_sample, threads_per_block_sample >> > (params);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			log_error("Error occur with SamplePixel: %s", cudaGetErrorString(err));
		}

		dim3 threads_per_block_mix(16);
		dim3 num_blocks_mix(width_ / threads_per_block_mix.x);
		Mix << <num_blocks_mix, threads_per_block_mix >> > (render_target_gpu_handle_,
			mix_buffer_gpu_handle_, output_buffer_gpu_handle_, cma.cur_sample_idx_, params.width, params.height);

		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			log_error("Error occur with Mix: %s", cudaGetErrorString(err));
		}

		cudaMemcpy(output_buffer_.get(), output_buffer_gpu_handle_, params.width * params.height * 4, cudaMemcpyDeviceToHost);
		task.Callback(output_buffer_.get(), params.width, params.height, task.cbParam);
	}
}

void PathTracer::InitPipeline() {
	if (scene_bvh_ == nullptr) {
		scene_bvh_.reset(new SceneBVH);
	}
	scene_bvh_gpu_handle_ = scene_bvh_->BuildBVH();
	InitBuffers(camera_->GetCopy());
	std::thread(&PathTracer::PipelineLoop, this).detach();
}

void PathTracer::DispatchRay(DispatchRayArgs args) {
	tasks_queue_.push_back(args);
	sem.Signal();
}

void PathTracer::SetCamera(std::shared_ptr<MotionalCamera>& camera) {
	camera_ = camera;
}

std::shared_ptr<MotionalCamera> PathTracer::GetCamera() {
	return camera_;
}