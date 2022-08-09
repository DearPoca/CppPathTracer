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
		sky_tex_obj_ = PocaTextureUtils::AddTexByFile("textures\\sky.png");
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

			err = cudaMalloc((void**)&normal_info_buffer_, height_ * width_ * sizeof(float4));
			if (err != cudaSuccess)
			{
				log_error("Error occur with InitGpuState: %s", cudaGetErrorString(err));
			}
		}

		{ // render target
			err = cudaMalloc((void**)(&render_target_gpu_handle_), height_ * width_ * sizeof(float4));
			if (err != cudaSuccess)
			{
				log_error("Error occur with InitGpuState: %s", cudaGetErrorString(err));
			}
		}

		{ // denoising temp buffer
			err = cudaMalloc((void**)(&denoising_buffer_gpu_handle_), height_ * width_ * sizeof(float3));
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
	payload.radiance = make_float3(PocaTextureUtils::GetTexture2D(payload.sky_tex_obj, u, v));
	payload.recursion_depth = MAX_RECURSION_DEPTH_SET;
}

__global__ void SamplePixel(PathTracerParams params) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = y * blockDim.x * gridDim.x + x;
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
			//printf("[%f, %f, %f]\n", payload.radiance.x, payload.radiance.y, payload.radiance.z);
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

__global__ void Denoising(
	float3* rt_buffer,
	float3* n_buffer,
	float* d_buffer,
	float3* dst_buffer,
	float stepwidth
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int width = blockDim.x * gridDim.x;
	int height = blockDim.y * gridDim.y;

	int dx[5] = { -2, -1, 0, 1, 2 };
	int dy[5] = { -2, -1, 0, 1, 2 };
	float kernel[5][5] = {
		{1.f,  4.f,  7.f,  4.f, 1.f},
		{4.f, 16.f, 26.f, 16.f, 4.f},
		{7.f, 26.f, 41.f, 26.f, 7.f},
		{4.f, 16.f, 26.f, 16.f, 4.f},
		{1.f,  4.f,  7.f,  4.f, 1.f}
	};

	int offset = y * width + x;

	float3 sum = make_float3(0.f);
	float2 step = make_float2(1.f / width, 1.f / height); // resolution
	float3 cval = rt_buffer[offset];
	float3 nval = n_buffer[offset];
	float pval = d_buffer[offset];
	float cum_w = 0.0;
	float c_w;
	float n_w;
	float p_w;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; ++j) {
			int u = x + dx[i];
			int v = y + dy[j];
			int cur_off = v * width + u;
			float3 ctmp;
			if (cur_off < 0 || cur_off >= width * height) {
				c_w = n_w = p_w = 0;
				ctmp = make_float3(0.f);
			}
			else {
				ctmp = rt_buffer[cur_off];
				float3 t = cval - ctmp;
				float dist2 = dot(t, t);
				c_w = min(exp(-(dist2) / M_PI), 1.0);
				float3 ntmp = n_buffer[cur_off];
				t = nval - ntmp;
				dist2 = max(dot(t, t), 0.0);
				n_w = min(exp(-(dist2) / M_PI), 1.0);
				float ptmp = d_buffer[cur_off];
				dist2 = (pval - ptmp) * (pval - ptmp);
				p_w = min(exp(-(dist2) / M_PI), 1.0);
			}
			float weight = c_w * n_w * p_w;
			sum += weight * kernel[i][j] * ctmp;
			cum_w += weight * kernel[i][j];
		}
	}
	dst_buffer[offset] = sum / cum_w;
}

__global__ void Mix(
	float3* src_gpu_handle_,
	float3* mix_buffer_gpu_handle_,
	uint8_t* dst_gpu_handle_,
	int cur_sample_idx
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y * blockDim.x * gridDim.x + x;
	mix_buffer_gpu_handle_[offset] = lerp(mix_buffer_gpu_handle_[offset], clamp(src_gpu_handle_[offset], 0.f, 1.f), 1.f / float(cur_sample_idx));
	dst_gpu_handle_[offset * 4 + 0] = 255.99f * mix_buffer_gpu_handle_[offset].z;
	dst_gpu_handle_[offset * 4 + 1] = 255.99f * mix_buffer_gpu_handle_[offset].y;
	dst_gpu_handle_[offset * 4 + 2] = 255.99f * mix_buffer_gpu_handle_[offset].x;
}

void PathTracer::PipelineLoop() {
	cudaError_t err;
	InitBuffers(camera_->GetCopy());
	while (running_.load()) {
		sem.Wait();
		long long start = Timer::GetMillisecondsTimeStamp();
		DispatchRayArgs task = tasks_queue_.front();
		tasks_queue_.pop_front();
		MotionalCamera cma = camera_->GetCopy();
		InitBuffers(cma);
		PathTracerParams params;
		params.sky_tex_obj = sky_tex_obj_;
		params.bvh_root = scene_bvh_gpu_handle_;
		params.max_recursion_depth = max_recursion_depth_;
		params.camera = cma;
		params.depth_info_buffer = depth_info_buffer_;
		params.normal_info_buffer = normal_info_buffer_;
		params.render_target = render_target_gpu_handle_;
		params.d_rng_states = d_rng_states_;

		dim3 threads_per_block(16, 16);
		dim3 num_blocks(width_ / threads_per_block.x, height_ / threads_per_block.y);
		SamplePixel << < num_blocks, threads_per_block >> > (params);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			log_error("Error occur with SamplePixel: %s", cudaGetErrorString(err));
		}

		Denoising << <num_blocks, threads_per_block >> > (render_target_gpu_handle_, normal_info_buffer_, depth_info_buffer_, denoising_buffer_gpu_handle_, 1.f);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			log_error("Error occur with Denoising: %s", cudaGetErrorString(err));
		}

		Mix << <num_blocks, threads_per_block >> > (denoising_buffer_gpu_handle_,
			mix_buffer_gpu_handle_, output_buffer_gpu_handle_, cma.cur_sample_idx_);
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			log_error("Error occur with Mix: %s", cudaGetErrorString(err));
		}
		else {
			log_info("Sample one pixel, cur sample idx: %d, used: %ldms", cma.cur_sample_idx_, Timer::GetMillisecondsTimeStamp() - start);
		}

		cudaMemcpy(output_buffer_.get(), output_buffer_gpu_handle_, width_ * height_ * 4, cudaMemcpyDeviceToHost);
		task.Callback(output_buffer_.get(), width_, height_, task.cbParam);
	}
}

void PathTracer::InitPipeline() {
	if (scene_bvh_ == nullptr) {
		scene_bvh_.reset(new SceneBVH);
	}
	scene_bvh_gpu_handle_ = scene_bvh_->BuildBVH();
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