#pragma once

#include <memory>
#include <deque>

#include "ray_tracing_common.h"
#include "ray_tracing_math.hpp"
#include "bvh.h"
#include "motional_camera.h"
#include "object.h"
#include "semaphore.h"

#define MAX_RECURSION_DEPTH_SET 32

class PathTracer {
public:
	void AddObject(Object* obj);

	void InitPipeline();

	void DispatchRay(DispatchRayArgs args);

	void SetCamera(std::shared_ptr<MotionalCamera>& camera);

	std::shared_ptr<MotionalCamera> GetCamera();

private:
	std::shared_ptr<uint8_t> output_buffer_;
	uint8_t* output_buffer_gpu_handle_;
	float* depth_info_buffer_;
	float3* normal_info_buffer_;
	float3* render_target_gpu_handle_;
	float3* denoising_buffer_gpu_handle_;
	float3* mix_buffer_gpu_handle_;
	curandState* d_rng_states_;

	std::unique_ptr<SceneBVH> scene_bvh_;
	SceneBVHGPUHandle scene_bvh_gpu_handle_;
	std::atomic_bool running_ = { true };
	std::deque<DispatchRayArgs> tasks_queue_;
	Semaphore sem;

	uint8_t max_recursion_depth_ = 8;
	int width_ = 0, height_ = 0;
	std::shared_ptr<MotionalCamera> camera_;
	cudaTextureObject_t sky_tex_obj_;
private:
	void InitBuffers(MotionalCamera &cam);
	void PipelineLoop();
};