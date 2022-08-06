#pragma once

#include <memory>

#include "ray_tracing_common.h"
#include "ray_tracing_math.hpp"
#include "bvh.h"
#include "motional_camera.h"
#include "object.h"

class PathTracer {
public:
	void AddObject(Object* obj);

	void Init();

	void Refresh();

	void DispatchRay(DispatchRayArgs args);

	void SetCamera(std::shared_ptr<MotionalCamera>& camera);

	std::shared_ptr<MotionalCamera> GetCamera();

private:
	uint8_t max_recursion_depth_ = 12;
	float* depth_info_buffer_;
	float3* normal_info_buffer_;
	uint8_t* output_buffer_gpu_handle_;

	std::unique_ptr<SceneBVHGPUHandle> scene_bvh_;

	std::shared_ptr<MotionalCamera> camera_;

	std::vector<Material*> materials_;
};