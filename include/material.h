#pragma once

#include "ray_tracing_common.h"

namespace MaterialType {
	enum Enum
	{
		Diffuse = 0,
		Metal,
		Mirror,
		Glass,
		Test,
		Count
	};
}  // namespace MaterialType

class Material {
private:
public:
	MaterialType::Enum type_;
	bool have_tex_;
	union {
		float3 kd_;
		cudaTextureObject_t tex_;
	};
	float refractive_index_;
	float emit_intensity_;
	float smoothness_;
	float reflectivity_;

	__COMMON_GPU_CPU__ Material();

	__device__ void EvalAttenuationAndCreateRay(RayPayload& payload, float3 position, float3 normal,
		float3 in_ray_dir, float x = 0.f, float y = 0.f);

	__device__ float3 GetKd(float x = 0.f, float y = 0.f);
};

