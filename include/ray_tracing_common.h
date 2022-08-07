#pragma once

#include <stdint.h>

#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>

#include "ray_tracing_math.hpp"

#define DEFAULT_RAY_TMAX 1e10f
#define BOUNCE_RAY_TMIN 2e-5f

struct Ray {
	float3 origin;
	float3 dir;
	float tmin;
	float tmax;
};

struct RayPayload {
	Ray ray;
	float3 radiance;	// �Է���
	float3 attenuation; // ˥��
	float3 hit_pos;
	float3 bounce_dir;	// ���䷽��
	uint recursion_depth;
	cudaTextureObject_t sky_tex_obj;
	curandState* d_rng_states;
};

struct IntersectionAttributes {
	float3 normal;
	float3 hit_pos;
};

struct DispatchRayArgs {
	void* cbParam;
	void (*Callback)(uint8_t* data, int width, int height, void* cbParam);
};