#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_functions.h>  
#include <helper_cuda.h> 
#include <helper_math.h>

#define __COMMON_GPU_CPU__ __device__ __host__
#define __COMMON_GPU_CPU_INLINE__ __COMMON_GPU_CPU__ inline

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#ifndef MIN
#define MIN(a, b) (a < b ? a : b)
#endif

#ifndef ABS
#define ABS(a) (a >= 0 ? a : -a)
#endif

typedef uint32_t uint;

inline float random() {
	static bool init = false;
	if (!init) {
		srand(static_cast<unsigned>(time(0)));
		init = true;
	}
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

inline float4 create_random_float4() { return make_float4(random(), random(), random(), random()); }

inline float3 create_random_float3() { return make_float3(random(), random(), random()); }

__COMMON_GPU_CPU_INLINE__ float cosine(float4 vec1, float4 vec2) {
	return dot(vec1, vec2) / length(vec1) / length(vec2);
}

__COMMON_GPU_CPU_INLINE__ float cosine(float3 vec1, float3 vec2) {
	return dot(vec1, vec2) / length(vec1) / length(vec2);
}

__COMMON_GPU_CPU_INLINE__ float3 to_world(float3 a, float3 N) {
	float3 B, C;
	if (abs(N.x) > abs(N.y)) {
		float invLen = 1.0f / sqrt(N.x * N.x + N.z * N.z);
		C = make_float3(N.z * invLen, 0.0f, -N.x * invLen);
	}
	else {
		float invLen = 1.0f / sqrt(N.y * N.y + N.z * N.z);
		C = make_float3(0.f, N.z * invLen, -N.y * invLen);
	}
	B = cross(C, N);
	return a.x * B + a.y * C + a.z * N;
}

__COMMON_GPU_CPU_INLINE__ float schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 *= r0;
	return r0 + (1 - r0) * pow(1 - cosine, 5);
}

__COMMON_GPU_CPU_INLINE__ bool refract(float3 v, float3 n, float ni_over_nt, float3& refracted) {
	float3 uv = normalize(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
	if (discriminant > 0) {
		refracted = normalize(ni_over_nt * (uv - n * dt) - n * sqrt(discriminant));
		return true;
	}
	return false;
}

__device__ inline curandState init_rand_state(int x, int y) {
	curandState rgnState;
	curand_init((unsigned long long)clock(), uint64_t(x) << 32 | uint64_t(y), 0, &rgnState);
	return rgnState;
}

__device__ inline curandState init_rand_state(unsigned long long clock, int x, int y) {
	curandState rgnState;
	curand_init(clock, uint64_t(x) << 32 | uint64_t(y), 0, &rgnState);
	return rgnState;
}

__device__ inline float random(curandState& state) {
	return curand_uniform(&state);
}

__device__ inline float3 device_create_random_float3(curandState& state) {
	return make_float3(curand_uniform(&state), curand_uniform(&state), curand_uniform(&state));
}

__device__ inline float4 device_create_random_float4(curandState& state) {
	return make_float4(curand_uniform(&state), curand_uniform(&state), curand_uniform(&state), curand_uniform(&state));
}