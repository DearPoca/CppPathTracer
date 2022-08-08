#include "material.h"

#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>

#include "textures.h"
#include "ray_tracing_math.hpp"
#include "ray_tracing_common.h"

__device__ float3 Material::GetKd(float x, float y) {
	if (have_tex_ == true) {
		return make_float3(GetTexture2D(tex_, x, y));
	}
	else {
		return kd_;
	}
}

__device__ void DiffuseHitShader(Material& self,
	RayPayload& payload, float3& position, float3& normal, float3& in_ray_dir, float x, float y) {
	float3 wi = -in_ray_dir;
	float x_1 = random(*payload.d_rng_states), x_2 = random(*payload.d_rng_states);
	float z = pow(x_1, 1.0 / 2);
	float r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
	float3 localRay = make_float3(r * cos(phi), r * sin(phi), z);

	payload.bounce_dir = to_world(localRay, normal);
	float cosalpha = dot(normal, payload.bounce_dir);
	if (cosalpha > 0.0f) {
		payload.attenuation = self.GetKd(x, y);
	}
	else {
		payload.attenuation = make_float3(0.0, 0.0, 0.0);
	}
	payload.radiance = self.emit_intensity_ * self.kd_;
	payload.hit_pos = position;
}

__device__ void MirrorHitShader(Material& self,
	RayPayload& payload, float3& position, float3& normal, float3& in_ray_dir, float x, float y) {
	float s = self.smoothness_;
	float alpha = pow(1000.0f, s);
	float x_1 = random(*payload.d_rng_states), x_2 = random(*payload.d_rng_states);
	float z = pow(x_1, 1.0 / alpha);
	float r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
	float3 localRay = make_float3(r * cos(phi), r * sin(phi), z);
	// end_common

	float3 reflect_dir = reflect(in_ray_dir, normal);
	float3 wo = to_world(localRay, reflect_dir);
	float cosalpha = dot(normal, wo);
	if (cosalpha > 0.0f) {
		float s = self.smoothness_;
		float alpha = pow(1000.0f, s);
		payload.attenuation = self.GetKd(x, y);
	}
	else {
		payload.attenuation = make_float3(0.0, 0.0, 0.0);
	}
	payload.bounce_dir = wo;
	payload.radiance = self.emit_intensity_ * self.kd_;
	payload.hit_pos = position;
}

__device__ void MetalHitShader(Material& self,
	RayPayload& payload, float3& position, float3& normal, float3& in_ray_dir, float x, float y) {
	float x_1 = random(*payload.d_rng_states), x_2 = random(*payload.d_rng_states);
	float s = self.smoothness_;
	float alpha = pow(1000.0f, s);
	float r, z, phi;
	float3 localRay;
	payload.bounce_dir = make_float3(0.f);

	float reflectivity = self.reflectivity_;

	if (random(*payload.d_rng_states) < reflectivity) {
		z = pow(x_1, 1.0 / alpha);
		r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
		localRay = make_float3(r * cos(phi), r * sin(phi), z);
		float3 reflect_dir = reflect(in_ray_dir, normal);
		payload.bounce_dir = to_world(localRay, reflect_dir);
	}
	else {
		z = pow(x_1, 1.0 / 2.0);
		r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
		localRay = make_float3(r * cos(phi), r * sin(phi), z);
		payload.bounce_dir = to_world(localRay, normal);
	}

	if (dot(payload.bounce_dir, normal) < 0) {
		payload.attenuation = make_float3(0.0, 0.0, 0.0);
	}
	else {
		payload.attenuation = self.GetKd(x, y);
	}
	payload.radiance = self.emit_intensity_ * self.kd_;
	payload.hit_pos = position;
}

__device__ void GlassHitShader(Material& self,
	RayPayload& payload, float3& position, float3& normal, float3& in_ray_dir, float x, float y) {
	float x_1 = random(*payload.d_rng_states), x_2 = random(*payload.d_rng_states);
	float alpha = pow(1000.0f, self.smoothness_);
	float z = pow(x_1, 1.0 / alpha);
	float r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
	float3 localRay = make_float3(r * cos(phi), r * sin(phi), z);

	float3 outward_normal;
	float3 refracted;
	float ni_over_nt;
	float reflect_prob;
	float cosine;

	in_ray_dir = normalize(in_ray_dir);
	if (dot(in_ray_dir, normal) > 0) {
		outward_normal = -normal;
		ni_over_nt = self.refractive_index_;
		cosine = dot(in_ray_dir, normal);
		cosine = sqrt(1 - self.refractive_index_ * self.refractive_index_ * (1 - cosine * cosine));
	}
	else {
		outward_normal = normal;
		ni_over_nt = 1.f / self.refractive_index_;
		cosine = -dot(in_ray_dir, normal);
	}
	if (refract(in_ray_dir, outward_normal, ni_over_nt, refracted)) {
		reflect_prob = schlick(cosine, self.refractive_index_);
	}
	else {
		reflect_prob = 1.0;
	}
	if (random(*payload.d_rng_states) < reflect_prob) {
		float3 reflected = reflect(in_ray_dir, normal);
		payload.bounce_dir = to_world(localRay, reflected);
	}
	else {
		payload.bounce_dir = to_world(localRay, refracted);
	}
	payload.attenuation = self.GetKd(x, y);
	payload.radiance = self.emit_intensity_ * self.kd_;
	payload.hit_pos = position;
}

__device__ void Material::EvalAttenuationAndCreateRay(RayPayload& payload, float3 position, float3 normal,
	float3 in_ray_dir, float x, float y) {
	switch (type_) {
	case MaterialType::Diffuse:
		DiffuseHitShader(*this, payload, position, normal, in_ray_dir, x, y);
		break;
	case MaterialType::Metal:
		MirrorHitShader(*this, payload, position, normal, in_ray_dir, x, y);
		break;
	case MaterialType::Mirror:
		MetalHitShader(*this, payload, position, normal, in_ray_dir, x, y);
		break;
	case MaterialType::Glass:
		GlassHitShader(*this, payload, position, normal, in_ray_dir, x, y);
		break;
	default:
		DiffuseHitShader(*this, payload, position, normal, in_ray_dir, x, y);
	}
}

__COMMON_GPU_CPU__ Material::Material() {
	type_ = MaterialType::Diffuse;
	have_tex_ = false;
	kd_ = make_float3(0.9, 0.9, 0.9);
	refractive_index_ = 0.f;
	emit_intensity_ = 0.f;
	smoothness_ = 0.f;
	reflectivity_ = 0.f;
}
