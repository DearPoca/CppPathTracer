#include "material.h"

__device__ void DiffuseHitShader(Material &self, Float4 &position, Float4 &normal, Float4 &in_ray_dir,
                                 RayPayload &payload) {
    Float4 wi = -in_ray_dir;
    float x_1 = curand_uniform(payload.d_rng_states), x_2 = curand_uniform(payload.d_rng_states);
    float z = pow(x_1, 1.0 / 2);
    float r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
    Float4 localRay = Float4(r * cos(phi), r * sin(phi), z);

    payload.bounce_dir = poca_mus::ToWorld(localRay, normal);
    float cosalpha = poca_mus::Dot(normal, payload.bounce_dir);
    if (cosalpha > 0.0f) {
        payload.attenuation = self.Kd_;
    } else {
        payload.attenuation = Float4(0.0, 0.0, 0.0);
    }
    payload.hit_pos = position;
}

__device__ void MirrorHitShader(Material &self, Float4 &position, Float4 &normal, Float4 &in_ray_dir,
                                RayPayload &payload) {
    float s = self.smoothness_;
    float alpha = pow(1000.0f, s);
    float x_1 = curand_uniform(payload.d_rng_states), x_2 = curand_uniform(payload.d_rng_states);
    float z = pow(x_1, 1.0 / alpha);
    float r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
    Float4 localRay = Float4(r * cos(phi), r * sin(phi), z);
    // end_common

    Float4 reflect_dir = poca_mus::Reflect(in_ray_dir, normal);
    Float4 wo = poca_mus::ToWorld(localRay, reflect_dir);
    float cosalpha = poca_mus::Dot(normal, wo);
    if (cosalpha > 0.0f) {
        float s = self.smoothness_;
        float alpha = pow(1000.0f, s);
        payload.attenuation = self.Kd_;
    } else {
        payload.attenuation = Float4(0.0, 0.0, 0.0);
    }
    payload.bounce_dir = wo;
    payload.hit_pos = position;
}

__device__ void PlasticHitShader(Material &self, Float4 &position, Float4 &normal, Float4 &in_ray_dir,
                                 RayPayload &payload) {
    float x_1 = curand_uniform(payload.d_rng_states), x_2 = curand_uniform(payload.d_rng_states);
    float s = self.smoothness_;
    float alpha = pow(1000.0f, s);
    float r, z, phi;
    Float4 localRay, H;
    payload.bounce_dir = 0.f;

    float reflectivity = self.reflectivity_;

    if (curand_uniform(payload.d_rng_states) < reflectivity) {
        z = pow(x_1, 1.0 / alpha);
        r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
        localRay = Float4(r * cos(phi), r * sin(phi), z);
        Float4 reflect_dir = poca_mus::Reflect(in_ray_dir, normal);
        payload.bounce_dir = poca_mus::ToWorld(localRay, reflect_dir);
    } else {
        z = pow(x_1, 1.0 / 2.0);
        r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
        localRay = Float4(r * cos(phi), r * sin(phi), z);
        payload.bounce_dir = poca_mus::ToWorld(localRay, normal);
    }

    if (poca_mus::Dot(payload.bounce_dir, normal) < 0) {
        payload.attenuation = Float4(0.0, 0.0, 0.0);
    } else {
        payload.attenuation = self.Kd_;
    }
    payload.hit_pos = position;
}

__device__ void GlassHitShader(Material &self, Float4 &position, Float4 &normal, Float4 &in_ray_dir,
                               RayPayload &payload) {
    float x_1 = curand_uniform(payload.d_rng_states), x_2 = curand_uniform(payload.d_rng_states);
    float alpha = pow(1000.0f, self.smoothness_);
    float z = pow(x_1, 1.0 / alpha);
    float r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
    Float4 localRay = Float4(r * cos(phi), r * sin(phi), z);

    Float4 outward_normal;
    Float4 refracted;
    float ni_over_nt;
    float reflect_prob;
    float cosine;

    poca_mus::Normalize(in_ray_dir);
    if (poca_mus::Dot(in_ray_dir, normal) > 0) {
        outward_normal = -normal;
        ni_over_nt = self.refractive_index_;
        cosine = poca_mus::Dot(in_ray_dir, normal);
        cosine = sqrt(1 - self.refractive_index_ * self.refractive_index_ * (1 - cosine * cosine));
    } else {
        outward_normal = normal;
        ni_over_nt = 1.f / self.refractive_index_;
        cosine = -poca_mus::Dot(in_ray_dir, normal);
    }
    if (poca_mus::CanRefract(in_ray_dir, outward_normal, ni_over_nt, refracted)) {
        reflect_prob = poca_mus::Schlick(cosine, self.refractive_index_);
    } else {
        reflect_prob = 1.0;
    }
    if (curand_uniform(payload.d_rng_states) < reflect_prob) {
        Float4 reflected = poca_mus::Reflect(in_ray_dir, normal);
        payload.bounce_dir = poca_mus::ToWorld(localRay, reflected);
    } else {
        payload.bounce_dir = poca_mus::ToWorld(localRay, refracted);
    }
    payload.attenuation = self.Kd_;
    payload.hit_pos = position;
}

__COMMON_GPU_CPU__ Material::Material() {
    type_ = Diffuse;
    Kd_ = Float4(0.9, 0.9, 0.9);
    refractive_index_ = 0.f;
    emit_intensity_ = 0.f;
    smoothness_ = 0.f;
    reflectivity_ = 0.f;
}

__device__ FuncEvalAttenuationAndCreateRayPtr fp_diffuse = DiffuseHitShader;
__device__ FuncEvalAttenuationAndCreateRayPtr fp_plastic = PlasticHitShader;
__device__ FuncEvalAttenuationAndCreateRayPtr fp_mirror = MirrorHitShader;
__device__ FuncEvalAttenuationAndCreateRayPtr fp_glass = GlassHitShader;

void MaterialMemCpyToGpu(Material *material_host, Material *material_gpu_handle) {
    cudaMemcpy((void *)material_gpu_handle, (void *)material_host, sizeof(Material), cudaMemcpyHostToDevice);
    switch (material_host->type_) {
        case MaterialType::Diffuse:
            cudaMemcpyFromSymbol(&material_gpu_handle->EvalAttenuationAndCreateRay, fp_diffuse,
                                 sizeof(FuncEvalAttenuationAndCreateRayPtr));
            break;
        case MaterialType::Plastic:
            cudaMemcpyFromSymbol(&material_gpu_handle->EvalAttenuationAndCreateRay, fp_plastic,
                                 sizeof(FuncEvalAttenuationAndCreateRayPtr));
            break;
        case MaterialType::Mirror:
            cudaMemcpyFromSymbol(&material_gpu_handle->EvalAttenuationAndCreateRay, fp_mirror,
                                 sizeof(FuncEvalAttenuationAndCreateRayPtr));
            break;
        case MaterialType::Glass:
            cudaMemcpyFromSymbol(&material_gpu_handle->EvalAttenuationAndCreateRay, fp_glass,
                                 sizeof(FuncEvalAttenuationAndCreateRayPtr));
            break;
        default:
            cudaMemcpyFromSymbol(&material_gpu_handle->EvalAttenuationAndCreateRay, fp_diffuse,
                                 sizeof(FuncEvalAttenuationAndCreateRayPtr));
    }
}