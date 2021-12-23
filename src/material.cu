#include "material.h"

__device__ void DiffuseProcess(Material &self, Float4 &position, Float4 &normal, Float4 &in_ray_dir,
                               RayPayload &payload) {
    Float4 wi = -in_ray_dir;

    float x_1 = curand_uniform(payload.d_rng_states), x_2 = curand_uniform(payload.d_rng_states);
    float z = pow(x_1, 1.0 / 2);
    float r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
    Float4 localRay = Float4(r * cos(phi), r * sin(phi), z);

    payload.hit_pos = position;
    payload.bounce_dir = poca_mus::ToWorld(localRay, normal);
    float cosalpha = poca_mus::Dot(normal, payload.bounce_dir);
    if (cosalpha > 0.0f) {
        payload.attenuation = self.Kd_;
    } else {
        payload.attenuation = Float4(0.0, 0.0, 0.0);
    }
}

Material::Material() {
    type_ = Diffuse;
    Kd_ = Float4(0.9, 0.9, 0.9);
    refraction_ = 0.f;
    emitIntensity_ = 0.f;
    smoothness_ = 0.f;
    reflectivity_ = 0.f;
    EvalAttenuationAndCreateRay = DiffuseProcess;
}

__device__ FuncEvalAttenuationAndCreateRayPtr fp_diffuse = DiffuseProcess;

void MaterialMemCpyToGpu(Material *material_host, Material *material_gpu_handle) {
    cudaMemcpy((void *)material_gpu_handle, (void *)material_host, sizeof(Material), cudaMemcpyHostToDevice);
    cudaMemcpyFromSymbol(&material_gpu_handle->EvalAttenuationAndCreateRay, fp_diffuse,
                         sizeof(FuncEvalAttenuationAndCreateRayPtr));
}
