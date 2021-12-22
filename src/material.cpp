#include "material.h"

void DiffuseProcess(Material &self, float4 &position, float4 &normal, float4 &ray_direction, RayPayload &payload) {
    Ray ray;
    ray.origin = position;
    float4 wi = -ray_direction;

    float x_1 = poca_mus::Random(), x_2 = poca_mus::Random();
    float z = pow(x_1, 1.0 / 2);
    float r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
    float4 localRay = float4(r * cos(phi), r * sin(phi), z);

    payload.hit_pos = position;
    payload.bounce_dir = poca_mus::ToWorld(localRay, normal);
    float cosalpha = poca_mus::Dot(normal, payload.bounce_dir);
    if (cosalpha > 0.0f) {
        payload.attenuation = self.Kd_;
    } else {
        payload.attenuation = float4(0.0, 0.0, 0.0);
    }
}

void Material::EvalAttenuationAndCreateRay(float4 &position, float4 &normal, float4 &in_ray_dir, RayPayload &payload) {
    EvalAttenuationAndCreateRayPtr(*this, position, normal, in_ray_dir, payload);
}

Material::Material() {
    type_ = Diffuse;
    Kd_ = float4(0.9, 0.9, 0.9);
    refraction_ = 0.f;
    emitIntensity_ = 0.f;
    smoothness_ = 0.f;
    reflectivity_ = 0.f;
    EvalAttenuationAndCreateRayPtr = DiffuseProcess;
}