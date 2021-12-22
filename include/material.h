#ifndef MATERIAL_H_432528432
#define MATERIAL_H_432528432

#include "path_tracing_common.h"

enum MaterialType
{
    Diffuse = 0,
    Mirror,
    Plastic,
    Glass
};

class Material {
private:
public:
    MaterialType type_;
    float4 Kd_;
    float refraction_;
    float emitIntensity_;
    float smoothness_;
    float reflectivity_;

    Material();

    void (*EvalAttenuationAndCreateRayPtr)(Material &self, float4 &position, float4 &normal, float4 &in_ray_dir,
                                           RayPayload &payload);

    void EvalAttenuationAndCreateRay(float4 &position, float4 &normal, float4 &in_ray_dir, RayPayload &payload);
};

#endif  // MATERIAL_H_432528432