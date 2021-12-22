#ifndef RAY_TRACING_COMMON_H
#define RAY_TRACING_COMMON_H

#include <stdint.h>

#include "material.h"
#include "ray_tracing_math.hpp"

struct Ray {
    float4 origin;
    float4 dir;
    float tmin;
    float tmax;
};

struct RayPayload {
    float4 radiance;
    float4 attenuation;
    float4 hit_pos;
    float4 bounce_dir;
    uint recursion_depth;
};

struct ProceduralPrimitiveAttributes {
    // float4 ray_dir;
    float4 normal;
    float4 hit_pos;
};

#endif  // RAY_TRACER_COMMON_H