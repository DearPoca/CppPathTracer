#ifndef RAY_TRACING_COMMON_H
#define RAY_TRACING_COMMON_H

#include <stdint.h>

#include "ray_tracing_math.hpp"

#define DEFAULT_RAY_TMAX 1e10f
#define BOUNCE_RAY_TMIN 2e-5f

struct Ray {
    Float4 origin;
    Float4 dir;
    float tmin;
    float tmax;
};

struct RayPayload {
    Float4 radiance;
    Float4 attenuation;
    Float4 hit_pos;
    Float4 bounce_dir;
    uint recursion_depth;
    curandState *d_rng_states;
};

struct IntersectionAttributes {
    // Float4 ray_dir;
    Float4 normal;
    Float4 hit_pos;
};

#endif  // RAY_TRACER_COMMON_H