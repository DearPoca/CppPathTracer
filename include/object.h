#ifndef OBJECT_H_45234263
#define OBJECT_H_45234263

#include "material.h"
#include "path_tracing_common.h"
#include "ray_tracing_math.hpp"

class Object {
private:
public:
    Material *material_;
    Object();
    ~Object();

    float minx_;
    float miny_;
    float minz_;
    float maxx_;
    float maxy_;
    float maxz_;

    float4 center_;
    float radius_;

    bool (*IntersectionTestPtr)(Object &self, Ray &ray, ProceduralPrimitiveAttributes &attr);

    bool IntersectionTest(Ray &ray, ProceduralPrimitiveAttributes &attr);
    void ClosetHit(Ray &ray, RayPayload &payload, ProceduralPrimitiveAttributes &attr);
};

#endif  // OBJECT_H_45234263