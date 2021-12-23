#ifndef OBJECT_H_45234263
#define OBJECT_H_45234263

#include "material.h"
#include "path_tracing_common.h"
#include "ray_tracing_math.hpp"

#define FUNC_TYPE_DEFINE_INTERSECTION \
    typedef bool (*FuncIntersectionTestPtr)(Object & self, Ray & ray, ProceduralPrimitiveAttributes & attr)
#define FUNC_TYPE_DEFINE_CLOSET_HIT                                                  \
    typedef void (*FuncClosetHitPtr)(Object & self, Ray & ray, RayPayload & payload, \
                                     ProceduralPrimitiveAttributes & attr)

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

    Float4 center_;
    float radius_;
    FUNC_TYPE_DEFINE_INTERSECTION;
    FUNC_TYPE_DEFINE_CLOSET_HIT;

    FuncIntersectionTestPtr IntersectionTestPtr;
    FuncClosetHitPtr ClosetHitPtr;

    // bool IntersectionTest(Ray &ray, ProceduralPrimitiveAttributes &attr);
    // void ClosetHit(Ray &ray, RayPayload &payload, ProceduralPrimitiveAttributes &attr);
};
FUNC_TYPE_DEFINE_INTERSECTION;
FUNC_TYPE_DEFINE_CLOSET_HIT;

void ObjectMemCpyToGpu(Object *object_host, Object *object_gpu_handle, std::map<void *, void *> &mp);

#endif  // OBJECT_H_45234263