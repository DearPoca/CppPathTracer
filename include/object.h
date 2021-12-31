#ifndef OBJECT_H_45234263
#define OBJECT_H_45234263

#include "material.h"
#include "path_tracing_common.h"
#include "ray_tracing_math.hpp"

#define FUNC_TYPE_DEFINE_INTERSECTION \
    typedef bool (*FuncIntersectionTestPtr)(Object & self, Ray & ray, IntersectionAttributes & attr)
#define FUNC_TYPE_DEFINE_CLOSET_HIT \
    typedef void (*FuncClosetHitPtr)(Object & self, Ray & ray, RayPayload & payload, IntersectionAttributes & attr)

namespace PrimitiveType {
    enum Enum
    {
        Sphere,
        Platform,
        Cylinder,
        Count
    };
}  // namespace PrimitiveType

class Object {
private:
public:
    PrimitiveType::Enum type_;

    Material *material_;

    Float4 AABB_min_;
    Float4 AABB_max_;

    Float4 center_;
    float radius_;

    float y_pos_;

    float height_;

    __COMMON_GPU_CPU__ void UpdataAABB();

    FUNC_TYPE_DEFINE_INTERSECTION;
    FUNC_TYPE_DEFINE_CLOSET_HIT;
    FuncIntersectionTestPtr IntersectionTest;
    FuncClosetHitPtr ClosetHit;
};

FUNC_TYPE_DEFINE_INTERSECTION;
FUNC_TYPE_DEFINE_CLOSET_HIT;

void ObjectMemCpyToGpu(Object *object_host, Object *object_gpu_handle, Material *material_gpu_handle);

#endif  // OBJECT_H_45234263