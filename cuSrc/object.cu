#include "object.h"

__device__ bool SphereIntersectionTest(Object &self, Ray &ray, ProceduralPrimitiveAttributes &attr) {
    Float4 A_C = ray.origin - self.center_;
    Float4 &B = ray.dir;
    float a = poca_mus::Dot(B, B);
    float b = poca_mus::Dot(A_C, B);
    float c = poca_mus::Dot(A_C, A_C) - self.radius_ * self.radius_;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < ray.tmax && temp > ray.tmin) {
            ray.tmax = temp;
            attr.hit_pos = ray.origin + temp * ray.dir;
            Float4 normal = (attr.hit_pos - self.center_);
            attr.normal = normal / self.radius_;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < ray.tmax && temp > ray.tmin) {
            ray.tmax = temp;
            attr.hit_pos = ray.origin + temp * ray.dir;
            Float4 normal = (attr.hit_pos - self.center_);
            attr.normal = normal / self.radius_;
            return true;
        }
    }
    return false;
}

__device__ void ClosetHit(Object &self, Ray &ray, RayPayload &payload, ProceduralPrimitiveAttributes &attr) {
    self.material_->EvalAttenuationAndCreateRay(*self.material_, attr.hit_pos, attr.normal, ray.dir, payload);
}

__COMMON_GPU_CPU__ Object::Object() : radius_(1000.f), center_(Float4(0.f, -1000.f, 0.f)) {
    this->material_ = new Material;
}
__device__ FuncIntersectionTestPtr fp_intersection_sphere = SphereIntersectionTest;
__device__ FuncClosetHitPtr fp_closet_hit = ClosetHit;

void ObjectMemCpyToGpu(Object *object_host, Object *object_gpu_handle, Material *material_gpu_handle) {
    Object tmp;
    memcpy(&tmp, object_host, sizeof(Object));
    tmp.material_ = material_gpu_handle;
    cudaMemcpy((void *)object_gpu_handle, (void *)&tmp, sizeof(Object), cudaMemcpyHostToDevice);
    cudaMemcpyFromSymbol(&object_gpu_handle->IntersectionTest, fp_intersection_sphere, sizeof(FuncIntersectionTestPtr));
    cudaMemcpyFromSymbol(&object_gpu_handle->ClosetHit, fp_closet_hit, sizeof(FuncIntersectionTestPtr));
}