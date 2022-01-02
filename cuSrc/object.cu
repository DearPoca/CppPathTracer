#include "object.h"

__device__ bool SphereIntersectionTest(Object &self, Ray &ray, IntersectionAttributes &attr) {
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

__device__ bool PlatformIntersectionTest(Object &self, Ray &ray, IntersectionAttributes &attr) {
    if (ray.origin.y < self.y_pos_ && ray.dir.y > 0.f || ray.origin.y > self.y_pos_ && ray.dir.y < 0.f) {
        float temp = (self.y_pos_ - ray.origin.y) / ray.dir.y;
        if (temp < ray.tmax && temp > ray.tmin) {
            ray.tmax = temp;
            attr.hit_pos = ray.origin + temp * ray.dir;
            attr.normal = poca_mus::GetNormalizeVec(Float4(0, -ray.dir.y, 0));
            return true;
        }
    }
    return false;
}

__device__ void ClosetHit(Object &self, Ray &ray, RayPayload &payload, IntersectionAttributes &attr) {
    self.material_->EvalAttenuationAndCreateRay(*self.material_, attr.hit_pos, attr.normal, ray.dir, payload);
}

__device__ bool CylinderIntersectionTest(Object &self, Ray &ray, IntersectionAttributes &attr) {
    bool ret = false;
    float upper_section_y_pos = self.center_.y + self.height_ / 2;
    if ((ray.origin.y < upper_section_y_pos && ray.dir.y > 0.f) ||
        (ray.origin.y > upper_section_y_pos && ray.dir.y < 0.f)) {
        float temp = (upper_section_y_pos - ray.origin.y) / ray.dir.y;
        Float4 hit_pos = ray.origin + temp * ray.dir;
        if (temp < ray.tmax && temp > ray.tmin &&
            sqrt((hit_pos.x - self.center_.x) * (hit_pos.x - self.center_.x) +
                 (hit_pos.z - self.center_.z) * (hit_pos.z - self.center_.z)) < self.radius_) {
            ray.tmax = temp;
            attr.hit_pos = hit_pos;
            attr.normal = poca_mus::GetNormalizeVec(Float4(0, -ray.dir.y, 0));
            ret = true;
        }
    }
    float lower_section_y_pos = self.center_.y - self.height_ / 2;
    if ((ray.origin.y < lower_section_y_pos && ray.dir.y > 0.f) ||
        (ray.origin.y > lower_section_y_pos && ray.dir.y < 0.f)) {
        float temp = (lower_section_y_pos - ray.origin.y) / ray.dir.y;
        Float4 hit_pos = ray.origin + temp * ray.dir;
        if (temp < ray.tmax && temp > ray.tmin &&
            sqrt((hit_pos.x - self.center_.x) * (hit_pos.x - self.center_.x) +
                 (hit_pos.z - self.center_.z) * (hit_pos.z - self.center_.z)) < self.radius_) {
            ray.tmax = temp;
            attr.hit_pos = hit_pos;
            attr.normal = poca_mus::GetNormalizeVec(Float4(0, -ray.dir.y, 0));
            ret = true;
        }
    }

    float &dx = ray.dir.x;
    float &dz = ray.dir.z;
    float &r = self.radius_;
    float cx = ray.origin.x - self.center_.x;
    float cz = ray.origin.z - self.center_.z;

    float a = dx * dx + dz * dz;
    float b = cx * dx + cz * dz;
    float c = cx * cx + cz * cz - r * r;
    float discriminant = b * b - a * c;
    if (discriminant > 0.f) {
        float temp = (-b - sqrt(discriminant)) / a;
        Float4 hit_pos = ray.origin + temp * ray.dir;
        if (temp < ray.tmax && temp > ray.tmin && hit_pos.y > lower_section_y_pos && hit_pos.y < upper_section_y_pos) {
            ray.tmax = temp;
            attr.hit_pos = hit_pos;
            Float4 normal = Float4(hit_pos.x - self.center_.x, 0.f, hit_pos.z - self.center_.z);
            attr.normal = poca_mus::GetNormalizeVec(normal);
            ret = true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        hit_pos = ray.origin + temp * ray.dir;
        if (temp < ray.tmax && temp > ray.tmin && hit_pos.y > lower_section_y_pos && hit_pos.y < upper_section_y_pos) {
            ray.tmax = temp;
            attr.hit_pos = hit_pos;
            Float4 normal = Float4(hit_pos.x - self.center_.x, 0.f, hit_pos.z - self.center_.z);
            attr.normal = poca_mus::GetNormalizeVec(normal);
            ret = true;
        }
    }
    return ret;
}

__COMMON_GPU_CPU__ void Object::UpdataAABB() {
    switch (type_) {
        case PrimitiveType::Sphere:
            AABB_max_ = center_ + Float4(ABS(radius_));
            AABB_min_ = center_ - Float4(ABS(radius_));
            break;
        case PrimitiveType::Platform:
            AABB_max_ = Float4(1e10f, y_pos_ + BOUNCE_RAY_TMIN * 5, 1e10f);
            AABB_min_ = Float4(-1e10f, y_pos_ - BOUNCE_RAY_TMIN * 5, -1e10f);
            break;
        case PrimitiveType::Cylinder:
            AABB_max_ = Float4(center_.x + radius_, center_.y + height_ / 2 + 0.5f, center_.z + radius_);
            AABB_min_ = Float4(center_.x - radius_, center_.y - height_ / 2 - 0.5f, center_.z - radius_);
            break;
        default:
            break;
    }
}

__device__ FuncIntersectionTestPtr fp_intersection_sphere = SphereIntersectionTest;
__device__ FuncIntersectionTestPtr fp_intersection_platform = PlatformIntersectionTest;
__device__ FuncIntersectionTestPtr fp_intersection_cylinder = CylinderIntersectionTest;

__device__ FuncClosetHitPtr fp_closet_hit = ClosetHit;

void ObjectMemCpyToGpu(Object *object_host, Object *object_gpu_handle, Material *material_gpu_handle) {
    Object tmp;
    memcpy(&tmp, object_host, sizeof(Object));
    tmp.material_ = material_gpu_handle;
    checkCudaErrors(cudaMemcpy((void *)object_gpu_handle, (void *)&tmp, sizeof(Object), cudaMemcpyHostToDevice));
    switch (object_host->type_) {
        case PrimitiveType::Sphere:
            checkCudaErrors(cudaMemcpyFromSymbol(&object_gpu_handle->IntersectionTest, fp_intersection_sphere,
                                                 sizeof(FuncIntersectionTestPtr)));
            break;
        case PrimitiveType::Platform:
            checkCudaErrors(cudaMemcpyFromSymbol(&object_gpu_handle->IntersectionTest, fp_intersection_platform,
                                                 sizeof(FuncIntersectionTestPtr)));
            break;
        case PrimitiveType::Cylinder:
            checkCudaErrors(cudaMemcpyFromSymbol(&object_gpu_handle->IntersectionTest, fp_intersection_cylinder,
                                                 sizeof(FuncIntersectionTestPtr)));
            break;
        default:
            break;
    }
    checkCudaErrors(cudaMemcpyFromSymbol(&object_gpu_handle->ClosetHit, fp_closet_hit, sizeof(FuncClosetHitPtr)));
}