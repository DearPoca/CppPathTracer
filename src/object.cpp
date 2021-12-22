#include "object.h"

bool SphereIntersectionTest(Object &self, Ray &ray, ProceduralPrimitiveAttributes &attr) {
    float4 A_C = ray.origin - self.center_;
    float4 &B = ray.dir;
    float a = poca_mus::Dot(B, B);
    float b = poca_mus::Dot(A_C, B);
    float c = poca_mus::Dot(A_C, A_C) - self.radius_ * self.radius_;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < ray.tmax && temp > ray.tmin) {
            ray.tmax = temp;
            attr.hit_pos = ray.origin + temp * ray.dir;
            float4 normal = (attr.hit_pos - self.center_);
            attr.normal = normal / self.radius_;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < ray.tmax && temp > ray.tmin) {
            ray.tmax = temp;
            attr.hit_pos = ray.origin + temp * ray.dir;
            float4 normal = (attr.hit_pos - self.center_);
            attr.normal = normal / self.radius_;
            return true;
        }
    }
    return false;
}

void Object::ClosetHit(Ray &ray, RayPayload &payload, ProceduralPrimitiveAttributes &attr) {
    material_->EvalAttenuationAndCreateRay(attr.hit_pos, attr.normal, ray.dir, payload);
}

bool Object::IntersectionTest(Ray &ray, ProceduralPrimitiveAttributes &attr) {
    return SphereIntersectionTest(*this, ray, attr);
}

Object::Object() : radius_(1000.f), center_(float4(0.f, -1000.f, 0.f)) {
    this->material_ = new Material;
    IntersectionTestPtr = SphereIntersectionTest;
}

Object::~Object() {}