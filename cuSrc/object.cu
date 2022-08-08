#include "object.h"

#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>

#include "ray_tracing_math.hpp"
#include "ray_tracing_common.h"

__device__ bool SphereIntersectionTest(Object& self, Ray& ray, IntersectionAttributes& attr) {
	float3 A_C = ray.origin - self.center_;
	float3& B = ray.dir;
	float a = dot(B, B);
	float b = dot(A_C, B);
	float c = dot(A_C, A_C) - self.radius_ * self.radius_;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < ray.tmax && temp > ray.tmin) {
			ray.tmax = temp;
			attr.hit_pos = ray.origin + temp * ray.dir;
			float3 normal = (attr.hit_pos - self.center_);
			attr.normal = normal / self.radius_;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < ray.tmax && temp > ray.tmin) {
			ray.tmax = temp;
			attr.hit_pos = ray.origin + temp * ray.dir;
			attr.normal = normalize(attr.hit_pos - self.center_);
			return true;
		}
	}
	return false;
}

__device__ bool PlatformIntersectionTest(Object& self, Ray& ray, IntersectionAttributes& attr) {
	if (ray.origin.y < self.y_pos_ && ray.dir.y > 0.f || ray.origin.y > self.y_pos_ && ray.dir.y < 0.f) {
		float temp = (self.y_pos_ - ray.origin.y) / ray.dir.y;
		if (temp < ray.tmax && temp > ray.tmin) {
			ray.tmax = temp;
			attr.hit_pos = ray.origin + temp * ray.dir;
			attr.normal = normalize(make_float3(0, -ray.dir.y, 0));
			return true;
		}
	}
	return false;
}

__device__ bool CylinderIntersectionTest(Object& self, Ray& ray, IntersectionAttributes& attr) {
	bool ret = false;
	float upper_section_y_pos = self.center_.y + self.height_ / 2;
	if ((ray.origin.y < upper_section_y_pos && ray.dir.y > 0.f) ||
		(ray.origin.y > upper_section_y_pos && ray.dir.y < 0.f)) {
		float temp = (upper_section_y_pos - ray.origin.y) / ray.dir.y;
		float3 hit_pos = ray.origin + temp * ray.dir;
		if (temp < ray.tmax && temp > ray.tmin &&
			sqrt((hit_pos.x - self.center_.x) * (hit_pos.x - self.center_.x) +
				(hit_pos.z - self.center_.z) * (hit_pos.z - self.center_.z)) < self.radius_) {
			ray.tmax = temp;
			attr.hit_pos = hit_pos;
			attr.normal = normalize(make_float3(0, -ray.dir.y, 0));
			ret = true;
		}
	}
	float lower_section_y_pos = self.center_.y - self.height_ / 2;
	if ((ray.origin.y < lower_section_y_pos && ray.dir.y > 0.f) ||
		(ray.origin.y > lower_section_y_pos && ray.dir.y < 0.f)) {
		float temp = (lower_section_y_pos - ray.origin.y) / ray.dir.y;
		float3 hit_pos = ray.origin + temp * ray.dir;
		if (temp < ray.tmax && temp > ray.tmin &&
			sqrt((hit_pos.x - self.center_.x) * (hit_pos.x - self.center_.x) +
				(hit_pos.z - self.center_.z) * (hit_pos.z - self.center_.z)) < self.radius_) {
			ray.tmax = temp;
			attr.hit_pos = hit_pos;
			attr.normal = normalize(make_float3(0, -ray.dir.y, 0));
			ret = true;
		}
	}

	float& dx = ray.dir.x;
	float& dz = ray.dir.z;
	float& r = self.radius_;
	float cx = ray.origin.x - self.center_.x;
	float cz = ray.origin.z - self.center_.z;

	float a = dx * dx + dz * dz;
	float b = cx * dx + cz * dz;
	float c = cx * cx + cz * cz - r * r;
	float discriminant = b * b - a * c;
	if (discriminant > 0.f) {
		float temp = (-b - sqrt(discriminant)) / a;
		float3 hit_pos = ray.origin + temp * ray.dir;
		if (temp < ray.tmax && temp > ray.tmin && hit_pos.y > lower_section_y_pos && hit_pos.y < upper_section_y_pos) {
			ray.tmax = temp;
			attr.hit_pos = hit_pos;
			float3 normal = make_float3(hit_pos.x - self.center_.x, 0.f, hit_pos.z - self.center_.z);
			attr.normal = normalize(normal);
			ret = true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		hit_pos = ray.origin + temp * ray.dir;
		if (temp < ray.tmax && temp > ray.tmin && hit_pos.y > lower_section_y_pos && hit_pos.y < upper_section_y_pos) {
			ray.tmax = temp;
			attr.hit_pos = hit_pos;
			float3 normal = make_float3(hit_pos.x - self.center_.x, 0.f, hit_pos.z - self.center_.z);
			attr.normal = normalize(normal);
			ret = true;
		}
	}
	return ret;
}

__device__ bool Object::IntersectionTest(Ray& ray, IntersectionAttributes& attr) {
	switch (type_) {
	case PrimitiveType::Sphere:
		return SphereIntersectionTest(*this, ray, attr);
		break;
	case PrimitiveType::Platform:
		return PlatformIntersectionTest(*this, ray, attr);
		break;
	case PrimitiveType::Cylinder:
		return CylinderIntersectionTest(*this, ray, attr);
		break;
	default:
		return false;
	}
}

__device__ void Object::ClosetHit(RayPayload& payload, IntersectionAttributes& attr) {
	material_.EvalAttenuationAndCreateRay(payload, attr.hit_pos, attr.normal, payload.ray.dir);
}

float3 Object::GetAABBMax() {
	float3 AABB_max;
	float tolerance = BOUNCE_RAY_TMIN * 5.f;
	switch (type_) {
	case PrimitiveType::Sphere:
		AABB_max = center_ + make_float3(ABS(radius_));
		break;
	case PrimitiveType::Platform:
		AABB_max = make_float3(DEFAULT_RAY_TMAX * 5, y_pos_ + tolerance, DEFAULT_RAY_TMAX * 5);
		break;
	case PrimitiveType::Cylinder:
		AABB_max = make_float3(center_.x + ABS(radius_), center_.y + height_ / 2 + tolerance, center_.z + ABS(radius_));
		break;
	default:
		break;
	}
	return AABB_max;
}

float3 Object::GetAABBMin() {
	float3 AABB_min;
	float tolerance = BOUNCE_RAY_TMIN * 5.f;
	switch (type_) {
	case PrimitiveType::Sphere:
		AABB_min = center_ - make_float3(ABS(radius_));
		break;
	case PrimitiveType::Platform:
		AABB_min = make_float3(-DEFAULT_RAY_TMAX * 5, y_pos_ - tolerance, -DEFAULT_RAY_TMAX * 5);
		break;
	case PrimitiveType::Cylinder:
		AABB_min = make_float3(center_.x - ABS(radius_), center_.y - height_ / 2 - tolerance, center_.z - ABS(radius_));
		break;
	default:
		break;
	}
	return AABB_min;
}

__device__ __host__ Object::Object() {
	type_ = PrimitiveType::Sphere;
	center_ = make_float3(0.f);
	radius_ = 0.f;
}