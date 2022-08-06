#pragma once

#include "material.h"
#include "ray_tracing_common.h"
#include "ray_tracing_math.hpp"

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
public:
	float3 GetAABBMin();
	float3 GetAABBMax();

	__device__ bool IntersectionTest(Ray& ray, IntersectionAttributes& attr);
	__device__ void ClosetHit(RayPayload& payload, IntersectionAttributes& attr);

	Object();

	PrimitiveType::Enum type_;
	Material material_;

	float3 center_;
	float radius_;	// for Sphere and Cylinder
	float y_pos_;	// for Platform
	float height_;	// for Cylinder
};


