#pragma once

#include <vector>

#include "object.h"
#include "ray_tracing_common.h"

class SceneBVH;

typedef SceneBVH* SceneBVHGPUHandle;

class SceneBVH {
public:
	static void AddObject(Object* obj);

	static SceneBVHGPUHandle* BuildBVH();

	static void UpdateObject(Object* obj);

	static void ReleaseBVH();

	__device__ Object* TraceRay(Ray& ray, IntersectionAttributes& attr);
private:
	static SceneBVH* Divide(std::vector<Object*>& objs, int l, int r);

	static SceneBVH* BuildBVHInCpu(std::vector<Object*>& objs);

	static SceneBVH* BuildBVHInGpu(SceneBVH* node_cpu_handle);

	static void UpdateSceneBVH(SceneBVH* node_cpu_handle);

	float3 AABB_min_;
	float3 AABB_max_;

	bool is_object_;
	SceneBVH* left_son_;
	SceneBVH* right_son_;
	Object obj_;
};





