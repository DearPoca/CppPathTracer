#pragma once

#include <vector>

#include "object.h"
#include "ray_tracing_common.h"

class SceneBVH;

typedef SceneBVH* SceneBVHGPUHandle;

class SceneBVH {
public:
	static void AddObject(Object* obj);

	static SceneBVHGPUHandle BuildBVH();

	static void UpdateObject(Object* obj);

	static void ReleaseBVH();

	__device__ bool TraceRay(Ray ray, IntersectionAttributes& attr, Object &ret);
private:
	static int Divide(std::vector<Object*>& objs, int l, int r);

	static void BuildBVHInCpu(std::vector<Object*>& objs);

	static SceneBVHGPUHandle BuildBVHInGpu();

	static void UpdateSceneBVH(int node_idx);

	float3 AABB_min_;
	float3 AABB_max_;

	bool is_object_;
	int left_son_;
	int right_son_;
	Object obj_;
};





