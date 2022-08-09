#include "bvh.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <unordered_map>
#include <set>
#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>

#include "logger.hpp"
#include "ray_tracing_math.hpp"
#include "ray_tracing_common.h"

static std::vector<SceneBVH> world_bvh_cpu_array;
static SceneBVHGPUHandle world_bvh_gpu_root;
static std::vector<Object*> bvh_objs;
static std::unordered_map<Object*, int> object_to_bvh_node_cpu_handle;
static std::unordered_map<int, int> bvh_node_pars;

void SceneBVH::AddObject(Object* obj) {
	static std::set<Object*> st;
	if (obj == nullptr || st.count(obj)) {
		return;
	}
	bvh_objs.push_back(obj);
	st.insert(obj);
}

int SceneBVH::Divide(std::vector<Object*>& objs, int l, int r) {
	if (l >= r) return -1;
	int ret_idx = world_bvh_cpu_array.size();
	world_bvh_cpu_array.push_back(SceneBVH());
	float3 obj_l_AABB_min = objs[l]->GetAABBMin();
	float3 obj_l_AABB_max = objs[l]->GetAABBMax();
	if (l == r - 1) {
		world_bvh_cpu_array[ret_idx].left_son_ = -1;
		world_bvh_cpu_array[ret_idx].right_son_ = -1;
		world_bvh_cpu_array[ret_idx].AABB_min_ = obj_l_AABB_min;
		world_bvh_cpu_array[ret_idx].AABB_max_ = obj_l_AABB_max;
		world_bvh_cpu_array[ret_idx].is_object_ = true;
		world_bvh_cpu_array[ret_idx].obj_ = *objs[l];
		object_to_bvh_node_cpu_handle[objs[l]] = ret_idx;
		return ret_idx;
	}
	float minx = obj_l_AABB_min.x;
	float miny = obj_l_AABB_min.y;
	float minz = obj_l_AABB_min.z;
	float maxx = obj_l_AABB_max.x;
	float maxy = obj_l_AABB_max.y;
	float maxz = obj_l_AABB_max.z;
	for (int i = l + 1; i < r; ++i) {
		float3 cur_AABB_min = objs[i]->GetAABBMin();
		float3 cur_AABB_max = objs[i]->GetAABBMax();
		minx = MIN(minx, cur_AABB_min.x);
		miny = MIN(miny, cur_AABB_min.y);
		minz = MIN(minz, cur_AABB_min.z);
		maxx = MAX(maxx, cur_AABB_max.x);
		maxy = MAX(maxy, cur_AABB_max.y);
		maxz = MAX(maxz, cur_AABB_max.z);
	}
	float span_x = maxx - minx;
	float span_y = maxy - miny;
	float span_z = maxz - minz;
	if (span_x >= span_y && span_x >= span_z) {
		std::sort(objs.begin() + l, objs.begin() + r, [](Object* a, Object* b) {
			return (a->GetAABBMin().x + a->GetAABBMax().x) / 2 < (b->GetAABBMin().x + b->GetAABBMax().x) / 2;
			});
	}
	else if (span_y >= span_z) {
		std::sort(objs.begin() + l, objs.begin() + r, [](Object* a, Object* b) {
			return (a->GetAABBMin().y + a->GetAABBMax().y) / 2 < (b->GetAABBMin().y + b->GetAABBMax().y) / 2;
			});
	}
	else {
		std::sort(objs.begin() + l, objs.begin() + r, [](Object* a, Object* b) {
			return (a->GetAABBMin().z + a->GetAABBMax().z) / 2 < (b->GetAABBMin().z + b->GetAABBMax().z) / 2;
			});
	}
	int mid = (l + r) / 2;
	world_bvh_cpu_array[ret_idx].left_son_ = Divide(objs, l, mid);
	world_bvh_cpu_array[ret_idx].right_son_ = Divide(objs, mid, r);
	bvh_node_pars[world_bvh_cpu_array[ret_idx].left_son_] = ret_idx;
	bvh_node_pars[world_bvh_cpu_array[ret_idx].right_son_] = ret_idx;
	world_bvh_cpu_array[ret_idx].AABB_min_ = make_float3(minx, miny, minz);
	world_bvh_cpu_array[ret_idx].AABB_max_ = make_float3(maxx, maxy, maxz);
	world_bvh_cpu_array[ret_idx].is_object_ = false;
	return ret_idx;
}

void SceneBVH::BuildBVHInCpu(std::vector<Object*>& objs) {
	world_bvh_cpu_array.clear();
	Divide(objs, 0, objs.size());
}

SceneBVHGPUHandle SceneBVH::BuildBVHInGpu() {
	cudaError_t err;

	SceneBVHGPUHandle ret;

	err = cudaMalloc((void**)&ret, world_bvh_cpu_array.size() * sizeof(SceneBVH));
	if (err != cudaSuccess) {
		log_error("cudaMalloc Failed: %s", cudaGetErrorString(err));
	}

	err = cudaMemcpy((void*)ret, (void*)world_bvh_cpu_array.data(), world_bvh_cpu_array.size() *
		sizeof(SceneBVH), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		log_error("cudaMemcpy Failed: %s", cudaGetErrorString(err));
	}
	world_bvh_gpu_root = ret;
	return ret;
}

SceneBVHGPUHandle SceneBVH::BuildBVH() {
	BuildBVHInCpu(bvh_objs);
	bvh_node_pars[0] = -1;
	return (SceneBVHGPUHandle)BuildBVHInGpu();
}

void SceneBVH::UpdateSceneBVH(int node_idx) {
	if (node_idx < 0) return;

	if (world_bvh_cpu_array[node_idx].is_object_) {
		world_bvh_cpu_array[node_idx].AABB_max_ = world_bvh_cpu_array[node_idx].obj_.GetAABBMax();
		world_bvh_cpu_array[node_idx].AABB_min_ = world_bvh_cpu_array[node_idx].obj_.GetAABBMin();
	}
	else {
#define UPDATE_AABB_FOR_CUR_NODE(target, axis, FUNC) \
			world_bvh_cpu_array[node_idx].target.axis = \
			FUNC(world_bvh_cpu_array[world_bvh_cpu_array[node_idx].left_son_].target.axis, \
			world_bvh_cpu_array[world_bvh_cpu_array[node_idx].right_son_].target.axis);

		UPDATE_AABB_FOR_CUR_NODE(AABB_max_, x, MAX);
		UPDATE_AABB_FOR_CUR_NODE(AABB_max_, y, MAX);
		UPDATE_AABB_FOR_CUR_NODE(AABB_max_, z, MAX);
		UPDATE_AABB_FOR_CUR_NODE(AABB_min_, x, MIN);
		UPDATE_AABB_FOR_CUR_NODE(AABB_min_, y, MIN);
		UPDATE_AABB_FOR_CUR_NODE(AABB_min_, z, MIN);
	}
}

void SceneBVH::UpdateObject(Object* obj) {
	if (!object_to_bvh_node_cpu_handle.count(obj))return;
	int cur_bvh_idx = object_to_bvh_node_cpu_handle[obj];
	world_bvh_cpu_array[cur_bvh_idx].obj_ = *obj;
	while (cur_bvh_idx != -1) {
		UpdateSceneBVH(cur_bvh_idx);
		cur_bvh_idx = bvh_node_pars[cur_bvh_idx];
	}
	cudaError_t err = cudaMemcpy((void*)world_bvh_gpu_root, (void*)world_bvh_cpu_array.data(), world_bvh_cpu_array.size() *
		sizeof(SceneBVH), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		log_error("cudaMemcpy Failed: %s", cudaGetErrorString(err));
	}
}

void SceneBVH::ReleaseBVH() {
	cudaFree(world_bvh_gpu_root);
	world_bvh_cpu_array.clear();
	bvh_objs.clear();
	object_to_bvh_node_cpu_handle.clear();
	bvh_node_pars.clear();
}

__device__ bool SceneBVH::TraceRay(Ray ray, IntersectionAttributes& attr, Object& obj) {
	int stack[512];
	int top = 0;
	stack[top++] = 0;
	bool ret = false;
	while (top > 0) {
		int node = stack[--top];
		if (node == -1) continue;
		if (this[node].is_object_) {
			if (this[node].obj_.IntersectionTest(ray, attr)) {
				obj = this[node].obj_;
				ret = true;
			}
		}
		float local_tmin = -DEFAULT_RAY_TMAX * 2, local_tmax = DEFAULT_RAY_TMAX * 2;
		if (ray.dir.x != 0.f) {
			float t0 = (this[node].AABB_min_.x - ray.origin.x) / ray.dir.x;
			float t1 = (this[node].AABB_max_.x - ray.origin.x) / ray.dir.x;
			local_tmin = MAX(local_tmin, MIN(t0, t1));
			local_tmax = MIN(local_tmax, MAX(t0, t1));
		}
		if (ray.dir.y != 0.f) {
			float t0 = (this[node].AABB_min_.y - ray.origin.y) / ray.dir.y;
			float t1 = (this[node].AABB_max_.y - ray.origin.y) / ray.dir.y;
			local_tmin = MAX(local_tmin, MIN(t0, t1));
			local_tmax = MIN(local_tmax, MAX(t0, t1));
		}
		if (ray.dir.z != 0.f) {
			float t0 = (this[node].AABB_min_.z - ray.origin.z) / ray.dir.z;
			float t1 = (this[node].AABB_max_.z - ray.origin.z) / ray.dir.z;
			local_tmin = MAX(local_tmin, MIN(t0, t1));
			local_tmax = MIN(local_tmax, MAX(t0, t1));
		}
		if (local_tmin > local_tmax || local_tmin > ray.tmax || local_tmax < ray.tmin) continue;
		stack[top++] = this[node].left_son_;
		stack[top++] = this[node].right_son_;
	}
	return ret;
}