#include "bvh.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>

#include "ray_tracing_math.hpp"
#include "ray_tracing_common.h"

static SceneBVH* bvh_cpu_root_handle;
static std::vector<Object*> bvh_objs;
static std::map<SceneBVH*, SceneBVH*> bvh_node_cpu_handle_to_gpu_handle;
static std::map<Object*, SceneBVH*> object_to_bvh_node_cpu_handle;
static std::map<SceneBVH*, Object*> bvh_node_cpu_handle_to_object;
static std::map<SceneBVH*, SceneBVH*> bvh_node_pars;

void SceneBVH::AddObject(Object* obj) {
	static std::set<Object*> st;
	if (obj == nullptr || st.count(obj)) {
		return;
	}
	bvh_objs.push_back(obj);
	st.insert(obj);
}

SceneBVH* SceneBVH::Divide(std::vector<Object*>& objs, int l, int r) {
	if (l >= r) return nullptr;
	SceneBVH* ret = new SceneBVH();
	float3 obj_l_AABB_min = objs[l]->GetAABBMin();
	float3 obj_l_AABB_max = objs[l]->GetAABBMax();
	if (l == r - 1) {
		ret->left_son_ = nullptr;
		ret->right_son_ = nullptr;
		ret->AABB_min_ = obj_l_AABB_min;
		ret->AABB_max_ = obj_l_AABB_max;
		ret->is_object_ = true;
		ret->obj_ = *objs[l];
		object_to_bvh_node_cpu_handle[objs[l]] = ret;
		bvh_node_cpu_handle_to_object[ret] = objs[l];
		return ret;
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
	ret->left_son_ = Divide(objs, l, mid);
	ret->right_son_ = Divide(objs, mid, r);
	bvh_node_pars[ret->left_son_] = ret;
	bvh_node_pars[ret->right_son_] = ret;
	ret->AABB_min_ = make_float3(minx, miny, minz);
	ret->AABB_max_ = make_float3(maxx, maxy, maxz);
	ret->is_object_ = false;
	return ret;
}

SceneBVH* SceneBVH::BuildBVHInCpu(std::vector<Object*>& objs) {
	return Divide(objs, 0, objs.size());
}

SceneBVH* SceneBVH::BuildBVHInGpu(SceneBVH* node_cpu_handle) {
	if (node_cpu_handle == nullptr) return nullptr;
	SceneBVH tmp;
	memcpy(&tmp, node_cpu_handle, sizeof(SceneBVH));
	tmp.left_son_ = BuildBVHInGpu(node_cpu_handle->left_son_);
	tmp.right_son_ = BuildBVHInGpu(node_cpu_handle->right_son_);

	SceneBVH* ret;
	checkCudaErrors(cudaMalloc((void**)&ret, sizeof(SceneBVH)));
	checkCudaErrors(cudaMemcpy((void*)ret, (void*)&tmp, sizeof(SceneBVH), cudaMemcpyHostToDevice));
	bvh_node_cpu_handle_to_gpu_handle[node_cpu_handle] = ret;
	return ret;
}

SceneBVHGPUHandle SceneBVH::BuildBVH() {
	bvh_cpu_root_handle = BuildBVHInCpu(bvh_objs);
	bvh_node_pars[bvh_cpu_root_handle] = nullptr;
	return (SceneBVHGPUHandle)BuildBVHInGpu(bvh_cpu_root_handle);
}

void SceneBVH::UpdateSceneBVH(SceneBVH* node_cpu_handle) {
	if (node_cpu_handle == nullptr) return;
	if (node_cpu_handle->is_object_) {
		node_cpu_handle->AABB_max_ = node_cpu_handle->obj_.GetAABBMax();
		node_cpu_handle->AABB_min_ = node_cpu_handle->obj_.GetAABBMin();
		checkCudaErrors(cudaMemcpy((void*)bvh_node_cpu_handle_to_gpu_handle[node_cpu_handle], (void*)node_cpu_handle, sizeof(SceneBVH), cudaMemcpyHostToDevice));
	}
	else {
#define UPDATE_AABB_FOR_CUR_NODE(target, axis, FUNC) node_cpu_handle->target.axis = \
			FUNC(node_cpu_handle->left_son_->target.axis, node_cpu_handle->right_son_->target.axis);

		UPDATE_AABB_FOR_CUR_NODE(AABB_max_, x, MAX);
		UPDATE_AABB_FOR_CUR_NODE(AABB_max_, y, MAX);
		UPDATE_AABB_FOR_CUR_NODE(AABB_max_, z, MAX);
		UPDATE_AABB_FOR_CUR_NODE(AABB_min_, x, MIN);
		UPDATE_AABB_FOR_CUR_NODE(AABB_min_, y, MIN);
		UPDATE_AABB_FOR_CUR_NODE(AABB_min_, z, MIN);

		SceneBVH tmp_host = *node_cpu_handle;
		SceneBVH tmp_device;
		SceneBVH* node_gpu_handle = bvh_node_cpu_handle_to_gpu_handle[node_cpu_handle];
		checkCudaErrors(cudaMemcpy((void*)&tmp_device, (void*)node_gpu_handle, sizeof(SceneBVH), cudaMemcpyDeviceToHost));
		tmp_host.left_son_ = tmp_device.right_son_;
		tmp_host.right_son_ = tmp_device.right_son_;
		checkCudaErrors(cudaMemcpy((void*)node_gpu_handle, (void*)&tmp_host, sizeof(SceneBVH), cudaMemcpyHostToDevice));
	}
}

void SceneBVH::UpdateObject(Object* obj) {
	if (!object_to_bvh_node_cpu_handle.count(obj))return;
	SceneBVH* cur_bvh_node_cpu_handle = object_to_bvh_node_cpu_handle[obj];
	cur_bvh_node_cpu_handle->obj_ = *obj;
	while (cur_bvh_node_cpu_handle != nullptr) {
		UpdateSceneBVH(cur_bvh_node_cpu_handle);
		cur_bvh_node_cpu_handle = bvh_node_pars[cur_bvh_node_cpu_handle];
	}
}

void SceneBVH::ReleaseBVH() {
	for (auto p : bvh_node_cpu_handle_to_gpu_handle) {
		cudaFree(p.second);
		delete p.first;
	}
	bvh_objs.clear();
	bvh_cpu_root_handle = nullptr;
	bvh_node_cpu_handle_to_gpu_handle.clear();
	object_to_bvh_node_cpu_handle.clear();
	bvh_node_cpu_handle_to_object.clear();
	bvh_node_pars.clear();
}

__device__ bool SceneBVH::TraceRay(Ray ray, IntersectionAttributes& attr, Object& obj) {
	SceneBVH* stack[512];
	int top = 0;
	stack[top++] = this;
	bool ret = false;
	while (top > 0) {
		SceneBVH* node = stack[--top];
		if (node == nullptr) continue;
		if (node->is_object_) {
			if (node->obj_.IntersectionTest(ray, attr)) {
				obj = node->obj_;
				ret = true;
			}
		}
		float local_tmin = -DEFAULT_RAY_TMAX, local_tmax = DEFAULT_RAY_TMAX;
		if (ray.dir.x != 0.f) {
			float t0 = (node->AABB_min_.x - ray.origin.x) / ray.dir.x;
			float t1 = (node->AABB_max_.x - ray.origin.x) / ray.dir.x;
			local_tmin = MAX(local_tmin, MIN(t0, t1));
			local_tmax = MIN(local_tmax, MAX(t0, t1));
		}
		if (ray.dir.y != 0.f) {
			float t0 = (node->AABB_min_.y - ray.origin.y) / ray.dir.y;
			float t1 = (node->AABB_max_.y - ray.origin.y) / ray.dir.y;
			local_tmin = MAX(local_tmin, MIN(t0, t1));
			local_tmax = MIN(local_tmax, MAX(t0, t1));
		}
		if (ray.dir.z != 0.f) {
			float t0 = (node->AABB_min_.z - ray.origin.z) / ray.dir.z;
			float t1 = (node->AABB_max_.z - ray.origin.z) / ray.dir.z;
			local_tmin = MAX(local_tmin, MIN(t0, t1));
			local_tmax = MIN(local_tmax, MAX(t0, t1));
		}
		if (local_tmin > local_tmax || local_tmin > ray.tmax || local_tmax < ray.tmin) continue;
		stack[top++] = node->left_son_;
		stack[top++] = node->right_son_;
	}
	return ret;
}