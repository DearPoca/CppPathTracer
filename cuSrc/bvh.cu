#include <algorithm>
#include <cstring>
#include <map>

#include "bvh.h"
#include "path_tracing_common.h"
#include "ray_tracing_math.hpp"

BVHNode *Divide(std::vector<Object *> &objs, int l, int r) {
    if (l >= r) return nullptr;
    BVHNode *ret = new BVHNode;
    if (l == r - 1) {
        ret->left_son_ = nullptr;
        ret->right_son_ = nullptr;
        ret->minx_ = objs[l]->minx_;
        ret->miny_ = objs[l]->miny_;
        ret->minz_ = objs[l]->minz_;
        ret->maxx_ = objs[l]->maxx_;
        ret->maxy_ = objs[l]->maxy_;
        ret->maxz_ = objs[l]->maxz_;
        ret->is_object_ = true;
        ret->obj_ = objs[l];
        return ret;
    }
    float minx = objs[l]->minx_;
    float miny = objs[l]->miny_;
    float minz = objs[l]->minz_;
    float maxx = objs[l]->maxx_;
    float maxy = objs[l]->maxy_;
    float maxz = objs[l]->maxz_;
    for (int i = l + 1; i < r; ++i) {
        minx = MIN(minx, objs[i]->minx_);
        miny = MIN(miny, objs[i]->miny_);
        minz = MIN(minz, objs[i]->minz_);
        maxx = MAX(maxx, objs[i]->maxx_);
        maxy = MAX(maxy, objs[i]->maxy_);
        maxz = MAX(maxz, objs[i]->maxz_);
    }

    float span_x = maxx - minx;
    float span_y = maxy - miny;
    float span_z = maxz - minz;
    if (span_x > span_y && span_x > span_z) {
        std::sort(objs.begin() + l, objs.begin() + r, [](const Object *a, const Object *b) {
            return (a->minx_ + a->maxx_) / 2 < (b->minx_ + b->maxx_) / 2;
        });
    } else if (span_y > span_x && span_y > span_z) {
        std::sort(objs.begin() + l, objs.begin() + r, [](const Object *a, const Object *b) {
            return (a->miny_ + a->maxy_) / 2 < (b->miny_ + b->maxy_) / 2;
        });
    } else {
        std::sort(objs.begin() + l, objs.begin() + r, [](const Object *a, const Object *b) {
            return (a->minz_ + a->maxz_) / 2 < (b->minz_ + b->maxz_) / 2;
        });
    }
    int mid = (l + r) / 2;
    ret->left_son_ = Divide(objs, l, mid);
    ret->right_son_ = Divide(objs, mid, r);
    ret->minx_ = minx;
    ret->miny_ = miny;
    ret->minz_ = minz;
    ret->maxx_ = maxx;
    ret->maxy_ = maxy;
    ret->maxz_ = maxz;
    ret->is_object_ = false;
    ret->obj_ = nullptr;
    return ret;
}

BVHNode *BuildBVHInCpu(std::vector<Object *> &objs) { return Divide(objs, 0, objs.size()); }

std::map<BVHNode *, BVHNode *> bvh_node_cpu_handle_to_gpu_handle;
std::map<Object *, Object *> object_cpu_handle_to_gpu_handle;
std::map<Material *, Material *> materials_cpu_handle_to_gpu_handle;

// 根据BVH结点CPU指针递归的分配GPU空间, 返回结点的GPU指针
BVHNode *BuildBVHInGpu(BVHNode *node_cpu_handle) {
    if (node_cpu_handle == nullptr) return nullptr;
    BVHNode tmp;
    memcpy(&tmp, node_cpu_handle, sizeof(BVHNode));
    tmp.left_son_ = BuildBVHInGpu(node_cpu_handle->left_son_);
    tmp.right_son_ = BuildBVHInGpu(node_cpu_handle->right_son_);

    // 需要为物体分配GPU内存空间
    if (tmp.is_object_) {
        Object *obj_gpu_handle;
        cudaMalloc((void **)&obj_gpu_handle, sizeof(Object));
        // 材质库中未保存当前材质
        if (!materials_cpu_handle_to_gpu_handle.count(node_cpu_handle->obj_->material_)) {
            Material *mat_gpu_handle;
            cudaMalloc((void **)&mat_gpu_handle, sizeof(Material));
            materials_cpu_handle_to_gpu_handle[node_cpu_handle->obj_->material_] = mat_gpu_handle;
            MaterialMemCpyToGpu(node_cpu_handle->obj_->material_, mat_gpu_handle);
        }
        object_cpu_handle_to_gpu_handle[node_cpu_handle->obj_] = obj_gpu_handle;
        ObjectMemCpyToGpu(node_cpu_handle->obj_, obj_gpu_handle,
                          materials_cpu_handle_to_gpu_handle[node_cpu_handle->obj_->material_]);
        tmp.obj_ = obj_gpu_handle;
    }
    BVHNode *ret;
    cudaMalloc((void **)&ret, sizeof(BVHNode));
    cudaMemcpy((void *)ret, (void *)&tmp, sizeof(BVHNode), cudaMemcpyHostToDevice);
    return ret;
}

BVHNode *poca_mus::BuildBVH(std::vector<Object *> &objs) {
    BVHNode *bvh_cpu_root = BuildBVHInCpu(objs);
    return BuildBVHInGpu(bvh_cpu_root);
}

__device__ Object *poca_mus::TraceRay(BVHNode *node, Ray &ray, ProceduralPrimitiveAttributes &attr) {
    if (node == nullptr) return nullptr;
    if (node->is_object_) {
        if (node->obj_->IntersectionTest(*node->obj_, ray, attr))
            return node->obj_;
        else
            return nullptr;
    }
    float local_tmin, local_tmax;
    if (ray.dir.x != 0.f) {
        float tmin = (node->minx_ - ray.origin.x) / ray.dir.x;
        float tmax = (node->maxx_ - ray.origin.x) / ray.dir.x;
        if (tmin > tmax) {
            float tmp = tmin;
            tmin = tmax;
            tmax = tmp;
        }
        local_tmin = tmin;
        local_tmax = tmax;
    }
    if (ray.dir.y != 0.f) {
        float tmin = (node->miny_ - ray.origin.y) / ray.dir.y;
        float tmax = (node->maxy_ - ray.origin.y) / ray.dir.y;
        if (tmin > tmax) {
            float tmp = tmin;
            tmin = tmax;
            tmax = tmp;
        }
        local_tmin = MIN(local_tmin, tmin);
        local_tmax = MAX(local_tmax, tmax);
    }
    if (ray.dir.z != 0.f) {
        float tmin = (node->miny_ - ray.origin.z) / ray.dir.z;
        float tmax = (node->maxy_ - ray.origin.z) / ray.dir.z;
        if (tmin > tmax) {
            float tmp = tmin;
            tmin = tmax;
            tmax = tmp;
        }
        local_tmin = MIN(local_tmin, tmin);
        local_tmax = MAX(local_tmax, tmax);
    }
    if (local_tmin > local_tmax || local_tmin > ray.tmax || local_tmax < ray.tmin) return nullptr;
    Object *ret = TraceRay(node->left_son_, ray, attr);
    Object *right_son = TraceRay(node->right_son_, ray, attr);
    if (right_son != nullptr) ret = right_son;
    return ret;
}