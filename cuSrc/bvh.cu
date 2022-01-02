#include <algorithm>
#include <cstring>
#include <map>

#include "bvh.h"
#include "path_tracing_common.h"
#include "ray_tracing_math.hpp"

BVHNode *bvh_cpu_root_handle;
std::map<BVHNode *, BVHNode *> bvh_node_cpu_handle_to_gpu_handle;
std::map<Object *, Object *> object_cpu_handle_to_gpu_handle;
std::map<Material *, Material *> materials_cpu_handle_to_gpu_handle;

BVHNode *Divide(std::vector<Object *> &objs, int l, int r) {
    if (l >= r) return nullptr;
    BVHNode *ret = new BVHNode;
    if (l == r - 1) {
        ret->left_son_ = nullptr;
        ret->right_son_ = nullptr;
        ret->AABB_min_ = objs[l]->AABB_min_;
        ret->AABB_max_ = objs[l]->AABB_max_;
        ret->is_object_ = true;
        ret->obj_ = objs[l];
        return ret;
    }
    float minx = objs[l]->AABB_min_.x;
    float miny = objs[l]->AABB_min_.y;
    float minz = objs[l]->AABB_min_.z;
    float maxx = objs[l]->AABB_max_.x;
    float maxy = objs[l]->AABB_max_.y;
    float maxz = objs[l]->AABB_max_.z;
    for (int i = l + 1; i < r; ++i) {
        minx = MIN(minx, objs[i]->AABB_min_.x);
        miny = MIN(miny, objs[i]->AABB_min_.y);
        minz = MIN(minz, objs[i]->AABB_min_.z);
        maxx = MAX(maxx, objs[i]->AABB_max_.x);
        maxy = MAX(maxy, objs[i]->AABB_max_.y);
        maxz = MAX(maxz, objs[i]->AABB_max_.z);
    }

    float span_x = maxx - minx;
    float span_y = maxy - miny;
    float span_z = maxz - minz;
    if (span_x > span_y && span_x > span_z) {
        std::sort(objs.begin() + l, objs.begin() + r, [](const Object *a, const Object *b) {
            return (a->AABB_min_.x + a->AABB_max_.x) / 2 < (b->AABB_min_.x + b->AABB_max_.x) / 2;
        });
    } else if (span_y > span_x && span_y > span_z) {
        std::sort(objs.begin() + l, objs.begin() + r, [](const Object *a, const Object *b) {
            return (a->AABB_min_.y + a->AABB_max_.y) / 2 < (b->AABB_min_.y + b->AABB_max_.y) / 2;
        });
    } else {
        std::sort(objs.begin() + l, objs.begin() + r, [](const Object *a, const Object *b) {
            return (a->AABB_min_.z + a->AABB_max_.z) / 2 < (b->AABB_min_.z + b->AABB_max_.z) / 2;
        });
    }
    int mid = (l + r) / 2;
    ret->left_son_ = Divide(objs, l, mid);
    ret->right_son_ = Divide(objs, mid, r);
    ret->AABB_min_ = Float4(minx, miny, minz);
    ret->AABB_max_ = Float4(maxx, maxy, maxz);
    ret->is_object_ = false;
    ret->obj_ = nullptr;
    return ret;
}

BVHNode *BuildBVHInCpu(std::vector<Object *> &objs) { return Divide(objs, 0, objs.size()); }

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
        checkCudaErrors(cudaMalloc((void **)&obj_gpu_handle, sizeof(Object)));
        // 材质库中未保存当前材质
        if (!materials_cpu_handle_to_gpu_handle.count(node_cpu_handle->obj_->material_)) {
            Material *mat_gpu_handle;
            checkCudaErrors(cudaMalloc((void **)&mat_gpu_handle, sizeof(Material)));
            materials_cpu_handle_to_gpu_handle[node_cpu_handle->obj_->material_] = mat_gpu_handle;
            MaterialMemCpyToGpu(node_cpu_handle->obj_->material_, mat_gpu_handle);
        }
        object_cpu_handle_to_gpu_handle[node_cpu_handle->obj_] = obj_gpu_handle;
        ObjectMemCpyToGpu(node_cpu_handle->obj_, obj_gpu_handle,
                          materials_cpu_handle_to_gpu_handle[node_cpu_handle->obj_->material_]);
        tmp.obj_ = obj_gpu_handle;
    }
    BVHNode *ret;
    checkCudaErrors(cudaMalloc((void **)&ret, sizeof(BVHNode)));
    checkCudaErrors(cudaMemcpy((void *)ret, (void *)&tmp, sizeof(BVHNode), cudaMemcpyHostToDevice));
    bvh_node_cpu_handle_to_gpu_handle[node_cpu_handle] = ret;
    return ret;
}

BVHNode *poca_mus::BuildBVH(std::vector<Object *> &objs) {
    bvh_cpu_root_handle = BuildBVHInCpu(objs);
    return BuildBVHInGpu(bvh_cpu_root_handle);
}

void UpdateBVHNode(BVHNode *node_cpu_handle) {
    if (node_cpu_handle == nullptr) return;
    if (node_cpu_handle->is_object_) {
        node_cpu_handle->AABB_max_ = node_cpu_handle->obj_->AABB_max_;
        node_cpu_handle->AABB_min_ = node_cpu_handle->obj_->AABB_min_;
    } else {
        UpdateBVHNode(node_cpu_handle->left_son_);
        UpdateBVHNode(node_cpu_handle->right_son_);
        node_cpu_handle->AABB_max_.x =
            MAX(node_cpu_handle->left_son_->AABB_max_.x, node_cpu_handle->right_son_->AABB_max_.x);
        node_cpu_handle->AABB_max_.y =
            MAX(node_cpu_handle->left_son_->AABB_max_.y, node_cpu_handle->right_son_->AABB_max_.y);
        node_cpu_handle->AABB_max_.z =
            MAX(node_cpu_handle->left_son_->AABB_max_.z, node_cpu_handle->right_son_->AABB_max_.z);
        node_cpu_handle->AABB_min_.x =
            MIN(node_cpu_handle->left_son_->AABB_min_.x, node_cpu_handle->right_son_->AABB_min_.x);
        node_cpu_handle->AABB_min_.y =
            MIN(node_cpu_handle->left_son_->AABB_min_.y, node_cpu_handle->right_son_->AABB_min_.y);
        node_cpu_handle->AABB_min_.z =
            MIN(node_cpu_handle->left_son_->AABB_min_.z, node_cpu_handle->right_son_->AABB_min_.z);
        BVHNode tmp;
        BVHNode *node_gpu_handle = bvh_node_cpu_handle_to_gpu_handle[node_cpu_handle];
        checkCudaErrors(cudaMemcpy((void *)&tmp, (void *)node_gpu_handle, sizeof(BVHNode), cudaMemcpyDeviceToHost));
        tmp.AABB_max_ = node_cpu_handle->AABB_max_;
        tmp.AABB_min_ = node_cpu_handle->AABB_min_;
        checkCudaErrors(cudaMemcpy((void *)node_gpu_handle, (void *)&tmp, sizeof(BVHNode), cudaMemcpyHostToDevice));
    }
}

void poca_mus::UpdateBVHInfos() {
    for (auto mat : materials_cpu_handle_to_gpu_handle) {
        MaterialMemCpyToGpu(mat.first, mat.second);
    }
    for (auto obj : object_cpu_handle_to_gpu_handle) {
        obj.first->UpdataAABB();
        ObjectMemCpyToGpu(obj.first, obj.second, materials_cpu_handle_to_gpu_handle[obj.first->material_]);
    }
    UpdateBVHNode(bvh_cpu_root_handle);
}

void poca_mus::ReleaseBVH() {
    for (auto p : materials_cpu_handle_to_gpu_handle) {
        cudaFree(p.second);
    }
    for (auto p : object_cpu_handle_to_gpu_handle) {
        cudaFree(p.second);
    }
    for (auto p : bvh_node_cpu_handle_to_gpu_handle) {
        cudaFree(p.second);
        delete p.first;
    }
}

__device__ Object *poca_mus::TraceRay(BVHNode *root, Ray &ray, IntersectionAttributes &attr) {
    BVHNode *stack[512];
    int top = 0;
    stack[top++] = root;
    Object *ret = nullptr;
    while (top > 0) {
        BVHNode *node = stack[--top];
        if (node == nullptr) continue;
        if (node->is_object_) {
            if (node->obj_->IntersectionTest(*node->obj_, ray, attr)) ret = node->obj_;
        }
        float local_tmin = -3e+30f, local_tmax = 3e+30f;
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