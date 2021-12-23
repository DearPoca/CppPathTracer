#ifndef RAY_TRACING_MATH_HPP
#define RAY_TRACING_MATH_HPP

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#ifndef MIN
#define MIN(a, b) (a < b ? a : b)
#endif

#ifndef ABS
#define ABS(a) (a >= 0 ? a : -a)
#endif

#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <utility>

#define __COMMON_GPU_CPU__ __device__ __host__
#define __COMMON_GPU_CPU_INLINE__ __COMMON_GPU_CPU__ inline

class Float4 {
private:
public:
    float x;
    float y;
    float z;
    float w;

    __COMMON_GPU_CPU_INLINE__ Float4() : x(0.f), y(0.f), z(0.f), w(0.f) {}
    __COMMON_GPU_CPU_INLINE__ Float4(float n) : x(n), y(n), z(n), w(n) {}
    __COMMON_GPU_CPU_INLINE__ Float4(float x, float y, float z, float w = 0.0f) {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
    __COMMON_GPU_CPU_INLINE__ Float4(const Float4 &vec) : x(vec.x), y(vec.y), z(vec.z), w(vec.w) {}

    __COMMON_GPU_CPU_INLINE__ float &operator[](int i) {
        switch (i) {
            case 0:
                return this->x;
            case 1:
                return this->y;
            case 2:
                return this->z;
            case 3:
                return this->w;
            default:
                break;
        }
        return this->x;
    }

    __COMMON_GPU_CPU_INLINE__ Float4 &operator=(Float4 vec) {
        x = vec.x;
        y = vec.y;
        z = vec.z;
        w = vec.w;
        return *this;
    }

    __COMMON_GPU_CPU_INLINE__ Float4 &operator+() { return *this; }
    __COMMON_GPU_CPU_INLINE__ Float4 operator-() { return Float4(-this->x, -this->y, -this->z, -this->w); }

    __COMMON_GPU_CPU_INLINE__ Float4 &operator+=(Float4 vec) {
        for (int i = 0; i < 4; ++i) (*this)[i] += vec[i];
        return *this;
    }
    __COMMON_GPU_CPU_INLINE__ Float4 &operator-=(Float4 vec) {
        for (int i = 0; i < 4; ++i) (*this)[i] -= vec[i];
        return *this;
    }
    __COMMON_GPU_CPU_INLINE__ Float4 &operator*=(Float4 vec) {
        for (int i = 0; i < 4; ++i) (*this)[i] *= vec[i];
        return *this;
    }
    __COMMON_GPU_CPU_INLINE__ Float4 &operator/=(Float4 vec) {
        for (int i = 0; i < 4; ++i)
            if (vec[i] != 0.0f) (*this)[i] /= vec[i];
        return *this;
    }
    __COMMON_GPU_CPU_INLINE__ Float4 &operator*=(float n) {
        for (int i = 0; i < 4; ++i) (*this)[i] *= n;
        return *this;
    }
    __COMMON_GPU_CPU_INLINE__ Float4 &operator/=(float n) {
        if (n == 0.0f) return *this;
        for (int i = 0; i < 4; ++i) (*this)[i] /= n;
        return *this;
    }
};

__COMMON_GPU_CPU_INLINE__ Float4 operator+(Float4 vec1, Float4 vec2) {
    return Float4(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z, vec1.w + vec2.w);
}
__COMMON_GPU_CPU_INLINE__ Float4 operator-(Float4 vec1, Float4 vec2) {
    return Float4(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z, vec1.w - vec2.w);
}
__COMMON_GPU_CPU_INLINE__ Float4 operator*(Float4 &vec1, Float4 &vec2) {
    return Float4(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z, vec1.w * vec2.w);
}
__COMMON_GPU_CPU_INLINE__ Float4 operator/(Float4 &vec1, Float4 &vec2) {
    Float4 ret(vec1);
    if (vec2.x != 0.0f) ret.x /= vec2.x;
    if (vec2.y != 0.0f) ret.y /= vec2.y;
    if (vec2.z != 0.0f) ret.z /= vec2.z;
    if (vec2.w != 0.0f) ret.w /= vec2.w;
    return ret;
}

__COMMON_GPU_CPU_INLINE__ Float4 operator*(float f, Float4 v) { return Float4(f * v.x, f * v.y, f * v.z, f * v.w); }

__COMMON_GPU_CPU_INLINE__ Float4 operator*(Float4 v, float f) { return Float4(f * v.x, f * v.y, f * v.z, f * v.w); }

__COMMON_GPU_CPU_INLINE__ Float4 operator/(Float4 &vec1, float f) {
    // if (f == 0.f) return float4(vec1);
    return Float4(vec1.x / f, vec1.y / f, vec1.z / f, vec1.w / f);
}

class float4x4 {
private:
public:
    Float4 x;
    Float4 y;
    Float4 z;
    Float4 w;

    float4x4();
    float4x4(const float &n);
    float4x4(const Float4 &n);
    float4x4(float xx, float xy, float xz, float xw, float yx, float yy, float yz, float yw, float zx, float zy,
             float zz, float zw, float wx, float wy, float wz, float ww);

    float &operator[](int i);
};

namespace poca_mus {

    __COMMON_GPU_CPU_INLINE__ float Length(const Float4 &vec) {
        return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    }

    __COMMON_GPU_CPU_INLINE__ Float4 GetNormalizeVec(Float4 vec) {
        float len = Length(vec);
        return vec / len;
    }

    __COMMON_GPU_CPU_INLINE__ void Normalize(Float4 &vec) {
        float len = Length(vec);
        vec.x /= len;
        vec.y /= len;
        vec.z /= len;
        vec.w /= len;
    }

    __COMMON_GPU_CPU_INLINE__ float Frac(float n) { return MAX(0.0f, MIN(0.9999999f, ABS(n))); }

    __COMMON_GPU_CPU_INLINE__ float Dot(const Float4 &vec1, const Float4 &vec2) {
        return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
    }

    __COMMON_GPU_CPU_INLINE__ float Cosine(Float4 &vec1, Float4 &vec2) {
        return Dot(vec1, vec2) / Length(vec1) / Length(vec2);
    }

    __COMMON_GPU_CPU_INLINE__ Float4 Cross(const Float4 &vec1, const Float4 &vec2) {
        return Float4(vec1.y * vec2.z - vec1.z * vec2.y, -(vec1.x * vec2.z - vec1.z * vec2.x),
                      vec1.x * vec2.y - vec1.y * vec2.x);
    }

    inline float Random() {
        static bool init = false;
        if (!init) {
            srand(static_cast<unsigned>(time(0)));
            init = true;
        }
        return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    inline Float4 CreateRandomFloat4() { return Float4(Random(), Random(), Random(), Random()); }

    __COMMON_GPU_CPU_INLINE__ Float4 ToWorld(Float4 &a, Float4 &N) {
        Float4 B, C;
        if (abs(N.x) > abs(N.y)) {
            float invLen = 1.0f / sqrt(N.x * N.x + N.z * N.z);
            C = Float4(N.z * invLen, 0.0f, -N.x * invLen);
        } else {
            float invLen = 1.0f / sqrt(N.y * N.y + N.z * N.z);
            C = Float4(0.0f, N.z * invLen, -N.y * invLen);
        }
        B = Cross(C, N);
        return a.x * B + a.y * C + a.z * N;
    }

}  // namespace poca_mus

#endif  // RAY_TRACING_MATH_HPP