#ifndef RAY_TRACING_MATH_HPP
#define RAY_TRACING_MATH_HPP

#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <utility>

class float4 {
private:
public:
    float x;
    float y;
    float z;
    float w;

    float4() : x(0.f), y(0.f), z(0.f), w(0.f) {}
    float4(float n) : x(n), y(n), z(n), w(n) {}
    float4(float x, float y, float z, float w = 0.0f) {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
    float4(const float4 &vec) : x(vec.x), y(vec.y), z(vec.z), w(vec.w) {}

    inline float &operator[](int i) {
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
                throw "Out of range";
        }
    }

    inline float4 &operator=(float4 vec) {
        x = vec.x;
        y = vec.y;
        z = vec.z;
        w = vec.w;
        return *this;
    }

    inline float4 &operator+() { return *this; }
    inline float4 operator-() { return float4(-this->x, -this->y, -this->z, -this->w); }

    inline float4 &operator+=(float4 vec) {
        for (int i = 0; i < 4; ++i) (*this)[i] += vec[i];
        return *this;
    }
    inline float4 &operator-=(float4 vec) {
        for (int i = 0; i < 4; ++i) (*this)[i] -= vec[i];
        return *this;
    }
    inline float4 &operator*=(float4 vec) {
        for (int i = 0; i < 4; ++i) (*this)[i] *= vec[i];
        return *this;
    }
    inline float4 &operator/=(float4 vec) {
        for (int i = 0; i < 4; ++i)
            if (vec[i] != 0.0f) (*this)[i] /= vec[i];
        return *this;
    }
    inline float4 &operator*=(float n) {
        for (int i = 0; i < 4; ++i) (*this)[i] *= n;
        return *this;
    }
    inline float4 &operator/=(float n) {
        if (n == 0.0f) return *this;
        for (int i = 0; i < 4; ++i) (*this)[i] /= n;
        return *this;
    }
};

inline float4 operator+(float4 vec1, float4 vec2) {
    return float4(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z, vec1.w + vec2.w);
}
inline float4 operator-(float4 vec1, float4 vec2) {
    return float4(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z, vec1.w - vec2.w);
}
inline float4 operator*(float4 &vec1, float4 &vec2) {
    return float4(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z, vec1.w * vec2.w);
}
inline float4 operator/(float4 &vec1, float4 &vec2) {
    float4 ret(vec1);
#define DIVIDE_FLOAT4(p) \
    if (vec2.p != 0.0f) ret.p /= vec2.p
    if (vec2.x != 0.0f) ret.x /= vec2.x;
    if (vec2.y != 0.0f) ret.y /= vec2.y;
    if (vec2.z != 0.0f) ret.z /= vec2.z;
    if (vec2.w != 0.0f) ret.w /= vec2.w;
    return ret;
}

inline float4 operator*(float f, float4 v) { return float4(f * v.x, f * v.y, f * v.z, f * v.w); }

inline float4 operator*(float4 v, float f) { return float4(f * v.x, f * v.y, f * v.z, f * v.w); }

inline float4 operator/(float4 &vec1, float f) {
    // if (f == 0.f) return float4(vec1);
    return float4(vec1.x / f, vec1.y / f, vec1.z / f, vec1.w / f);
}

class float4x4 {
private:
public:
    float4 x;
    float4 y;
    float4 z;
    float4 w;

    float4x4();
    float4x4(const float &n);
    float4x4(const float4 &n);
    float4x4(float xx, float xy, float xz, float xw, float yx, float yy, float yz, float yw, float zx, float zy,
             float zz, float zw, float wx, float wy, float wz, float ww);

    float &operator[](int i);
};

namespace poca_mus {

    inline float Length(const float4 &vec) { return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z); }

    inline float4 GetNormalizeVec(float4 vec) {
        float len = Length(vec);
        return vec / len;
    }

    inline void Normalize(float4 &vec) {
        float len = Length(vec);
        vec.x /= len;
        vec.y /= len;
        vec.z /= len;
        vec.w /= len;
    }

    inline float Frac(float n) { return std::max(0.0f, std::min(0.99999f, std::abs(n))); }

    inline float Dot(const float4 &vec1, const float4 &vec2) {
        return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
    }

    inline float Cosine(float4 &vec1, float4 &vec2) { return Dot(vec1, vec2) / Length(vec1) / Length(vec2); }

    inline float4 Cross(const float4 &vec1, const float4 &vec2) {
        return float4(vec1.y * vec2.z - vec1.z * vec2.y, -(vec1.x * vec2.z - vec1.z * vec2.x),
                      vec1.x * vec2.y - vec1.y * vec2.x);
    }

    inline float Random() {
        uint64_t now =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        now %= 0xfffffff;
        float res = float(now) / 23.14069263277926f;
        return Frac(cos(res));
    }

    inline float4 CreateRandomFloat4() { return float4(Random(), Random(), Random(), Random()); }

    inline float4 ToWorld(float4 &a, float4 &N) {
        float4 B, C;
        if (abs(N.x) > abs(N.y)) {
            float invLen = 1.0f / sqrt(N.x * N.x + N.z * N.z);
            C = float4(N.z * invLen, 0.0f, -N.x * invLen);
        } else {
            float invLen = 1.0f / sqrt(N.y * N.y + N.z * N.z);
            C = float4(0.0f, N.z * invLen, -N.y * invLen);
        }
        B = Cross(C, N);
        return a.x * B + a.y * C + a.z * N;
    }

}  // namespace poca_mus

#endif  // RAY_TRACING_MATH_HPP