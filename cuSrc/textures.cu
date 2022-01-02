#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "textures.h"

#define DECLARE_TEX_REF(i) texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> texture##i

#define DEFINE_DEFAULT_TEX_REF(i)                              \
    texture##i.addressMode[0] = cudaAddressModeMirror;         \
    texture##i.addressMode[1] = cudaAddressModeMirror;         \
    texture##i.normalized = true;                              \
    texture##i.filterMode = cudaFilterModeLinear;              \
    cudaMallocArray(&cuArray, &cuDesc, img_width, img_height); \
    cudaBindTextureToArray(texture##i, cuArray)

#define SWITCH_TEX_INDEX_DEFINE(i) \
    case (i):                      \
        DEFINE_DEFAULT_TEX_REF(i); \
        break

#define SWITCH_TEX_INDEX_GET(ret, i, u, v) \
    case (i):                              \
        ret = tex2D(texture##i, (u), (v)); \
        break

DECLARE_TEX_REF(0);
DECLARE_TEX_REF(1);
DECLARE_TEX_REF(2);
DECLARE_TEX_REF(3);
DECLARE_TEX_REF(4);
DECLARE_TEX_REF(5);
DECLARE_TEX_REF(6);
DECLARE_TEX_REF(7);

#define SWITCH_FOR_ADD_TEX(i)          \
    switch (i) {                       \
        SWITCH_TEX_INDEX_DEFINE(0);    \
        SWITCH_TEX_INDEX_DEFINE(1);    \
        SWITCH_TEX_INDEX_DEFINE(2);    \
        SWITCH_TEX_INDEX_DEFINE(3);    \
        SWITCH_TEX_INDEX_DEFINE(4);    \
        SWITCH_TEX_INDEX_DEFINE(5);    \
        SWITCH_TEX_INDEX_DEFINE(6);    \
        SWITCH_TEX_INDEX_DEFINE(7);    \
        default:                       \
            DEFINE_DEFAULT_TEX_REF(0); \
    }

#define SWITCH_FOR_GET_TEX(ret, i, u, v)    \
    switch (i) {                            \
        SWITCH_TEX_INDEX_GET(ret, 0, u, v); \
        SWITCH_TEX_INDEX_GET(ret, 1, u, v); \
        SWITCH_TEX_INDEX_GET(ret, 2, u, v); \
        SWITCH_TEX_INDEX_GET(ret, 3, u, v); \
        SWITCH_TEX_INDEX_GET(ret, 4, u, v); \
        SWITCH_TEX_INDEX_GET(ret, 5, u, v); \
        SWITCH_TEX_INDEX_GET(ret, 6, u, v); \
        SWITCH_TEX_INDEX_GET(ret, 7, u, v); \
        default:                            \
            ret = tex2D(texture0, u, v);    \
            break;                          \
    }

int poca_mus::AddTexByFile(std::string file_path) {
    static int textures_size = 0;
    int tex_index = textures_size;
    textures_size++;

    cv::Mat mat = cv::imread(file_path);
    cv::cvtColor(mat, mat, CV_BGR2RGBA);

    cudaArray* cuArray;
    cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar4>();

    int img_width = mat.cols;
    int img_height = mat.rows;
    int channels = mat.channels();

    SWITCH_FOR_ADD_TEX(tex_index);

    cudaMemcpyToArray(cuArray, 0, 0, mat.data, img_width * img_height * sizeof(uchar) * channels,
                      cudaMemcpyHostToDevice);

    return tex_index;
}

__device__ Float4 poca_mus::tex2D(int tex_index, float u, float v) {
    float4 rgba;
    SWITCH_FOR_GET_TEX(rgba, tex_index, u, v);
    return Float4(rgba.x, rgba.y, rgba.z, rgba.w);
}
