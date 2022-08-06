#include "textures.h"

#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>


cudaTextureObject_t AddTexByFile(std::string file_path, cudaTextureAddressMode addr_mode = cudaAddressModeMirror, cudaTextureFilterMode filter_mode = cudaFilterModeLinear) {
	cv::Mat src = cv::imread(file_path);
	cv::Mat dst;
	cv::cvtColor(src, dst, CV_BGR2RGBA);

	int width = dst.cols;
	int height = dst.rows;

	cudaArray* cu_array;
	cudaChannelFormatDesc cu_desc = cudaCreateChannelDesc<uchar4>();
	checkCudaErrors(cudaMallocArray(&cu_array, &cu_desc, width, height));
	checkCudaErrors(cudaMemcpy2DToArray(cu_array, 0, 0, dst.data, width * 4, width * sizeof(uint8_t), height,
		cudaMemcpyHostToDevice));

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = cu_array;

	cudaTextureDesc tex_desc;
	tex_desc.addressMode[0] = addr_mode;
	tex_desc.addressMode[1] = addr_mode;
	tex_desc.filterMode = filter_mode;
	tex_desc.readMode = cudaReadModeNormalizedFloat;
	tex_desc.normalizedCoords = 1;

	cudaTextureObject_t tex_object;

	checkCudaErrors(cudaCreateTextureObject(&tex_object, &res_desc, &tex_desc, NULL));
	return tex_object;
}

__device__ float4 GetTexture2D(cudaTextureObject_t tex_obj, float u, float v) {
	float4 tex = tex2D<float4>(tex_obj, u, v);
	return tex;
}