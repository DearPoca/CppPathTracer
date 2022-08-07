#include "textures.h"

#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>

#include "logger.hpp"

cudaTextureObject_t AddTexByFile(std::string file_path, cudaTextureAddressMode addr_mode, cudaTextureFilterMode filter_mode) {
	cv::Mat src = cv::imread(file_path, CV_LOAD_IMAGE_GRAYSCALE);

	int width = src.cols;
	int height = src.rows;

	cudaError_t err;

	float* h_data = (float*)std::malloc(sizeof(float) * width * height);
	for (int i = 0; i < height * width; ++i)
		h_data[i] = i;

	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray_t cuArray;
	err = cudaMallocArray(&cuArray, &channelDesc, width, height);
	if (err != cudaSuccess)
	{
		log_error("Error occur with AddTexByFile: %s", cudaGetErrorString(err));
	}

	// Set pitch of the source (the width in memory in bytes of the 2D array pointed
	// to by src, including padding), we dont have any padding
	const size_t spitch = width * sizeof(float);
	// Copy data located at address h_data in host memory to device memory
	err = cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(float),
		height, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		log_error("Error occur with AddTexByFile: %s", cudaGetErrorString(err));
	}

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = addr_mode;
	texDesc.addressMode[1] = addr_mode;
	texDesc.filterMode = filter_mode;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	if (err != cudaSuccess)
	{
		log_error("Error occur with AddTexByFile: %s", cudaGetErrorString(err));
	}

	return texObj;
}

__device__ float4 GetTexture2D(cudaTextureObject_t tex_obj, float u, float v) {
	float4 tex = tex2D<float4>(tex_obj, u, v);
	return tex;
}