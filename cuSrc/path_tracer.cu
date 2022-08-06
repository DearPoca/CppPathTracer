#include "path_tracer.h"

#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include "ray_tracing_math.hpp"
#include "ray_tracing_common.h"

void PathTracer::AddObject(Object* obj) {

}

void PathTracer::Init() {

}

void PathTracer::Refresh() {

}

void PathTracer::DispatchRay(DispatchRayArgs args) {
	cv::Mat src = cv::imread("textures\\sky.png");
	cv::Mat dst;
	cv::cvtColor(src, dst, CV_BGR2RGBA);
	cv::Mat dst2;
	cv::resize(dst, dst2, cv::Size(1280, 720));
	memcpy(args.buf, dst2.data, 1280 * 720 * 4);
}

void PathTracer::SetCamera(std::shared_ptr<MotionalCamera>& camera) {
	camera_ = camera;
}

std::shared_ptr<MotionalCamera> PathTracer::GetCamera() {
	return camera_;
}