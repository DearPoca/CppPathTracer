
#include "motional_camera.h"

#include <cuda_runtime.h>
#include <helper_functions.h>  
#include <helper_cuda.h>

#include "ray_tracing_math.hpp"
#include "ray_tracing_common.h"

MotionalCamera::MotionalCamera() :
	width_(1920),
	height_(1080),
	origin_(make_float3(0.f)),
	look_at_(make_float3(0.f, 0.f, 1.f)),
	cur_sample_idx_(0) {

}

MotionalCamera::MotionalCamera(int width, int height) :
	width_(width),
	height_(height),
	origin_(make_float3(0.f)),
	look_at_(make_float3(0.f, 0.f, 1.f)),
	cur_sample_idx_(0) {

}

MotionalCamera::MotionalCamera(int width, int height, float3 ori, float3 at) :
	width_(width),
	height_(height),
	origin_(ori),
	look_at_(at),
	cur_sample_idx_(0) {

}

MotionalCamera::~MotionalCamera() {}

void MotionalCamera::Refresh() {
	cur_sample_idx_ = 0;
}

void MotionalCamera::SetViewFov(float fov) {
	view_fov_ = fov;
	Refresh();
}

void MotionalCamera::Resize(int width, int height) {
	width_ = width;
	height_ = height;
	Refresh();
}
void MotionalCamera::SetOrigin(float3 ori) {
	origin_ = ori;
	Refresh();
}
void MotionalCamera::SetOrigin(float x, float y, float z) {
	origin_.x = x;
	origin_.y = y;
	origin_.z = z;
	Refresh();
}

void MotionalCamera::SetLookAt(float3 look_at) {
	look_at_ = look_at;
	Refresh();
}

void MotionalCamera::SetLookAt(float x, float y, float z) {
	look_at_.x = x;
	look_at_.y = y;
	look_at_.z = z;
	Refresh();
}

void MotionalCamera::MoveEyeLeft(float coefficient, bool refresh) {
	float3 w = normalize(origin_ - look_at_);
	float3 left = normalize(cross(vup, w));

	origin_ += coefficient * move_speed_ * left;
	look_at_ += coefficient * move_speed_ * left;

	if (refresh)
		Refresh();
}

void MotionalCamera::MoveEyeRight(float coefficient, bool refresh) {
	float3 w = normalize(origin_ - look_at_);
	float3 left = normalize(cross(vup, w));

	origin_ -= coefficient * move_speed_ * left;
	look_at_ -= coefficient * move_speed_ * left;

	if (refresh)
		Refresh();
}

void MotionalCamera::MoveEyeForward(float coefficient, bool refresh) {
	float3 w = normalize(origin_ - look_at_);
	float3 left = normalize(cross(vup, w));
	float3 back = normalize(cross(left, vup));

	origin_ -= coefficient * move_speed_ * back;
	look_at_ -= coefficient * move_speed_ * back;

	if (refresh)
		Refresh();
}

void MotionalCamera::MoveEyeBackward(float coefficient, bool refresh) {
	float3 w = normalize(origin_ - look_at_);
	float3 left = normalize(cross(vup, w));
	float3 back = normalize(cross(left, vup));

	origin_ += coefficient * move_speed_ * back;
	look_at_ += coefficient * move_speed_ * back;

	if (refresh)
		Refresh();
}

void MotionalCamera::MoveEyeUp(float coefficient, bool refresh) {
	origin_ += coefficient * move_speed_ * vup;
	look_at_ += coefficient * move_speed_ * vup;

	if (refresh)
		Refresh();
}

void MotionalCamera::MoveEyeDown(float coefficient, bool refresh) {
	origin_ -= coefficient * move_speed_ * vup;
	look_at_ -= coefficient * move_speed_ * vup;

	if (refresh)
		Refresh();
}

void MotionalCamera::RotateAroundUp(float dy, bool refresh) {
	look_at_ = origin_ + normalize(look_at_ - origin_);
	look_at_ += dy * vup;
	look_at_ = origin_ + normalize(look_at_ - origin_);

	if (refresh)
		Refresh();
}

void MotionalCamera::RotateAroundDown(float dy, bool refresh) {
	look_at_ = origin_ + normalize(look_at_ - origin_);
	look_at_ -= dy * vup;
	look_at_ = origin_ + normalize(look_at_ - origin_);

	if (refresh)
		Refresh();
}

void MotionalCamera::RotateAroundLeft(float dx, bool refresh) {
	look_at_ = origin_ + normalize(look_at_ - origin_);

	float3 w = normalize(origin_ - look_at_);
	float3 left = normalize(cross(vup, w));

	look_at_ += dx * left;
	look_at_ = origin_ + normalize(look_at_ - origin_);

	if (refresh)
		Refresh();
}

void MotionalCamera::RotateAroundRight(float dx, bool refresh) {
	look_at_ = origin_ + normalize(look_at_ - origin_);

	float3 w = normalize(origin_ - look_at_);
	float3 left = normalize(cross(vup, w));
	float3 back = normalize(cross(left, vup));

	look_at_ -= dx * left;
	look_at_ = origin_ + normalize(look_at_ - origin_);

	if (refresh)
		Refresh();
}

void MotionalCamera::ScaleFov(float d, bool refresh) {
	view_fov_ += d * M_PI / 180.0f;

	if (refresh)
		Refresh();
}

void MotionalCamera::Updata() {
	float theta = view_fov_ * M_PI / 180;
	float aspectRatio = float(width_) / float(height_);
	float half_height = tan(theta / 2);
	float half_width = aspectRatio * half_height;

	w_ = normalize(origin_ - look_at_);
	u_ = normalize(cross(vup, w_));
	v_ = cross(w_, u_);

	dist_to_focus_ = length(origin_ - look_at_);

	top_left_corner_ =
		origin_ - half_width * dist_to_focus_ * u_ + half_height * dist_to_focus_ * v_ - dist_to_focus_ * w_;
	horizontal_ = 2 * half_width * dist_to_focus_ * u_;
	vertical_ = -2 * half_height * dist_to_focus_ * v_;

	cur_sample_idx_++;
}

__device__ Ray MotionalCamera::RayGen(int x, int y, curandState& state) {
	Ray ray;
	float3 rd = lens_radius_ * device_create_random_float3(state);
	float3 offset = u_ * rd.x + v_ * rd.y;
	float dx = float(x) / float(width_);
	float dy = float(y) / float(height_);
	ray.origin = origin_ + offset;
	ray.dir = top_left_corner_ + dx * horizontal_ + dy * vertical_ - origin_ - offset;
	ray.tmin = 0.f;
	ray.tmax = DEFAULT_RAY_TMAX;
	return ray;
}
