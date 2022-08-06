#pragma once

#include "ray_tracing_math.hpp"
#include "ray_tracing_common.h"

class MotionalCamera {
private:
	const float3 vup = make_float3(0.0f, 1.0f, 0.0f);

	int width_;
	int height_;
	uint cur_sample_idx_;

	float3 origin_;
	float3 look_at_;
	float view_fov_ = 20;        //ÊÓ½Ç
	float dist_to_focus_ = 10;   //½¹¾à
	float lens_radius_ = 0.05f;  //¿×°ë¾¶
	float move_speed_ = 0.005f;

	float3 u_, v_, w_;
	float3 top_left_corner_;
	float3 horizontal_;
	float3 vertical_;

public:
	MotionalCamera();
	MotionalCamera(int width, int height);
	MotionalCamera(int width, int height, float3 ori, float3 at);

	~MotionalCamera();

	void Resize(int width, int height);
	void SetOrigin(float3 ori);
	void SetOrigin(float x, float y, float z);

	void SetLookAt(float3 lookAt);
	void SetLookAt(float x, float y, float z);

	void SetViewFov(float fov);

	void MoveEyeLeft(float coefficient = 1.f, bool refresh = true);
	void MoveEyeRight(float coefficient = 1.f, bool refresh = true);
	void MoveEyeForward(float coefficient = 1.f, bool refresh = true);
	void MoveEyeBackward(float coefficient = 1.f, bool refresh = true);
	void MoveEyeUp(float coefficient = 1.f, bool refresh = true);
	void MoveEyeDown(float coefficient = 1.f, bool refresh = true);

	void RotateAroundUp(float dy, bool refresh = true);
	void RotateAroundDown(float dy, bool refresh = true);
	void RotateAroundLeft(float dx, bool refresh = true);
	void RotateAroundRight(float dx, bool refresh = true);
	void ScaleFov(float d, bool refresh = true);

	void Updata();
	void Refresh();

	__device__ Ray RayGen(int x, int y, curandState& state);
};
