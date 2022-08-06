#include "video_renderer.h"

#include <string>
#include <Windows.h>

VideoRenderer::VideoRenderer(HWND wnd, int width, int height) {
	ZeroMemory(&bmi_, sizeof(bmi_));
	bmi_.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi_.bmiHeader.biPlanes = 1;
	bmi_.bmiHeader.biBitCount = 32;
	bmi_.bmiHeader.biCompression = BI_RGB;
	bmi_.bmiHeader.biWidth = width;
	bmi_.bmiHeader.biHeight = -height;
	bmi_.bmiHeader.biSizeImage =
		width * height * (bmi_.bmiHeader.biBitCount >> 3);
	image_.reset(new uint8_t[bmi_.bmiHeader.biSizeImage]);
	key_left_pressed_ = 0;
	key_right_pressed_ = 0;
	key_up_pressed_ = 0;
	key_down_pressed_ = 0;
	key_front_pressed_ = 0;
	key_back_pressed_ = 0;

	path_tracer_.reset(new PathTracer);
	path_tracer_->SetCamera(std::shared_ptr<MotionalCamera>(new MotionalCamera(width, height)));
}

VideoRenderer::~VideoRenderer() {

}

void VideoRenderer::lock() {
	mux_.lock();
}
void VideoRenderer::unlock() {
	mux_.unlock();
}

void VideoRenderer::OnRender() {
	int l_r = key_left_pressed_ - key_right_pressed_;
	int f_b = key_front_pressed_ - key_back_pressed_;
	int u_d = key_up_pressed_ - key_down_pressed_;
	if (l_r || f_b || u_d) {
		auto camera = path_tracer_->GetCamera();
		float div = sqrt(float(l_r * l_r + f_b * f_b + u_d * u_d));
		camera->MoveEyeLeft(float(l_r) / div, false);
		camera->MoveEyeForward(float(f_b) / div, false);
		camera->MoveEyeUp(float(u_d) / div, false);
		camera->Refresh();
	}
	dispatch_ray_args_.type = PocaFormatType::ARGB;
	dispatch_ray_args_.buf = image_.get();
	path_tracer_->DispatchRay(dispatch_ray_args_);
}

void VideoRenderer::SetSize(int width, int height) {
	std::lock_guard<VideoRenderer> lock(*this);

	if (width == bmi_.bmiHeader.biWidth && height == bmi_.bmiHeader.biHeight) {
		return;
	}

	bmi_.bmiHeader.biWidth = width;
	bmi_.bmiHeader.biHeight = -height;
	bmi_.bmiHeader.biSizeImage =
		width * height * (bmi_.bmiHeader.biBitCount >> 3);
	image_.reset(new uint8_t[bmi_.bmiHeader.biSizeImage]);

	path_tracer_->GetCamera()->Resize(width, height);
	path_tracer_->Refresh();
}

void VideoRenderer::OnMouseDown(WPARAM btnState, int x, int y) {

}

void VideoRenderer::OnMouseUp(WPARAM btnState, int x, int y) {

}

void VideoRenderer::OnMouseMove(WPARAM btnState, int x, int y) {
	if ((btnState & MK_LBUTTON) != 0)
	{
		// Make each pixel correspond to a quarter of a degree.
		float dx = 0.025f * static_cast<float>(x - last_mouse_pos_.x);
		float dy = 0.025f * static_cast<float>(y - last_mouse_pos_.y);

		// Rotate camera.
		auto camera = path_tracer_->GetCamera();
		camera->RotateAroundUp(dx, false);
		camera->RotateAroundLeft(dy, false);
		camera->Refresh();
	}

	else if ((btnState & MK_RBUTTON) != 0)
	{
		// Make each pixel correspond to 0.5 unit in the scene.
		float dx = 0.5f * static_cast<float>(x - last_mouse_pos_.x);
		float dy = 0.5f * static_cast<float>(y - last_mouse_pos_.y);

		// Zoom in or out.
		auto camera = path_tracer_->GetCamera();
		camera->ScaleFov(dx - dy);
	}

	last_mouse_pos_.x = x;
	last_mouse_pos_.y = y;
}

void VideoRenderer::OnKeyUp(UINT8 key) {
	switch (key)
	{
	case 'W':
		key_front_pressed_ = 0;
		break;
	case 'S':
		key_back_pressed_ = 0;
		break;
	case 'A':
		key_left_pressed_ = 0;
		break;
	case 'D':
		key_right_pressed_ = 0;
		break;
	case 'Q':
		key_up_pressed_ = 0;
		break;
	case 'E':
		key_down_pressed_ = 0;
		break;
	default:
		break;
	}
}

void VideoRenderer::OnKeyDown(UINT8 key) {
	switch (key)
	{
	case 'W':
		key_front_pressed_ = 1;
		break;
	case 'S':
		key_back_pressed_ = 1;
		break;
	case 'A':
		key_left_pressed_ = 1;
		break;
	case 'D':
		key_right_pressed_ = 1;
		break;
	case 'Q':
		key_up_pressed_ = 1;
		break;
	case 'E':
		key_down_pressed_ = 1;
		break;
	case 'R':
		path_tracer_->GetCamera()->Refresh();
	default:
		break;
	}
}