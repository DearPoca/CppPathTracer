#include "video_renderer.h"

#include <string>
#include <Windows.h>

#include "logger.hpp"

VideoRenderer::VideoRenderer(HWND wnd, int width, int height) {
	wnd_ = wnd;
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
	path_tracer_->SetCamera(std::shared_ptr<MotionalCamera>(
		new MotionalCamera(
			width, height,
			make_float3(13.f, 103.f, 13.f),
			make_float3(0.f, 0.f, 0.f))));
	{
		// 创建材质库
		std::vector<Material*> materials = { new Material };
		materials[0]->type_ = MaterialType::Diffuse;
		materials[0]->kd_ = make_float3(0.95f, 0.95f, 0.95f);
		for (int i = 1; i < 20; ++i) {
			materials.push_back(new Material);
			materials[i]->kd_ = create_random_float3();
			int rnd = int(random() * 2048) % MaterialType::Count;
			switch (rnd) {
			case 1:
				materials[i]->type_ = MaterialType::Metal;
				materials[i]->smoothness_ = random() * 4 + 1.f;
				materials[i]->reflectivity_ = random() * 0.8f;
				break;
			case 2:
				materials[i]->type_ = MaterialType::Mirror;
				materials[i]->kd_ = 0.5f + 0.5f * create_random_float3();
				materials[i]->smoothness_ = random() * 4 + 0.5f;
				break;
			case 3:
				materials[i]->type_ = MaterialType::Glass;
				materials[i]->smoothness_ = random() * 4 + 2.f;
				materials[i]->refractive_index_ = random() * 2 + 1.2f;
				materials[i]->kd_ = make_float3(1.f);
				break;
			default:
				materials[i]->type_ = MaterialType::Diffuse;
			}
		}

		// 创建物体库
		Object* floor = new Object();
		floor->material_ = *materials[0];
		floor->type_ = PrimitiveType::Platform;
		floor->y_pos_ = 0.f;
		floor->center_ = make_float3(0, -10000.f, 0);
		floor->radius_ = 10000.f;
		path_tracer_->AddObject(floor);

		for (int i = -550; i < 550; i += 15) {
			Object* obj = new Object();
			int rnd = int(random() * 2048) % 2;
			if (rnd == 0) {
				obj->type_ = PrimitiveType::Sphere;
				obj->material_ = *materials[rand() % 20];
				obj->radius_ = random() * 15.f + 1.f;
				obj->center_ = make_float3(random() * 300.f - 150.f, obj->radius_, float(i));
				path_tracer_->AddObject(obj);
				if (obj->material_.type_ == MaterialType::Glass && random() > 0.5f) {
					Object* inball = new Object();
					inball->material_ = obj->material_;
					inball->center_ = obj->center_;
					inball->radius_ = 0.01f - obj->radius_;
					path_tracer_->AddObject(inball);
				}
			}
			else {
				obj->type_ = PrimitiveType::Cylinder;
				obj->material_ = *materials[rand() % 20];
				obj->radius_ = random() * 15.f + 1.f;
				obj->height_ = obj->radius_ / 2 + random() * 20.f;
				obj->center_ = make_float3(random() * 300.f - 150.f, obj->height_ / 2, float(i));
				path_tracer_->AddObject(obj);
				if (obj->material_.type_ == MaterialType::Glass && random() > 0.5f) {
					Object* inside = new Object();
					inside->material_ = obj->material_;
					inside->center_ = obj->center_;
					inside->radius_ = 0.01f - obj->radius_;
					inside->height_ = 0.01f - obj->height_;
					path_tracer_->AddObject(inside);
				}
			}
		}
	}
	path_tracer_->InitPipeline();
}

VideoRenderer::~VideoRenderer() {

}

void VideoRenderer::lock() {
	mux_.lock();
}
void VideoRenderer::unlock() {
	mux_.unlock();
}

void VideoRenderer::OnFrame(uint8_t* data, int width, int height) {
	std::lock_guard<VideoRenderer> lock(*this);
	if (width == bmi_.bmiHeader.biWidth && height == bmi_.bmiHeader.biHeight) {
		bmi_.bmiHeader.biWidth = width;
		bmi_.bmiHeader.biHeight = -height;
		bmi_.bmiHeader.biSizeImage =
			width * height * (bmi_.bmiHeader.biBitCount >> 3);
		image_.reset(new uint8_t[bmi_.bmiHeader.biSizeImage]);
	}
	memcpy(image_.get(), data, bmi_.bmiHeader.biSizeImage);
	InvalidateRect(wnd_, nullptr, true);
}

void OnFrameCallback(uint8_t* data, int width, int height, void* cb) {
	VideoRenderer* rder = reinterpret_cast<VideoRenderer*>(cb);
	rder->OnFrame(data, width, height);
}

void VideoRenderer::OnRender() {
	int l_r = key_left_pressed_ - key_right_pressed_;
	int f_b = key_front_pressed_ - key_back_pressed_;
	int u_d = key_up_pressed_ - key_down_pressed_;
	if (l_r || f_b || u_d) {
		auto camera = path_tracer_->GetCamera();
		camera->Lock();
		float div = sqrt(float(l_r * l_r + f_b * f_b + u_d * u_d));
		camera->MoveEyeLeft(float(l_r) / div);
		camera->MoveEyeForward(float(f_b) / div);
		camera->MoveEyeUp(float(u_d) / div);
		camera->Refresh();
		camera->Unlock();
	}
	dispatch_ray_args_.cbParam = this;
	dispatch_ray_args_.Callback = OnFrameCallback;
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

	auto camera = path_tracer_->GetCamera();
	camera->Resize(width, height);
	camera->Refresh();
}

void VideoRenderer::OnMouseDown(WPARAM btnState, int x, int y) {

}

void VideoRenderer::OnMouseUp(WPARAM btnState, int x, int y) {

}

void VideoRenderer::OnMouseMove(WPARAM btnState, int x, int y) {
	if ((btnState & MK_LBUTTON) != 0)
	{
		// Make each pixel correspond to a quarter of a degree.
		tagRECT rect;
		GetWindowRect(wnd_, &rect);
		float dx = static_cast<float>(x - last_mouse_pos_.x) / float(rect.right - rect.left);
		float dy = static_cast<float>(y - last_mouse_pos_.y) / float(rect.bottom - rect.top);

		// Rotate camera.
		auto camera = path_tracer_->GetCamera();
		camera->Lock();
		camera->RotateAroundUp(dy);
		camera->RotateAroundLeft(dx);
		camera->Refresh();
		camera->Unlock();
	}

	else if ((btnState & MK_RBUTTON) != 0)
	{
		// Make each pixel correspond to 0.5 unit in the scene.
		float dx = 0.5f * static_cast<float>(x - last_mouse_pos_.x);
		float dy = 0.5f * static_cast<float>(y - last_mouse_pos_.y);

		// Zoom in or out.
		auto camera = path_tracer_->GetCamera();
		camera->Lock();
		camera->ScaleFov(dx - dy);
		camera->Refresh();
		camera->Unlock();
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