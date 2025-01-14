#pragma once

#include <string>
#include <mutex>
#include <Windows.h>

#include "path_tracer.h"

class VideoRenderer {
public:
	VideoRenderer(HWND wnd, int width = 1280, int height = 720);
	~VideoRenderer();

	void lock();
	void unlock();

	void OnRender();
	void Create();

	void OnMouseDown(WPARAM btnState, int x, int y);
	void OnMouseUp(WPARAM btnState, int x, int y);
	void OnMouseMove(WPARAM btnState, int x, int y);
	void OnKeyUp(UINT8 key);
	void OnKeyDown(UINT8 key);
	void SetSize(int width, int height);
	void OnFrame(uint8_t* data, int width, int height);

	const BITMAPINFO& bmi() const { return bmi_; }
	const uint8_t* image() const { return image_.get(); }

private:
	HWND wnd_;
	BITMAPINFO bmi_;
	std::unique_ptr<uint8_t> image_;
	std::mutex mux_;

	int width_ = 1280;
	int height_ = 720;
	std::shared_ptr<PathTracer> path_tracer_;
	DispatchRayArgs dispatch_ray_args_;
	void ResetOutBufSize(int width, int height);

	POINT last_mouse_pos_;
	int key_left_pressed_;
	int key_right_pressed_;
	int key_up_pressed_;
	int key_down_pressed_;
	int key_front_pressed_;
	int key_back_pressed_;
};