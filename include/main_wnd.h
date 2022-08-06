#pragma once

#include <direct.h>
#include <string>
#include <Windows.h>

#include "video_renderer.h"

class MainWnd {
public:
	static const wchar_t class_name_[];

	MainWnd();
	~MainWnd();

	int Create();
	bool Destroy();

	bool IsWindow();
	void MessageBox(const char* caption, const char* text, bool is_error);

	std::shared_ptr<VideoRenderer> GetRenderer() { return renderer_; }

	HWND handle() const { return wnd_; }

	friend LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

protected:
	void OnPaint();
	void OnDestroyed();
	void OnInit();

private:
	std::shared_ptr<VideoRenderer> renderer_;
	HWND wnd_;
};