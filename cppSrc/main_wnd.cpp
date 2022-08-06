#include "main_wnd.h"
#include <WindowsX.h>

HFONT GetDefaultFont() {
	static HFONT font = reinterpret_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
	return font;
}

std::string GetWindowText(HWND wnd) {
	char text[MAX_PATH] = { 0 };
	::GetWindowTextA(wnd, &text[0], ARRAYSIZE(text));
	return text;
}

void AddListBoxItem(HWND listbox, const std::string& str, LPARAM item_data) {
	LRESULT index = ::SendMessageA(listbox, LB_ADDSTRING, 0,
		reinterpret_cast<LPARAM>(str.c_str()));
	::SendMessageA(listbox, LB_SETITEMDATA, index, item_data);
}

MainWnd::MainWnd() :wnd_(nullptr) {

}

MainWnd::~MainWnd() {

}

int MainWnd::Create() {
	// Initialize the window class.
	WNDCLASSEX windowClass = { 0 };
	windowClass.cbSize = sizeof(WNDCLASSEX);
	windowClass.style = CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc = WindowProc;
	windowClass.hInstance = GetModuleHandle(nullptr);
	windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	windowClass.lpszClassName = "MainWnd";
	if (!RegisterClassEx(&windowClass)) {
		exit(-1);
	}

	RECT windowRect = { 0, 0, 1280, 720 };
	AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

	wnd_ = CreateWindow(
		windowClass.lpszClassName,
		"CUDAPathTracer",
		WS_OVERLAPPEDWINDOW | WS_VISIBLE | WS_CLIPCHILDREN,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		windowRect.right - windowRect.left,
		windowRect.bottom - windowRect.top,
		nullptr,		// We have no parent window.
		nullptr,		// We aren't using menus. 
		GetModuleHandle(NULL),
		this);

	OnInit();

	// Main sample loop.
	MSG msg = {};
	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

	}

	// Return this part of the WM_QUIT message to Windows.
	return static_cast<int>(msg.wParam);
}

LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	MainWnd* main_wnd = reinterpret_cast<MainWnd*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

	switch (message)
	{
	case WM_CREATE:
	{
		LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
		SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
		return 0;
	}

	case WM_SIZE:
		if (wParam != SIZE_MINIMIZED)
		{

		}
		return 0;

	case WM_LBUTTONDOWN:

	case WM_MBUTTONDOWN:

	case WM_RBUTTONDOWN:
		main_wnd->GetRenderer()->OnMouseDown(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		return 0;

	case WM_LBUTTONUP:
		main_wnd->GetRenderer()->OnMouseUp(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		return 0;
	case WM_MBUTTONUP:
	case WM_RBUTTONUP:
		main_wnd->GetRenderer()->OnMouseUp(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		return 0;
	case WM_MOUSEMOVE:
		main_wnd->GetRenderer()->OnMouseMove(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
		return 0;
	case WM_KEYDOWN:
		main_wnd->GetRenderer()->OnKeyDown(static_cast<UINT8>(wParam));
		if (static_cast<UINT8>(wParam) == VK_ESCAPE)
			PostQuitMessage(0);
		return 0;
	case WM_KEYUP:
		main_wnd->GetRenderer()->OnKeyUp(static_cast<UINT8>(wParam));
		return 0;
	case WM_PAINT:
		main_wnd->GetRenderer()->OnRender();
		main_wnd->OnPaint();
		return 0;

	case WM_DESTROY:
		main_wnd->OnDestroyed();
		return 0;
	}

	// Handle any messages the switch statement didn't.
	return DefWindowProc(hWnd, message, wParam, lParam);
}

bool MainWnd::Destroy() {
	BOOL ret = FALSE;
	if (IsWindow()) {
		ret = ::DestroyWindow(wnd_);
	}

	return ret != FALSE;
}

bool MainWnd::IsWindow() {
	return wnd_ && ::IsWindow(wnd_) != FALSE;
}

void MainWnd::MessageBox(const char* caption, const char* text, bool is_error) {
	DWORD flags = MB_OK;
	if (is_error)
		flags |= MB_ICONERROR;

	::MessageBoxA(handle(), text, caption, flags);
}

void MainWnd::OnInit() {
	renderer_.reset(new VideoRenderer(wnd_));
}

void MainWnd::OnPaint() {
	PAINTSTRUCT ps;
	::BeginPaint(handle(), &ps);

	RECT rc;
	::GetClientRect(handle(), &rc);

	VideoRenderer* renderer = renderer_.get();
	if (renderer) {
		std::lock_guard<VideoRenderer> lock(*renderer);

		renderer->OnRender();
		const BITMAPINFO& bmi = renderer->bmi();
		int height = abs(bmi.bmiHeader.biHeight);
		int width = bmi.bmiHeader.biWidth;

		const uint8_t* image = renderer->image();
		if (image != NULL) {
			HDC dc_mem = ::CreateCompatibleDC(ps.hdc);
			::SetStretchBltMode(dc_mem, HALFTONE);

			HDC all_dc[] = { ps.hdc, dc_mem };
			for (size_t i = 0; i < 2; ++i) {
				SetMapMode(all_dc[i], MM_ISOTROPIC);
				SetWindowExtEx(all_dc[i], width, height, NULL);
				SetViewportExtEx(all_dc[i], rc.right, rc.bottom, NULL);
			}

			HBITMAP bmp_mem = ::CreateCompatibleBitmap(ps.hdc, rc.right, rc.bottom);
			HGDIOBJ bmp_old = ::SelectObject(dc_mem, bmp_mem);

			POINT logical_area = { rc.right, rc.bottom };
			DPtoLP(ps.hdc, &logical_area, 1);

			HBRUSH brush = ::CreateSolidBrush(RGB(0, 0, 0));
			RECT logical_rect = { 0, 0, logical_area.x, logical_area.y };
			::FillRect(dc_mem, &logical_rect, brush);
			::DeleteObject(brush);

			int x = (logical_area.x / 2) - (width / 2);
			int y = (logical_area.y / 2) - (height / 2);

			StretchDIBits(dc_mem, x, y, width, height, 0, 0, width, height, image,
				&bmi, DIB_RGB_COLORS, SRCCOPY);

			BitBlt(ps.hdc, 0, 0, logical_area.x, logical_area.y, dc_mem, 0, 0,
				SRCCOPY);

			// Cleanup.
			::SelectObject(dc_mem, bmp_old);
			::DeleteObject(bmp_mem);
			::DeleteDC(dc_mem);
		}
	}
	::EndPaint(handle(), &ps);
	InvalidateRect(wnd_, nullptr, false);
}

void MainWnd::OnDestroyed() {
	PostQuitMessage(0);
}

