#include <Windows.h>

#include "main_wnd.h"
#include "logger.hpp"

Logger logger;

int main() {
	MainWnd wnd;
	return wnd.Create();
}
