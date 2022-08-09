#pragma once
//判断操作系统
#ifdef _WIN32
#pragma warning(disable:4996)
#include <windows.h>
#elif __linux__
#include <unistd.h>
#endif
//时间和字符串头文件
#include <time.h>
#include <string>

typedef unsigned int uint;

class Timer {

public:

	struct {
		uint year;
		uint mon;
		uint day;
		uint hour;
		uint min;
		uint sec;
	};
	//程序等待(单位:毫秒)
	inline void wait(uint ms) {
#ifdef _WIN32
		Sleep(ms);
#elif __linux__
		usleep(ms * 1000);
#endif
	}
	//程序开始的计时
	inline void start() {
		clocks = clock();
	};
	//程序结束的计时 返回消耗时间
	inline uint end() {
		return clock() - clocks;
	};
	//获取系统当前时间
	inline void getTime() {
		time(&rawtime);
		ptminfo = localtime(&rawtime);

		year = ptminfo->tm_year + 1900;
		mon = ptminfo->tm_mon + 1;
		day = ptminfo->tm_mday;
		hour = ptminfo->tm_hour;
		min = ptminfo->tm_min;
		sec = ptminfo->tm_sec;
	}
	//获取当前时间 返回字符串类型
	inline std::string getTimeStr() {
		getTime();
		return std::to_string(year) + "-" + std::to_string(mon) + "-" + std::to_string(day) + " "
			+ std::to_string(hour) + ":" + std::to_string(min) + ":" + std::to_string(sec);
	}

	inline static long long GetMillisecondsTimeStamp() {
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	}

private:
	//时间相关结构体
	clock_t clocks;
	time_t rawtime;
	tm* ptminfo;
};