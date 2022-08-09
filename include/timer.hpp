#pragma once
//�жϲ���ϵͳ
#ifdef _WIN32
#pragma warning(disable:4996)
#include <windows.h>
#elif __linux__
#include <unistd.h>
#endif
//ʱ����ַ���ͷ�ļ�
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
	//����ȴ�(��λ:����)
	inline void wait(uint ms) {
#ifdef _WIN32
		Sleep(ms);
#elif __linux__
		usleep(ms * 1000);
#endif
	}
	//����ʼ�ļ�ʱ
	inline void start() {
		clocks = clock();
	};
	//��������ļ�ʱ ��������ʱ��
	inline uint end() {
		return clock() - clocks;
	};
	//��ȡϵͳ��ǰʱ��
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
	//��ȡ��ǰʱ�� �����ַ�������
	inline std::string getTimeStr() {
		getTime();
		return std::to_string(year) + "-" + std::to_string(mon) + "-" + std::to_string(day) + " "
			+ std::to_string(hour) + ":" + std::to_string(min) + ":" + std::to_string(sec);
	}

	inline static long long GetMillisecondsTimeStamp() {
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	}

private:
	//ʱ����ؽṹ��
	clock_t clocks;
	time_t rawtime;
	tm* ptminfo;
};