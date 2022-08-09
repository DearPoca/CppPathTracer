#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <mutex>

#include "timer.hpp"

enum {
	Error,
	Warning,
	Info,
	Debug,
	Trace
};

#define MAX_LOGGER_BUF 512
#define log_error(fmt, ...) logger.Log(Error, __FILE__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__) 
#define log_warning(fmt, ...) logger.Log(Warning, __FILE__, __LINE__, __FUNCTION__  , fmt, ##__VA_ARGS__) 
#define log_info(fmt, ...) logger.Log(Info, __FILE__, __LINE__, __FUNCTION__  , fmt, ##__VA_ARGS__) 
#define log_debug(fmt, ...) logger.Log(Debug, __FILE__, __LINE__, __FUNCTION__  , fmt, ##__VA_ARGS__) 
#define log_trace(fmt, ...) logger.Log(Trace, __FILE__, __LINE__, __FUNCTION__  , fmt, ##__VA_ARGS__) 

class Logger {

public:
	inline Logger(std::string _path = "./logs/CUDAPathTracer.log") :path(_path) {
		memset(&timer, 0, sizeof(Timer));
		ofs.open(_path, std::ios::out | std::ios::app);
		assert(ofs.is_open());
	};

	~Logger() {
		ofs.close();
	}

	void Log(const int& Level, char const* const file, long line, char const* const func, char const* const fmt, ...) {
		char logger_buffer[MAX_LOGGER_BUF];
		char* log_level;
		switch (Level) {
		case Error:
			log_level = "Error";
			break;
		case Warning:
			log_level = "Warning";
			break;
		case Info:
			log_level = "Info";
			break;
		case Debug:
			log_level = "Debug";
			break;
		case Trace:
			log_level = "Trace";
			break;
		default:
			log_level = "Info";
			break;
		}
		va_list args;
		va_start(args, fmt);
		sprintf(logger_buffer, "[%s]", timer.getTimeStr().c_str());
		sprintf(logger_buffer + strlen(logger_buffer), "[%s]", log_level);
		sprintf(logger_buffer + strlen(logger_buffer), "[%s:%d]", file, line);
		vsprintf(logger_buffer + strlen(logger_buffer), fmt, args);
		va_end(args);

		ofs << std::string(logger_buffer) << std::endl;
	}

private:
	std::ofstream ofs;
	std::string path;
	Timer timer;
};

extern Logger logger;