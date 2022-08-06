#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

class Semaphore
{
public:
	inline explicit Semaphore(int count = 0):count_(count) {
		
	}

	inline void Signal() {
		std::unique_lock<std::mutex> lock(mutex_);
		++count_;
		cv_.notify_one();
	}

	inline void Wait() {
		std::unique_lock<std::mutex> lock(mutex_);
		cv_.wait(lock, [=]
			{ return count_ > 0; });
		--count_;
	}

private:
	std::mutex mutex_;
	std::condition_variable cv_;
	int count_;
};
