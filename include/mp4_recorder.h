#ifndef MINI_PATH_TRACER_MP4_RECORDER_H
#define MINI_PATH_TRACER_MP4_RECORDER_H

#include <atomic>
#include <fstream>
#include <thread>
#include "semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>
#ifdef __cplusplus
};
#endif

class Mp4Recoder
{
  public:
	Mp4Recoder();

	//设定分辨率
	void set_resolution(int width, int height);

	//设定帧率
	void set_frame_rate(int frame_rate);

	//重新设定输出文件
	void set_dst_file(std::string dst_filename);

	//以异步方式启动编码器
	void Start();
	void AddFrame(const uint8_t *data, const int size);
	void Finish();

	bool IsInitialized();

  private:
	AVPixelFormat output_pix_fmt_ = AV_PIX_FMT_YUV420P;
	AVPixelFormat input_pix_fmt_ = AV_PIX_FMT_RGB24;
	int input_fmt_bytes_ = 3;

	std::atomic_bool data_end_;
	std::atomic_bool is_initialized_;

	Semaphore wait_for_data_;
	Semaphore wait_for_encoder_;
	Semaphore wait_for_finish_;

	std::string dst_filename_;

	int width_ = 0, height_ = 0;
	int frame_rate_ = 0;

	uint8_t *raw_data_buff_;
	size_t raw_data_frame_size_;

	AVFrame *dst_frame_;
	struct SwsContext *sws_context_;

	AVPacket *dst_packet_;
	AVFormatContext *dst_fmt_ctx_;
	AVCodecContext *enc_ctx_;
	AVStream *dst_stream_;

	void InitConversion();
	void InitEncode();
	void Process();
	void Release();
};

#endif