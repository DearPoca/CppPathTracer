#include "mp4_recorder.h"

#include <cstring>
#include <memory>
#include <string>

#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "stdio.h"

void Mp4Recoder::set_resolution(int width, int height) {
    width_ = width;
    height_ = height;
}

void Mp4Recoder::set_frame_rate(int frame_rate) { frame_rate_ = frame_rate; }

void Mp4Recoder::set_dst_file(std::string dst_filename) { dst_filename_ = dst_filename; }

Mp4Recoder::Mp4Recoder() { is_initialized_.store(false); }

void Mp4Recoder::InitConversion() {
    dst_frame_ = av_frame_alloc();
    if (!dst_frame_) {
        return;
    }

    dst_frame_->format = output_pix_fmt_;
    dst_frame_->width = width_;
    dst_frame_->height = height_;

    // init sws context
    if (av_image_alloc(dst_frame_->data, dst_frame_->linesize, width_, height_, output_pix_fmt_, 1) < 0) {
        return;
    }

    sws_context_ = sws_getContext(width_, height_, input_pix_fmt_, width_, height_, output_pix_fmt_, SWS_FAST_BILINEAR,
                                  nullptr, nullptr, nullptr);
    if (!sws_context_) {
        return;
    }
}

void Mp4Recoder::InitEncode() {
    AVCodec *encoder;
    AVDictionary *param = 0;
    // av_dict_set(&param, "preset", "superfast", 0);

    dst_fmt_ctx_ = avformat_alloc_context();
    if (!dst_fmt_ctx_) {
        return;
    }
    if (avformat_alloc_output_context2(&dst_fmt_ctx_, NULL, "mp4", dst_filename_.c_str()) < 0) {
        return;
    }

    dst_stream_ = avformat_new_stream(dst_fmt_ctx_, NULL);
    if (!dst_stream_) {
        return;
    }

    encoder = const_cast<AVCodec *>(avcodec_find_encoder(AV_CODEC_ID_H264));
    if (!encoder) {
        return;
    }

    enc_ctx_ = avcodec_alloc_context3(encoder);
    if (!enc_ctx_) {
        return;
    }

    enc_ctx_->height = height_;
    enc_ctx_->width = width_;
    enc_ctx_->pix_fmt = output_pix_fmt_;
    enc_ctx_->time_base = (AVRational){1, frame_rate_};

    if (dst_fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) enc_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(enc_ctx_, encoder, &param) < 0) {
        return;
    }

    if (avcodec_parameters_from_context(dst_stream_->codecpar, enc_ctx_) < 0) {
        return;
    }

    dst_stream_->time_base = (AVRational){1, frame_rate_};

    if (!(dst_fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&dst_fmt_ctx_->pb, dst_filename_.c_str(), AVIO_FLAG_WRITE) < 0) {
            return;
        }
    }

    /* init muxer, write output file header */
    if (avformat_write_header(dst_fmt_ctx_, NULL) < 0) {
        return;
    }

    dst_packet_ = av_packet_alloc();
    if (!dst_packet_) {
        return;
    }
}

void Mp4Recoder::AddFrame(const uint8_t *data, const int size) {
    wait_for_encoder_.Wait();
    if (size < raw_data_frame_size_) return;
    // flat_ring_fifo_->enqueue(data, raw_data_frame_size_, true, true);
    memcpy(raw_data_buff_, data, size < raw_data_frame_size_ ? size : raw_data_frame_size_);
    wait_for_data_.Signal();
}

void Mp4Recoder::Finish() {
    wait_for_encoder_.Wait();
    data_end_.store(true);
    wait_for_data_.Signal();
    wait_for_finish_.Wait();
    is_initialized_.store(false);
}

void Mp4Recoder::Process() {
    int ret, frame_cnt = 0, interval = 125;
    int line_size[1] = {input_fmt_bytes_ * width_};
    int64_t start_encode = av_gettime();
    wait_for_encoder_.Signal();
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    while (true) {
        wait_for_data_.Wait();
        bool data_end = data_end_.load();
        if (data_end == true) break;

        cv::Mat src(height_, width_, CV_8UC3, raw_data_buff_);
        cv::Mat dst_gaussian;
        cv::Mat dst_bilateral;
        cv::Mat dst_filter;

        cv::GaussianBlur(src, dst_gaussian, cv::Size(3, 3), 5, 5);
        cv::bilateralFilter(dst_gaussian, dst_bilateral, 5, 50, 50);
        cv::filter2D(dst_bilateral, dst_filter, CV_8UC3, kernel);

        ret = sws_scale(sws_context_, &dst_filter.data, line_size, 0, height_, dst_frame_->data, dst_frame_->linesize);

        wait_for_encoder_.Signal();

        dst_frame_->pts = frame_cnt * 1000;
        ret = avcodec_send_frame(enc_ctx_, dst_frame_);

        while (ret >= 0) {
            ret = avcodec_receive_packet(enc_ctx_, dst_packet_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_packet_unref(dst_packet_);
                break;
            } else if (ret < 0)
                printf("Error during encoding\n");

            dst_packet_->stream_index = 0;
            av_interleaved_write_frame(dst_fmt_ctx_, dst_packet_);
            av_packet_unref(dst_packet_);
        }

        printf("Mp4Recoder Process, frame[%d] finished\n", frame_cnt);

        frame_cnt++;
    }
    printf("Average speed: %.2f ms/frame\n", (av_gettime() - start_encode) / 1000.0 / frame_cnt);

    ret = avcodec_send_frame(enc_ctx_, nullptr);

    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx_, dst_packet_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_packet_unref(dst_packet_);
            break;
        }

        dst_packet_->stream_index = 0;
        av_interleaved_write_frame(dst_fmt_ctx_, dst_packet_);
        av_packet_unref(dst_packet_);
    }
    av_write_trailer(dst_fmt_ctx_);
    wait_for_finish_.Signal();
    Release();
}

void Mp4Recoder::Release() {
    av_free(raw_data_buff_);
    av_frame_free(&dst_frame_);
    sws_freeContext(sws_context_);
    av_packet_free(&dst_packet_);
    avformat_close_input(&dst_fmt_ctx_);
    avcodec_close(enc_ctx_);
    avcodec_free_context(&enc_ctx_);
}

void Mp4Recoder::Start() {
    data_end_.store(false);
    is_initialized_.store(true);

    raw_data_frame_size_ = input_fmt_bytes_ * width_ * height_;
    raw_data_buff_ = static_cast<uint8_t *>(av_malloc(raw_data_frame_size_));
    if (!raw_data_buff_) {
        return;
    }

    InitConversion();
    InitEncode();

    std::thread work_thread(&Mp4Recoder::Process, this);
    work_thread.detach();
}

bool Mp4Recoder::IsInitialized() {
    return is_initialized_.load() && width_ && height_ && frame_rate_ && dst_filename_.size();
}
