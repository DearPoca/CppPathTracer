#include <string>

#include "mp4_recorder.h"
#include "path_tracer.h"

std::string dst_file_path = "./video.mp4";
int width = 1280;
int height = 720;

float time_to_ori_x(int64_t t) {
    float ft = float(t) / 10.f;
    return -ft * ft + ft;
}
float time_to_ori_y(int64_t t) {
    float ft = float(t) / 10.f;
    return 33.f + ft * ft + ft;
}
float time_to_ori_z(int64_t t) {
    float ft = float(t) / 10.f;
    return 13.f - ft * ft + 6.f * ft;
}

float time_to_look_at_x(int64_t t) {
    float ft = float(t) / 10.f;
    return 0.f;
}
float time_to_look_at_y(int64_t t) {
    float ft = float(t) / 10.f;
    return 0.f;
}
float time_to_look_at_z(int64_t t) {
    float ft = float(t) / 10.f;
    return 0.f;
}

int main(int argc, char **argv) {
    Mp4Recoder *recoder = new Mp4Recoder;
    recoder->set_dst_file(dst_file_path);
    recoder->set_frame_rate(25);
    recoder->set_resolution(width, height);
    recoder->Start();

    int buf_size = width * height * 3;
    uint8_t *buf = new uint8_t[buf_size];
    PathTracer *path_tracer = new PathTracer;
    Scene *scene = new Scene;
    MotionalCamera *camera = new MotionalCamera(width, height);

    camera->SetViewFov(35);

    path_tracer->ReSize(width, height);
    path_tracer->SetScene(scene);
    path_tracer->SetCamera(camera);

    {
        std::vector<Material *> materials = {new Material};
        for (int i = 1; i < 10; ++i) {
            materials.push_back(new Material);
            materials[i]->Kd_ = poca_mus::CreateRandomFloat4();
        }

        Object *earth = new Object();
        earth->material_ = materials[0];
        scene->AddObject(earth);

        for (int i = -50; i < 50; i += 5) {
            Object *ball = new Object();
            ball->material_ = materials[rand() % 10];
            ball->center_ = float4(poca_mus::Random() * 100.f - 50.f, 1.f + poca_mus::Random() * 2.f, i);
            ball->radius_ = poca_mus::Random() * 2.f + 1.f;
            scene->AddObject(ball);
        }
    }

    path_tracer->time_to_look_at_x = time_to_look_at_x;
    path_tracer->time_to_look_at_y = time_to_look_at_y;
    path_tracer->time_to_look_at_z = time_to_look_at_z;
    path_tracer->time_to_ori_x = time_to_ori_x;
    path_tracer->time_to_ori_y = time_to_ori_y;
    path_tracer->time_to_ori_z = time_to_ori_z;

    for (int i = 0; i < 200; ++i) {
        path_tracer->DispatchRay(buf, buf_size, i);
        recoder->AddFrame(buf, buf_size);
    }

    recoder->Finish();
    return 0;
}