#include <string>

#include "mp4_recorder.h"
#include "path_tracer.h"

std::string dst_file_path = "./video.mp4";
int width = 1920;
int height = 1080;

float time_to_ori_x(int64_t t) {
    float ft = float(t) / 10.f;
    return 13.f + 0.3f * (ft - 5) * (ft - 15);
}
float time_to_ori_y(int64_t t) {
    float ft = float(t) / 10.f;
    return 3.f + 0.05 * ft * ft;
}
float time_to_ori_z(int64_t t) {
    float ft = float(t) / 10.f;
    return 13.f + 0.3f * (ft - 5) * (ft - 15);
    ;
}

float time_to_look_at_x(int64_t t) {
    float ft = float(t) / 10.f;
    return 3.f + 0.5 * ft * (ft - 20);
}
float time_to_look_at_y(int64_t t) {
    float ft = float(t) / 10.f;
    return 5.f + 0.02 * ft * ft;
}
float time_to_look_at_z(int64_t t) {
    float ft = float(t) / 10.f;
    return 3.f - 0.5 * ft * (ft - 20);
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
    MotionalCamera *camera = new MotionalCamera(width, height);

    camera->SetViewFov(35);

    path_tracer->ReSize(width, height);
    path_tracer->SetCamera(camera);

    {
        // 创建材质库
        std::vector<Material *> materials = {new Material};
        for (int i = 1; i < 20; ++i) {
            materials.push_back(new Material);
            materials[i]->Kd_ = poca_mus::CreateRandomFloat4();
            int rnd = int(poca_mus::Random() * 4) % 4;
            switch (rnd) {
                case 1:
                    materials[i]->type_ = MaterialType::Plastic;
                    materials[i]->smoothness_ = poca_mus::Random() * 4 + 1.f;
                    break;
                case 2:
                    materials[i]->type_ = MaterialType::Mirror;
                    materials[i]->smoothness_ = poca_mus::Random() * 5;
                    break;
                case 3:
                    materials[i]->type_ = MaterialType::Glass;
                    materials[i]->smoothness_ = poca_mus::Random() * 4 + 1.f;
                    materials[i]->refraction_ = poca_mus::Random() * 2 + 1.f;
                    materials[i]->Kd_ = 1.f;
                    break;
                default:
                    materials[i]->type_ = MaterialType::Diffuse;
            }
        }
        for (Material *material : materials) path_tracer->AddMeterial(material);

        // 创建物体库
        Object *earth = new Object();
        earth->material_ = materials[0];
        path_tracer->AddObject(earth);

        for (int i = -150; i < 150; i += 3) {
            Object *ball = new Object();
            ball->material_ = materials[rand() % 20];
            ball->center_ = Float4(poca_mus::Random() * 300.f - 150.f, 1.f + poca_mus::Random() * 5.f, float(i));
            ball->radius_ = poca_mus::Random() * 5.f + 1.f;
            path_tracer->AddObject(ball);
        }
    }

    path_tracer->AllocateGpuMemory();
    path_tracer->SetSamplePerPixel(10);

    for (int i = 0; i < 200; ++i) {
        camera->SetOrigin(time_to_ori_x(i), time_to_ori_y(i), time_to_ori_z(i));
        camera->SetLookAt(time_to_look_at_x(i), time_to_look_at_y(i), time_to_look_at_z(i));
        path_tracer->DispatchRay(buf, buf_size, i);
        recoder->AddFrame(buf, buf_size);
    }

    recoder->Finish();
    return 0;
}