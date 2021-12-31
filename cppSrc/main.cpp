#include <string>

#include "mp4_recorder.h"
#include "path_tracer.h"

std::string dst_file_path = "./video.mp4";
int width = 1920;
int height = 1080;

float time_to_ori_x(int64_t t) {
    float ft = float(t) / 90.f * M_PI;
    return cos(ft) * 10.f;
}
float time_to_ori_y(int64_t t) {
    float ft = float(t) / 7.f;
    return 3.f + 0.1f * ft * ft;
}
float time_to_ori_z(int64_t t) {
    float ft = float(t) / 90.f * M_PI;
    return sin(ft) * 10.f;
}

float time_to_look_at_x(int64_t t) {
    float ft = float(t) / 90.f * M_PI + M_PI / 4.f;
    return cos(ft) * 100.f;
}
float time_to_look_at_y(int64_t t) {
    float ft = float(t) / 7.f;
    return 5.f + 0.08f * ft * ft;
}
float time_to_look_at_z(int64_t t) {
    float ft = float(t) / 90.f * M_PI + M_PI / 4.f;
    return sin(ft) * 100.f;
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
            int rnd = int(poca_mus::Random() * 2048) % MaterialType::Count;
            switch (rnd) {
                case 1:
                    materials[i]->type_ = MaterialType::Plastic;
                    materials[i]->smoothness_ = poca_mus::Random() * 4 + 1.f;
                    materials[i]->reflectivity_ = poca_mus::Random() * 0.8f;
                    break;
                case 2:
                    materials[i]->type_ = MaterialType::Mirror;
                    materials[i]->Kd_ = 0.5f + 0.5f * poca_mus::CreateRandomFloat4();
                    materials[i]->smoothness_ = poca_mus::Random() * 4 + 0.5f;
                    break;
                case 3:
                    materials[i]->type_ = MaterialType::Glass;
                    materials[i]->smoothness_ = poca_mus::Random() * 4 + 2.f;
                    materials[i]->refractive_index_ = poca_mus::Random() * 2 + 1.2f;
                    materials[i]->Kd_ = 1.f;
                    break;
                default:
                    materials[i]->type_ = MaterialType::Diffuse;
            }
        }
        {
            materials.push_back(new Material);
            materials[20]->Kd_ = Float4(0.8f, 1.f, 1.f);
            materials[20]->type_ = MaterialType::Mirror;
            materials[20]->smoothness_ = 2.5f;
        }
        {
            materials.push_back(new Material);
            materials[21]->Kd_ = 0.2f + 0.8f * poca_mus::CreateRandomFloat4();
            materials[21]->type_ = MaterialType::Plastic;
            materials[21]->smoothness_ = 0.8f;
            materials[21]->reflectivity_ = 0.7f;
        }
        {
            materials.push_back(new Material);
            materials[22]->Kd_ = 1.f;
            materials[22]->type_ = MaterialType::Glass;
            materials[22]->smoothness_ = 3.f;
            materials[22]->refractive_index_ = 1.5f;
        }
        {
            materials.push_back(new Material);
            materials[23]->Kd_ = 0.2f + 0.8f * poca_mus::CreateRandomFloat4();
            materials[23]->type_ = MaterialType::Diffuse;
        }
        {
            materials.push_back(new Material);
            materials[24]->Kd_ = 0.7f + 0.3f * poca_mus::CreateRandomFloat4();
            materials[24]->type_ = MaterialType::Diffuse;
        }

        for (Material *material : materials) path_tracer->AddMeterial(material);

        // 创建物体库
        Object *floor = new Object();
        floor->material_ = materials[0];
        floor->type_ = PrimitiveType::Platform;
        floor->y_pos_ = 0.f;
        floor->center_ = Float4(0, -10000.f, 0);
        floor->radius_ = 10000.f;
        floor->UpdataAABB();
        path_tracer->AddObject(floor);

        for (int i = -550; i < 550; i += 3) {
            Object *obj = new Object();
            int rnd = int(poca_mus::Random() * 2048) % 2;
            if (rnd == 0) {
                obj->type_ = PrimitiveType::Sphere;
                obj->material_ = materials[rand() % 20];
                obj->radius_ = poca_mus::Random() * 5.f + 1.f;
                obj->center_ = Float4(poca_mus::Random() * 300.f - 150.f, 1.f + poca_mus::Random() * 20.f, float(i));
                obj->UpdataAABB();
                path_tracer->AddObject(obj);
                if (obj->material_->type_ == MaterialType::Glass && poca_mus::Random() > 0.5f) {
                    Object *inball = new Object();
                    inball->material_ = obj->material_;
                    inball->center_ = obj->center_;
                    inball->radius_ = 0.01f - obj->radius_;
                    inball->UpdataAABB();
                    path_tracer->AddObject(inball);
                }
            } else {
                obj->type_ = PrimitiveType::Cylinder;
                obj->material_ = materials[int(poca_mus::Random() * 2048) % 5 + 20];
                obj->radius_ = poca_mus::Random() * 5.f + 1.f;
                obj->height_ = obj->radius_ / 2 + poca_mus::Random() * 20.f;
                obj->center_ =
                    Float4(poca_mus::Random() * 300.f - 150.f, obj->height_ + poca_mus::Random() * 20.f, float(i));
                obj->UpdataAABB();
                path_tracer->AddObject(obj);
                if (obj->material_->type_ == MaterialType::Glass && poca_mus::Random() > 0.5f) {
                    Object *inside = new Object();
                    inside->material_ = obj->material_;
                    inside->center_ = obj->center_;
                    inside->radius_ = 0.01f - obj->radius_;
                    inside->height_ = 0.01f - obj->height_;
                    inside->UpdataAABB();
                    path_tracer->AddObject(inside);
                }
            }
        }
    }

    path_tracer->AllocateGpuMemory();
    path_tracer->SetSamplePerPixel(40);

    for (int i = 0; i < 200; ++i) {
        camera->SetOrigin(time_to_ori_x(i), time_to_ori_y(i), time_to_ori_z(i));
        camera->SetLookAt(time_to_look_at_x(i), time_to_look_at_y(i), time_to_look_at_z(i));
        path_tracer->DispatchRay(buf, buf_size, i);
        recoder->AddFrame(buf, buf_size);
    }

    recoder->Finish();
    return 0;
}