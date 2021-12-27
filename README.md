# CUDAPathTracer

一个由C++与CUDA实现的路径追踪渲染器, 使用opencv进行滤波降噪, 使用ffmpeg进行编码录制。

材质:
```
Diffuse:漫反射
Plastic:不限于塑料,也可是金属,部分漫反射部分镜面反射
Mirror:镜面反射材质
Glass:玻璃,可以折射光线
```
物体:
```
目前只支持计算几何
Sphere:球体
```

需要根据系统环境修改以下代码
```
CMakeLists.txt:
include_directories(/usr/local/cuda-11.5/include)
```

效果:

![image](https://user-images.githubusercontent.com/44687653/147464629-39e3629b-49db-4f6d-a237-2799043a2eae.png)
