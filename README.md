# CUDAPathTracer

一个由C++与CUDA实现的路径追踪渲染器, 使用opencv进行滤波降噪, 使用ffmpeg进行编码录制。

材质:
```
Diffuse: 漫反射
Metal: 金属,同时拥有漫反射和镜面反射性质
Mirror:镜面反射材质
Glass:玻璃,可以折射光线
```
物体:
```
目前只支持计算几何
Sphere: 球体
Platform: 垂直于y轴的平面
Cylinder: 圆柱体
```

需要根据系统环境修改以下代码
```
CMakeLists.txt:
include_directories(/usr/local/cuda-11.5/include)
```

效果:

![image](https://user-images.githubusercontent.com/44687653/147867714-b9b0bf8a-09ea-4dbd-92e2-9544ebd8528b.png)

