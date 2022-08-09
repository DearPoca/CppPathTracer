# CUDAPathTracer

一个由C++与CUDA实现的路径追踪渲染器, 使用opencv进行读取图片获取纹理, 使用Windows窗口显示。

材质:
```
Diffuse: 漫反射
Metal: 金属, 同时拥有漫反射和镜面反射性质
Mirror: 镜面反射材质
Glass: 玻璃, 可以折射光线
```
物体:
```
目前只支持Procedural Geometry
Sphere: 球体
Platform: 垂直于y轴的平面
Cylinder: 圆柱体
```

环境
```
Windows Visual Studio 2019, 使用前需要解压 lib\opencv_world460d.zip 文件
```

效果:

![image](https://user-images.githubusercontent.com/44687653/148175016-21c0daba-3003-4393-8598-44c3be3cc507.png)

