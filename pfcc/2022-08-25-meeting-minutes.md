# Paddle Frawework Contributor Club 第九次会议纪要

## 会议概况

- 会议时间：2022-08-25 19：00 - 20：00
- 会议地点：线上会议
- 参会人：本次会议共有 28 名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由轮值主席任子跻（[OccupyMars2025](https://github.com/OccupyMars2025)）主持。

## 会议分享与讨论

### 新成员的自我介绍
PFCC 新成员 [peachlcy](https://github.com/peachlcy) 、[Rayman96](https://github.com/Rayman96)进行了自我介绍，欢迎加入 PFCC！

### Paddle框架中Python端与C++端交互专题分享
本次会议，邀请了两位开发者分享相关知识，然后飞桨官方向社区发布了新的开发任务。

- [OccupyMars2025](https://github.com/OccupyMars2025)：以第三期黑客松的第2个任务[为 Paddle 新增 iinfo API](https://github.com/PaddlePaddle/Paddle/issues/44073#task2)为例，完整的介绍了他完成该任务的[Pull Request](https://github.com/PaddlePaddle/Paddle/pull/45321)，以此来讲解Paddle框架和Pytorch框架中iinfo API内部是如何实现python端与C++端的交互。
- 飞桨研发工程师 [wanghuancoder](https://github.com/wanghuancoder) 主题分享:[PYTHON_C_API](https://github.com/PaddlePaddle/Paddle/pull/32524), 完整深入地介绍了在Paddle框架中python端和C++端是如何通过pybind11和Python C API实现交互的，他介绍相关代码全在paddle\fluid\pybind文件夹下，其中核心是paddle\fluid\pybind\pybind.cc，编译后该文件会生成python module "core_avx" 或者 "core_noavx"，里面包含了大量的编译后绑定到python端的C++函数或类
- [luotao1](https://github.com/luotao1): 发布 [Call for Contributions 任务介绍](https://github.com/PaddlePaddle/community/tree/master/pfcc/call-for-contributions) 

### 下次会议安排
确定下次会议的时间为两周后的同一个时间段。并确定下次会议的主席为：徐晓健 ([Nyakku Shigure](https://github.com/SigureMo))，副主席待定。
