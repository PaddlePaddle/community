# Paddle Frawework Contributor Club 第八次会议纪要

## 会议概况

- 会议时间：2022-08-11 19：00 - 20：00
- 会议地点：线上会议
- 参会人：本次会议共有 26 名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由轮值主席王志宇（[fuqianya](https://github.com/fuqianya)，湖南大学）主持。

## 会议分享与讨论

### 新成员的自我介绍
PFCC 新成员 [Ainavo](https://github.com/Ainavo)、[ReganYue](https://github.com/ReganYue) 、[mrcangye](https://github.com/mrcangye) 进行了自我介绍，欢迎加入 PFCC！

### CUDA编程专题分享
CUDA编程是提升飞桨性能的关键。本次会议，邀请了三位成员分享CUDA编程的相关知识。

- [OccupyMars2025](https://github.com/OccupyMars2025)：以第三期黑客松的第四个任务[为 Paddle 新增 cummax API](https://github.com/PaddlePaddle/Paddle/issues/44073#task4)为例，完整的介绍了他完成该任务的整个过程，并深入CUDA代码，讲解了一些他对于CUDA代码的个人理解。
- [thunder95](https://github.com/thunder95)：介绍了一些CUDA编程的基础知识，并分享了他在实现 [Pixelunshuffle](https://github.com/PaddlePaddle/Paddle/pull/40774)、[Rrelu](https://github.com/PaddlePaddle/Paddle/pull/41823)、[Nanmedian](https://github.com/PaddlePaddle/Paddle/pull/42385)、[IndexFill](https://github.com/PaddlePaddle/Paddle/pull/42453) 算子时的经验，最后他指出了CUDA加速在企业中的一些应用。
-  [ZzSean](https://github.com/ZzSean)：从优化访存效率、优化block和grid设置策略、优化计算效率、优化指令并行度四个方面介绍了多维 Reduce 的优化方法。

### 下次会议安排
确定下次会议的时间为两周后的同一个时间段。并确定下次会议的主席为：任子跻（[OccupyMars2025](https://github.com/OccupyMars2025)），副主席待定。
