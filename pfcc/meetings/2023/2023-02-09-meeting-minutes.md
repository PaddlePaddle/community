# Paddle Frawework Contributor Club 第十六次会议纪要

## 会议概况

- 本次会议时间：2022-02-09 19:00
- 会议地点：线上会议
- 参会人：本次会议共有37名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由轮值主席黄子豪（[DrRyanHuang](https://github.com/DrRyanHuang)）主持。
- 会议主题：《低精度算子开发与优化经验》



## 会议分享与讨论

本次会议以 Paddle 低精度算子开发与优化为主题进行相关内容的分享与讨论。

以下为主要内容：


### 1、低精度算子开发与优化经验

飞桨研发[liuruyan](https://github.com/liuruyan)，主要分享低精度算子开发与优化经验。

交流环节：

- Q：我们在 paddle 中开启 amp 训练, 遇到 loss 出现 NaN 时, 是否主要调整 Grad Scale 参数?
- A：是的, 首先要保证不开启 amp 训练没有异常问题。其次, 我们在训练时遇到上溢的问题次数较少(在amp O1模式下, 可能会产生上溢问题的算子会自动转化为 fp32 进行计算), 下溢问题主要可以通过调整 Grad Scale 参数来缓解这个问题。同时在训练过程中, 如果遇到 loss 为 NaN, 框架会自动调整 loss 的 scale 来尝试解决这个问题。



### 2、飞桨 API 动静行为统一经验分享

PFCC成员[DrRyanHuang](https://github.com/DrRyanHuang) 进行飞桨 API 动静行为统一经验分享。


### 3、春节开源经历分享

PFCC成员[RedContritio](https://github.com/RedContritio) 进行春节开源经历分享。


### 4、"改进飞桨框架的logging系统"志愿者征集

PFCC成员[engineer1109](https://github.com/engineer1109) 进行 "改进飞桨框架的logging系统" 的志愿者征集。


### 5、飞桨快乐开源活动进度同步

PFCC组织者[Ligoml](https://github.com/Ligoml) 介绍两大开源活动（快乐开源+黑客松），收集贡献指南和框架开发需求。


### 下次会议安排

确定下次会议的时间为两周后的同一个时间段。主席为[sanbuphy](https://github.com/sanbuphy), 副主席[待定]()，主题待定。
