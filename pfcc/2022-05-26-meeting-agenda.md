# Paddle Frawework Contributor Club 第四次会议

- 本次会议时间：2022-5-26 19：00
- 本次会议接入方式： 
  - 腾讯会议：763-545-348
  - 会议密码：**** (群内通知)
  - [点击链接入会](https://meeting.tencent.com/dm/N62pGjjeyQb6)，或添加至会议列表
- 本次会议主席：石华榜（[S-HuaBomb](https://github.com/S-HuaBomb)，广西大学）
- 本次会议副主席：王儒婷（[xiaoguoguo626807](https://github.com/xiaoguoguo626807)，中科院计算所）
- 本次拟参会Paddle Framework Contributor列表：
    - [jzhang533](https://github.com/jzhang533)
    - [jeff41404](https://github.com/jeff41404)
    - [luotao1](https://github.com/luotao1)
    - [TCChenlong](https://github.com/TCChenlong)
    - [Ligoml](https://github.com/Ligoml)
    - [guguguzi](https://github.com/guguguzi)
    - [gsq7474741](https://github.com/gsq7474741)
    - [chenyanlann](https://github.com/chenyanlann)
    - [GeYuYao-hub](https://github.com/GeYuYao-hub)
    - [isLinXu](https://github.com/isLinXu)
    - [yangguohao](https://github.com/yangguohao)
    - [Li-fAngyU](https://github.com/Li-fAngyU)
    - [BrilliantYuKaimin](https://github.com/BrilliantYuKaimin)
    - [jinyouzhi](https://github.com/jinyouzhi)
    - [thunder95](https://github.com/thunder95)
    - [Liyulingyue](https://github.com/Liyulingyue)
    - [Asthestarsfalll](https://github.com/Asthestarsfalll)
    - [liqitong-a](https://github.com/liqitong-a)
    - [unseenme](https://github.com/unseenme)
    - [greatv](https://github.com/greatv)
    - [fuqianya](https://github.com/fuqianya)
    - [liyongchao911](https://github.com/liyongchao911)
    - [Yulv-git](https://github.com/Yulv-git)

# 会议议程


- PFCC 新同学 [Yulv-git](https://github.com/Yulv-git) 的一句话自我介绍（30 seconds）
- [GitLink](https://github.com/PaddlePaddle/Paddle/issues/42843) 编程夏令营（GLCC）活动介绍（[jzhang533](https://github.com/jzhang533)，5 minutes）
- PFCC-Roadmap：API单测增强、算子性能扩展、算子数据类型扩展专题介绍 （5-10 minutes）
- 文档方向启动信息（[TCChenlong](https://github.com/TCChenlong)，5-10 minutes）
- 提了一个新需求的 issue [【PFCC-Roadmap】API高维功能扩展 #42904](https://github.com/PaddlePaddle/Paddle/issues/42904)，为部分paddle api增加3D输入能力（API范围，增加方式，应用举例，性能考虑等）（10 minutes）
  - `F.affine_grrid` 和 `F.grid_sample()`，目前不支持 3D 处理。
    
    它们的组合是实现STN（Spatial Transformer Network）的基础组件，常用在图像变形中。增加3D支持对实现可微3D图像（特别是医学图像）变形和配准十分关键。相关issue详见：[【论文复现】paddle的affine_grid如何实现三维功能 · Issue #38670](https://github.com/PaddlePaddle/Paddle/issues/38670)。

  - `paddle.logsumexp` 目前不支持 5D Tensor。
    
    此 OP 计算 x 的指数的总和的对数：logsumexp(x) = log∑exp(x)。常用于对数softmax的计算。相关issue详见：[【论文复现】paddle.logsumexp不支持5D tensor · Issue #34742](https://github.com/PaddlePaddle/Paddle/issues/34742)。
- 框架贡献者的分享，内容不限：（15-25 minutes）
  - [fuqianya](https://github.com/fuqianya)：分享一些关于贡献飞桨高层 API & 论文复现的一些经历
  - [Yulv-git](https://github.com/Yulv-git)：Search for typos 介绍
- 自由发言，可以提想法、提问和讨论（10 minutes）
- 确定下次会议的主席与副主席，以及召开时间 和 部分议题预告（3 minutes)
