# Paddle Framework Contributor Club 第三十四次会议纪要

## 会议概况

- 本次会议时间：2024/01/25 20:00-21:00 (GMT+08:00) 中国标准时间 - 北京
- 会议地点：线上会议
- 参会人：本次会议共有 28 名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由轮值主席[Turingg](https://github.com/Turingg)主持。
- 会议主题：《黑客马拉松的编程集训、PaddleOCR算法挑战、Pytorch Conference 会议翻译项目以及飞桨社区开发工作组的季度会议内容和议题征集》

## 会议分享与讨论

以下为本次会议的主要内容：

## 会议议程

### 1、新成员的自我介绍

欢迎新同学[Tsaiyue](https://github.com/Tsaiyue)的加入！

### 2、[@sunzhongkai](https://github.com/sunzhongkai588) [【HACKATHON 6th Code Camp】黑客松护航计划集训营](https://github.com/PaddlePaddle/Paddle/issues/61006)的详情以及细节介绍

1. 活动流程
   - 在issue下进行报名以及意向
   - 报完名之后会进行面试
   - 进行三个月开发
   - 三个月结束后会有相应考核
2. 注意事项
   - 参加集训营的同时要提交周报
   - 简历投递邮箱与上期的不同[ext_paddle_oss@baidu.com](mailto:ext_paddle_oss@baidu.com)
   - 每周开发时间至少要有25h
   - 注意项目介绍，人数以及锁定营员（锁定说明已被录取，不需要再报名）
   - 提交简历的时候最好附上之前参与的开源项目或者在社区参与开源项目的PR链接

### 3、[@tink2123](https://github.com/tink2123) [PaddleOCR 算法模型挑战赛](https://competition.atomgit.com/competitionInfo?id=d25e62a0d7f27876a8c4219bfc0be90e)活动介绍

1. 活动流程

   - 打榜比赛，分为两个赛题
   - 在飞桨星河社区提交对应赛题的结果

2. 赛题介绍

   - 基于[ppocr](https://github.com/Turingg/PaddleOCR/tree/release/2.7/ppocr)、[ppstructure](https://github.com/Turingg/PaddleOCR/tree/release/2.7/ppstructure) 进行一次技术上的升级和迭代

     1.赛题一：OCR端到端识别任务

     * 输入一张图片，输出完整图片里框的内容以及文字内容
     * 结果在评测脚本上算出来端到端的指标，最后的Hmean or Hscore 超过62.24%为符合要求
     * 提供开源的数据集大概在5-10万张文本识别，1000张文本检测——推荐数据合成工具（需提交）
     * 不要人工标注数据
     * A榜为实时计算，B榜为内测指标

     2.赛题二：通用表格识别任务

     * 输入一张图片形式的表格转成html格式
     * 提交模型要求在PubTabNet评估集上ACC超过76.31%
     * 不要把评估集的数据加到训练集里

3. 注意事项

   * 在[开放原子开源大赛](https://competition.atomgit.com/competitionInfo?id=d25e62a0d7f27876a8c4219bfc0be90e)官网进行报名
   * 获奖队伍需要把代码or模型提交到开源大赛的网址上

4. 答疑

   - OCR合成数据的数据量要多大合适

     1.对识别来说是取决于识别大小的，字典里每一个字在训练集里出现的频率不超过500次

     2.数据的质量较为重要

   - 人工标注的数据集可能出错

     合成工具构建，出错的可能性会较低

   - 推荐的字典PaddleOCR里面的[字典](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppocr/utils/ppocr_keys_v1.txt)

### 4、[@sanbuphy](https://github.com/sanbuphy) Pytorch Conference 系列翻译活动（社区开发者个人项目）

1.内容

* 翻译工具流

  1.散步师傅提供翻译流程还有python文件-指定视频转录成英文结果

  2.使用openAI开源工具whisper

  3.结果为视频→到文本结果

* 如何建设Pytorch Conference 系列翻译的想法

  1.2023 Pytorch Conference 转为中文的形式

2.散步师傅之前翻译的：https://mp.weixin.qq.com/s/uSLye5vmL8jM0I5PMYt0ww

### 5、[飞桨社区开源发展工作组](https://github.com/PaddlePaddle/community/tree/master/pposdwg) 2024 Q1 例会预告及议题征集。

* 工作

  1.社区开源之星评选

  2.每个季度都会开会

  3.讨论社区接下来的发展

  4.预告开会时间1.31

  5.有议题想法可以在Github community issue 区提相关的issue

### 6、自由讨论时间

1. 飞桨新开了个小红书账号：飞桨xiaoxiao
2. Pytorch Conference 系列翻译活动产生的原因

