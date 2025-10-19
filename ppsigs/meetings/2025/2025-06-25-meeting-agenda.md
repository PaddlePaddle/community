## 第二十二次 PaddleOCR 开源共建研讨会

* 会议时间：2025-06-25 19:30
* 本次会议接入方式：
    * 腾讯会议：758-578-854
    * [点击链接入会](https://meeting.tencent.com/dm/KPTfyrAYOKCc)，或添加至会议列表
* 本次拟参会 member list：
    * [cuicheng01](https://github.com/cuicheng01)
    * [jzhang533](https://github.com/jzhang533)
    * [Topdu](https://github.com/Topdu)
    * [greatv](https://github.com/greatv)
    * [Gmgge](https://github.com/Gmgge)
    * [SWHL](https://github.com/SWHL)
    * [Liyulingyue](https://github.com/Liyulingyue)
    * [luotao1](https://github.com/luotao1)
    * [mattheliu](https://github.com/mattheliu)
    * [UserWangZz](https://github.com/UserWangZz)
    * [jingsongliujing](https://github.com/jingsongliujing)
    * [Alex Zhang](https://github.com/openvino-book)

## 会议议程

### 1. Issue 与 PR 回顾

### 2. 其他议题
Paddle C API 讨论
- 问题：C API没有做好C++异常隔离，也无法将错误信息报告给上层的C# 程序——必须从控制台看日志，目前C API虽然维护变少了，但希望官方不要完全停止维护，这对其它编程语言接入很重要
- PR review：
  - 1、有些关键新API，如NewIREnabled不支持C API，patch会增加这个C API：https://github.com/PaddlePaddle/Paddle/pull/73629
  - 2、修复new ir（JSON模型）无法从内存中加载的问题：https://github.com/PaddlePaddle/Paddle/pull/73630
  - 3、修复macos-13（x64）使用了ONNXRUNTIME时编译报错的问题：https://github.com/PaddlePaddle/Paddle/pull/73631 

### 3. 自由发言，可以提需求给大家讨论、提问等

### 4. 确定下次的部分议题及召开时间

### 轮值值班安排

* 职责：处理 issue 区和 Discussion 区的新增问题，召集，主持例会并记录。
* 期间：每隔两周轮换。
* 排班：@greatv, @SWHL, @jingsongliujing, @cuicheng01， @jzhang533, @mattheliu， @Topdu

序号|GITHUB ID|值班开始|值班结束
:------:|:------|:------:|:------:
1|@greatv|2025-03-06|2025-03-19
2|@SWHL |2025-03-20|2025-04-02
3|@jingsongliujing |2025-04-03|2025-04-23
4|@cuicheng01 |2025-04-17|2025-05-16
5|@jzhang533 |2025-05-17|2025-06-03
6|@mattheliu |2025-06-04|2025-06-18
7|@Topdu |2025-06-19|2025-07-02
