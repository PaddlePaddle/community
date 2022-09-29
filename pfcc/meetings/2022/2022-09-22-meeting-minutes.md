# Paddle Frawework Contributor Club 第十一次会议纪要

## 会议概况

- 会议时间：2022-09-22 19:00 - 20:00
- 会议地点：线上会议
- 参会人：本次会议共有 34名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由轮值主席林旭（[isLinXu](https://github.com/isLinXu)）主持。
- 会议主题：《Paddle的编译构建》



## 会议分享与讨论

本次会议以Paddle的编译构建为主题进行相关内容的分享与讨论。

以下为主要内容：

### 1、新成员的自我介绍

首先，PFCC 新成员[engineer1109](https://github.com/engineer1109)、[enkilee](https://github.com/enkilee)、[Zheng-Bicheng](https://github.com/Zheng-Bicheng)进行了自我介绍，欢迎加入 PFCC！

### 2、Paddle 的编译构建与优化

飞桨研发工程师[pangyoki](https://github.com/pangyoki) ，分享主题《Paddle 的编译构建与优化》，深入浅出的介绍了Paddle的编译与构建流程，以及优化的一些方式。


### 3、使用AIStudio完成编译的体验分享

PFCC成员[xiaohemaikoo](https://github.com/xiaohemaikoo) ，分享主题《float64类型扩展开发分享》，主要是在windows平台下编译出现的一些问题，以及一些个人的理解与建议。

**问题如下**：

- 编译最后链接失败
- 依赖文件缺失。(缺失文件并非来自漏安装的依赖包，缺失文件系统本身不存在)
- 编译用到的工具环境缺失或者异常

**建议如下**：

- 仔细检查比对自己搭建环境和编译步骤和官方指导文档是否存在不合理出入。

- 针对环境文件缺失，建议检查方法。

- 针对环境异常， 编译报代码错误。

    检查该paddle版本是否是可用基线。

    检查报错代码，查找是否是依赖的环境工具版本不兼容。

**理解如下**：
这部分内容介绍了一些理解和思考，也在QA中得到了积极响应，并展开讨论。

- windows和linux编译工具未统一
- 代码框架结构中多用模板和宏，算子c++代码中未见面向对象

主要是分析了一些编译工具链以及设计模式上的问题，针对这些问题与研发人员也进行了沟通交流，分析其优点与缺点。



### 4、float64类型扩展开发分享

PFCC成员[Li-fAngyU](https://github.com/Li-fAngyU) ，分享主题《使用AIStudio完成编译的体验分享》

- 在二次开发中，编译paddle是一个重要环节和前置条件。
- 讲解了参与热身打卡任务的几个步骤与流程，主要包括：初次编译、二次编译、安装whl包以及运行单元测试
- 使用AI Studio二次开发项目，通过VScode界面的终端进行执行。

热身运动：[链接](https://www.paddlepaddle.org.cn/contributionguide?docPath=hackathon_warming_up_cn)
编译文档：[链接](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile.html)

### 5、编译吐槽

PFCC成员[engineer1109](https://github.com/engineer1109) ，分享主题《编译吐槽》，基于[engineer1109](https://github.com/engineer1109) 资深的开发经验，总结了一些在使用paddle过程中发现的一些问题。

- 命名空间格式问题

  > 命名空间嵌套与使用格式，详情可见[issue](https://github.com/PaddlePaddle/Paddle/issues/4588)

- 链接顺序混乱

  > 高层API要靠前，底层API要靠后

- docker与本地开发差异

  > Linux本地用户是不能直接访问/usr 文件夹，需要s udo
  > docker是可以访问任何文件夹

  

### 6、编译warning消除任务介绍

PFCC成员[isLinXu](https://github.com/isLinXu) ，介绍并开发任务 《编译warning消除任务介绍》
主要介绍了该任务的目的以及现状，同时也介绍了消除warning的解决方法与原则，还有推进解决的方式和一些已知的问题。

更多内容，可见**任务开放**：[链接](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/code_style_compiler_warning.md)


最后，关于以上内容部分的ppt及资料，已上传至PFCC的百度云盘，可从群信息或公告中获取。

### 7、QA 交流环节

- [xiaohemaikoo](https://github.com/xiaohemaikoo) 继续讨论了一些自己的心得体会与编译理解。
- [engineer1109](https://github.com/engineer1109) ，咨询并讨论了关于C++调用PaddleLite的问题，并表示希望后续fluid可以开放C++接口。

### 下次会议安排

确定下次会议的时间为两周后的同一个时间段。并确定下次会议的主席为[tizhou86](https://github.com/tizhou86)，副主席为[OuyangChao](https://github.com/OuyangChao)，主题暂定

