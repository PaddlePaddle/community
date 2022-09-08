

#### 开源软件开发及实践课程

这是一个给学生介绍开源的基础知识，和基本开发流程的课程，我们希望能够通过这个课程促进学生了解开源软件开发，并加入到自己感兴趣的开源社区。

##### 课程的应用场景

这个课程分为两节小课，每一节都可以单独拿来作为插入到本科或者研究生的计算机程序设计或者其他相关的课程当中，用来当做对学校的系统性课程的补充。第一节课程介绍开源的基础知识，和需要用到的工具（git）和平台（github）；第二节课程以参与飞桨框架的研发为例，教学生如何在一个真实的开源项目当中进行实践。

##### 课程的前置要求

- 拥有github账号。
- 有一些简单的软件开发的基础知识。

##### Course1 开源的基础知识和工具

- Lecture1：通识知识：开源项目基础，git及github的简介。
    - 快速互动：以下哪些是开源软件
        - linux操作系统
        - gcc编译器
        - 飞桨深度学习框架
        - MySQL数据库
    - 贡献开源的动机
        - 对于学生：
            - 提升个人技能
            - 丰富个人履历（丰富的github记录会是面试时很好的加分项）
            - 认识更多有趣的人
            - 获得成就感
        - 对于工程师
            - 改进自己的生产工具
        - 对于公司
            - 构建基于开源的生态（例：基于安卓的生态）
            - 形成基于开源的商业模式（例：各类大数据公司）
    - 如何加入开源社区
        - 从用户到开发者
        - Star、Fork、PR、Issue、
        - CLA协议、DCO、Code of Conduct
        - Communication：maillist、微信群、论坛、等等
            - public and transparent communication
    - 开源贡献的多种形式
        - 代码贡献：案例，pull requests of paddle
        - 社区布道：案例，
        - code review：案例，
        - bug report：案例，
        - issue triage：案例，
        - 文档修复：案例，
        - Tutorials：案例，
        - Case Study: [numpy's call for contributions](https://github.com/numpy/numpy/blob/main/README.md#call-for-contributions)
    
    
    - upstream first 原则
    
    - 贡献开源需要的工具和平台
    
      - git
    
      - github
    
      - 国内的gitee
    
      - 其他方式：提交patch（如linux kernel、gcc）
    


- Lecture2：开源项目开发的实践。

  - 寻找需要开发的功能
    - issue 标签为 [good first issue](https://github.com/PaddlePaddle/Paddle/labels/good%20first%20issue) 或 [PR is welcome](https://github.com/PaddlePaddle/Paddle/labels/PR%20is%20welcome) ：通常是一些小功能的开发或者 bug 的修复，你可以通过完成这个 ISSUE 来踏出贡献代码的第一步。
    - 开发新的 feature：可以参考 [issue 指南](https://github.com/PaddlePaddle/Paddle/issues/41281) 发起新的 issue，描述新 feature 的背景和特性，发起相关讨论。也可以通过 label 的 issue：[feature-request](https://github.com/PaddlePaddle/Paddle/labels/type%2Ffeature-request) 和 [new-feature](https://github.com/PaddlePaddle/Paddle/labels/type%2Fnew-feature) 来了解其它社区开发者提出的 feature 。
    - 感兴趣的相关 issue：可以在 issue 页面搜索感兴趣的相关 issue 来改进，可以重点关注 Pinned issues（置顶位的重要议题）。
    - [报告安全问题](https://github.com/PaddlePaddle/Paddle/blob/develop/SECURITY_cn.md)：特别地，若发现飞桨项目中有任何的安全漏洞（或潜在的安全问题），请第一时间通过 paddle-security@baidu.com 邮箱私下联系我们。在对应代码修复之前，请不要将对应安全问题对外披露，也不鼓励公开提 issue 报告安全问题。
  - 搭建飞桨开发环境
    - 本地开发环境：飞桨提供了 Linux/MacOS/Windows 等多种本地环境的源码编译方式，可参考 [文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html)。
    - 线上开发环境：为解决资源和环境等前置难题，百度飞桨 AI Studio 面向社区开发者提供飞桨镜像环境、在线 IDE 与专属 GPU 算力。此功能为专属授权功能，申请使用请通过邮件（ext_paddle_oss@baidu.com）联系管理员，资源有限，请按需申请；
    - 如果准备就绪，欢迎参与[【热身打卡：开发框架，从编译 paddle 开始】](https://www.paddlepaddle.org.cn/contributionguide?docPath=hackathon_warming_up_cn)小试牛刀！不仅能让你以最快速度上手框架开发，成功打卡还有飞桨周边礼品送出哦～
  - 完成一个具体的功能开发
  - 提交代码，并创建PullRequest
  - 通过Code Review，CI及代码合入
  - 软件版本发布及分发




##### 课程作业

- 在github上完成PR合入流程：提交一个PR修改到作业的仓库。
    - 推荐给开源项目的文档进行修改。（例如Paddle的docs仓库的修改）
- 完成paddle的编译。
    - 附件项：跑通单测。
- 在fork出来的paddle仓库里，完成一个fake API的实现，并完成PR提交。
- 给飞桨做贡献
