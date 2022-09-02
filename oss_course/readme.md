

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
  - 搭建本地的飞桨开发环境
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
