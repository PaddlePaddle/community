# 开源软件开发及实践课程

这是一个给学生介绍开源的基础知识，和基本开发流程的课程，我们希望能够通过这个课程促进学生了解开源软件开发，并加入到自己感兴趣的开源社区。

We have offered this course in:
- 2022-9-28， 北京大学， [第4次课](https://github.com/osslab-pku/OSSDevelopment/blob/main/Syllabus.md), offered by [jzhang533](https://github.com/jzhang533)
- 2022-10-20， 浙江大学AI研究生院， offered by [luotao1](https://github.com/luotao1)
- 2022-11-9 北京航空航天大学， offered by  [jzhang533](https://github.com/jzhang533)

[Ligoml](https://github.com/Ligoml) kindly served as TA in these offerings.

## 课程的应用场景

这个课程分为两节小课，每一节都可以单独拿来作为插入到本科或者研究生的计算机程序设计或者其他相关的课程当中，用来当做对学校的系统性课程的补充。第一节课程介绍开源的基础知识，和需要用到的工具（git）和平台（github）；第二节课程以参与飞桨框架的研发为例，教学生如何在一个真实的开源项目当中进行实践。

如果需要第二节课程的 PPT ，请发送邮件到 ext_paddle_oss@baidu.com ，说明用途。

## 课程的前置要求

- 拥有github账号。
- 有一些简单的软件开发的基础知识。

## Lecture1： 开源的基础知识和工具

- 通识知识：开源项目基础，git及github的简介。
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
        - 代码贡献：案例，[飞桨黑客松活动](https://github.com/PaddlePaddle/Paddle/issues/43938)
        - 社区布道：案例，[飞桨领航团](https://www.paddlepaddle.org.cn/ppdenavigategroup)
        - code review：案例，[pull requests of paddle](https://github.com/PaddlePaddle/Paddle/pulls)
        - bug report：案例，[bug report issues of Paddle](https://github.com/PaddlePaddle/Paddle/issues?q=is%3Aissue+is%3Aopen+label%3Atype%2Fbug-report)
        - issue triage：案例，[needs triage issues of pip](https://github.com/pypa/pip/issues?q=is%3Aissue+is%3Aopen+label%3A%22S%3A+needs+triage%22)
        - 文档修复：案例，[pull requests of paddle docs](https://github.com/PaddlePaddle/docs/pulls)
        - Tutorials：案例，[Tutorials of paddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/index_cn.html)
        - Case Study
          -  [numpy's call for contributions](https://github.com/numpy/numpy/blob/main/README.md#call-for-contributions)
          -  [report a bug to pip](https://github.com/pypa/pip/issues/11423)
    - 贡献开源需要的工具和平台
      - git的简要介绍
      - github的简要介绍
      - 国内的gitee的简要介绍
      - 其他方式：提交patch（如linux kernel、gcc）

## Lecture2: 开源项目开发的实践

#### 寻找需要开发的功能
- issue 标签为 [good first issue](https://github.com/PaddlePaddle/Paddle/labels/good%20first%20issue) 或 [PR is welcome](https://github.com/PaddlePaddle/Paddle/labels/PR%20is%20welcome) ：通常是一些小功能的开发或者 bug 的修复，你可以通过完成这个 ISSUE 来踏出贡献代码的第一步。
- 开发新的 feature：可以参考 [issue 指南](https://github.com/PaddlePaddle/Paddle/issues/41281) 发起新的 issue，描述新 feature 的背景和特性，发起相关讨论。也可以通过 label 的 issue：[feature-request](https://github.com/PaddlePaddle/Paddle/labels/type%2Ffeature-request) 和 [new-feature](https://github.com/PaddlePaddle/Paddle/labels/type%2Fnew-feature) 来了解其它社区开发者提出的 feature 。
- 感兴趣的相关 issue：可以在 issue 页面搜索感兴趣的相关 issue 来改进，可以重点关注 Pinned issues（置顶位的重要议题），如[飞桨社区活动总览-飞桨黑客松](https://github.com/PaddlePaddle/Paddle/issues/42410)。
- [Roadmap](https://github.com/PaddlePaddle/Paddle/issues/42571) 和 [Call-for-Contributions](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/README.md)：为了让大家能深入地了解飞桨、在飞桨收获更多成长、解决更有挑战性的问题，飞桨团队将正在开展的一些重点工作和技术方向陆续发布。每个技术方向都会有工程师支持，和该方向中的同学一起确定目标、规划和分工。
- [报告安全问题](https://github.com/PaddlePaddle/Paddle/blob/develop/SECURITY_cn.md)：特别地，若发现飞桨项目中有任何的安全漏洞（或潜在的安全问题），请第一时间通过 paddle-security@baidu.com 邮箱私下联系我们。在对应代码修复之前，请不要将对应安全问题对外披露，也不鼓励公开提 issue 报告安全问题。
#### 搭建飞桨开发环境
- 本地开发环境：飞桨提供了 Linux/MacOS/Windows 等多种本地环境的源码编译方式，可参考 [文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html)。
- 线上开发环境：为解决资源和环境等前置难题，百度飞桨 AI Studio 面向社区开发者提供飞桨镜像环境、在线 IDE 与专属 GPU 算力。此功能为专属授权功能，申请使用请通过邮件（ext_paddle_oss@baidu.com）联系管理员，资源有限，请按需申请；
- 如果准备就绪，欢迎参与[【热身打卡：开发框架，从编译 paddle 开始】](https://www.paddlepaddle.org.cn/contributionguide?docPath=hackathon_warming_up_cn)小试牛刀！不仅能让你以最快速度上手框架开发，成功打卡还有飞桨周边礼品送出哦～
#### 完成一个具体的功能开发
- 一些方向的功能开发已经有完整的 [贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，如新增 API 开发、算子性能优化、算子数据类型扩展、自定义硬件接入、文档贡献等，请遵守这些贡献流程。
- 如果你的 PR 包含非常大的变更，比如模块的重构或者添加新的组件，请**务必先提出相关 issue 发起详细讨论，达成一致后再进行变更**，并为其编写详细的文档来阐述其设计、解决的问题和用途。注意一个 PR 尽量不要过于大，如果的确需要有大的变更，可以将其按功能拆分成多个单独的 PR。
- 重要的代码需要有完善的测试用例（单元测试、集成测试），对应的衡量标准是测试覆盖率，飞桨要求 [增量代码需满足行覆盖率大于 90%](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/git_guides/paddle_ci_manual_cn.html#pr-ci-coverage) 。
- 代码需要有可读性、易用性和健壮性。重要代码要有详细注释，代码尽量简练、复用度高、有着完善的设计，代码风格要整洁、规范，请参考 [飞桨代码规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/style_guides_cn.html) 和 [代码风格检查指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/git_guides/codestyle_check_guide_cn.html)。
#### 提交代码，并创建PullRequest
- 不论是否熟悉 GitHub 相关操作，建议先浏览一遍 [飞桨提交代码的流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)，以便了解飞桨项目贡献的一些差异点。
- 提交 PR 的时候请参考 [PR 模板](https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/.github/PULL_REQUEST_TEMPLATE.md)，同时请遵循 [提交 PR 约定](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html#span-id-caution1-pr-span)。在进行较大的变更的时候请确保 PR 有一个对应的 Issue。
- 若你是初次提交 PR，请先签署 [CLA](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html#cla)（PR 页面会有自动回复指引）。
#### 通过Code Review，CI及代码合入
- CI: 在提交 PR 后，系统会自动运行持续集成，CI 测试进程一般会在几个小时内完成。请确保所有的 CI 均为 pass 状态（除 PR-CI-APPROVAL 和 PR-CI-Static-Check 这两个 CI 测试项可能需要飞桨相关开发者 approve 才能通过），如果没有通过，请通过报错信息自查代码，详细测试内容可参见 [Paddle CI 测试详解](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/git_guides/paddle_ci_manual_cn.html)。 
- Code Review：CI 测试通过后，接下来请等待 Code Review，一般会在三个工作日内回复。但是若 CI 测试不通过，评审人一般不做评审。收到 Code Review 意见后，请参考 [Code Review 注意事项](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html#CodeReview) 回复评审人的意见，并根据意见修改代码。
- 代码合入：PR Merge 后本次贡献结束，飞桨团队相关人员会对整个框架功能进行集成测试，集成测试用于模型、API、OP 等的功能和性能测试。如果测试通过，恭喜你贡献流程已经全部完成；如果测试不通过，我们会在 GitHub 发 Issue 联系你进行代码修复，请及时关注 GitHub 上的最新动态。
#### 软件版本发布及分发
-  我们使用 develop 分支作为我们的开发分支，这代表它是不稳定的分支。每个版本区间（如 2.1.x）都会创建一个 release 分支（如 release-2.1）作为稳定的发布分支。每发布一个新版本都会将其合并到对应的 release 分支并打上对应的 tag。阅读 [release note](https://github.com/PaddlePaddle/Paddle/releases) 有助于了解每个版本的功能。
- 代码合入 Paddle develop 分支后的第二天，即可从飞桨官网下载 [develop 版本的编译安装包](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)体验此功能。通过测试的代码会被纳入正式版的发版计划。
- 跟进据国际主流的惯例，[「版本号」的格式](https://github.com/guobinhit/cg-blog/blob/master/articles/others/version.md)是：`<major>.<minor>.<patch>`。`<major>`即主版本号，俗称大版本升级；`<minor>`即次版本号，俗称小版本升级；`<patch>`即修订号，俗称 bug 修复。飞桨2018年发布核心框架 v1.0版本，2021年发布v2.0版本，即将发布2.4版本。

## 课程作业

- 文档贡献
    - [Bug-fix](https://shimo.im/sheets/e1Az48XnO4t6g7qW/akF3x/)：修正至少一组飞桨 API 文档 bug，参考资料：[API 文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html#id8)
- 编译打卡
    - 在本地/线上环境完成 Paddle 编译体验，并输出一份编译报告，参考资料：[热身打卡 issue](https://github.com/PaddlePaddle/Paddle/issues/45347)
- 开发任务：[call-for-contributions](https://github.com/PaddlePaddle/community/tree/master/pfcc/call-for-contributions)
    -  [【编译 warning 的消除】](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/code_style_compiler_warning.md)
    -  [【flake8 代码风格检查工具的引入】](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/code_style_flake8.md)
    -  [【Python 2.7 相关代码退场】](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/legacy_python2.md)
    -  [【Type Hint类型注释】](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/type_hint.md)
    -  注：此类任务对代码仓库改动比较大，可以拆分成若干子项，完成一个 PR 合入则算完成，有飞桨研发工程师指导
