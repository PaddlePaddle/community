# 飞桨框架代码仓库的角色及权限介绍

作为一个开放的开源社区，飞桨社区鼓励每一位开发者在这里学习与贡献，与飞桨共同成长。随着你在飞桨社区的参与与贡献的增多，你可以逐步的建立在飞桨社区的影响力，并逐步承担更多的飞桨社区的职责。参与飞桨社区的贡献，有多种形式，包括宣传布道、组织本地活动、参与讨论、进行代码开发，等等方式。特别的，在代码仓库的管理上（[Paddle](https://github.com/PaddlePaddle/Paddle)、[docs](https://github.com/PaddlePaddle/docs)），飞桨社区设定了以下的职责与权限，来便于社区成员进行项目开发与协作。

| 成员角色    | 社区职责                                               | 获得方式                                                     | 社区权限                                                     |
| ----------- | ------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Anyone      | 参与飞桨社区的成员。                                   | 无                                                           | 代码仓库的 read 权限。PR 及 ISSUE 的创建与评论。             |
| Triager     | 管理代码仓库中的PR与ISSUE。                            | 因实际项目的需要，对一部分PR或者ISSUE进行分类、编辑与分派（e.g.：[intel label](https://github.com/PaddlePaddle/Paddle/issues?q=label%3Aintel+)）。流程：请联系项目的接口人。 | 代码仓库的 triage 权限。PR 及 ISSUE 的分派与编辑。           |
| Contributor | 以 PR 的方式，为飞桨的代码仓库贡献代码或文档。         | 所提交的 PR 被成功合入代码仓库。                             | 如果你有意愿，欢迎加入 [PFCC](https://github.com/PaddlePaddle/community/tree/master/pfcc)，参与飞桨代码仓库贡献的技术讨论。 |
| Committer   | 为飞桨社区的 PR 贡献提供修改建议，并判断是否可以合入。 | 若干条高质量的已合入 PR，若干条高质量的PR Review (或者Issue），2 名 Committer 的认可。获取流程：在代码仓库下按照 模板 提 Issue，取得 2 名 Committer 的正向评价后，由 Maintainer 开通权限。 | 代码仓库的 write 权限。PR 及 ISSUE 的分派与编辑。            |
| Maintainer  | 领导飞桨框架的发展方向与技术规划。                     | 在飞桨开源社区内获取广泛且充分的影响力。                     | 同上                                                         |

## Anyone

任何拥有 GitHub 账号的人都可以对 PaddlePaddle 作出贡献，飞桨开源社区欢迎所有新的贡献者。

任何人都可以：

- 在 PaddlePaddle 下的所有代码仓库报告 Issue，如果你第一次接触，欢迎查看 [【必读】如何正确的提出一个 Issue](https://github.com/PaddlePaddle/Paddle/issues/41281)；
- 对他人的 Issue & PR 提出反馈信息，就你感兴趣的话题进行讨论与学习；
- star & fork 代码仓库，拉取代码到本地并进行使用与修改。

在 [签署了 CLA](https://cla-assistant.io/PaddlePaddle/Paddle) 之后，任何人还可以向代码仓库提交 PR，具体操作欢迎参考 [飞桨官网-代码贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)。

## Triager

该角色，会获得代码仓库的triage权限，用来管理github当中的实际项目相关的ISSUE或者PR。多用在实际的项目当中，对涉及到的ISSUE或者PR进行分类与指派。请联系相应的项目接口人来获取。

## Contributor

成为飞桨的 Contributor，你可以通过：

- 修复日常 Issue 中发现的框架 bug；
- 认领 [PFCC-RoadMap](https://github.com/PaddlePaddle/Paddle/issues/42571)，自选并完成框架任务；
- 参与 [飞桨黑客松](https://github.com/PaddlePaddle/Paddle/issues/42410)，完成指定框架开发任务；
- 基于个人需求，为飞桨框架新增功能。

为 [Paddle](https://github.com/PaddlePaddle/Paddle) 仓库提交 PR 并被 merge 的开发者会自动成为 Contributor，获得飞桨社区开发者的认可与飞桨开源贡献证书。如果你有意愿长期贡献，可以发送邮件至 [ext_paddle_oss@baidu.com](mailto:ext_paddle_oss@baidu.com)，会收到邀请加入 [PFCC](https://github.com/PaddlePaddle/community/tree/master/pfcc)。

## Committer

在完成若干高质量的 PR，并高质量的完成了若干他人 PR 的 review 工作（提出合理的修改意见，并在他人完成后给出 approve）后，你可以以 Issue 的形式发起 Committer 身份的申请。请注意：

1. 你需要找到 2 名 Committer 为你的申请做担保，他们需要是你的 PR Reviewer，并对你提交的 PR 质量进行把控；
2. 不同代码仓库下的权限是分开的，因此你需要在申请 Committer 身份的代码仓库下，参考附录中的 **Committer 身份申请模板** 提交一个 Issue；
3. 告知你的担保人并邀请他们在 Issue 下回复，对你的申请做出正向的评价。获得 2 名担保人的评价与同意后，代码仓库的 Maintainer 会为你添加相应的权限；如果你的请求未被通过，你依旧会获得反馈，当处理完这些反馈后，你可以重新发起申请。

成为飞桨的 Committer 后，你会拥有：

- 社区 Issue & PR 的分派与判别权限，你可以为 Issue & PR 添加标签、分配负责人、关闭与重新打开；
- 社区 PR 的评审权限，你可以对他人 PR 提出有约束力的反馈信息，修改他人 PR 内容，合入通过全部 CI 的 PR。

具体权限内容可参考 [GitHub 社区 write 权限](https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-roles-for-an-organization)。

## Maintainer

领导飞桨框架的发展方向与技术规划。

## 附录A
代码仓库的各项操作权限说明请参考：[Manage access to repositories](https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-roles-for-an-organization)。

## 附录B

### Committer 身份申请模板

标题：【社区治理】XXX（GitHub ID） 发起 Committer 身份申请

正文：

#### 基本信息

| 申请人 GitHub ID | **Paddle Repo 整体 merge PR 数** | **Paddle Repo 整体 review PR 数** | **Paddle Repo 整体报告 Issue 数** |
| ---------------- | -------------------------------- | --------------------------------- | --------------------------------- |
|                  |                                  |                                   |                                   |

#### 社区贡献概况

简述你在所申请的代码仓库下的行为，你所参与的社区贡献，以及你申请 Commiter 身份的理由

##### 高质量 merge PR 展示

| PR 号          | PR 标题                        | PR 简介            | Reviewer                      |
| -------------- | ------------------------------ | ------------------ | ----------------------------- |
| （可跳转链接） | （标题需要清晰展示 PR 的工作） | （主要内容及价值） | （主要的 Reviewer GitHub ID） |
|                |                                |                    |                               |
|                |                                |                    |                               |

##### 高质量 review PR 展示

| PR 号          | PR 标题                        | PR 简介            | review 详情                            |
| -------------- | ------------------------------ | ------------------ | -------------------------------------- |
| （可跳转链接） | （标题需要清晰展示 PR 的工作） | （主要内容及价值） | （简单介绍你 review 的内容及有效建议） |
|                |                                |                    |                                        |

##### 代表性 Issue 展示

| Issue 号       | Issue 标题                            | Issue 简介               | 解决情况                              |
| -------------- | ------------------------------------- | ------------------------ | ------------------------------------- |
| （可跳转链接） | （标题需要清晰展示 Issue 的主要表现） | （主要内容及暴露的问题） | （你所了解到的 Issue 的后续解决情况） |
|                |                                       |                          |                                       |

#### 担保人意见

你需要在这里 @ 至少 2 名 Committer 为你的申请做担保（他们需要是你的 PR Reviewer），并邀请他们在评论区回复。
