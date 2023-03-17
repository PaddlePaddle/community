# Call for Contributions

| **[Guide to Call for Contributions](./guide-to-call-for-contribution.md)** | **[Call For Contribution 指引](./guide-to-call-for-contribution_cn.md)** |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

为了让大家能深入地了解飞桨、在飞桨收获更多成长、解决更有挑战性的问题，飞桨团队计划将正在开展的一些重点工作和技术方向陆续发布。
每个技术方向都会有工程师支持，和该方向中的同学一起确定目标、规划和分工，希望 PFCC 的成员能逐渐成为方向骨干甚至是带头人，带领更多人一起开发。

- 详细项目进展：https://github.com/orgs/PaddlePaddle/projects/7 
- 已结项 & 即将结项的内容： https://github.com/PaddlePaddle/Paddle/issues/48019#issuecomment-1457650067 
- 进行中的项目： [飞桨快乐开源活动表单->命题任务](https://github.com/PaddlePaddle/Paddle/issues/48019)


## 飞桨线上开发环境——AI Studio
AI Studio是基于百度深度学习开源平台飞桨的人工智能学习与实训社区，为开发者免费提供功能强大的线上训练环境、云端超强GPU算力及存储资源。云集200万AI开发者共同学习、交流、竞技与成长。每日提供8小时免费GPU算力供飞桨开发者学习实践。

为支持开发者完成框架开发任务，AI Studio推出「框架开发任务」，为开发者提供飞桨镜像环境、在线 IDE 与专属 GPU 算力。你可以在这里便捷地从 GitHub 拉取代码、基于飞桨框架开发并参与开源共建，飞桨团队愿与你一起执桨破浪，让深度学习技术的创新与应用更简单。
<img width="500" alt="image" src="https://user-images.githubusercontent.com/39876205/201044617-2dbcb752-42c1-40f7-b634-2e4c776b55f9.png">

一个使用case：[利用 AI Studio 完成 Paddle 编译](https://aistudio.baidu.com/aistudio/projectdetail/4572885)

申请方式：
发邮件到 ext_paddle_oss@baidu.com，附上 GitHub username 和 AI Studio uid 并说明用途。

AI Studio uid 参考：
<img width="453" alt="image" src="https://user-images.githubusercontent.com/39876205/201087539-4f1cecb1-8261-46e6-b425-13d21cceb45b.png">

传送门：https://aistudio.baidu.com/aistudio/index


## Project Ideas

一些在社区发现的可以进行贡献的想法，先简单的记录在这里。需要先把这些想法明确成社区的项目描述，来方便开展具体的开源贡献项目。

#### IDEA：改进飞桨框架的logging系统

飞桨框架在C++层，python层的多个模块中会产生日志，以进行信息提示，或者告警。这些日志产生的方式（例如，C++层和python层没有统一，有些日志甚至在用`print`打印，在python层甚至有多个`get_logger`的定义）、日志的分级（哪些属于warning，哪些属于information，等）、日志的清晰程度，等多方面都有值得改进的地方。

- 社区中的相关issue：[#46622](https://github.com/PaddlePaddle/Paddle/issues/46622)、[#46554](https://github.com/PaddlePaddle/Paddle/pull/46554#pullrequestreview-1122960171)、[#44857](https://github.com/PaddlePaddle/Paddle/pull/44857)、[45756](https://github.com/PaddlePaddle/Paddle/issues/45756)、[#43610](https://github.com/PaddlePaddle/Paddle/issues/43610)
- 可参考的材料：[pytorch/rfcs/RFC-0026-logging-system.md](https://github.com/pytorch/rfcs/blob/4b75803bf90c16b0120787fa0557bfe79ace1ef3/RFC-0026-logging-system.md)
- [Paddle报错信息文案书写规范](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-Error-Message-Writing-Specification)
