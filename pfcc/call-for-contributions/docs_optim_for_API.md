# API文档体验优化

> This project will be mentored by [@Ligoml](https://github.com/ligoml) and [@momozi1996](https://github.com/momozi1996) 

## 1. 背景与意义

飞桨框架为用户提供中英双语的API文档，英文文档以docstring的方式维护在[Paddle](https://github.com/PaddlePaddle/Paddle)仓库，中文文档以rst文档的方式维护在[docs](https://github.com/PaddlePaddle/docs)仓库，最终统一展示在飞桨官网[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html)目录下。在维护双语API文档的过程中，出现许多细节上的问题，严重影响到中英双语的API文档使用体验。

本项目发源于飞桨框架体验评估，在[第一期飞桨框架体验评测](https://github.com/PaddlePaddle/Paddle/issues/38865)中，**API文档与目录大纲评测任务**从抽样选取了100组API中英文文档，从结构、内容、格式、红线和主观评价5个维度进行打分，总得分仅为64.8分。为了摸底飞桨全量API文档中存在的问题，社区开发者组成[文档工作小组](https://shimo.im/sheets/e1Az48XnO4t6g7qW/akF3x/)，经过8轮评估，共计**完成1005组API中英文文档评估，产生434个issue，优化414组API中英文文档**。在[第六期飞桨框架体验评测](https://github.com/PaddlePaddle/Paddle/issues/45962)中，完成优化的API文档抽样后得分86.3分，符合预期。在全量评估过程中，我们还发起了[飞桨 API 文档的不规范之处及其规范化方案汇总](https://github.com/PaddlePaddle/docs/discussions/5243)、[pylint docstring checker 不工作原因及可能的解决方案](https://github.com/PaddlePaddle/Paddle/issues/47821)等高价值的讨论内容。

API文档是用户使用飞桨的第一手册，它就像一本「字典」，对用户学习和使用飞桨框架发挥着至关重要的作用。飞桨团队十分重视API文档的建设和维护，不仅提供英文文档，还同步提供中文版本，更方便国内用户的查阅和使用。然人力有时尽，而问题无穷尽，因此发布本计划，呼吁更多有兴趣参与的开发者同我们一起，优化飞桨API文档体验，为飞桨的用户提供更舒适的文档阅读体验。

## 2. 目标

本项目的目标是完成飞桨全量API文档的修复，并为飞桨API文档贡献1-2个建设性成果。

## 3. 工作及执行思路

API文档体验优化项目分两条线同步推进：

- 一条线是评估问题修复，即对文档工作小组未完成的优化工作的继续。这部分工作很适合作为你的第一次开源贡献尝试，任务明确且难度较低，主要用来练习git操作和熟悉飞桨贡献流程；
- 一条线是建设性问题讨论与优化，目前正在进行的项目是**飞桨API文档写作规范手册制定**。这部分工作由社区开发者自发启动，例如更新pre-commit中的pylint使得docstring更加规范、使用docstring2yaml2rst的方式使中英文API文档更便于对齐等，我们非常欢迎你来发起一些建设性的议题并贡献自己的智慧，优化飞桨API文档。



参考链接：

- [飞桨官网-贡献指南-代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/code_contributing_path_cn.html)
- [飞桨官网-贡献指南-新增API开发&提交流程-API文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html)
- 参考PR：[Paddle#42942](https://github.com/PaddlePaddle/Paddle/pull/42942)；[docs#4850](https://github.com/PaddlePaddle/docs/pull/4850)

参与讨论：

- [飞桨 API 文档的不规范之处及其规范化方案汇总](https://github.com/PaddlePaddle/docs/discussions/5243)
- [pylint docstring checker 不工作原因及可能的解决方案](https://github.com/PaddlePaddle/Paddle/issues/47821)
