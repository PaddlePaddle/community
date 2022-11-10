# Call for Contributions

| **[Guide to Call for Contributions](./guide-to-call-for-contribution.md)** | **[Call For Contribution 指引](./guide-to-call-for-contribution_cn.md)** |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

为了让大家能深入地了解飞桨、在飞桨收获更多成长、解决更有挑战性的问题，飞桨团队计划将正在开展的一些重点工作和技术方向陆续发布。
每个技术方向都会有工程师支持，和该方向中的同学一起确定目标、规划和分工，希望 PFCC 的成员能逐渐成为方向骨干甚至是带头人，带领更多人一起开发。

- [单测报错信息优化](code_style_improvement_for_unittest.md)【已完成】
- [编译 warning 的消除](code_style_compiler_warning.md)【进行中】
- [flake8 代码风格检查工具的引入](code_style_flake8.md)【进行中】
- [clang-tidy 代码风格检查工具的引入](code_style_clang_tidy.md)
- [Python 2.7 相关代码退场](legacy_python2.md)【进行中】
- [Type Hint类型注释](type_hint.md)【进行中】
- [建设更多的Tutorial](Call_For_Tutorials.md)【进行中】
- [API文档体验优化](docs_optim_for_API.md)【进行中】
- [PHI算子库独立编译](decoupled_phi_from_fluid.md)
- [DenseTensor 概念统一](tensor_concept_unification.md)
- [CINN基础算子开发](CINN_base_operator.md)

## Project Ideas

一些在社区发现的可以进行贡献的想法，先简单的记录在这里。需要先把这些想法明确成社区的项目描述，来方便开展具体的开源贡献项目。

#### IDEA：改进飞桨框架的logging系统

飞桨框架在C++层，python层的多个模块中会产生日志，以进行信息提示，或者告警。这些日志产生的方式（例如，C++层和python层没有统一，有些日志甚至在用`print`打印，在python层甚至有多个`get_logger`的定义）、日志的分级（哪些属于warning，哪些属于information，等）、日志的清晰程度，等多方面都有值得改进的地方。

- 社区中的相关issue：[#46622](https://github.com/PaddlePaddle/Paddle/issues/46622)、[#46554](https://github.com/PaddlePaddle/Paddle/pull/46554#pullrequestreview-1122960171)、[#44857](https://github.com/PaddlePaddle/Paddle/pull/44857)、[45756](https://github.com/PaddlePaddle/Paddle/issues/45756)、[#43610](https://github.com/PaddlePaddle/Paddle/issues/43610)
- 可参考的材料：[pytorch/rfcs/RFC-0026-logging-system.md](https://github.com/pytorch/rfcs/blob/4b75803bf90c16b0120787fa0557bfe79ace1ef3/RFC-0026-logging-system.md)
- [Paddle报错信息文案书写规范](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-Error-Message-Writing-Specification)

#### IDEA：iScan 流水线退场

[PR-CI-iScan-C](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/builds/18485?module=PaddlePaddle/Paddle&pipeline=PR-CI-iScan-C&branch=branches)、[PR-CI-iScan-Python](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/builds/18500?module=PaddlePaddle/Paddle&pipeline=PR-CI-iScan-Python&branch=branches) 是分别用 `cppcheck==1.8.3` 和 `pylint==1.9.4` 两个工具做代码的静态分析和检查。我们近期会 review 一下这两条流水线的必要性和替代这些检查的方式。
