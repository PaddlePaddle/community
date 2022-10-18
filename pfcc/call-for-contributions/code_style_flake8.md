# flake8 代码风格检查工具的引入

> This project will be mentored by [@luotao1](http://github.com/luotao1)
>
> Tracking issue: [PaddlePaddle/Paddle#46039](https://github.com/PaddlePaddle/Paddle/issues/46039)

## 背景

Paddle 目前使用的代码风格检查工具及 hooks 版本较低，导致开发者在提交代码时会碰到一些问题，加强代码规范检查，有利于提高 Paddle 代码质量，增强代码的可读性。

内部已经完成了 pre-commit、pylint、remove-crlf、cpplint、cmake-format、yapf、cmakelint、cmake-format 8 大检查工具的升级，还剩下两大检查工具 clang-tidy 和 flake8 还未引入，期待社区开发者主导完成。

## flake8 调研情况

【新增】flake8：4.0.1 （总错误数：45784，需修改文件数：2773）

- 原因：4.0.1 于 2021.10 发布，3.8.2 于 2020.5 发布，都比较稳定。测试过 4.0.1 和 3.8.2 的结果，相差很小
- 收益：规范项目代码风格，帮助 RD 更快地发现语法错误和少许程序逻辑错误
- 风险：

  - 根据历史全量检测结果 flake8 表，可知待修改文件和错误总数很多（2700+）。先用 yapf 对代码进行格式化，再用 flake8 检测，总报错数减少很少（45000+ –> 43000+）。主要原因是 yapf 作为 formatter 只能在不修改代码逻辑的前提下进行格式上的修改，因此对于语法上的问题和代码逻辑上的问题都是无法修改的。另外，由于 yapf 本身格式化力度一般，因此一些常见的代码风格问题 yapf 也是无法自动修复的。
  - flake8 只检查但不能自动修复，对部分存量较大的问题，需要借助其他自动化工具进行修复。
    - 如 [PR#44474](https://github.com/PaddlePaddle/Paddle/pull/44474) 可借助 precommit 自动删除结尾多余空格，但因为存量没有修完，这个 PR 就没有合入。
    - 有一些用来修复的自动化的工具，比如 [autoflake](https://github.com/PyCQA/autoflake)，也可以尝试来使用。
  - 取数量前三的错误列举如下表：

    | 错误类型                            | 错误代码 | 数量  | 含义             |
    | ----------------------------------- | -------- | ----- | ---------------- |
    | line too long (100 > 80 characters) | E501     | 20993 | 列宽超限         |
    | xx imported but unused              | F401     | 6445  | 模块引入但未使用 |
    | trailing whitespace                 | W291     | 5561  | 结尾有多余空格   |

- 检查步骤：参考文档 https://flake8.pycqa.org/en/latest/

```shell
pip install flake8==4.0.1
flake8 path/to/code/to/check.py
# 或者
flake8 path/to/code/
```

## 可行性分析和规划排期

1. 因为存量较大，需要进行存量修复的可行性分析（本任务最有难度的地方），并制定如何修复的步骤
   - 将检查工具的检测项逐一打开，并逐步修复对应的存量问题。如可以先解决 flake8 中结尾有多余空格的问题。
   - 部分存量较大的问题，需要有自动化工具进行修复。
2. 格式检查工具升级和存量修复，可以参考 [code format check upgrade](https://github.com/PaddlePaddle/Paddle/search?q=code%20format%20check%20upgrade&type=commits) 已有工作
3. 仅格式检查工具同步到 release 分支，参考 [PR43732](http://agroup.baidu.com/paddle-ci/md/article/2https://github.com/PaddlePaddle/Paddle/pull/43732)
4. 将检查工具升级同步到 CI 镜像中，避免重复安装下载，参考 [PR43534](https://github.com/PaddlePaddle/Paddle/pull/43534)
5. 代码风格检查指南文档，[docs#4933](https://github.com/PaddlePaddle/docs/pull/4933)
