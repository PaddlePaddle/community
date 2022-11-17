# Python 引入 import 区域自动重排工具 isort 方案

| 任务名称     | Python 引入 import 区域自动重排工具 isort 方案 |
| ------------ | ---------------------------------------------- |
| 提交作者     | @SigureMo                                      |
| 提交时间     | 2022-11-11                                     |
| 版本号       | v1.0                                           |
| 依赖飞桨版本 | develop                                        |
| 文件名       | 20221111_introducing_isort.md                  |

# 一、概述

## 1、相关背景

Paddle 在近期已经连续引入了 Flake8、Black 工具，Python 端代码风格已经得到了极大的改善，在可读性和可维护性上都有了很大的提升。

但是目前 Paddle 的 import 区域的代码仍然不是很规范，虽然在 [Flake8 引入计划](https://github.com/PaddlePaddle/Paddle/issues/46039)过程中 F401 已经将大量未使用的 import 都移除了，但 import 部分仍然存在顺序不规范的问题，这需要我们引入新的修复工具来实现这一点。

[isort](https://pycqa.github.io/isort/index.html) 是一个对文件中 imports 部分进行重排的工具，会依照 [PEP 8 - imports](https://peps.python.org/pep-0008/#imports) 规范以下规则对 imports 进行重排

> Imports should be grouped in the following order:
>
> 1. Standard library imports.
> 2. Related third party imports.
> 3. Local application/library specific imports.

这可以极大改善 imports 部分的代码风格，使得开发人员更容易了解模块之间的依赖关系，且能够自动去除 imports 部分重复的导入。

此外，isort [对 Black 有着非常好的支持](https://pycqa.github.io/isort/docs/configuration/black_compatibility.html)，我们已经在 [PaddlePaddle/Paddle#46014](https://github.com/PaddlePaddle/Paddle/pull/46014) 引入了 Black 并对全量代码格式化，因此我们可以在此基础上引入 isort 来对 import 区域进行重排。

下面是一个文件使用 isort 重排前后的对比：

![python/paddle/fluid/tests/unittests/asp/test_fleet_with_asp_sharding.py diff view](https://user-images.githubusercontent.com/38436475/201512682-33e9111d-ac36-41b4-bc25-cceba940b1a3.png)

## 2、功能目标

引入 isort 工具，对 Paddle 的 import 区域进行重排。

isort 格式化文件范围为 Paddle 全量 Python 代码，但排除以下目录：

- `python/paddle/fluid` 目录，但不包含其中的单测文件（即单测文件仍然需要格式化）
- 第三方库
- 自动生成的文件
- NPU、MLU 相关即将移动到 [PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice) 的代码

## 3、意义

- 保证 import 区域的顺序规范，提高代码可读性
- 避免开发者考虑 import 区域的顺序手动排序，提高开发效率
- 完全去除 Flake8 E401（multiple imports on one line）错误码（存量 9 -> 0），由于其自动修复功能，可以自动修复全部增量
- 消除部分 Flake8 F811（redefinition of unused name from line N）错误码（存量 123 -> 46），方便 F811 错误码的进一步修复

# 二、飞桨现状

在 C++ 端，Paddle 已经有 clang-format 同时对代码进行格式化和对头文件进行排序。

在 Python 端，Paddle 目前已经引入了 Flake8、Black 两个工具，isort 作为最受欢迎的三大工具之一，目前尚未引入。

# 三、业内方案调研

Black、Flake8、isort 是目前 Python 社区最受欢迎的三个工具，深度学习社区也不例外，PyTorch、Keras 均使用了这三个工具：

- [PyTorch - `.isort.cfg`](https://github.com/pytorch/pytorch/blob/master/.isort.cfg)
- [Keras - `shell/format.sh`](https://github.com/keras-team/keras/blob/master/shell/format.sh)

TensorFlow 并没有使用这些工具，目前主要是推荐手动格式化，见 [how to auto format python code~](https://github.com/tensorflow/tensorflow/issues/50304)

isort 在开源社区非常受欢迎，截止至 2022.11.12，isort 在 GitHub 上 Star 数 5.3k，两倍于 Flake8 2.4k，使用数（Used by）为 280k，高于 Flake8 253K。

# 四、设计思路与实现方案

## 1、主体设计思路与折衷

由于 Paddle 在此之前并没有对 import 区域进行排序过，因此所有的文件都是开发者自觉手动排序的，真正符合规范的文件非常少，基本上所有 Python 文件都需要重排（3000+ 文件）。

对于近乎全量文件的格式化，我们已经有了两次经验，一次是 Flake8 F401 错误码的存量修复，另一次是引入 Black 时的全量格式化，下面对比两者所使用的方案的原因以及 isort 所需采取的折衷方案：

| 影响一次性修复的原因 | Flake8 F401 | Black | isort |
| - | - | - | - |
| CI 错误不容易排查 | 存在，修复 F401 问题很容易导致 API 变动等问题，一次性修复很难排查问题 | 几乎不存在，Black 的格式化可以保证代码运行时语义不变，除部分依赖于格式的代码外（动转静、读取 docstring 等）不会产生任何问题 | 存在但较少，少数依赖于 import 顺序的情况是可能产生问题的（如前一个 import 修改了全局状态，后一个 import 依赖于这个全局状态） |
| 一次性改动文件过多导致的 PR-CI-Coverage 流水线崩溃在参数传递处 | 不存在，通过 33 个 PR，对逐步细分的各个目录进行修复，每个 PR 改动文件都比较少 | 存在，因为一次性修改了 3000+ 文件，文件改动数过多 | 可通过拆分成 5 个以内低于 1000 个文件的 PR 避免本问题 |
| 影响面很大，需要专门找时间来 merge | 不存在，一方面每个 PR 改动文件都比较少，另一方面改动仅仅涉及 import 区域 | 存在，Black 格式化基本将所有文件都进行了改动，而且改动区域遍布于整个代码文件 | 同 F401，改动仅涉及 import 区域，影响较小 |

也就是说，isort 的引入采取两者的折衷，即分目录来做，但不必像 F401 那样分的过于细致，每个 PR 保持在修改 1000 左右数量的文件。

具体实施步骤如下：

- pre-commit 配置添加：

  ```yaml
  - repo: https://github.com/pycqa/isort
  rev: 5.10.1
  hooks:
      - id: isort
  ```

- `pyproject.toml` 配置添加

  ```toml
  [tool.isort]
  profile = "black"
  line_length = 80
  known_first_party = ["paddle"]
  skip = ["build", "__init__.py"]
  extend_skip_glob = [
      # These files do not need to be formatted,
      # see .flake8 for more details
      "python/paddle/fluid/[!t]**",
      "python/paddle/fluid/tra**",
      "*_pb2.py",
      "python/paddle/utils/gast/**",
      "python/paddle/fluid/tests/unittests/npu/**",
      "python/paddle/fluid/tests/unittests/mlu/**",

      # These files will be fixed in the future
      "cmake/**",
      "paddle/**",
      "r/**",
      "tools/**",
      "python/paddle/[!f]**",
      "python/paddle/fluid/tests/unittests/[k-z]**",
      "python/paddle/fluid/tests/unittests/dygraph_to_static/test_error.py",
  ]
  ```

> **Note**
>
> - 类似 Flake8 的 F401 引入，先在配置中 ignore 一部分，之后逐步移除 ignore 部分并修复
> - `profile`、`line_length` 选项用于兼容 Black
> - `known_first_party` 选项可以让 isort 在排序时将 `paddle` 识别为第一方模块而不是第三方模块

- 「全量」格式化

  ```bash
  isort .
  ```

由于配置里 ignore 了部分路径，因此只会格式化一部分路径，之后每个 PR 解开一部分路径之后 `isort .` 即可，针对有问题的先 ignore 掉在之后 PR 里解决

## 2、关键技术点/子模块设计与实现方案

### 可行性验证

[PaddlePaddle/Paddle#46475](https://github.com/PaddlePaddle/Paddle/pull/46475) 已经尝试了对第一步 900+ 文件进行格式化，能够通过全部单测。

### 推进方式

第一个 PR（即 [PaddlePaddle/Paddle#46475](https://github.com/PaddlePaddle/Paddle/pull/46475)）添加配置并修复部分文件，之后每个 PR 修复一部分文件（尽可能保持在 1000 左右，以避免 PR-CI-Coverage 流水线崩溃在参数传递处），利用大概 5 个以内 PR 修复绝大多数文件格式，之后利用一些 PR 专注于解决需要手动解决的问题。

对于直接格式化会出错的文件，需要通过添加[适当的注释](https://pycqa.github.io/isort/docs/configuration/action_comments.html)（如 `isort: skip`、`isort: skip-file`）来跳过格式化。

由于 fluid 目录预计在 release/2.5 中移除，目前也已经有很多 PR 在推进，因此该目录同 Flake8 采取相同方案，除单测外不进行修复。此外，NPU、MLU 目录也因为相同理由不进行修复。

由于 isort 的影响范围较小（仅仅为代码 import 区域），因此针对 release/2.4 分支无需将配置 Cherry-pick 过去，也不需要进行全量修复。

## 3、主要影响的模块接口变化

仅仅对 import 重排，不会对模块接口产生影响。

# 五、测试和验收的考量

通过 CI 各条流水线即可。

# 六、影响面

## 对用户的影响

用户对于框架内部 import 重排不会有任何感知，不会有任何影响。

## 对二次开发用户的影响

可以提高 import 区域代码可读性，方便开发者了解框架内部模块依赖关系，极大提高开发体验。

## 对框架架构的影响

在 pre-commit 工作流中引入 isort，因此在该 hook 引入后开发者首次 commit 需要稍微等一段时间用于初始化 isort 环境，后续提交代码不受影响。

## 对性能的影响

对性能不会产生任何影响。

## 对比业内深度学习框架的差距与优势的影响

与 Keras、PyTorch 同样采用 isort 工具，代码风格上可以与其保持一致。

## 其他风险

无。

# 七、排期规划

预计 1 ～ 2 周内完成，预计在 10 个以内 PR 即可完成。

# 名词解释

无。

# 附件及参考资料

1. [Flake8 tracking issue](https://github.com/PaddlePaddle/Paddle/issues/46039)
2. [isort 引入 part1 测试 PR](https://github.com/PaddlePaddle/Paddle/pull/46475)
