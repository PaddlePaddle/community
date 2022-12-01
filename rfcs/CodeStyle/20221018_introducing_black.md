# python 代码检查工具 yapf 升级 black 方案
| 任务名称     | python 代码检查工具 yapf 升级 black 方案 |
| ------------ | -------------------------------- |
| 提交作者     | @luotao1 @SigureMo        |
| 提交时间     | 2022-10-18                       |
| 版本号       | v1.0                             |
| 依赖飞桨版本 | develop                          |
| 文件名       | 20221018_introducing_black.md   |

## 一、概要
### 1、相关背景
Paddle 在 Q2 完成了《代码检查机制完善和工具升级一期项目》，整体代码质量和可读性有了较大提升，升级版本后的自动化代码风格检查，也不再成为贡献者提交代码时的痛点。但仍然需要持续地完善，一期的后续规划是引入 clang-tidy 、flake8 等检查工具，检查 C++ 和 Python 代码的静态逻辑错误。

flake8 作为一个 python linter 不提供自动修复的能力。当前 flake8 包含大量的格式问题需要修复，而格式化工具 yapf 并不能修复一些 PEP 8 中描述的格式规范问题，因此可能需要考虑使用其他格式化工具进行格式化。

black 可以兼顾 PEP 8 和格式上的统一，相比于 yapf 格式化力度更高，可以自动修复较多的格式问题，大大减少开发者手动解决 flake8 问题的频率。将yapf 升级为 black 是有利于 flake8 检查工具的集成。

### 2、功能目标

 将python代码检查工具yapf升级为black，有助于 flake8 检查工具的集成。

### 3、方案要点
- yapf 升级为 black 后，对集成 flake8 带来的好处。
- 如何进行存量代码的修复和增量代码的拦截。

## 二、意义
- 当前 flake8 包含大量的格式问题需要修复：flake8 默认启用的三个工具（pycodestyle、pyflakes、mccabe）共包含了一共 132 个错误子项，**Paddle 中存在问题的错误子项共 62 个，涉及4w多行python代码。使用 black 替换 yapf 可以一次性解决26个子项，减少7k行代码**，大大减少开发者手动解决 flake8 问题的频率，有助于 flake8 检查工具的集成。
- black 在生态上比 yapf 更好，Github stars 数、使用量、开源工具的支持都远超 yapf ，有助于后续加入更多的检查工具。

## 三、业内调研
### Paddle引入flake8初始状态
flake8 默认启用的三个工具（pycodestyle、pyflakes、mccabe）共包含了一共 132 个子项。Paddle 中存在问题的子项共 62 个，涉及代码约4w多行。

-  错误码参考：
   - E、W：[PyCodeStyle Error code](https://pycodestyle.pycqa.org/en/latest/intro.html#error-codes)
   - F：[flake8 Error code](https://flake8.pycqa.org/en/latest/user/error-codes.html)
- 62个错误码子项清单见：[flake8 tracking issue](https://github.com/PaddlePaddle/Paddle/issues/46039#issue-1372620386)
- 推进方式：采用在 [.flake8](https://github.com/PaddlePaddle/Paddle/blob/develop/.flake8) 配置文件中 ignore 现存的错误码来推进，修复一类错误就去掉一个错误码

### yapf & black 在Github的使用情况

flake8 作为一个 linter 不提供自动修复的能力，而目前采用的格式化工具 yapf 并不能修复一些 PEP 8 中描述的格式规范问题，因此可能需要考虑使用其他格式化工具进行格式化。

目前主流的格式化工具包含 yapf、black 和 autopep8，三者对比如下：

|-|autopep8|yapf|black|
|-|-|-|-|
|GitHub Stars|4.1k|12.8k|29.1k|
|GitHub Used by|132k|36k|178k|
|速度|非常慢|慢（Paddle 全量格式化需 13min）|快（同全量格式化 1min）|
|格式化力度|主要修复 PEP 8 的问题，但太过于关注 PEP 8 导致可能改变代码语义，不算「合格」的格式化工具|不注重 PEP 8，主要注重代码格式的统一|兼顾 PEP 8 和代码格式的统一|
|可配置性|可选择忽略的 Error Code，修复的激进等级等|各式各样的配置，可以设置各式各样的风格，有点类似 clang-format|基本没有配置项|

此外，有很多工具对 black 支持/兼容性更好，比如：

-  [isort](https://pycqa.github.io/isort/docs/configuration/black_compatibility.html) （对头文件的引入顺序做排序）专门有个 profile 就是针对 black 的，可以保证与 black 没有冲突，而 yapf 则很难与 isort 一起工作（见 [pull/46475#issuecomment](https://github.com/PaddlePaddle/Paddle/pull/46475#issuecomment-1257246946) ）
-  [blacken-docs](https://github.com/asottile/blacken-docs) （docstring、rst 文档示例代码格式化工具）也是集成了 black。

结论：

- autopep8 可能改变代码语义，不安全；
- **black 的 stars 数、使用量、速度、开源工具的支持都远超 yapf**，也有助于后续加入更多的集成工具

### 业内情况

- pytorch 的 python 代码检查工具为：`flake8==3.8.2, black==22.3.0, mypy==0.950`，没有使用 yapf
- keras 的 python [代码检查工具](https://github.com/keras-team/keras/blob/master/requirements.txt)为：`flake8==4.0.1, black==22.3.0, isort==5.10.1` 
- tensorflow 没有使用 flake8、yapf 和 black，只是推荐手动 yapf 格式化。见issue： [how to auto format python code](https://github.com/tensorflow/tensorflow/issues/50304)，[Some check that we have enabled in .pylintrc are silently not executed anymore](https://github.com/tensorflow/tensorflow/issues/55442) 
### yapf & black 在Paddle的错误码数量对比

统计说明：

- 共统计59个子项：1）初始62个子项中，排除已经解决的E999和trailing whitespace（W191和W293）
- 由于fluid将在2.5版本移除，排除 `python/paddle/fluid`目录，但包含 `python/paddle/fluid/tests`目录
- 表格中的数字是：格式化后剩余的不规范的代码行数

```
# yapf/black/autopep8 所在列是：对应工具格式化后剩余的不规范的代码行数
code    yapf    black   diff    percent autopep8        diff    percent

Type: E (yapf 26436 -> black 20023 & autopep8 19200) 
E101    11      10      -1      -9%     10      -1      -9%
E121    8       0       -8      -100%   0       -8      -100%
E122    81      0       -81     -100%   0       -81     -100%
E123    12      0       -12     -100%   0       -12     -100%
E125    168     0       -168    -100%   0       -168    -100%
E126    723     0       -723    -100%   0       -723    -100%
E127    140     0       -140    -100%   1       -139    -99%
E128    207     0       -207    -100%   0       -207    -100%
E129    9       0       -9      -100%   0       -9      -100%
E131    45      0       -45     -100%   10      -35     -78%
E201    29      0       -29     -100%   0       -29     -100%
E202    11      0       -11     -100%   0       -11     -100%
E225    61      0       -61     -100%   0       -61     -100%
E226    93      0       -93     -100%   0       -93     -100%
E228    3       0       -3      -100%   0       -3      -100%
E231    60      3       -57     -95%    0       -60     -100%
E241    2       0       -2      -100%   0       -2      -100%
E251    109     0       -109    -100%   0       -109    -100%
E261    11      0       -11     -100%   0       -11     -100%
E262    238     17      -221    -93%    0       -238    -100%
E265    925     48      -877    -95%    218     -707    -76%
E266    116     116     0       0%      57      -59     -51%
E271    4       0       -4      -100%   0       -4      -100%
E272    1       0       -1      -100%   0       -1      -100%
E301    7       0       -7      -100%   5       -2      -29%
E302    3       0       -3      -100%   0       -3      -100%
E303    7       0       -7      -100%   2       -5      -71%
E305    2       0       -2      -100%   0       -2      -100%
E306    1       1       0       0%      0       -1      -100%
E401    19      19      0       0%      0       -19     -100%
E402    2666    2666    0       0%      341     -2325   -87%
E501    19252   16239   -3013   -16%    17714   -1538   -8%
E502    400     0       -400    -100%   0       -400    -100%
E701    108     0       -108    -100%   0       -108    -100%
E711    166     166     0       0%      166     0       0%
E712    340     340     0       0%      340     0       0%
E713    22      22      0       0%      22      0       0%
E714    4       4       0       0%      4       0       0%
E721    8       8       0       0%      8       0       0%
E722    149     149     0       0%      149     0       0%
E731    62      62      0       0%      0       -62     -100%
E741    153     153     0       0%      153     0       0%

Type: F (yapf 9895 -> black 9913 & autopep8 9858)
F401    6750    6768    18      0%      6739    -11     -0%
F402    1       1       0       0%      1       0       0%
F403    57      57      0       0%      57      0       0%
F405    556     556     0       0%      556     0       0%
F522    1       1       0       0%      1       0       0%
F524    1       1       0       0%      1       0       0%
F541    33      33      0       0%      33      0       0%
F601    7       7       0       0%      7       0       0%
F631    2       2       0       0%      2       0       0%
F632    18      18      0       0%      18      0       0%
F811    177     177     0       0%      151     -26     -15%
F821    88      88      0       0%      88      0       0%
F841    2204    2204    0       0%      2204    0       0%

Type: W (yapf 1135 -> black 185 & autopep8 1482)
W191    11      10      -1      -9%     10      -1      -9%
W504    949     0       -949    -100%   1297    348     37%
W601    3       3       0       0%      3       0       0%
W605    172     172     0       0%      172     0       0%
```

总结：格式化后剩余的不规范的代码行数 & 错误码数量 对比

|-|yapf|black|备注|
|-|-|-|-|
|E|26436行，42子项|20023行，17子项| black 优势很明显，减少6k多行代码，25个子项|
|F|9895行，13子项|9913行，13子项|F 可能会影响语义，不应该被格式化，两者效果类似|
|W|1135行，4子项|185行，3子项| black 优势很明显，减少1k行代码，1个子项|

black 相比于 yapf 格式化力度更高，可以自动修复较多的格式问题，大大减少开发者手动解决 flake8 问题的频率，有利于 flake8 检查工具的集成。

## 四、设计思路与实现方案
### 1、主体设计思路与折衷
**增量监控**：

- 替换配置文件，将`.style.yapf`改成`pyproject.toml`。其中`skip-string-normalization`参数：black 原来是强制字符串用双引号的，后来做出了妥协，这个参数就是允许保持原来的单引号。

<img width="766" alt="image" src="https://user-images.githubusercontent.com/6836917/196332356-90a8ee55-0387-409f-b5ee-77ea8c149a0e.png">

- 在 `.pre-commit-config.yaml` 中，将 yapf 格式工具替换为 black。该方式会在 pre-commit 时自动触发，同时适用于 CI 和本地机器，且无需用户额外操作。（动转静的两个验证行号的单测，需要过滤）

![image](https://user-images.githubusercontent.com/6836917/196331579-5f25ea96-63c0-45a5-9e85-352f049eb7c3.png)

**存量修复**：

- 本地安装 black 进行自动化修复

```bash
pip install black
black .

git checkout develop -- 'python/paddle/fluid/tests/unittests/dygraph_to_static/test_error.py'
git checkout develop -- 'python/paddle/fluid/tests/unittests/dygraph_to_static/test_origin_info.py'
```
- 在`.flake8`中，将 black 工具修复过的26个子项错误码去除。

### 2、 关键技术点/子模块设计与实现方案

**可行性验证**：[pull/46014](https://github.com/PaddlePaddle/Paddle/pull/46014) 过滤了动转静的两个验证行号的单测（`dygraph_to_static/test_error.py`和`dygraph_to_static/test_origin_info.py`）后，接近 24 万行 change 不经任何人工调整可直接通过单测。

**推进方式**：按照《代码检查机制完善和工具升级一期项目》的推进方式。

- 由于该 PR 较大，引起其他PR冲突的概率很高，为了减少对其他PR的影响，会提前通知大家该PR合入的时间点。
- develop 分支修复存量问题，并将配置从 yapf 改成 black，即合入本 PR。release/2.4 分支只改配置，不修存量问题。

## 五、影响和风险总结
### 对用户的影响
yapf 升级 black 是对开发者的影响

- black 可以自动修复较多的格式问题，大大减少开发者手动解决 flake8 问题的频率。可以让开发者专注于编写代码逻辑，而不是浪费时间在调整代码格式上。
- 在正式升级后初次提交代码时，pre-commit需要初始化 black 检查环节，等待的时间会稍微长一点（大概1min），后续提交代码不受影响。
- 提交代码时 black 检查时间取决于修改文件的数量和类型。最长不超过1min（全量代码格式化的时间），在完成存量修复后，该时间会大大缩短。

### 风险
只影响格式，CI 通过即可。

## 附件及参考资料
 1. [flake8 tracking issue](https://github.com/PaddlePaddle/Paddle/issues/46039)
 1. [use black instead of yapf 测试代码](https://github.com/PaddlePaddle/Paddle/pull/46014)
