# 将 xdoctest 引入到飞桨框架工作流中

|领域 | 将 xdoctest 引入到飞桨框架工作流中                       |
|---|--------------------------------|
|提交作者 | megemini (柳顺)             |
|提交时间 | 2023-05-21                     |
|版本号 | V1.0                           |
|依赖飞桨版本 | paddlepaddle>2.4               |
|文件名 | 将 xdoctest 引入到飞桨框架工作流中.md |


# 一、概述
## 1、相关背景

在学习与使用 Paddle 框架的时候，python 源代码与官方文档中的 `Example`，也就是示例代码，是非常重要的学习与参考依据，示例代码的正确性其重要程度也就不言而喻了。[xdoctest](https://xdoctest.readthedocs.io/en/latest/) 是一个示例代码自动执行和检查工具，可以自动执行 Python docstring 中的示例代码，并对示例代码输出进行检查。

[中国软件开源创新大赛：飞桨框架任务挑战赛 赛题5](https://github.com/PaddlePaddle/Paddle/issues/53172#paddlepaddle05) 要求将 xdoctest 引入到飞桨框架的工作流中，利用 xdoctest 来自动检查示例代码运行正确，且与输出结果匹配，以确保示例代码输出的一致性，进一步提高飞桨框架示例代码的质量。

## 2、功能目标

### 2.1 文档建设

更新 Paddle 贡献指南中的文档： [开发 API Python 端](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/new_python_api_cn.html#api-python) 。以此规范后续代码的开发。

添加 `Example` 示例代码的写作要求，要求符合 `xdoctest` 中的 `google` style，即，在示例 `Example` 中代码需要以 `>>>` 开头。且保留目前的 `code-block` 提示，从而不影响中文文档的生成工作。

添加 `Example` 示例代码特殊情况的处理指导或链接，如使用 `xdoctest` 的 `# xdoctest +SKIP` 跳过随机生成的输出检查。

更新 Paddle API 文档中 `代码示例` 的页面显示特性，使其能够正确显示带有 `>>>` 的示例代码，并能够复制不带有 `>>>` 的示例代码，以方便用户运行示例代码。

### 2.2 CI 流水线

在 Paddle 的 CI 流水线中引入 `xdoctest`，使其在代码（python PR，新提交）提交 PR 时触发检查。`xdoctest` 的引入，需要不影响现有代码，只针对后续 PR。现有代码的示例，如不符合规范，则修改后重新提交 PR，并需要通过检查。

由于在代码的 PR 中进行了示例代码的检查，建议后续可取消 Paddle docs 中的示例代码检查。

`xdoctest` 的引入，保留目前的 `code-block` 提示，不影响现有文档生成流程。

### 2.3 更新现存代码

分批次更新现存的 Paddle 示例代码。以自动修改+人工复审的方式进行。

更新时需要在参考原有示例的同时，关注版本变化带来的输出结果变化。

## 3、意义

提高飞桨框架示例代码的质量。

# 二、飞桨现状

## 1、文档建设

目前 [开发 API Python 端](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/new_python_api_cn.html#api-python) 中，对于示例代码的写作方式，是通过一张代码截图说明的: ![代码截图](https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/dev_guides/api_contributing_guides/images/zeros_python_api.png?raw=true)

其中的代码示例对于普通 python 语句没有提示符要求，对于输出则使用 `#` 进行注释，这不符合 `xdoctest` 中的 `google` style要求。

建议使用 `截图+文字` 说明的方式更新此开发文档。

另外，中英文 [API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html#api) 可以复制代码示例，此特性需要保留。

## 2、CI 流水线

涉及此设计文档的 CI 流水线，主要包括:
- [Paddle 代码](https://github.com/PaddlePaddle/Paddle)
- [Paddle docs](https://github.com/PaddlePaddle/docs)

两个部分。

CI 的构建主要通过 [百度效率云 iPipe](https://xly.bce.baidu.com/paddlepaddle/paddle/_detail) 实现。

### 2.1 Paddle 代码

Paddle 代码的 CI 流水线相关工具放置在 [Paddle/tools/](https://github.com/PaddlePaddle/Paddle/tree/develop/tools) 目录下。

目前对于 python 示例代码的检查，主要通过 [Paddle/tools/codestyle/docstring_checker.py](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/codestyle/docstring_checker.py) 完成。

主要是检查 docstring 的格式，并无示例代码正确性检查。

### 2.2 Paddle docs

Paddle docs 的 CI 流水线相关工具放置在 [docs/ci_scripts/](https://github.com/PaddlePaddle/docs/tree/develop/ci_scripts) 目录下。

目前对于 python 示例代码的检查，主要通过 [docs/ci_scripts/chinese_samplecode_processor.py](https://github.com/PaddlePaddle/docs/blob/develop/ci_scripts/chinese_samplecode_processor.py) 完成。

相关工具将 python 代码中的示例提取出来，并单独封装为一个 python 文件，以执行此文件是否正确(是否报错)为依据，判断此示例代码的正确性。

``` python
...

def extract_sample_code(srcfile, status_all):
    filename = srcfile.name
    srcc = srcfile.read()
    srcfile.seek(0, 0)
    srcls = srcfile.readlines()
    srcls = remove_desc_code(
        srcls, filename
    )  # remove description info for samplecode
    status = []
    sample_code_begins = find_all(srcc, " code-block:: python")
    if len(sample_code_begins) == 0:
        status.append(-1)

    else:
        for i in range(0, len(srcls)):
            if srcls[i].find(".. code-block:: python") != -1:
                content = ""
                start = i

                blank_line = 1
                while srcls[start + blank_line].strip() == '':
                    blank_line += 1

                startindent = ""
                # remove indent error
                if srcls[start + blank_line].find("from") != -1:
                    startindent += srcls[start + blank_line][
                        : srcls[start + blank_line].find("from")
                    ]
                elif srcls[start + blank_line].find("import") != -1:
                    startindent += srcls[start + blank_line][
                        : srcls[start + blank_line].find("import")
                    ]
                else:
                    startindent += check_indent(srcls[start + blank_line])
                content += srcls[start + blank_line][len(startindent) :]
                for j in range(start + blank_line + 1, len(srcls)):
                    # planish a blank line
                    if (
                        not srcls[j].startswith(startindent)
                        and srcls[j] != '\n'
                    ):
                        break
                    if srcls[j].find(" code-block:: python") != -1:
                        break
                    content += srcls[j].replace(startindent, "", 1)
                status.append(run_sample_code(content, filename))

    status_all[filename] = status
    return status_all

def run_sample_code(content, filename):
    # three status ,-1:no sample code; 1: running error; 0:normal
    fname = (
        filename.split("/")[-1].replace("_cn", "").replace(".rst", "") + ".py"
    )
    tempf = open("temp/" + fname, 'w')
    content = "# -*- coding: utf-8 -*-\n" + content
    tempf.write(content)
    tempf.close()
    cmd = ["python", "temp/" + fname]

    subprc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _, error = subprc.communicate()
    err = "".join(error.decode(encoding='utf-8'))

    if subprc.returncode != 0:
        print("\nSample code error found in ", filename, ":\n")
        print(err)
        status = 1
    else:
        status = 0
    os.remove("temp/" + fname)
    return status

def test(file):
...

if not ci_pass:
    print("Mistakes found in sample codes.")
    exit(1)
else:
    print("Sample code check is successful!")
```

此方法存在较多问题，比如，无法验证代码与示例中的结果是否一致，无法处理本应报错的示例代码等。

## 3、现存代码

目前 Paddle 中 python 相关代码，主要放置在 [Paddle/python/paddle/](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle) 目录下。

其中包括 `2334` 个 python 文件，包括示例代码 `341` 段。(commit `8acbf10bd51026c0a41423c2826b7cc886ad1e74`)

使用 `xdoctest` 对目前的 Paddle 模块进行测试：

``` shell
$ xdoctest --style=google paddle
...
=== 6 failed, 1945 skipped, 2 warnings in 4.73 seconds ===
```

由于目前对于示例的要求(无提示符，用 `#` 注释输出)不在 `xdoctest` 的捕获样式中，所以大部分示例均被跳过检测。

另外，示例代码的格式虽然总体符合目前对于示例的要求，但是仍存在多种具体写法：

- 输出与代码同一行

    ``` python
    # ISTFT
    x_ = istft(y, n_fft=512)  # [8, 48000]
    ```

- 输出的注释与实际输出不一致

    ``` python
    p = lognormal_a.probs(value_tensor)
    # [0.48641577] with shape: [1]
    ```

- 不使用 `print` 提示

    ``` python
    conv = paddle.nn.Conv1D(3, 2, 3, weight_attr=attr)
    conv.weight
    # Tensor(shape=[2, 3, 3], dtype=float32, place=CPUPlace, stop_gradient=False,
    #       [[[0., 1., 0.],
    #         [0., 0., 0.],
    #         [0., 0., 0.]],
    #
    #        [[0., 0., 0.],
    #         [0., 1., 0.],
    #         [0., 0., 0.]]])
    ```

- 使用 `print` 提示

    ``` python
    res = linear(data)
    print(linear.weight)
    # Tensor(shape=[2, 4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
    #        [[2., 2., 2., 2.],
    #         [2., 2., 2., 2.]])
    ```
- 使用 `google` 样式

    ``` python
    >>> import paddle.fluid as fluid
    >>> p, g = backward(...)
    >>> with program.lr_schedule_guard():
    >>>     lr = lr * decay
    ```

等。

另外，对于无法验证输出一致性的示例(随机分布)、需要特殊环境(如需要GPU、文件存储)等均无特殊处理。


# 三、设计思路与实现方案

## 0、 总述

综合考虑新旧流水线更替、文档生成等问题，将 `xdoctest` 的引入拆分为以下三个阶段以及子任务：

1. 前期准备阶段：Paddle doc 与 Paddle 代码 的检查共存，不进行代码示例的修改。
    - 修改目前 Paddle docs 的代码检查方式，兼容 `google` 样式。
    - Paddle 代码 CI 中引入 `xdoctest` 检查，兼容目前的代码示例格式。

2. 中期切换阶段：规范后续 python 开发，修改现有示例代码。
    - 分批次修改已有代码的示例，提交 PR 并通过检查。
    - 更新文档《开发 API Python 端》，规范后续 python 开发行为。
    - 不再兼容旧格式(可选)，Paddle 代码 CI 的 `docstring` 检查代码示例需要符合 `google` 样式。

3. 后期收尾阶段：切换流水线至 Paddle 代码中，可移除 Paddle docs 的代码检查。
    - 中英文 [API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html#api) 特性更新，可以复制带有 `>>>` 提示符的代码示例，包含代码与注释，不含输出。
    - 代码检查移交(可选)，将代码检查的工作全部从 Paddle docs 移交至 Paddle 代码的 CI 流水线中进行。

## 1、前期准备阶段

### 1.1 修改目前 Paddle docs 的代码检查方式

考虑到新旧流水线的更替，需要不影响目前 Paddle docs 的生成与检测，因此，需要首先修改 Paddle docs 的代码检查方式，使其兼容 `google` 样式的代码运行检测。

目前 Paddle docs 中的流水线，通过 `extract_sample_code` 方法抽取示例代码，并通过 `run_sample_code` 运行代码。此时如果引入 `google` 样式的代码，如：

``` python
def test(a):
    """this is docstring...

    Note:
        this is note...

    Args:
        other (Tensor): some args...

    Returns:
        Tensor: returns...

    Some code not in the `Examples` should NOT be test.
    >>> print(5)
    6

    Some code should be test in the `Examples`.
    There should be 3 passed, 1 failed but skipped.
    The final result should be SUCCESS!

    Examples:

        .. code-block:: python

            >>> a = 3
            >>> print(a)
            3

            # the 3 lines below should NOT be test.
            a = 1
            print(a)
            1

            >>> b = 4
            >>> print(a+b)
            7

            >>> for i in range(3):
            >>>     print(i)
            0
            1
            2

            >>> # xdoctest: +SKIP
            >>> print(a)
            10000
    """
    pass
```

会报语法错误：

``` bash
  File "temp/test.py.py", line 2
    >>> a = 3
     ^
SyntaxError: invalid syntax
```

因此，需要首先修改 Paddle doc 的代码检查工具，使其能够正确抽取不带有 `>>>` 的代码。

### 1.2 Paddle 代码 CI 中引入 `xdoctest` 检查

由于目前的代码示例格式，不在 `xdoctest` 的捕获样式中，所以，可以直接引入 `xdoctest` 检查，在兼容目前的代码示例格式的同时，可对符合 `google` 样式的代码进行检查。

另外，建议将示例代码的检查，后续逐步从 Paddle docs 移交至 Paddle 的 CI 流水线中，Paddle doc 后续只需要复制示例代码即可。原因是，示例代码是同代码一同提交的，如果在 Paddle docs 中检查示例，却发现存在问题，那么需要重新修改已经 merge 的代码并 review 等操作。

参考 Paddle docs 目前的代码检查方案，可以有两种实现方式：

- 在 `PR-CI-Codestyle-Check` 中增加 `xdoctest` 的检查
- 增加一个新的 CI 流程

> **Review 注意:** 由于我这里没有 iPipe 的相关权限，目前看不到 Paddle 是如何配置 CI 的，所以这里主要以 Paddle docs 的 CI 流程为主要参考，Paddle doc 的流程相对比较简单。

具体实现方式参考 Paddle docs `ci_scripts/check_api_cn.sh` 目前的方式：

``` shell
for file in $need_check_files;do
    xdoctest --style=google ../docs/$file
    if [ $? -ne 0 ];then
        EXIT_CODE=5
    fi
done
```

对于需要检查的文件，执行 `xdoctest` 检查，使用样式为 `google`。

## 2、中期切换阶段

### 2.1 分批次修改已有代码的示例

分批次、分模块对已有的代码示例进行修改，可以采用 `脚本修改+人工复审` 的方式进行。

可以先使用脚本将已有的示例代码，加以 `>>>` 提示符，然后使用 `xdoctest` 检查是否通过。

对于有 `print` 的示例代码，可以通过注销掉含有 `#` 符号的输出，再进行 `xdoctest` 验证的方式。

但是，总体来说，人工审核占主要工作量，暂无更好的方式。

### 2.2 更新文档《开发 API Python 端》

为了规范后续 python 端代码的开发行为。需要更新文档 [开发 API Python 端](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/new_python_api_cn.html#api-python)：
- 修改代码截图中的示例样式为 `google` 样式
- 单独配以文字部分章节，讲解此处应该符合的规范
- 可以通过链接的方式，引导开发者至 `xdoctest` 官方文档处。

后续如果发现有需要特殊说明的部分，如 `SKIP` 指令使用的情况等，可以进一步更新开发文档。

### 2.3 不再兼容旧格式(可选)

在 `PR-CI-Codestyle-Check` 中增加示例代码格式的检查，要求代码必须符合 `google` 样式，可以使用以下简单的约束进行检查：

- 以 `>>>` 开头的为代码段。
- 后续行中没有 `>>>` 开头的语句视为输出，其上一行必须以 `>>>` 开头。
- 空行视为新的代码段开始

但是，由于 `xdoctest` 中也暂无此类强行的格式检查，所以，此设计项作为可选。

## 3、后期收尾阶段

### 3.1 中英文 API 文档特性更新

目前中英文 API 文档中的代码示例可以直接一键复制，当完成 `google` 样式的示例代码更新之后，需要保留此特性，但是复制的代码不能包含 `>>>` 提示符以及代码的输出。

### 3.2 代码检查移交(可选)

在以上 `xdoctest` 的引入过程中，Paddle doc 与 Paddle 代码的检查都同时存在，且互不干扰。当完成 `xdoctest` 的引入，以及 `google` 样式的示例代码的修改之后，可以考虑移除 Paddle doc 中的代码检查，以减少重复的检查流程。

## 4、日常更新与维护

`xdoctest` 中使用多种指令指导代码的检查，如 `SKIP`、`IGNORE_WANT`、`REQUIRES` 等。后续在日常更新与维护过程中需要具体分析何时使用。

如，在 `pytorch` 中，对于随机采样，使用了 `IGNORE_WANT` 指令：

``` python
class Chi2(Gamma):
    r"""
    Creates a Chi-squared distribution parameterized by shape parameter :attr:`df`.
    This is exactly equivalent to ``Gamma(alpha=0.5*df, beta=0.5)``

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = Chi2(torch.tensor([1.0]))
        >>> m.sample()  # Chi2 distributed with shape df=1
        tensor([ 0.1046])

    Args:
        df (float or Tensor): shape parameter of the distribution
    """
    ...
```

对于外部包的依赖，使用了 `REQUIRES` 指令：

``` python
def meshgrid(*tensors, indexing: Optional[str] = None) -> Tuple[Tensor, ...]:
    r"""
    ...

    Example::
        ...

        `torch.meshgrid` is commonly used to produce a grid for
        plotting.
        >>> # xdoctest: +REQUIRES(module:matplotlib)
        >>> # xdoctest: +REQUIRES(env:DOCTEST_SHOW)
        >>> import matplotlib.pyplot as plt
        >>> xs = torch.linspace(-5, 5, steps=100)
        >>> ys = torch.linspace(-5, 5, steps=100)
        >>> x, y = torch.meshgrid(xs, ys, indexing='xy')
        >>> z = torch.sin(torch.sqrt(x * x + y * y))
        >>> ax = plt.axes(projection='3d')
        >>> ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
        >>> plt.show()

    .. image:: ../_static/img/meshgrid.png
        :width: 512

    """
    return _meshgrid(*tensors, indexing=indexing)

```

对于简化的示例，使用了 `SKIP` 指令：

``` python
def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
    r"""
    ...

    Example::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> self.register_buffer('running_mean', torch.zeros(num_features))

    """
```

# 四、测试和验收的考量

- 修改目前 Paddle docs 的代码检查方式

    需要增加单元测试，保证修改后的工具能够同时兼容目前的代码示例样式与 `google` 样式的检查。

- Paddle 代码 CI 中引入 `xdoctest` 检查

    需要增加单元测试，保证修改/新增的 CI 流程，不影响目前的 Paddle 代码的检查，且能够进行 `google` 样式的代码检查。

- 分批次修改已有代码的示例

    需要保证修改后的代码能够通过新的 CI 流程检查。

- 更新文档《开发 API Python 端》

    需要保证更新后的开发文档能够指导新的代码示例的开发。

- 不再兼容旧格式(可选)

    需要增加单元测试，保证修改后的 CI 流程不能使用目前旧格式的示例代码。

- 中英文 API 文档特性更新

    需要保证修改后的 API 文档页面能够正确复制示例代码。

- 代码检查移交(可选)

    需要保证示例代码的检查工作完全移交至 Paddle 代码的 CI 流程中，需要保证 Paddle docs 的 CI 流程能够正确复制示例代码。

# 五、排期规划

- 修改目前 Paddle docs 的代码检查方式
- Paddle 代码 CI 中引入 `xdoctest` 检查
- 分批次修改已有代码的示例
- 更新文档《开发 API Python 端》
- 不再兼容旧格式(可选)
- 中英文 API 文档特性更新
- 代码检查移交(可选)

# 六、影响面

- 影响 Paddle 代码与 Paddle docs 的 CI 流水线
- 影响目前 python API 的示例代码写作方式
- 影响文档 `开发 API Python 端` 的页面显示
- 影响中英文 API 文档的示例代码显示与代码复制
