# 将 xdoctest 引入到飞桨框架工作流中

|领域 | 将 xdoctest 引入到飞桨框架工作流中                       | 
|---|--------------------------------|
|提交作者 | megemini (柳顺) | 
|提交时间 | 2023-06-14 | 
|版本号 | V1.3 | 
|依赖飞桨版本 | develop 分支 | 
|文件名 | 将 xdoctest 引入到飞桨框架工作流中.md | 

**v1.3 修订记录**

- 增加 任务排期的优先级
- 修改 部分格式

**v1.2 修订记录**

- 增加 修改目前 Paddle docs 中 `COPY-FROM` 的逻辑至整个 docstring。
- 增加 Paddle docs 中的示例代码统计。
- 增加 Paddle docs 中 `code-block` 至 `COPY-FROM` 的修改任务。
- 增加 对于 Paddle tensor `place` 的处理方案。
- 修改 使用 `google` 样式为 `google/freeform` 样式。
- 修改 `中英文 API 文档` 的复制示例特性为 `推荐`。
- 修改 部分描述。

**v1.1 修订记录**

- 增加 整体对于 Paddle 代码 CI 的调研。


# 一、概述
## 1、相关背景

`python` 源代码与官方文档中的 `Example`，也就是示例代码，是学习与使用 Paddle 框架非常重要的学习与参考依据，示例代码的正确性其重要程度也就不言而喻。[xdoctest](https://xdoctest.readthedocs.io/en/latest/) 是一个示例代码自动执行和检查工具，可以自动执行 Python docstring 中的示例代码，并对示例代码输出进行检查。

[中国软件开源创新大赛：飞桨框架任务挑战赛 赛题5](https://github.com/PaddlePaddle/Paddle/issues/53172#paddlepaddle05) 要求将 xdoctest 引入到飞桨框架的工作流中，利用 xdoctest 来自动检查示例代码运行正确，且与输出结果匹配，以确保示例代码输出的一致性，进一步提高飞桨框架示例代码的质量。

## 2、功能目标

### 2.1 文档建设

更新 Paddle 贡献指南：
- [开发 API Python 端](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/new_python_api_cn.html#api-python) 中的 `Python API 的代码开发示例`。
- [API 文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html#api) 中的 `API 代码示例`。

以此规范后续代码的开发。

添加 `Example` 示例代码的写作要求，要求符合 `xdoctest` 中的 `google/freeform` 样式，即，在示例 `Example` 中代码需要以 `>>>` 开头。且保留目前的 `".. code-block:: python"` 提示(sphinx 代码块的指令)，从而不影响中文文档的生成工作。

添加 `Example` 示例代码特殊情况的处理指导或链接，如使用 `xdoctest` 的 `# xdoctest +SKIP` 跳过随机生成的输出检查。

修改 `sphinx` 对于示例代码的复制特性，更新 Paddle API 文档中 `代码示例` 的页面显示特性，使其能够正确显示带有 `>>>` 的示例代码，并能够复制不带有 `>>>` 的示例代码，以方便用户运行示例代码。

### 2.2 CI 流水线

更新 CI 流水线：
- Paddle 代码的 CI
- Paddle docs 的 CI

在 Paddle 的 CI 流水线中引入 `xdoctest`，使其在代码（python PR，新提交）提交 PR 时触发检查。`xdoctest` 的引入，需要不影响现有代码，只针对后续 PR。现有代码的示例，如不符合规范，则修改后重新提交 PR，并需要通过检查。

由于在代码的 PR 中进行了示例代码的检查，后续可取消 Paddle docs 中的示例代码检查。

`xdoctest` 的引入，保留目前的 `code-block` 提示，不影响现有文档生成流程。

### 2.3 现存代码

更新示例代码：

- Paddle 代码中的示例代码
- Paddle docs 中示例代码的引用方式

分批次更新现存的 Paddle 示例代码。以自动修改+人工复审的方式进行。

更新时需要在参考原有示例的同时，关注版本变化带来的输出结果变化。

更新 Paddle docs 中的示例代码统一使用 `COPY-FROM` 指令。

## 3、意义

`xdoctest` 的引入，可以规范飞桨框架示例代码的写作格式、提高示例代码的写作质量、保证示例代码输出的正确性。对于飞桨框架示例代码的维护将更加便捷，对于学习飞桨框架的开发者具有更好的参考价值。

# 二、飞桨现状

## 1、文档建设

### 1.1 贡献指南

目前 [开发 API Python 端](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/new_python_api_cn.html#api-python) 中，对于示例代码的写作方式，是通过一张代码截图说明的: ![代码截图](https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/dev_guides/api_contributing_guides/images/zeros_python_api.png?raw=true) 

其中的代码示例对于普通 python 语句没有提示符要求，对于输出则使用 `#` 进行注释，这不符合 `xdoctest` 中的 `google/freeform` 样式要求。

需要使用 `截图+文字` 说明的方式更新此开发文档，并有链接至 [API 文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html#api)。

目前 [API 文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html#api) 中的 `API 代码示例` 是关于示例代码的书写要求:

> API 代码示例：中英文文档当中的代码示例完全一致（注释可不用翻译），中文文档建议使用 COPY-FROM 的方式与英文文档做同步。代码示例使用 2.0 版本中的 API，可运行。尽量不用随机输入，注释形式给出输出值。构造输入数据时，尽量使用 paddle 提供的 API，如: paddle.zeros、paddle.ones、paddle.full、paddle.arange、paddle.rand、paddle.randn、paddle.randint、paddle.normal、paddle.uniform，尽量不要引入第三方库（如 NumPy）；

需要更新此说明，以符合 `google/freeform` 样式要求，以及对于特殊情况，如 `SKIP`、`REQUIRES(env:GPU)` 等，有相应的示例。

> **Xdoctest 示例样式说明**
>
> - A `Google-style <https://sphinxcontrib-napoleon.readthedocs.io>`__ doctest is
expected to exist in  Google "docblock" with an ``Example:`` or ``Doctest:``
tag. All code in this block is parsed out as a single doctest.
>
> - A `freeform style` doctest is any contiguous block of lines prefixed by ``>>>``.
This is the original parsing style of the builtin doctest module. Each block is
listed as its own test. 
>

目前 [API 代码示例（重要）](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html#id8) 中列举了部分注意事项，需要在此补充新样式的示例说明。

### 1.2 sphinx 构建文档

目前 Paddle 的文档使用 `sphinx` 进行构建。主要配置文件为：
- `docs/ci_scripts/doc-build-config/zh/conf.py`
- `docs/ci_scripts/doc-build-config/en/conf.py`

中英文 [API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html#api) 可以复制代码示例，此特性需要保留。

## 2、CI 流水线

涉及此设计文档的 CI 流水线，主要包括:
- [Paddle 代码](https://github.com/PaddlePaddle/Paddle)
- [Paddle docs](https://github.com/PaddlePaddle/docs)

两个部分。

CI 的构建主要通过 [百度效率云 iPipe](https://xly.bce.baidu.com/paddlepaddle/paddle/_detail) 实现。

### 2.1 Paddle 代码

#### 2.1.1 代码检查

Paddle 代码的 CI 流水线相关工具放置在 [Paddle/tools/](https://github.com/PaddlePaddle/Paddle/tree/develop/tools) 目录下。

目前对于 python 示例代码的检查，主要通过 

[Paddle/tools/sampcd_processor.py](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/sampcd_processor.py) 

完成。

整个流程主要分为三个主要阶段：

1. **接口抽取**
2. **示例执行**
3. **结果比对**

主要执行流程为（以 `PR-CI-Static-Check` 为例）：

- shell 部分 `Paddle/paddle/scripts/paddle_build.sh`

    - 编译脚本 

        1. 执行命令：`paddle/scripts/paddle_build.sh build_and_check_cpu`

        2. `paddle_build.sh` 脚本执行 `generate_api_spec ${PYTHON_ABI:-""} "PR"` 生成 `PR` 对应的 API `paddle/fluid/API_PR.spec`

        3. `generate_upstream_develop_api_spec` 中的 `generate_api_spec "$1" "DEV"` 生成 `DEV` 对应的 API `paddle/fluid/API_DEV.spec`

    - 测试脚本

        1. 执行命令：`paddle/scripts/paddle_build.sh build_and_check_gpu`

        2. `paddle_build.sh` 脚本执行 `build_and_check_gpu` 命令，并调用 `exec_samplecode_test`

        3. `exec_samplecode_test` 调用 `sampcd_processor.py` 脚本进行代码检查：
        `python sampcd_processor.py --threads=${SAMPLE_CODE_EXEC_THREADS} gpu; example_error=$?`

- python 部分 `Paddle/tools/sampcd_processor.py`

    1. `sampcd_processor.py` 首先获取到当前的运行环境： `get_test_capacity`，
    这里判断当前环境是否适合 `gpu` 运行等。

    2. 创建临时代码运行目录 `SAMPLECODE_TEMPDIR`，并根据是否全量测试，获取/生成 API 及相应示例代码：`filenames = get_filenames(args.full_test)`。

    3. 对于全量测试

        - `get_full_api_from_pr_spec` 利用之前的 `API_PR_SPEC_FN` (`paddle/fluid/API_PR.spec`) 生成 API 列表。

        - 如果没有，则利用 `Paddle/tools/print_signatures.py` 中的 `get_all_api` 生成全量 API。

    4. 对于增量测试，`get_incrementapi` 利用之前的 

        - `API_PR_SPEC_FN(paddle/fluid/API_PR.spec)`

        - `API_DEV_SPEC_FN(paddle/fluid/API_DEV.spec)`

        生成增量 API 列表。

    5. `get_filenames` 在生成 API 列表之后，逐个 `eval` 判断接口的可执行性，对于有 `__doc__` 的接口，利用 
        
        - `sampcd_extract_to_file` 
        
        - `extract_code_blocks_from_docstr`
        
        抽取示例代码并保存至文件。

    6. `sampcd_extract_to_file` 会判断 docstring 中是否有示例代码，示例代码的 `required` 是否满足，是否跳过 `skiptest` 等。

    7. 示例代码生成之后，利用 `multiprocessing` 的 `map_async(execute_samplecode, filenames.keys())` 逐个执行示例，并判断是否出错。

    8. `execute_samplecode` 利用 `python xxx.py` 的模式执行示例代码，并根据 `subprc.returncode` 判断是否出错。同时，在执行时对运行时间进行记录。

    9. 最后对总体结果进行输出。

    10. 利用 `exec_gen_doc` 生成预览文档。

其中：

1. 接口抽取

    `print_signatures.py` :: `get_all_api`

    `sampcd_processor.py` :: `get_filenames`

2. 示例执行

    `sampcd_processor.py` :: `execute_samplecode`

3. 结果比对

    `sampcd_processor.py` :: `execute_samplecode`

这里对比 `pytorch` 对于示例代码的检查。

`pytorch` 通过 `pytorch/test/run_test.py` 中的 `run_doctests` 进行代码检查：

``` python
def run_doctests(test_module, test_directory, options):

    import pathlib

    import xdoctest

    pkgpath = pathlib.Path(torch.__file__).parent

    exclude_module_list = []
    enabled = {
        "lapack": 0,
        "cuda": 0,
        "cuda1": 0,
        "qengine": 0,
        "autograd_profiler": 0,
        "cpp_ext": 0,
        "monitor": 0,
        "onnx": "auto",
    }

    # Resolve "auto" based on a test to determine if the feature is available.
    if enabled["cuda"] == "auto" and torch.cuda.is_available():
        enabled["cuda"] = True

    ... 

    pkgpath = os.path.dirname(torch.__file__)

    xdoctest_config = {
        "global_exec": r"\n".join(
            [
                "from torch import nn",
                "import torch.nn.functional as F",
                "import torch",
            ]
        ),
        "analysis": "static",  # set to "auto" to test doctests in compiled modules
        "style": "google",
        "options": "+IGNORE_WHITESPACE",
    }
    xdoctest_verbose = max(1, options.verbose)
    run_summary = xdoctest.runner.doctest_module(
        os.fspath(pkgpath),
        config=xdoctest_config,
        verbose=xdoctest_verbose,
        command=options.xdoctest_command,
        argv=[],
        exclude=exclude_module_list,
    )
    result = 1 if run_summary.get("n_failed", 0) else 0
    return result
```

可以看到，`pytorch` 对于示例代码的检查是通过 `xdoctest` 进行的，基本上是 `全量测试`。

并且，`pytorch` 利用 `xdoctest` 自动搜索 API 并进行测试，没有单独的 API 抽取过程。这与 `Paddle` 目前的 API 抽取方式与测试方式是不同的。

使用 `xdoctest` 对于 `torch==1.13.1` 进行统计：

``` shell
$ xdoctest torch --style=google --command=list > tmp.txt
```

能够检测出 `602` 个含有示例的接口（这个命令没有统计信息，具体数量通过统计 `tmp.txt` 中的行数得出）。

这里需要对比一下 `xdoctest` 与 `Paddle` 目前对于示例代码检查的区别。

#### 2.1.2 对比 xdoctest 与 Paddle 的示例检查

进一步看一下 Paddle 对于 API 的抽取过程，主要的脚本为：

`Paddle/tools/print_signatures.py`

1. `print_signatures.py` 通过 `get_all_api` 获取 API 列表。

2. `get_all_api` 利用 `pkgutil.walk_packages` 遍历 `paddle.__path__` 中的各个模块。

3. 对于正常引入的模块，利用 `process_module` 解析模块中的 API。

4. `process_module` 只对含有属性 `__all__` 的模块进行解析，并跳过 `__all__` 中私有方法。

5. 通过 `insert_api_into_dict` 对于正常 `eval` 的接口进行记录。

6. `insert_api_into_dict` 通过 `inspect.getdoc` 获取到接口的文档。

7. 回到 `process_module` 中，对于通过 `insert_api_into_dict` 记录的接口，判断接口类型为 `inspect.isclass(api_info['object'])` 的，进一步通过 `inspect.getmembers` 获取内部方法。同样需要过滤掉私有方法。

8. 当遍历完 `paddle.__path__` 目录下的所有模块后，还需要遍历 `paddle` 模块本身的所有属性：

    `api_counter += process_module(paddle, attr)`

    这是由于，`paddle` 通过 `globals` 直接赋值等方式，暴露很多接口在 `paddle` 模块下，这些接口是无法通过 `paddle.__path__` 获取的。比如： 
    
    `Paddle/python/paddle/tensor/ops.py` 中的

    ``` python
    add_sample_code(
        globals()["abs"],
        r"""
    Examples:
        .. code-block:: python

            import paddle

            x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
            out = paddle.abs(x)
            print(out)
            # [0.4 0.2 0.1 0.3]
    """,
    )
    ```

至此，Paddle 的 `print_signatures.py` 通过 `get_all_api` 将所有接口捕获出来。而对于接口示例代码的检查，在上一节中有相应介绍，这里不再赘述。

可以看到，Paddle 对于接口的统计主要有两部分：

- `paddle.__path__`
- `paddle` 模块下

然后，是 `xdoctest` 对于接口的遍历过程。

`xdoctest` 的入口为 `xdoctest.runner.doctest_module`:

1. 对于传入 `xdoctest.runner.doctest_module` 的 `module_identifier` 有两种类型：

    - `types.ModuleType`

    - `str`

2. 对于 `types.ModuleType` 类型的模块，比如： `paddle`，直接设置模块的信息：

    ``` python
    modinfo['module'] = module_identifier
    modinfo['modpath'] = modinfo['module'].__file__
    ```

3. 对于 `str` 类型的模块，比如：

    - `"paddle.abs"` 直接通过 `core._rectify_to_modpath` 识别其模块路径。

    - `"paddle::abs"` 这类带有 `::` 的字符串，后半部分表示模块下的测试目标，如 `方法` 等。而前半部分，则同样通过 `core._rectify_to_modpath` 识别其模块路径。

4. 通过判断 `modinfo['modpath']` 是否为 `None` 作为后续的 `parsable_identifier` 标识依据。

    ``` python
    if modinfo['modpath'] is None:
        parsable_identifier = modinfo['module']
    else:
        parsable_identifier = modinfo['modpath']
    ```

    也就是说，如果一个模块是正常的，如 `paddle`，能够获取到 `modpath`，则后续以此为解析的根。而不是类似 Paddle 以此模块下的 `__all__` 为依据。`core.parse_doctestables` 根据此标识解析接口。

5. 之后利用 `parse_google_docstr_examples` 或者 `parse_auto_docstr_examples`(默认) 对文档进行示例的解析。

6. 其中，对于 `.py` 文件使用 `static_analysis`， 而对于 `.so` 等动态接口使用 `dynamic_analysis` 解析示例代码。

7. 对于解析出来的示例，利用 `eval`, `compile`, `exec` 等进行运行。

8. 利用 `check_output`, `check_exception` 等方法检查

    - `got` 示例输出
    
    - `want` 期望输出

    的一致性，并给出最终结果。

其中：

1. 接口抽取

    依赖 `parsable_identifier`

2. 示例执行

    `eval`, `compile`, `exec`

3. 结果比对

    `check_output`, `check_exception` 等


`xdoctest` 的这种接口解析方式，与 Paddle 存在几个较大的差别：

- 对于不是模块的接口，通过解析模块路径的方法进行查找。这对于 `paddle.abs` 之类的方法无法捕获。 `paddle.abs` 本身是方法而不是模块，所以 `xdoctest` 会进而寻找 `paddle.abs` 的路径，查找不到后便会报错。

- `xdoctest` 遵循 `__init__.py` 与模块的关系，对于不含 `__init__.py` 的包不做寻找。比如
`Paddle/python/paddle/incubate/nn/layer/` 下面没有 `__init__.py` 导致找不到这个包

- 对于 `__file__` 不存在的模块会报错，比如 `paddle.fluid.libpaddle.eager.ops`，虽然是模块，但不包含 `__file__`，解析出错。

- `xdoctest` 不区分是否为私有方法，也就是下划线符号(`"_"`)开头的方法。

另一方面，`xdoctest` 对于示例代码输出的判断更全面，能够捕获到 `ValueError` 等 python 的异常。

一言以蔽之：

1. `xdoctest` 以文件为寻找接口的主要依据，Paddle 以 `__all__` 属性为主要依据。
2. `xdoctest` 对于代码输出的判断更全面，Paddle 相对简单。

鉴于以上差别，这里建议引入 `xdoctest` 对代码进行更全面的检查，但不建议 Paddle 采用与 `pytorch` 一样的 `xdoctest` 方式进行示例代码的检查。

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

此方法存在较多问题，比如，无法验证代码与示例中的结果是否一致，无法处理本应报错的示例代码（如 python 的 `raise`）等。

另外，Paddle docs 在 `docs-build.sh` 的过程中，会调用到 `docs/docs/api/gen_doc.py` 脚本，此脚本中同样存在抽取示例代码和运行检查的方法：

- `extract_sample_codes_into_dir`
- `run_all_sample_codes`

这些方法，与 Paddle 代码 CI 中的 `sampcd_processor.py` 中的方法，**逻辑是一样的**。

也就是说，仅仅是示例代码的检查，就至少有：

- **Paddle 代码的 CI**
- **Paddle docs 的构建**
- **Paddle docs 的 CI**

然而，具体分析这三个部分的执行逻辑，会发现：

- `Paddle 代码的 CI` 与 `Paddle docs 的构建` 是相同的

    他们会在 docstring 的 `Examples` 中抽取示例代码
    
- `Paddle docs 的 CI` 会抽取 docstring 中所有的 `code-block`

如：

``` python
def test(a):
    """this is docstring...
        
    .. code-block:: python

        >>> print('Some code in docstring...')
        Some code in docstring...

    Examples:

        .. code-block:: python

            >>> a = 3
            >>> print(a)
            3
    """
    pass

```

使用 `Paddle 代码的 CI` 与 `Paddle docs 的构建` 的方法，抽取到的是一段代码：

``` python
>>> a = 3
>>> print(a)
3
```

而使用 `Paddle docs 的 CI` 的方法，抽取到的是两段代码：

``` python
>>> print('Some code in docstring...')
Some code in docstring...

>>> a = 3
>>> print(a)
3
```

同样的示例代码检查，存在多个地方，是比较大的设计冗余，对于后续的代码维护带来较多不便，建议后续只保留 Paddle 代码的 CI 这一个地方的代码检查即可。

## 3、现存代码

### 3.1 Paddle 代码

目前 Paddle 中 python 相关代码，主要放置在 [Paddle/python/paddle/](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle) 目录下。

其中包括(Paddle 官方 develop 版本，`sampcd_processor.py` 内统计)：
- `3648` 个接口
- `1409` 段示例代码

使用 `xdoctest` 对目前的 Paddle 模块进行测试：

``` shell
$ xdoctest --style=google paddle
...
=== 5 failed, 1967 skipped, 1 warnings in 5.81 seconds ===
```

对于 `1409` 段示例代码与 `xdoctest` 的 `5+1967+1=1973` 差距，主要是由于两者对于接口的抽取方式不同导致，上文已经介绍过，这里不再赘述。

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
- 使用 `google/freeform` 样式

    ``` python
    >>> import paddle.fluid as fluid
    >>> p, g = backward(...)
    >>> with program.lr_schedule_guard():
    >>>     lr = lr * decay
    ```

等。

对于无法验证输出一致性的示例(随机分布)、需要特殊环境(如需要GPU、文件存储)等均无特殊处理。

### 3.2 Paddle docs

目前 Paddle docs 中的相关 API 主要放置在 [docs/docs/api/paddle/](https://github.com/PaddlePaddle/docs/tree/develop/docs/api/paddle) 目录下。

其中包括(Paddle docs 的 commit 为 `0132db11`)：

- `1534` 个 `rst` 文档
- `1325` 个文档使用了 `COPY-FROM` 指令
- `174` 个文档仍存留 `code-block` 指令
- `53` 个文档同时包含 `COPY-FROM` 指令和 `code-block`

单独统计 `fluid` 目录：
- `321` 个 `rst` 文档
- `226` 个文档使用了 `COPY-FROM` 指令
- `76` 个文档仍存留 `code-block` 指令
- `8` 个文档同时包含 `COPY-FROM` 指令和 `code-block`

这里需要重点关注同时包含 `COPY-FROM` 指令和 `code-block` 的文档，如 `gather_cn.rst`：

``` reStructuredText
.. _cn_api_paddle_tensor_gather:

gather
-------------------------------

.. py:function:: paddle.gather(x, index, axis=None, name=None)

根据索引 index 获取输入 ``x`` 的指定 ``aixs`` 维度的条目，并将它们拼接在一起。

.. code-block:: text

        Given:

        X = [[1, 2],
             [3, 4],
             [5, 6]]

        Index = [1, 2]

        axis = 0

        Then:

        Out = [[3, 4],
               [5, 6]]

参数
::::::::::::
        ...

返回
::::::::::::
...


代码示例
::::::::::::

COPY-FROM: paddle.gather 
```

由于示例代码本身不需要翻译，所以引入 `COPY-FROM` 指令，可以提高中文文档的书写速度与质量。

但是，由于 `COPY-FROM` 指令只在 `代码示例` 部分起作用，导致在 docstring 的描述部分如果存在代码，仍需要使用 `code-block` 指令进行显性的书写。

`COPY-FROM` 的实现主要在 Paddle docs 的 `docs/docs/api/copy_codes_from_en_doc.py` 中：

- `filter_all_files` 方法为主入口
- `instert_codes_into_cn_rst_if_need` 抽取代码至文档中
- `read_rst_lines_and_copy_info` 根据 `COPY-FROM` 相关正则，分析需要抽取代码的具体 python API
- `find_codeblock_needed` 负责具体查找 API 的 docstring
- `find_codeblock_needed` 复用 `docs/docs/api/gen_doc.py` 中的 `extract_code_blocks_from_docstr` 方法，对 docstring 中的示例进行抽取
- `extract_code_blocks_from_docstr` 抽取 `Examples` 中的 `code-block`

此上，由于 `COPY-FROM` 复用的 `extract_code_blocks_from_docstr` 方法，导致 `code-block` 只能在 `代码示例` 部分使用。

# 三、设计思路与实现方案

## 0、 总述

综合考虑新旧流水线更替、文档生成等问题，将 `xdoctest` 的引入拆分为以下三个阶段以及子任务：

1. 前期准备阶段：Paddle doc 与 Paddle 代码 的检查共存，新旧代码样式共存，不进行代码示例的修改。

    - 修改目前 Paddle docs 中 `COPY-FROM` 的逻辑，使其兼容 docstring 除 `Examples` 外的代码抽取。

    - 修改目前 Paddle docs 中仍使用 `code-block` 的示例代码为 `COPY-FROM`。

    - 修改目前 Paddle docs 的代码检查方式，兼容 `google/freeform` 样式。

    - Paddle 代码 CI 中引入 `xdoctest` 检查，兼容目前的代码示例格式。

2. 中期切换阶段：规范后续 python 开发，修改现有示例代码。

    - 分批次修改已有代码的示例，提交 PR 并通过检查。

    - 更新文档《开发 API Python 端》与《API 文档书写规范》，规范后续 python 开发行为。

    - 不再兼容旧格式，移除原有检查方式，Paddle 代码 CI 的检查代码示例需要符合 `google/freeform` 样式。

3. 后期收尾阶段：切换流水线至 Paddle 代码中，可移除 Paddle docs 的代码检查。

    - 中英文 [API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html#api) 特性更新，可以复制带有 `>>>` 提示符的代码示例，包含代码与注释，不含输出。（*注，此特性不在本项目中实现，仅为 `推荐` 方案）

    - 代码检查移交，将代码检查的工作全部从 Paddle docs 移交至 Paddle 代码的 CI 流水线中进行。

## 1、前期准备阶段

### 1.1 修改目前 Paddle docs 中 `COPY-FROM` 的逻辑

由于目前 `COPY-FROM` 的逻辑仅支持在 `Examples` 中进行提取，使得中文文档中同时存在

- `code-block`
- `COPY-FROM`

之前提到，`COPY-FROM` 对于代码的提取，依赖 `extract_code_blocks_from_docstr` 的实现，而此方法只会提取 `Examples` 的部分，由此，需要从至少两个方面着手修改，使其兼容 docstring 除 `Examples` 外的代码抽取：

- API 的 docstring 中 `code-block` 需要标识此示例的名称，如：

    ``` python
    .. code-block:: python

        :name: code-example-1

        ...
    ```

    此格式的 `code-block` 目前可以解析为：

    ``` python
    {
        'codes': '...',
        'id': 1,
        'name': 'code-example-1',
        'required': None
    }
    ```

    对应 `COPY-FROM` 可以为：

    ``` reStructuredText
    COPY-FROM： paddle.xxx:code-example-1
    ```

    可以利用 `name` 属性作为示例代码段的标识。

- 修改 `extract_code_blocks_from_docstr` 方法提取 docstring 中的所有代码段。

### 1.2 修改目前 Paddle docs 中仍使用 `code-block` 的示例代码为 `COPY-FROM`

由于目前仍有约 `174` 个文档仍存留 `code-block` 指令，所以，需要人工修改这部分文档为新的 `COPY-FROM` 指令。

这里需要修改 `rst` 文档中 

- `代码示例` 
- 其他 docstring （依赖上面任务的完成： `1.1 修改目前 Paddle docs 中 COPY-FROM 的逻辑`）

两部分的 `code-block` 指令。

如 `docs/api/paddle/distribution/Bernoulli_cn.rst`，

``` reStructuredText
.. _cn_api_distribution_Bernoulli:

Bernoulli

...

代码示例
::::::::::::

.. code-block:: python

    import paddle
    from paddle.distribution import Bernoulli

    # init `probs` with a float
    rv = Bernoulli(probs=0.3)

    print(rv.mean)
    # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        0.30000001)

...
```

需要修改其中的 `.. code-block:: python` 为 `COPY-FROM: paddle.distribution.Bernoulli`。

以及，如 `gather_cn.rst` 描述中的 `.. code-block:: text` 以及代码，为 `COPY-FROM: paddle.gather:code_0`。

### 1.3 修改目前 Paddle docs 的代码检查方式

考虑到新旧流水线的更替，需要不影响目前 Paddle docs 的生成与检测，因此，需要首先修改 Paddle docs 的代码检查方式，使其兼容 `google/freeform` 样式的代码运行检测。

目前 Paddle docs 中的流水线，通过 `extract_sample_code` 方法抽取示例代码，并通过 `run_sample_code` 运行代码。此时如果引入 `google/freeform` 样式的代码，如：

``` python
def test(a):
    """this is docstring...
    
    Some code not in the `Examples`, SHOULD be test.
    
    .. code-block:: python

        :name: example_0

        >>> print('Some code in docstring...')
        Some code in docstring...

    Note:
        this is note...

    Args:
        other (Tensor): some args...

    Returns:
        Tensor: returns...

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

这里根据示例代码中是否带有 `>>>` 起始的行，如有则跳过，以此对新旧样式进行分流。

鉴于后续计划移除 Paddle docs 中的此处代码检查，建议不对此处进一步针对 `xdoctest` 进行设计。

### 1.4 Paddle 代码 CI 中引入 `xdoctest` 检查

由于目前的代码示例格式，不在 `xdoctest` 的捕获样式中，所以，可以直接引入 `xdoctest` 检查，在兼容目前的代码示例格式的同时，可对符合 `google/freeform` 样式的代码分流至 `xdoctest` 进行检查。

另外，建议将示例代码的检查，后续逐步从 Paddle docs 移交至 Paddle 的 CI 流水线中，Paddle doc 后续只需要复制示例代码即可。原因是，示例代码是同代码一同提交的，如果在 Paddle docs 中检查示例，却发现存在问题，那么需要重新修改已经 merge 的代码并 review 等操作。

具体的实现方式，以 `最小影响、最少修改` 为设计依据，参考示例代码检查整个流程的三个主要阶段：

1. 接口抽取

    沿用原有方式

    `print_signatures.py` :: `get_all_api`

    `sampcd_processor.py` :: `get_filenames`

2. 示例执行

    增加 `xdoctest` 的代码检查

3. 结果比对

    增加 `xdoctest` 的结果比对

#### 1.4.1 接口抽取

需要在 `sampcd_processor.py` :: `get_filenames` 中增加 `api_obj.__doc__` 的抽取。

简化代码演示：

``` python
all_xdoctest = []

def get_filenames(full_test=False):
    ...
            if hasattr(api_obj, '__doc__') and api_obj.__doc__:
                sample_code_filenames = sampcd_extract_to_file(
                    api_obj.__doc__, api
                )
                for tfname in sample_code_filenames:
                    all_sample_code_filenames[tfname] = api
                    
                all_xdoctest.append((api_obj.__doc__, api)) # 这里追加需要抽取的文档
                    
    return all_sample_code_filenames
```

#### 1.4.2 示例执行

示例执行首先需要进行运行时环境检查，也就是 `REQUIRES` 与 `os.environ` 的对应关系。

这里为了兼容 `xdoctest` 的 `REQUIRES` 指令，需要根据

`SAMPLE_CODE_TEST_CAPACITY`

设置环境变量：

``` python
# add requires envs
if 'gpu' in SAMPLE_CODE_TEST_CAPACITY:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
```

然后，单独增加一个 `xdoctest` 的检查方法：

``` python
def execute_xdoctest(docs):
    
    import xdoctest
    
    xdoctest_config = {
        "global_exec": r"\n".join(
            [
                "import paddle",
            ]
        ),
        "analysis": "auto", 
        "options": "+IGNORE_WHITESPACE",
    }
    all_test = []
    for doc, api in docs:
        for example in xdoctest.core.parse_google_docstr_examples(doc, callname=api):
            example.mode = 'native'
            example.config.update(xdoctest_config)
            
            start_time = time.time()
            result = example.run(verbose=1, on_error='return')
            end_time = time.time()
            all_test.append((str(example), result, end_time-start_time))

    return all_test
```

方法中单独对于每个 API 中的文档先进行解析，后执行并获得检查结果。

同时记录运行时时间。

另外，如果后续需要检查示例代码是否符合 `google/freeform` 格式，可以判断：

- 是否有 `code-block` 代码段
- 如果有，但是 `example._parts` 为空

则说明示例代码不符合 `google/freeform` 格式。

`xdoctest` 返回的结果中包括：

- `passed`
- `skipped`
- `failed`

关键字，可利用此对结果进行统计。

这里将上述代码插入 `sampcd_processor.py` 后，运行代码检查：

``` shell
$ python sampcd_processor.py cpu --full-test
```

可以得到
``` shell
passed: 0 skipped: 1605 failed: 0
```

这里 `1605` 多于之前 `1409` 段示例代码的统计，主要是由于 `xdoctest` 会将示例代码分多个段执行进而统计导致，所以会较之前多。

另外，这里尝试验证 `xdoctest` 的正确性，修改 `paddle.abs` 接口的文档，原为：

``` python
add_sample_code(
    globals()["abs"],
    r"""
Examples:
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.abs(x)
        print(out)
        # [0.4 0.2 0.1 0.3]

""",
)
```

`xdoctest` 会直接跳过，修改为：

``` python
add_sample_code(
    globals()["abs"],
    r"""
Examples:
    .. code-block:: python

        >>> import paddle
        >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        >>> out = paddle.abs(x)
        >>> print(out)
        [0.4 0.2 0.1 0.3]

""",
)
```

``` shell
$ python sampcd_processor.py cpu --full-test
passed: 0 skipped: 1604 failed: 1
```

这里可以正确检查出一个错误的示例代码。

进一步修改为：

``` python
add_sample_code(
    globals()["abs"],
    r"""
Examples:
    .. code-block:: python

        >>> import paddle
        >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        >>> out = paddle.abs(x)
        >>> print(out)
        Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
        [0.40000001, 0.20000000, 0.10000000, 0.30000001])

""",
)
```

``` shell
$ python sampcd_processor.py cpu --full-test
passed: 1 skipped: 1604 failed: 0
```

这里可以正确检查出正确的示例代码。

总之，对于 Paddle docs 与 Paddle 代码的 CI 改造，主要是：

- 分流新旧样式的示例代码
- 引入 `xdoctest` 的检查 (Paddle docs 建议不实现此特性)

#### 1.4.3 Paddle Tensor place 的处理方式

由于 Paddle Tensor 格式化后的字符串里有 `place` 信息，而此信息在不同平台有不同的输出方式，如，在 `CPU` 环境中：

``` python
Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
        [0.40000001, 0.20000000, 0.10000000, 0.30000001])
```

而在 `GPU` 环境中：

``` python
Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
        [0.40000001, 0.20000000, 0.10000000, 0.30000001])
```

Paddle CI 的示例代码检测，主要在 `CPU` 环境中执行，所以，这里采用：

- 加上 `paddle.device.set_device('cpu')` 全局命令
- 开发文档中引导使用 `...` 的写作方式

#### 1.4.3.1 加上 `paddle.device.set_device('cpu')` 全局命令

`execute_xdoctest` 的 `global_exec` 加一个 `paddle.device.set_device("cpu")`，也就是说，每次执行前统一默认设置为 "cpu" 环境执行：

``` python
def execute_xdoctest(docs):
    
    import xdoctest
    
    xdoctest_config = {
        "global_exec": r"\n".join(
            [
                "import paddle",
                "paddle.device.set_device('cpu')",
            ]
        ),
        ...
    }
    ...
```

另外，简单分析一下运行时状况（有 place 的示例代码）：

1. 不全局执行 `paddle.device.set_device('cpu')`，跟随运行时环境

| 执行环境 | 示例代码 | 检查结果|
| - | - | - |
| cpu | 无REQUIRES | 正常检查 |
| cpu | +REQUIRES(env:gpu) 和 paddle.device.set_device("gpu") | 跳过检查 |
| gpu | 无REQUIRES | 正常检查，如果有 place 则可能报错 |
| gpu | +REQUIRES(env:gpu) 和 paddle.device.set_device("gpu") | 正常检查 |

2. 全局执行 `paddle.device.set_device('cpu')`

| 执行环境 | 示例代码 | 检查结果|
| - | - | - |
| cpu | 无REQUIRES | 正常检查 |
| cpu | +REQUIRES(env:gpu) 和 paddle.device.set_device("gpu") | 跳过检查 |
| gpu | 无REQUIRES | 正常检查 |
| gpu | +REQUIRES(env:gpu) 和 paddle.device.set_device("gpu") | 正常检查 |

#### 1.4.3.2 开发文档中引导使用 `...` 的写作方式

开发文档里面要写清楚：

比如，明确开发者在写示例的时候，是否关注 `place`：
- 关注：如特指 gpu 环境等，开发者应该写明：
    - `+REQUIRES(env:gpu)`
    - `paddle.device.set_device("gpu")`
- 不关注：那么，默认环境为 “cpu”。此处为开发者认为示例检查的默认环境，而不是实际代码检查的环境。

另外，`xdoctest` 还支持 `...` 的方式，如：
``` python
add_sample_code(
    globals()["abs"],
    r"""
Examples:
    .. code-block:: python

        >>> import paddle
        >>> x = paddle.to_tensor([[0.1, 0.2],[-0.4, -0.2]])
        >>> out = paddle.abs(x)
        >>> print(out)
        ...
        [[0.10000000, 0.20000000],
        [0.40000001, 0.20000000]]
        ...
""",
)
```
这样也是可以通过测试的。也就是说，在关注的输出内容前后，加上 `...` ，这样 xdoctest 可以不比对前面和后面的内容。

## 2、中期切换阶段

### 2.1 分批次修改已有代码的示例

分批次、分模块对已有的代码示例进行修改，可以采用 `脚本修改+人工复审` 的方式进行。

可以先使用脚本将已有的示例代码，加以 `>>>` 提示符，然后使用 `xdoctest` 检查是否通过。

对于有 `print` 的示例代码，可以通过注销掉含有 `#` 符号的输出，再进行 `xdoctest` 验证的方式。

但是，总体来说，人工审核占主要工作量，暂无更好的方式。

> **建议**，可以通过 `快乐开源` 等活动，发动广大开发者对代码进行修改，以修改个数与质量进行相应奖励。

### 2.2 更新文档《开发 API Python 端》与《API 文档书写规范》

为了规范后续 python 端代码的开发行为。需要

更新文档 [开发 API Python 端](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/new_python_api_cn.html#api-python)：

- 修改代码截图中的示例样式为 `google/freeform` 样式

- 链接至 [API 文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html#api) 对具体样式进行说明

更新文档 [API 文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html#api) ：

- 单独配以文字部分章节，讲解此处应该符合的规范

- 通过链接的方式，引导开发者至 `xdoctest` 官方文档处

- 后续如果发现有需要特殊说明的部分，如 `SKIP` 指令使用的情况等，可以进一步更新开发文档。

### 2.3 不再兼容旧格式

在 `sampcd_processor.py` 中增加示例代码格式的检查，要求代码必须符合 `google/freeform` 样式，由于引入了 `xdoctest`，可以使用以下逻辑进行检查：

- 是否有 `code-block` 代码段
- 如果有，但是 `example._parts` 为空

则说明示例代码不符合 `google/freeform` 格式。

## 3、后期收尾阶段

### 3.1 中英文 API 文档特性更新（推荐）

目前中英文 API 文档中的代码示例可以直接一键复制，当完成 `google/freeform` 样式的示例代码更新之后，需要保留此特性，但是复制的代码不能包含 `>>>` 提示符以及代码的输出。

由于目前 Paddle docs 是使用 `sphinx` 进行文档的构建，可以使用现有的工具 [Sphinx-copybutton](https://sphinx-copybutton.readthedocs.io/en/latest/) 进行扩展。

![](https://user-images.githubusercontent.com/1839645/150200219-73663c59-08fd-4185-b157-62f3769c02ac.gif)

在相应的配置文件 `conf.py` 中添加：

``` python
extensions = [
    ...
    'sphinx_copybutton'
    ...
]
```

以此实现兼容 `google/freeform` 样式示例代码的一键复制特性。

另外，`.. code-block::` 此指令建议保留：

- 此指令是 `sphinx` 对于代码块的识别指令

- Paddle 代码与 Paddle docs 中多处使用此指令识别代码块

虽然 `.. code-block::` 存在与否不会影响示例代码的检查，但是若移除此指令，`sphinx` 则无法识别代码块，而且需要修改 Paddle 代码与 Paddle docs 中大量相关依赖，所以建议保留。

最后，在完成示例代码的修改之前，由于新旧样式示例代码共存，文档中可能同时出现新旧两种样式的示例代码。这里建议不对这种情况再进行处理：

- 新旧样式的示例代码没有好的手段进行统一显示(如果有，则也无需进行大量的人工修改工作)。

- 如果修改新旧样式的示例代码为统一显示格式，则必然会出现某一种的显示与实际代码块中不符的情况。

- 如果能够保证单一 API 的显示统一，则不同 API 的示例代码显示即使不同，对于用户的使用体验影响也较小。

### 3.2 代码检查移交

在以上 `xdoctest` 的引入过程中，Paddle doc 与 Paddle 代码的检查都同时存在，且互不干扰。当完成 `xdoctest` 的引入，以及 `google/freeform` 样式的示例代码的修改之后，可以考虑移除 Paddle doc 中的代码检查，以减少重复的检查流程。

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

- 修改目前 Paddle docs 中 `COPY-FROM` 的逻辑

    需要增加单元测试，保证修改后的工具可以正确抽取 `Examples` 与其他部分的示例代码，并插入到中英文文档正确的位置。

- 修改目前 Paddle docs 中仍使用 `code-block` 的示例代码为 `COPY-FROM`

    需要保证修改后的文档可以正确引入示例代码，示例代码检测通过。

- 修改目前 Paddle docs 的代码检查方式

    需要增加单元测试，保证修改后的工具能够同时兼容目前的代码示例样式与 `google/freeform` 样式的检查。

- Paddle 代码 CI 中引入 `xdoctest` 检查

    需要增加单元测试，保证修改/新增的 CI 流程，不影响目前的 Paddle 代码的检查，且能够进行 `google/freeform` 样式的代码检查。

- 分批次修改已有代码的示例

    需要保证修改后的代码能够通过新的 CI 流程检查。

- 更新文档《开发 API Python 端》与《API 文档书写规范》

    需要保证更新后的开发文档能够指导新的代码示例的开发。

- 不再兼容旧格式

    需要增加单元测试，保证修改后的 CI 流程不能使用目前旧格式的示例代码。

- 中英文 API 文档特性更新

    需要保证修改后的 API 文档页面能够正确复制示例代码。

- 代码检查移交

    需要保证示例代码的检查工作完全移交至 Paddle 代码的 CI 流程中，需要保证 Paddle docs 的 CI 流程能够正确复制示例代码。

# 五、排期规划

- (1) 修改目前 Paddle docs 中 `COPY-FROM` 的逻辑 (a)
- (2) 修改目前 Paddle docs 中仍使用 `code-block` 的示例代码为 `COPY-FROM` (b)
- (3) 修改目前 Paddle docs 的代码检查方式 (a)
- (4) Paddle 代码 CI 中引入 `xdoctest` 检查 (a)
- (5) 分批次修改已有代码的示例 (b)
- (6) 更新文档《开发 API Python 端》与《API 文档书写规范》(b)
- (7) 不再兼容旧格式 (c)
- (8) 中英文 API 文档特性更新 (c)
- (9) 代码检查移交 (d)

圆括号中的字母 `(x)` 表示任务优先级，原则上后一等级的任务需要依赖前一等级任务的完成。

其中:

- 任务 `(2)` 和 `(5)`，不涉及具体框架逻辑的修改，且任务量较大，建议借助 `快乐开源` 等活动进行。

- 任务 `(1) (3) (4) (6) (7) (9) `，涉及 CI 流水线的修改，建议分批次（依优先级）提交 `ISSUE`，开发者认领并开发。

- 任务 `(8)`，不在本项目的范围内，另作安排。

# 六、影响面

- 修改目前 Paddle docs 中 `COPY-FROM` 的逻辑

    1. 对用户的影响

        不涉及

    2. 对开发者的影响

        - 开发者需要统一使用 `COPY-FROM` 指令

        - 开发者可以在 docstring 的任意地方使用 `COPY-FROM` 指令

    3. 对框架架构的影响

        - 修改 Paddle docs 中 `docs/docs/api/copy_codes_from_en_doc.py` 示例抽取逻辑

        - 改变 Paddle docs 对于代码的 编写/抽取 方式，包括 `Examples` 以及整个 docstring

- 修改目前 Paddle docs 中仍使用 `code-block` 的示例代码为 `COPY-FROM`

    1. 对用户的影响

        不涉及

    2. 对开发者的影响

        开发者需要统一使用 `COPY-FROM` 指令

    3. 对框架架构的影响

        可以使用 `xdoctest` 的 `freeform style`，用以检查 docstring 中所有使用 `>>>` 的示例代码检查。

- 修改目前 Paddle docs 的代码检查方式

    1. 对用户的影响

        不涉及

    2. 对开发者的影响

        - 前期，开发者需要对 `xdoctest` 的引入无感知
        - 中后期，开发者需要满足 `xdoctest` 对于示例代码样式的要求

    3. 对框架架构的影响

        修改 Paddle docs 中的 `docs/ci_scripts/chinese_samplecode_processor.py` 分流 `google/freeform` 样式代码，但不做检查。

    4. 其他风险

        后续移除 Paddle docs 中的示例代码检查

- Paddle 代码 CI 中引入 `xdoctest` 检查

    1. 对用户的影响

        不涉及

    2. 对开发者的影响

        - 前期，开发者需要对 `xdoctest` 的引入无感知
        - 中后期，开发者需要满足 `xdoctest` 对于示例代码样式的要求

    3. 对框架架构的影响

        - `Paddle/tools/sampcd_processor.py` 引入 `xdoctest` 检查。前期，分流 `google/freeform` 样式代码至 `xdoctest`，其他流程不变。中后期，全部代码检查移至 `xdoctest`，删除旧的代码检查。

        - `Paddle/python/unittest_py/requirements.txt` 增加 `xdoctest` 依赖项。

    4. 其他风险

        `sampcd_processor.py` 中存在 `>>>` 样式的检查，显示已经遗弃，不知道中间有什么考量？

- 分批次修改已有代码的示例

    1. 对用户的影响

        在 新/旧 示例代码样式切换阶段，用户可能在官网文档中看到两种样式的示例代码。

    2. 对开发者的影响

        开发者可以使用新样式编写示例代码。

    3. 对框架架构的影响

        框架中逐步将就样式的示例代码替换为新样式。

    4. 其他风险

        - 由于示例代码是跟随版本发布的，所以，存在 `现有示例代码` 不适应 `最新版本` 的情况

        - 现有示例代码存量较大(近2000个)，存在修改工作量风险。

- 更新文档《开发 API Python 端》与《API 文档书写规范》

    1. 对用户的影响

        用户可以查看新的文档编写规范。

    2. 对开发者的影响

        开发者可以根据更新的文档编写示例代码。

    3. 对框架架构的影响

        不涉及

    4. 其他风险

        - 新样式存在很多具体的情况（如：CPU/GPU等环境如何处理之类），需要及时更新。

        - 新旧样式的切换，需要设定一个时间点，最好随某一个版本发布，否则可能出现大量 PR 提交失败的情况。

- 不再兼容旧格式

    1. 对用户的影响

        不涉及

    2. 对开发者的影响

        开发者需要使用新样式编写示例代码。

    3. 对框架架构的影响

        框架不再兼容旧格式的示例代码

    4. 其他风险

        对于仍使用旧格式的开发者提交的 PR 需要进行引导。

- 中英文 API 文档特性更新（推荐）

    1. 对用户的影响

        用户可以正确的复制 `>>>` 标识的代码。

    2. 对开发者的影响

        开发者可以正确的复制 `>>>` 标识的代码。

    3. 对框架架构的影响

        不涉及（需要官网开发支持）

    4. 其他风险

        此特性为完成之前，用户可能无法正确复制示例代码

- 代码检查移交

    1. 对用户的影响

        用户从官网只能看到新样式的示例代码

    2. 对开发者的影响

        不涉及

    3. 对框架架构的影响

        框架中现存的示例代码检查，需要处理：

        - 存在于 Paddle 代码的 CI： **保留**
        - 存在于 Paddle docs 的 CI： **移除**
        - 存在于 Paddle docs 的 build： **移除**

    4. 其他风险

        不能在 更新/移除 各部分的示例代码检查后，引入遗漏检查等问题。
