# 框架易用性提升——单测报错信息优化

| 任务名称     | 单测报错信息优化                                |
| ------------ | ----------------------------------------------- |
| 提交作者     | Nyakku Shigure(@SigureMo)、何双池（@Yulv-git）  |
| 提交时间     | 2022-08-05                                      |
| 版本号       | v0.2                                            |
| 依赖飞桨版本 | develop                                         |
| 文件名       | 20220805_code_style_improvement_for_unittest.md |

## 一、概述

### 1、相关背景

来源于 GitHub Paddle repo 下的一个 issue [Recommend to use np.testing.assert_allclose instead of assertTrue(np.allclose(...)) #44641](https://github.com/PaddlePaddle/Paddle/issues/44641)

由于 Paddle repo 中的单测在进行比较时大多使用的是 `self.assertTrue(np.allclose(a, b))` 来断言两个 np.ndarray 之间在容忍误差范围内相等，而该报错信息是由 `self.assertTrue` 控制的，也就是说在 `self.assertTrue` 看来只有 `True` 和 `False` 的分别，报错信息也只会提示 `np.allclose(...)` 返回的是 `False`，就像下面这样：

```python
# in unittest class
self.assertTrue(np.allclose(x, y), "compare x and y")
# AssertionError: False is not true : compare x and y
```

这对于单测的报错信息是不友好的，既不能知道 `a` 和 `b` 的的值是多少，也不知道到底问题（diff）出现在哪里，另外，`np.allclose` 对于 `shape` 也是不敏感的，`np.allclose` 会在比较时自动应用广播机制，因此有些时候测试的检查并不是全面的。

NumPy 还有一个函数 `np.testing.assert_allclose` 是专门用于单元测试的，它可以提供更友好的报错信息，比如下面的示例：

```python
np.testing.assert_allclose(x, y, err_msg="compare x and y")
# AssertionError:
# Not equal to tolerance rtol=0, atol=0
# compare x and y
# Mismatched elements: 1 / 3 (33.3%)
# Max absolute difference: 1
# Max relative difference: 0.33333333
#  x: array([1, 2, 3])
#  y: array([1, 3, 3])
```

该函数不仅可以展示两个 ndarray 各自的值，而且可以将差异的一些统计信息展示出来，帮助开发人员快速定位和解决问题。

### 2、功能目标

#### 目标一：修改现有单测，优化报错信息

修改 Paddle 现有单测中的利用 `self.assertEqual` 或者 `self.assertTrue` 来对 np.ndarray 进行比较断言的代码，替换成合适的 `np.testing` 函数，这主要包含了以下几种情况（模式）。

- `self.assertTrue(np.allclose(...))`，需要修改为 `np.testing.assert_allclose(...)`
- `self.assertTrue(np.array_equal(...))`，视情况而定
  - 如果测试的数据的数据类型为 `float`（含 `float16`、`float32` 等）类型，则需要修改为 `np.testing.assert_allclose(...)`
  - 如果测试的数据的数据类型为 `int`（含 `int32`、`int64` 等）类型，则需要修改为 `np.testing.assert_array_equal(...)`

> **Note**
>
> 还需要考虑一些等价情况，比如 `self.assertEqual(..., True)` 与 `self.assertTrue(...)` 等价，因此 `self.assertEqual(np.allclose(...), True)` 等价于 `self.assertTrue(np.allclose(...))`，`self.assertEqual(np.array_equal(...), True)` 等价于 `self.assertTrue(np.array_equal(...))`。
>
> 另外，还有部分单测仅仅使用了 `import numpy` 而非 `import numpy as np`，因此上述模式中的 `np` 在替换为 `numpy` 时也是等价的。

#### 目标二：添加 CI 检测脚本，阻止增量问题

为了避免新增的测试用例再次使用 `self.assertEqual` 或 `self.assertTrue` 来对 np.ndarray 进行比较断言，可在 CI 中添加一个对增量代码中字段进行监控的检查，由于该问题的模式比较简单，可直接利用 grep 来检测 `self.assertTrue(np.allclose(` 字段即可。这可以参考 Paddle CI 中现有的报错信息监控部分：

```bash
# https://github.com/PaddlePaddle/Paddle/blob/ce9d2a9ec4daa8e0809eac7f44d731ed8189dc66/tools/check_file_diff_approvals.sh#L247-L254
# tools/check_file_diff_approvals.sh
ALL_ADDED_LINES=`git diff -U0 upstream/$BRANCH |grep "^+" || true`
ALL_PADDLE_CHECK=`echo $ALL_ADDED_LINES |grep -zoE "(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" || true`
VALID_PADDLE_CHECK=`echo "$ALL_PADDLE_CHECK" | grep -zoE '(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\((.[^,;]+,)*.[^";]*(errors::).[^"]*".[^";]{20,}.[^;]*\);\s' || true`
INVALID_PADDLE_CHECK=`echo "$ALL_PADDLE_CHECK" |grep -vxF "$VALID_PADDLE_CHECK" || true`
if [ "${INVALID_PADDLE_CHECK}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="The error message you wrote in PADDLE_ENFORCE{_**} or PADDLE_THROW does not meet our error message writing specification. Possible errors include 1. the error message is empty / 2. the error message is too short / 3. the error type is not specified. Please read the specification [ https://github.com/PaddlePaddle/Paddle/wiki/Paddle-Error-Message-Writing-Specification ], then refine the error message. If it is a mismatch, please request chenwhql (Recommend), luotao1 or lanxianghit review and approve.\nThe PADDLE_ENFORCE{_**} or PADDLE_THROW entries that do not meet the specification are as follows:\n${INVALID_PADDLE_CHECK}\n"
    check_approval 1 6836917 47554610 22561442
fi
```

### 3、意义

修改为 `np.testing` 模块下的函数来进行提示可以极大优化单测的提示信息，为开发人员定位错误问题提供更全面的参考信息。

## 二、实现方案

### 目标一

#### 文本替换问题

对于现有代码来说 `self.assertTrue(np.allclose(...))` 是一个非常简单的模式，其前缀 `self.assertTrue(np.allclose(` 是完全可以通过正则甚至简单的文本搜索来搜索到，但如果想要无错漏地将整个模式匹配出来进行替换，可能正则表达式并不能很好地完成（要考虑到括号是可以无限嵌套的，而正则表达式是不能表达无限嵌套的，除非将其嵌套限制在一个深度，但那样写出来的正则可读性也极差）

而对于 Python 代码的解析，当然最好的方式是直接将其翻译为 Python 的语法树，然后在语法树上匹配相应的模式并进行替换即可。Python 代码到语法树的解析，我们可以利用 builtin 的 `ast` 模块，以下是一个目前实现的 `self.assertTrue(np.allclose(...))` 替换的简单 demo：

```python
# required: python >= 3.10
import ast
from typing import Optional

class TransformAssertTrueAllClose(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        transformed_node: ast.AST
        match node:
            case ast.Call(
                func=ast.Attribute(value=ast.Name(id="self"), attr="assertTrue"),
                args=[
                    ast.Call(
                        func=ast.Attribute(value=ast.Name(id="np"), attr="allclose"),
                        args=allclose_args,
                        keywords=allclose_kwargs,
                    ),
                    *assert_true_args,
                ],
                keywords=assert_true_kwargs,
            ):
                actual: ast.AST
                desired: ast.AST
                rtol: Optional[ast.AST] = None
                atol: Optional[ast.AST] = None
                equal_nan: Optional[ast.AST] = None
                err_msg: Optional[ast.AST] = None

                # https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertEqual
                # self.assertTrue(np.allclose(...), assert_true_args)
                # self.assertTrue(np.allclose(...), msg=assert_true_kwargs)
                if assert_true_args:
                    err_msg = assert_true_args[0]
                for kw in assert_true_kwargs:
                    if kw.arg == "msg":
                        err_msg = kw.value

                # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
                assert allclose_args, "allclose_args is empty"
                # parse actual and desired
                actual = allclose_args[0]
                desired = allclose_args[1]
                # parse rtol and atol from remaining args
                if len(allclose_args) > 2:
                    rtol = allclose_args[2]
                    if len(allclose_args) > 3:
                        atol = allclose_args[3]
                        if len(allclose_args) > 4:
                            equal_nan = allclose_args[4]
                # or parse from kwargs
                for kw in allclose_kwargs:
                    if kw.arg == "rtol":
                        rtol = kw.value
                    elif kw.arg == "atol":
                        atol = kw.value
                    elif kw.arg == "equal_nan":
                        equal_nan = kw.value

                # https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html
                # testing.assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True, err_msg='', verbose=True)
                keyword_args: list[ast.AST] = []
                if rtol is not None:
                    keyword_args.append(ast.keyword(arg="rtol", value=rtol))
                if atol is not None:
                    keyword_args.append(ast.keyword(arg="atol", value=atol))
                if equal_nan is not None:
                    keyword_args.append(ast.keyword(arg="equal_nan", value=equal_nan))
                if err_msg is not None:
                    keyword_args.append(ast.keyword(arg="err_msg", value=err_msg))
                transformed_node = ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="testing", ctx=ast.Load()),
                        attr="assert_allclose",
                        ctx=ast.Load(),
                    ),
                    args=[actual, desired],
                    keywords=keyword_args,
                )
            case _:
                transformed_node = node
        return transformed_node



code = """
self.assertTrue(
    np.allclose(res[0],
        feed_add,
        rtol=1e-5),
    # 这个字符串里括号都不匹配，正则可是很难写的
    msg='blabla((((()()((xxxdfdf('
)
"""
tree = ast.parse(code)
new_tree = ast.fix_missing_locations(TransformAssertTrueAllClose().visit(tree))

print("Before:")
print(ast.dump(tree, indent=4))

# Before:
# Module(
#     body=[
#         Expr(
#             value=Call(
#                 func=Attribute(
#                     value=Attribute(
#                         value=Name(id='np', ctx=Load()),
#                         attr='testing',
#                         ctx=Load()),
#                     attr='assert_allclose',
#                     ctx=Load()),
#                 args=[
#                     Subscript(
#                         value=Name(id='res', ctx=Load()),
#                         slice=Constant(value=0),
#                         ctx=Load()),
#                     Name(id='feed_add', ctx=Load())],
#                 keywords=[
#                     keyword(
#                         arg='rtol',
#                         value=Constant(value=1e-05)),
#                     keyword(
#                         arg='err_msg',
#                         value=Constant(value='blabla((((()()((xxxdfdf('))]))],
#     type_ignores=[])

print("After:")
print(ast.dump(new_tree, indent=4))

# After:
# Module(
#     body=[
#         Expr(
#             value=Call(
#                 func=Attribute(
#                     value=Attribute(
#                         value=Name(id='np', ctx=Load()),
#                         attr='testing',
#                         ctx=Load()),
#                     attr='assert_allclose',
#                     ctx=Load()),
#                 args=[
#                     Subscript(
#                         value=Name(id='res', ctx=Load()),
#                         slice=Constant(value=0),
#                         ctx=Load()),
#                     Name(id='feed_add', ctx=Load())],
#                 keywords=[
#                     keyword(
#                         arg='rtol',
#                         value=Constant(value=1e-05)),
#                     keyword(
#                         arg='err_msg',
#                         value=Constant(value='blabla((((()()((xxxdfdf('))]))],
#     type_ignores=[])

print("Transformed code:", ast.unparse(new_tree))
# Transformed code: np.testing.assert_allclose(res[0], feed_add, rtol=1e-05, err_msg='blabla((((()()((xxxdfdf(')
```

可以看到，通过 AST 解析的方式可以轻松实现对 Python 代码的转换，可读性也非常好，也可以轻松涵盖 Python 中既支持位置参数也支持关键字参数的各种情况。

但由于 AST 转换并不是无损的（在 Python 语法层面无损，但比如注释之类的无法保留），因此不能直接将代码文件转换后整个写回，而是应当对匹配到的位置进行局部替换。由于 AST 上是包含 `lineno`、`col_offset` 等信息的，这些目前也已经实现。

#### 类型判断问题

对于 `self.assertTrue(np.array_equal(...))` 是要判断类型来进一步替换的，而在本问题中，在代码语法分析阶段（或者说编译时）基本不可能把所有类型没有疏漏地推断出来，最可靠的只有运行时来进行类型判断，这需要将 `np.array_equal` 参数的 `dtype` 打印出来，因此需要在 `np.array_equal(a, b)` 代码下面插入类似 `print(__file__, a.dtype, b.dtype)` 的代码来将类型及文件名打印出来，这样可以知道各个文件中的数据类型是什么了。

但这样有一个问题是依赖于运行时，而且需要预先插入 `print` 相关的代码，在实现上稍有麻烦。

另一种解决方案是直接用肉眼对代码上下文进行判断，进而判断比较变量的数据类型，但由于上下文一般比较复杂，有些数据的数据类型信息隐藏的很深，因此可能判断并不准确。

为了避免全部肉眼判断与手动替换带来的巨大工作量，这里首先将 `self.assertTrue(np.array_equal(...))` 全部自动替换为 `np.testing.assert_array_equal(...)`，之后对现存 `np.testing.assert_array_equal(...)` 进行搜索，手动将 `float` 类型数据的替换为 `np.testing.assert_allclose(...)`。

在已有的测试中（https://github.com/PaddlePaddle/Paddle/pull/44947），已经尝试将 `self.assertTrue(np.array_equal(...))` 全部自动替换为 `np.testing.assert_array_equal(...)`，CI 没有任何问题，因此该替换是安全的，因此在不进行手工替换的情况下也是没有任何问题的。但为了让测试更加严谨，应当手动再将 `float` 的替换为 `np.testing.assert_allclose(...)`。在无法立即判断的情况下保持现状。

<!-- TODO: 这里的描述需要优化，方案目前还没有完全确定 -->

### 目标二

如目标一中所述，`self.assertTrue(np.allclose(...))` 这一文本模式前缀（`self.assertTrue(np.allclose(`）的匹配是非常简单的，使用简单的正则足矣。在 yapf 自动格式化的前提下，该模式不会过于复杂化，唯一需要额外考虑的情况是有可能如下折行的情况：

```python
self.assertTrue(
    np.allclose(...))

# 一个现有的案例
# https://github.com/PaddlePaddle/Paddle/blob/ce9d2a9ec4daa8e0809eac7f44d731ed8189dc66/python/paddle/fluid/tests/unittests/test_sparse_elementwise_op.py#L112-L123
# python/paddle/fluid/tests/unittests/test_sparse_elementwise_op.py
self.assertTrue(
    np.allclose(expect_res.numpy(),
                actual_res.to_dense().numpy(),
                equal_nan=True))
self.assertTrue(
    np.allclose(dense_x.grad.numpy(),
                coo_x.grad.to_dense().numpy(),
                equal_nan=True))
self.assertTrue(
    np.allclose(dense_y.grad.numpy(),
                coo_y.grad.to_dense().numpy(),
                equal_nan=True))
```

因此正则需要覆盖这一情况。此外当然需要考虑 `np.array_equal` 的情况及之前提到的一些等价情况，根据这些目前拟定的正则为 `self\.assert(True|Equal)\(\s*(np|numpy)\.(allclose|array_equal)`，可在后续开发过程中根据其他边界情况进行细化。

在关键词触发时应当正确地阻止提交并给出明确的提示信息，并给出 Wiki 链接以详细说明问题，因此需要编写相应的 Wiki 页面。

check_approval 的人是 qili93 (Recommend), luotao1。

## 三、任务分工和排期规划

整个任务可以根据前文所述的目标一和目标二进行划分，两个目标分别由两人各自主导，另外一人进行辅助（视具体工作量和完成进度而定）

- 目标一：存量修改
  - 主要负责人：Nyakku（@SigureMo）
  - 主要工作内容
    - 编写脚本修改现有单测代码「半周内」
      - `self.assertTrue(np.allclose(...))` -> `np.testing.assert_allclose(...)`
      - `self.assertTrue(np.array_equal(...))` -> `np.testing.assert_array_equal(...)`
    - 根据 CI 结果进行调试，尽可能使其通过「一周内」
    - 肉眼判断现有的 `np.testing.assert_array_equal(...)`，将判断为 `float` 类型的数据修改为 `np.testing.assert_allclose(...)`「一周半内」
- 目标二：增量阻止
  - 主要负责人：何双池（@Yulv-git）
  - 主要工作内容
    - 编写 Wiki 页面（可参考本 RFC 和 [#44641](https://github.com/PaddlePaddle/Paddle/issues/44641)）「一周内」
    - 编写 CI 增量阻止脚本「一周内」
    - 调试脚本使其能够正确工作（需测试存在关键词的 commit 确实无法提交）「一周内」

工作小组可互相配合，共同完成目标一和目标二的工作。

整体工作大概为三周，可根据实际情况提前完成，主要工作时间为 8 月 8 日（周一）到 8 月 29 日（周一），预计在八月末完成全部工作。

## 四、其他注意事项及风险评估

由于 Paddle 内部已经有部分 IPU、NPU、XPU、MLU 相关单测的优化，因此在修改过程中应当注意避开这些单测文件（`*_mlu.py,*_ipu.py,*_npu.py,*_xpu.py`）。

CI 实在过不去的单测可交给 Paddle 内部修复，但这应当是在充分测试后确定自身无法修复的情况下。

## 五、影响面

不会也不应该对现有模块产生任何影响，会极大优化现有单测的报错信息，提高开发人员的开发效率。

## 名词解释

- AST（Abstract Syntax Tree）：抽象语法树

## 附件及参考资料

- [issue #44641](https://github.com/PaddlePaddle/Paddle/issues/44641)
- [NumPy testing module docs](https://numpy.org/doc/stable/reference/routines.testing.html)
- [Python ast reference](https://docs.python.org/3/library/ast.html)
- [Paddle CI `check_file_diff_approvals` script](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/check_file_diff_approvals.sh)
