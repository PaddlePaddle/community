# 框架易用性提升——单测报错信息优化

| 任务名称     | 单测报错信息优化                                |
| ------------ | ----------------------------------------------- |
| 提交作者     | Nyakku Shigure(@SigureMo)、何双池（@Yulv-git）  |
| 提交时间     | 2022-08-05                                      |
| 版本号       | v1.2                                            |
| 依赖飞桨版本 | develop                                         |
| 文件名       | 20220805_code_style_improvement_for_unittest.md |

## 一、概述

### 1、相关背景<a id='background'></a>

来源于 GitHub Paddle repo 下的一个 issue [Recommend to use np.testing.assert_allclose instead of assertTrue(np.allclose(...)) #44641](https://github.com/PaddlePaddle/Paddle/issues/44641)

由于 Paddle repo 中的单测在进行比较时大多使用 `self.assertTrue(np.allclose(a, b))` 来断言两个 np.ndarray 在容忍误差范围内相等，而 `self.assertTrue` 只会区分 `True` 和 `False`，即报错信息只会提示 `np.allclose(...)` 返回的是 `False`，就像下面这样：

```python
# in unittest class
self.assertTrue(np.allclose(x, y), "compare x and y")
# AssertionError: False is not true : compare x and y
```

这样的单测报错信息是不友好的，既不能知道 `a` 和 `b` 的值是多少，也不知道到底问题（diff）出现在哪里。另外，`np.allclose` 对于 `shape` 也是不敏感的，`np.allclose` 会在比较时自动应用广播机制，因此有些时候测试的检查并不是全面的。

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

因此，建议使用 `np.testing.assert_allclose(...)` 来替代 `self.assertEqual(np.allclose(...))`，并使用 `np.testing.assert_array_equal` 来替代 `self.assertTrue(np.array_equal(...))`，以提供更全面的错误信息。

> **Note**
>
> 在进行替换时，应当注意两者在默认参数上的差异，这可能是导致替换后单测无法通过的主要原因。

### 2、功能目标

#### 目标一：修改现有单测，优化报错信息

修改 Paddle 现有单测中的利用 `self.assertEqual` 或者 `self.assertTrue` 来对 np.ndarray 进行比较断言的代码，替换成合适的 `np.testing` 函数，这主要包含了以下几种情况（模式）。

- `self.assertTrue(np.allclose(...))`，需要修改为 `np.testing.assert_allclose(...)`
- `self.assertTrue(np.array_equal(...))`，需要修改为 `np.testing.assert_array_equal(...)`，如果修改后因为硬件本身的精度问题导致测试无法通过，需要修改为 `np.testing.assert_allclose(...)`，并搭配其参数 `atol` 和 `rtol` 来保证 CI 通过。

> **Note**
>
> 还需要考虑一些等价情况，比如 `self.assertEqual(..., True)` 与 `self.assertTrue(...)` 等价，因此 `self.assertEqual(np.allclose(...), True)` 等价于 `self.assertTrue(np.allclose(...))`，`self.assertEqual(np.array_equal(...), True)` 等价于 `self.assertTrue(np.array_equal(...))`。
>
> 再比如说 `self.assertTrue(np.isclose(...).all())` 也是与 `self.assertTrue(np.allclose(...))` 等价的。
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

对于现有代码来说 `self.assertTrue(np.allclose(...))` 是一个非常简单的模式，其前缀 `self.assertTrue(np.allclose(` 是完全可以通过正则甚至简单的文本搜索来搜索到，但如果想要无错漏地将整个模式匹配出来进行替换，可能正则表达式并不能很好地完成（要考虑到括号是可以无限嵌套的，而正则表达式是不能表达无限嵌套的，除非将其嵌套限制在一个深度，但那样写出来的正则可读性也极差）。

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

                # https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertTrue
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

print("Before:")
print(ast.dump(tree, indent=4))

# Before:
# Module(
#     body=[
#         Expr(
#             value=Call(
#                 func=Attribute(
#                     value=Name(id='self', ctx=Load()),
#                     attr='assertTrue',
#                     ctx=Load()),
#                 args=[
#                     Call(
#                         func=Attribute(
#                             value=Name(id='np', ctx=Load()),
#                             attr='allclose',
#                             ctx=Load()),
#                         args=[
#                             Subscript(
#                                 value=Name(id='res', ctx=Load()),
#                                 slice=Constant(value=0),
#                                 ctx=Load()),
#                             Name(id='feed_add', ctx=Load())],
#                         keywords=[
#                             keyword(
#                                 arg='rtol',
#                                 value=Constant(value=1e-05))])],
#                 keywords=[
#                     keyword(
#                         arg='msg',
#                         value=Constant(value='blabla((((()()((xxxdfdf('))]))],
#     type_ignores=[])

new_tree = ast.fix_missing_locations(TransformAssertTrueAllClose().visit(tree))
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

可以看到，通过 AST 解析的方式可以轻松实现对 Python 代码的转换，可读性和可调试性也非常好，也可以轻松涵盖 Python 中既支持位置参数也支持关键字参数的各种情况。

但由于 AST 转换并不是无损的（在 Python 语法层面无损，但比如注释之类的无法保留），因此不能直接将代码文件转换后整个写回，而是应当对匹配到的位置进行局部替换。由于 AST 上是包含 `lineno`、`col_offset` 等信息的，因此是完全可以实现的，目前也已经尝试了相应的实现。

#### 测试通过性问题

经过测试发现，`np.array_equal` -> `np.testing.assert_array_equal` 基本上没有问题，仅仅会在某些特殊硬件上会出现些精度的问题，在这种情况下应当使用 `np.testing.assert_allclose`，并根据两者误差调整 `rtol` 和 `atol`。

而 `np.allclose` -> `np.testing.assert_allclose` 出现了大量测试（CI）失败，经排查问题主要出在以下两个方面：

- 精度问题

  这是由于两者默认值不同

  - `np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)`
  - `np.testing.assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True, err_msg='', verbose=True)`

  明显 `np.testing.assert_allclose` 精度要求更高，所以当 `np.allclose` 无 `rtol`（即设为默认值）时将值修改为 `1e-5`，使其与原来行为一致，本着在测试能通过的情况下尽可能使用默认参数的原则，`atol` 暂时没有修改。

  修改后测试失败减少了一半，余量已经很少了（20 个左右），经检查，余量大多都是些误差非常小的（`1e-10` 以下），针对这些手动加上 `atol=1e-8` 即可解决。

- shape 不对齐

  这是由于 `np.allclose` 在比较时会自动 broadcast，而 `np.testing.allclose` 不会，因此需要手动对这些数据进行检查及修改。

  修改精度问题后，本问题占了 90% 以上，仍然很难逐个手动修复。

  目前发现最主要的问题是，静态图执行结果是一个 list，但有的开发者直接将返回值进行比较，这会在比较时认为静态图结果的 shape 比预期值多一维度。

  ```python
  # https://github.com/PaddlePaddle/Paddle/blob/c91aaced74aa1a34c8bde2e53b3072baf8012e73/python/paddle/fluid/tests/unittests/test_softmax2d.py#L32-L41
  def test_static_api(self):
      paddle.enable_static()
      with paddle.static.program_guard(paddle.static.Program()):
          x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
          m = paddle.nn.Softmax2D()
          out = m(x)
          exe = paddle.static.Executor(self.place)
          res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
      out_ref = ref_softmax(self.x_np, self.axis)
      self.assertTrue(np.allclose(out_ref, res))
  ```

  以及在 `dygraph_to_static` 目录下的测试中经常使用的 `PredictorTools` 实例对象调用后的返回值也是一个 list，解决方案同上。

  此外还有少许误操作，需要逐一排查。

此外还发现有的测试静态图代码报错，经排查发现是由于 shape 不对齐问题导致测试不通过而无法转为静态图，进而后续静态图测试全部无法通过，本质上还是问题二。

对于精度问题，上面已经给出解决方案，修改后已经减少到了一个非常小的数字。

对于 shape 不对齐问题，需要考虑问题是否是因为静态图返回的原因导致的，目前该原因导致的问题基本上报错类似下面这种：

```text
2022-08-09 00:59:56 ======================================================================
2022-08-09 00:59:56 FAIL: test_static_api (test_softmax2d.TestSoftmax2DAPI)
2022-08-09 00:59:56 ----------------------------------------------------------------------
2022-08-09 00:59:56 Traceback (most recent call last):
2022-08-09 00:59:56   File "/workspace/Paddle/build/python/paddle/fluid/tests/unittests/test_softmax2d.py", line 41, in test_static_api
2022-08-09 00:59:56     np.testing.assert_allclose(out_ref, res, rtol=1e-05)
2022-08-09 00:59:56   File "/opt/_internal/cpython-3.7.0/lib/python3.7/site-packages/numpy/testing/_private/utils.py", line 1531, in assert_allclose
2022-08-09 00:59:56     verbose=verbose, header=header, equal_nan=equal_nan)
2022-08-09 00:59:56   File "/opt/_internal/cpython-3.7.0/lib/python3.7/site-packages/numpy/testing/_private/utils.py", line 763, in assert_array_compare
2022-08-09 00:59:56     raise AssertionError(msg)
2022-08-09 00:59:56 AssertionError:
2022-08-09 00:59:56 Not equal to tolerance rtol=1e-05, atol=0
2022-08-09 00:59:56
2022-08-09 00:59:56 (shapes (2, 6, 5, 4), (1, 2, 6, 5, 4) mismatch)
2022-08-09 00:59:56  x: array([[[[0.099922, 0.185298, 0.217424, 0.078946],
2022-08-09 00:59:56          [0.09362 , 0.356357, 0.193014, 0.055791],
2022-08-09 00:59:56          [0.313798, 0.074912, 0.173342, 0.101163],...
2022-08-09 00:59:56  y: array([[[[[0.099922, 0.185298, 0.217424, 0.078946],
2022-08-09 00:59:56           [0.09362 , 0.356357, 0.193014, 0.055791],
2022-08-09 00:59:56           [0.313798, 0.074912, 0.173342, 0.101163],...
```

可以看到 y 相对于 x 多了第一个维度，而且该维度为 1，可以认为这种模式的错误都是由于 y 是静态图返回的结果导致的，因此可以从 log 中分别提取出 `test_function`（`test_static_api`）、`test_file`（`test_softmax2d`）、`test_case`（`TestSoftmax2DAPI`）以及静态图返回的结果变量（本例中是右值 y），之后在替换时在该变量后加上 `[0]`。

此外，在统计后可以发现该现象发生的聚集性较强，一般出问题的整个测试代码文件都会有此问题，因此也可以尝试手动对剩余文件进行修复。具体方案将在后续开发中进一步确定。

### 目标二

#### 关键词匹配

如目标一中所述，`self.assertTrue(np.allclose(...))` 这一文本模式前缀（`self.assertTrue(np.allclose(`）的匹配是非常简单的，使用简单的正则即可。在 yapf 自动格式化的前提下，该模式不会过于复杂化，唯一需要额外考虑的情况是有可能如下折行的情况：

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

因此正则需要覆盖这一情况。此外当然需要考虑 `np.array_equal` 的情况及之前提到的一些等价情况，根据这些目前拟定的正则如下：

```text
self\.assert(True|Equal)\(\s*(np|numpy)\.(allclose|array_equal)
                 │         │    │                 │
                 │         │    │                 └─────────  两种需要修改替换的函数
                 │         │    └───────────────────────────  等价情况：np 与 numpy
                 │         └────────────────────────────────  边界情况：折行
                 └──────────────────────────────────────────  等价情况：self.assertTrue(...) 与 self.assertEqual(..., True)
```

这里未阻止 `np.isclose(...).all()` 与 `np.allclose(...)` 这一等价情况是为了避免在有 `np.isclose(...).any()` 的使用需求时的误检问题，比如下面的测试就是这样使用的：

```python
# https://github.com/PaddlePaddle/Paddle/blob/9b35f03572867bbca056da93698f36035106c1f3/python/paddle/fluid/tests/custom_op/test_custom_relu_op_setup.py#L323-L326
# python/paddle/fluid/tests/custom_op/test_custom_relu_op_setup.py
self.assertTrue(
    np.isclose(predict, predict_infer, rtol=5e-5).any(),
    "custom op predict: {},\n custom op infer predict: {}".format(
        predict, predict_infer))
```

#### 信息提示

为了获取增量代码，可首先使用 `git diff` 来获取当前 PR 的 diff，并通过 `grep` 匹配出开头为 `+` 的行。因此此时所有行都是以 `+` 开头的，在考虑折行情况时需要对正则进行调整。

为了匹配折行的情况，需要开启 `grep` 的 `-z` 选项，以忽略换行符，但相应的由于没有换行符界定，所以返回的结果将会是整个输入，而不是匹配到的几行，这样的结果是无法直接进行提示的。因此需要开启 `-o` 选项仅仅返回匹配的部分，并且调整正则使其匹配的部分不仅仅是关键词部分，而是包含关键词的行。

根据这些，可拟定以下的 `grep` 命令：

```text
grep -zoE '\+\s+self\.assert(True|Equal)\((\s*\+\s*)?(np|numpy)\.(allclose|array_equal)[^+]*'
            │                                  │                                       │
            │                                  │                                       └──────  尾行剩余部分内容（要求不包含 +）
            │                                  └──────────────────────────────────────────────  折行部分，包含一个 +
            └─────────────────────────────────────────────────────────────────────────────────  首行开头的 +
```

由于使用 `-z` 无法通过换行符来界定不同行，这里利用了增量代码每行开头是 `+` 的特性，以起到替代换行符的作用。但该方法有一个很明显的缺陷，就是要求尾行剩余内容不包含 `+`，否则将会认为是一个换行而提前折断。而在经过对已有代码的统计后，发现该情况非常少（目前一共约有 10 处），因此认为这种方式是可行的。

在关键词触发时应当正确地阻止提交并给出明确的提示信息，在提示信息中附上本 RFC 的链接以详细说明问题。之后将 `grep` 匹配到的结果打印出来，以便于开发人员定位问题。

如果出现了误报需要能够让相应检查人员通过手动在 PR approve 使该 CI 通过，本问题 `check_approval` 的人是 `qili93 (Recommend), luotao1`。

## 三、任务分工和排期规划

整个任务可以根据前文所述的目标一和目标二进行划分，两个目标分别由两人各自主导，另外一人进行辅助（视具体工作量和完成进度而定）

- 目标一：存量修改

  - 主要负责人：Nyakku（@SigureMo）
  - 主要工作内容
    - 编写脚本修改现有单测代码「半周内」
      - `self.assertTrue(np.allclose(...))` -> `np.testing.assert_allclose(...)`
      - `self.assertTrue(np.array_equal(...))` -> `np.testing.assert_array_equal(...)`
    - 根据 CI 结果进行调试，尽可能使其通过「两周内」

- 目标二：增量阻止

  - 主要负责人：何双池（@Yulv-git）
  - 主要工作内容
    - 编写 CI 增量阻止脚本「一周内」
    - 调试脚本使其能够正确工作（需测试存在关键词的 commit 确实无法提交）「一周内」

工作小组可互相配合，共同完成目标一和目标二的工作。

整体工作大概为 2~3 周，可根据实际情况提前完成，主要工作时间为 8 月 8 日（周一）到 8 月 29 日（周一），预计在八月底之前完成全部工作。

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
