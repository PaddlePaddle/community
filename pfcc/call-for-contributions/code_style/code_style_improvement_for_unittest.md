# 框架易用性提升——单测报错信息优化

> This project will be mentored by [@luotao1](http://github.com/luotao1)

## 背景

来源于NVIDIA开发者提的 [issue#44641](https://github.com/PaddlePaddle/Paddle/issues/44641)，NVIDIA开发者每月需要上线一次飞桨到NGC官网，他们会在不同的GPU卡上跑全量单测。使用 `np.testing.assert_allclose` 代替 `assertTrue(np.allclose(...))`，可以获得更全面的单测报错信息，便于验证不同卡上的Op精度。

```python
------ 现有问题 ------
飞桨现有单测代码为
self.assertTrue(np.allclose(x, y), "compare x and y")
错误信息如下，对用户定位问题不够友好
AssertionError: False is not true : compare x and y
------ 修改建议 ------
建议全部修改为如下代码
np.testing.assert_allclose(x, y, err_msg="compare x and y")
错误信息如下，方便用户定位
AssertionError:
Not equal to tolerance rtol=0, atol=0
compare x and y
Mismatched elements: 1 / 3 (33.3%)
Max absolute difference: 1
Max relative difference: 0.33333333
x: array([1, 2, 3])
y: array([1, 3, 3])
```

## 可行性分析和规划排期

1. 存在以上问题的单测列表，具体修复步骤如下：

   - 对于 `self.assertTrue(np.allclose(...))` 类型的代码，统一修改为 `np.testing.assert_allclose(...)` 格式
   - 对于 `self.assertTrue(np.array_equal(...))` 类型的代码，先判断其中的数据类型：
     - 如果是 float 类型(包括 fp16)，则统一修改为 `np.testing.assert_allclose(...)` 格式
     - 如果是 int 数据类型，这统一修改为 `np.testing.assert_array_equal(...)` 格式
   - 验证修改后的单测代码，保证单测可以通过
     - 在单测通过的情况下，请尽量使用 `numpy.testing.assert_allclose` 接口中默认的 rtol 与 atol 的值。
       修复 PR 参考：https://github.com/PaddlePaddle/Paddle/pull/44135

2. CI 中增加对 `assertTrue(np.allclose(` 和 `self.assertTrue(np.array_equal(` 的检查，即增量不能使用该方式。
   可参考 `tools/check_file_diff_approvals.sh` 对字段进行监控，并给出修复建议，如报错信息的监控。

```shell
ALL_ADDED_LINES=git diff -U0 upstream/$BRANCH |grep "^+" || true
ALL_PADDLE_CHECK=echo $ALL_ADDED_LINES |grep -zoE "(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" || true
VALID_PADDLE_CHECK=echo "$ALL_PADDLE_CHECK" | grep -zoE '(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\((.[^,;]+,)*.[^";]*(errors::).[^"]*".[^";]{20,}.[^;]*\);\s' || true
INVALID_PADDLE_CHECK=echo "$ALL_PADDLE_CHECK" |grep -vxF "$VALID_PADDLE_CHECK" || true
if [ "${INVALID_PADDLE_CHECK}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
echo_line="The error message you wrote in PADDLE_ENFORCE{**} or PADDLE_THROW does not meet our error message writing specification. Possible errors include 1. the error message is empty / 2. the error message is too short / 3. the error type is not specified. Please read the specification [ https://github.com/PaddlePaddle/Paddle/wiki/Paddle-Error-Message-Writing-Specification ], then refine the error message. If it is a mismatch, please request chenwhql (Recommend), luotao1 or lanxianghit review and approve.\nThe PADDLE_ENFORCE{**} or PADDLE_THROW entries that do not meet the specification are as follows:\n${INVALID_PADDLE_CHECK}\n"
check_approval 1 6836917 47554610 22561442
fi
```

## 项目总结<a id='summary'></a>
### 意义
这是一个从社区中来到社区中去的代表性项目 （成员：SigureMo（组长） 和 Yulv-git）
- 背景来源于NVIDIA开发者的真实需求 [issue#44641](https://github.com/PaddlePaddle/Paddle/issues/44641)：NVIDIA开发者每月需要上线一次飞桨到NGC官网，他们会在不同的GPU卡上跑全量单测。使用 `np.testing.assert_allclose` 代替 `assertTrue(np.allclose(...))`，可以获得更全面的单测报错信息，便于验证不同卡上的Op精度。
- 社区开发者 SigureMo（组长） 和 Yulv-git 帮飞桨和NVIDIA解决了这个问题：使用自己开发的AST解析脚本，存量（批量自动化）共修复了400+文件近2700+处单测的断言函数，增量使用CI检查项阻止新增不合规断言函数的出现，大幅提升了单测的报错信息丰富度。
  - 从8月5日发布计划，8月8日提出RFC文档，8月19日全部完成，开发者的热情和速度都非常感人。
  - 此项目会应用在9月份NGC官网上线中。
  - 如果没有准确性高的AST解析脚本，400+单测文件需要几十个RD手动进行修复，沟通成本和修复时间都会大大增加。

### 成果概述

#### 存量代码单测断言转换

经过 4 个 PR（[#44947](https://github.com/PaddlePaddle/Paddle/pull/44947)、[#44988](https://github.com/PaddlePaddle/Paddle/pull/44988)、[#45213](https://github.com/PaddlePaddle/Paddle/pull/45213)、[#45251](https://github.com/PaddlePaddle/Paddle/pull/45251)）的逐步替换，已经将 400+ 文件中的近 2000 处 `self.assertTrue(np.allclose(...))` 和 700+ 处 `self.assertTrue(np.array_equal(...))` 替换为合适的 `np.testing` 模块下的断言函数。并对替换后出现的一些问题进行了修复，修复后 CI 上的测试可全部通过。

#### 增量代码关键词阻止

通过一个 PR（[#45126](https://github.com/PaddlePaddle/Paddle/pull/45126)）增加了阻止增量代码中出现 `self.assertTrue(np.allclose(` 前缀的 CI 检查项，并在 [#45184](https://github.com/PaddlePaddle/Paddle/pull/45184) 测试阻止的效果以及 Approve 的效果，两者均有效。

在该 PR 合入后很快就成功地阻止了一个使用 `self.assertTrue(np.allclose(...))` 进行断言的 PR（见 [#45168 (comment)](https://github.com/PaddlePaddle/Paddle/pull/45168#discussion_r948767123)）。成功避免了增量代码中出现新的问题。

#### PR 情况


RFC：[20220805_code_style_improvement_for_unittest.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/CodeStyle/20220805_code_style_improvement_for_unittest.md)


| 提交时间   | 合入时间   | 作者     | 题目                                                                                 | 链接                                              |
| ---------- | ---------- | -------- | ------------------------------------------------------------------------------------ | ------------------------------------------------- |
| 8 月 8 日  | 8 月 10 日 | SigureMo | use np.testing.assert_array_equal instead of self.assertTrue(np.array_equal(...))    | https://github.com/PaddlePaddle/Paddle/pull/44947 |
| 8 月 8 日  | 8 月 17 日 | SigureMo | use np.testing.assert_allclose instead of self.assertTrue(np.allclose(...)) (part 1) | https://github.com/PaddlePaddle/Paddle/pull/44988 |
| 8 月 14 日 | 8 月 17 日 | Yulv-git | Add CI for self.assertTrue(np.allclose(...))                                         | https://github.com/PaddlePaddle/Paddle/pull/45126 |
| 8 月 17 日 | 8 月 19 日 | SigureMo | use np.testing.assert_allclose instead of self.assertTrue(np.allclose(...)) (part 2) | https://github.com/PaddlePaddle/Paddle/pull/45213 |
| 8 月 18 日 | 8 月 19 日 | SigureMo | use np.testing.assert_allclose instead of self.assertTrue(np.allclose(...)) (part 3) | https://github.com/PaddlePaddle/Paddle/pull/45251 |

### 遗留问题

部分替换为 `np.testing.assert_allclose` 后的单测出现了 shape 不匹配的问题，这是由于 `np.allclose` 是对 shape 不敏感的，会在比较时自动应用广播机制。对于这些问题大多数已经修复完毕，最终剩余 4 个暂不清楚修复方式的单测暂未修改，交给 Paddle 内部 RD 修改。

剩余 4 个单测的问题如下（详情见 [#45213 (review)](https://github.com/PaddlePaddle/Paddle/pull/45213#pullrequestreview-1076390386)）：

- `test_layers`：疑似 `layers.embedding` 返回值确实比其余的少一个维度
- `test_group_norm_op_v2`：应该是 NumPy 的 reference（`group_norm_naive_for_general_dimension`）有问题，其返回值经过广播后维度 >= 3，因此维度为 2 的单测无法通过
- `test_layer_norm_op`：同样应该是 NumPy 的 reference（`_reference_layer_norm_grad`）有问题，其返回的 `d_scale` 和 `d_bias` 是多一个维度的
- `test_imperative_tensor_clear_gradient`：一个很奇怪的问题，断言失败位置并不是修改处，暂不清楚为何修改处会影响其他位置的断言
