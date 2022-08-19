# 框架易用性提升——单测报错信息优化

## 背景

来源于[issue#44641](https://github.com/PaddlePaddle/Paddle/issues/44641)，使用`np.testing.assert_allclose代替assertTrue(np.allclose(...))`，
可以获得更全面的单测报错信息。

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

- 对于`self.assertTrue(np.allclose(...))`类型的代码，统一修改为 `np.testing.assert_allclose(...)`格式
- 对于`self.assertTrue(np.array_equal(...))`类型的代码，先判断其中的数据类型：
  - 如果是 float 类型(包括 fp16)，则统一修改为 `np.testing.assert_allclose(...)`格式
  - 如果是 int 数据类型，这统一修改为`np.testing.assert_array_equal(...)`格式
- 验证修改后的单测代码，保证单测可以通过
  - 在单测通过的情况下，请尽量使用`numpy.testing.assert_allclose`接口中默认的 rtol 与 atol 的值。
    修复 PR 参考：https://github.com/PaddlePaddle/Paddle/pull/44135

2. CI 中增加对`assertTrue(np.allclose(`和`self.assertTrue(np.array_equal(`的检查，即增量不能使用该方式。
   可参考`tools/check_file_diff_approvals.sh`对字段进行监控，并给出修复建议，如报错信息的监控。

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

## 项目进展

- 成员：SigureMo（组长），Yulv-git
- RFC：[20220805_code_style_improvement_for_unittest.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/CodeStyle/20220805_code_style_improvement_for_unittest.md)
- PR 情况：

| 提交时间   | 合入时间   | 作者     | 题目                                                                              | 链接                                              |
| ---------- | ---------- | -------- | --------------------------------------------------------------------------------- | ------------------------------------------------- |
| 8 月 8 日  | 8 月 10 日 | SigureMo | use np.testing.assert_array_equal instead of self.assertTrue(np.array_equal(...)) | https://github.com/PaddlePaddle/Paddle/pull/44947 |
| 8 月 8 日  | 8 月 17 日 | SigureMo | use np.testing.assert_allclose instead of self.assertTrue(np.allclose(...))       | https://github.com/PaddlePaddle/Paddle/pull/44988 |
| 8 月 14 日 | 8 月 17 日 | Yulv-git | Add CI for self.assertTrue(np.allclose(...))                                      | https://github.com/PaddlePaddle/Paddle/pull/45126 |
