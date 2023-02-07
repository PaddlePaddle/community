yj/dygraph_test_upgrade# 飞桨老动态图测试迁移至新动态图

> This project will be mentored by [wanghuancoder](https://github.com/wanghuancoder) and [@yjjiang11](https://github.com/yjjiang11)
> 

## 一、概要
### 1.背景
 自 2022 年 7 月 1 日以来，新动态图切换为默认模式，在 CPU 和 GPU多 场景、多维度经过充分验证，确保了 2.4 及预后版本的稳定性和安全性。为了进一步降低框架的维护成本和提升 Python 端的简洁性，已于2022 年 12 月 20 日正式下线老动态图功能。作为老动态图下线延续工作，现需要集中进行部分老动态图测试迁移至新动态图，以下简称为动态图测试迁移工作。

为了更加清晰地参与开发工作，现补充下背景算子：当前 Paddle 框架算子主要有两种实现，静态图实现和动态图实现。动态图实现又分为中间态实现和最终态实现。动态图算子中间态通过 paddle._legacy_C_ops 调用，最终态通过 paddle._C_ops 调用，两者功能相同，但参数的传入方式不同，差异对比如下：

```python
  # 中间态调用形式，tensor 类可直接传入，非tensor的参数需要通过key-value形式传入
  paddle._legacy_C_ops.matmul_v2(x, weight, 'trans_x', False, 'trans_y', False)
  # 最终态调用形式，可直接传入参数
  paddle._C_ops.matmul(x, y，transpose_x, transpose_y)

```
当前的动态图测迁移优先在最终态中查找相关算子进行适配，其次在中间态中查找。
### 2.功能目标
对 Paddle 现有的[算子单元测试](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests)进行动态图迁移，迁移的内容主要包括进行新动态图适配、修复算子，确保老动态图能通过的测试新动态图测试通过。

### 3.方案要点
测试迁移工作主要分为三个阶段：

#### 3.1 第一阶段：动态图测试接口统一
此前测试动态图分为老动态图测试和新动态图测试，控制开关分别为 check_dygraph 和 check_eager。随着老动态图下线 ，代表老动态图测试开关语义的 check_dygraph 失效。现将开关进行统一仅保留 check_dygraph 作为新动态图测试开关，默认打开，去除原来的的老动态图测试代码。

目前这部分工作已完成，见[PR49877](https://github.com/PaddlePaddle/Paddle/pull/49877)

#### 3.2 第二阶段：老动态图测试迁移至新动态图（社区重点参与）

在完成第一阶段的工作后发现有不少算子测试失败，失败原因主要为：

1.尚未适配新动态图测试，即测试代码中尚未添加 python_api，python_api 为可直接调用函数，需要用户写对应的适配代码

2.已经适配了新动态图测试，即已经添加了 python_api，但尚有新动态图不支持的场景，需要修复
    

##### 老动态图测试迁移至新动态图的迁移规则：

        1. 测试代码中有 check_eager=False，则表示此前不支持新动态图测试，分析原因调通测试
        2. 测试代码中有 check_eager=True,则表示此前已支持新动态图测试，可以直接删除check_eager=True
        3. 测试代码中有check_dygraph=False，则表示此前不支持老动态图测试, 新动态图也不要求测试，可以将check_dygraph=False
        4.测试代码中尚不设置check_eager 和 check_dygraph，则表示此前仅支持测试老动态图，需要添加 python_api，调通测试

#### 3.3 第三阶段：老动态图测试代码完全移除（这部分工作暂不需要社区参与）

## 二、主要工作

需要将尚未迁移的算子进行迁移，算子列表另外公布。

本次工作主要需要社区开发者进行动态图测试迁移，主要内容为为测试算子添加 `python_api` 并确保测试通过，工作可以分为以下几个步骤。

### 2.1 把新动态图测试开关打开，分析报错算子
按照以下方式进行代码修改
```python
  # 将
  from op_test import OpTest
  
  # 改为
  from eager_op_test import OpTest
```
如果代码中有 `check_eager` 需要全局替换为 `check_dygraph` 并设置为  `True`，运行 python path/to/test/file， 复现报错场景
，如：
```python
python  python/paddle/fluid/tests/unittests/test_eig_op.py 
  AssertionError: Detect there is KernelSignature for `eig` op, please set the `self.python_api` if you set check_dygraph = True
```
此时报错提示需要为 `eig` 算子设置 `python_api` , `python_api` 为可调用函数，形如 `paddle.sum`
### 2.2 根据测试文件的算子类型 op_type 查找相关算子
根据2.1中的报错信息查找相关算子。比如[test_slice.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_slice_op.py) 代码中定义的`op_type='slice'` 即为需要测试 slice 算子。此时开发者可以在 Paddle 代码库中采用以下三种方式搜索；
   
1. 【优先】全局搜索 op_type，看是否有定义 python 接口

2. 【优先】在最终态库中查找 op_type，看是否有相关实现，可以通过python 终端 `python -c "import paddle; getattr(paddle._C_ops， op_type)` 查看是否有相关实现
3. 在中间态算子库中查找 op_type, 看是否有相关实现，可以用过 python 终端 `python -c "import paddle; getattr(paddle._legacy_C_ops, op_type)`

如果以上三种方法均找不到算子实现，则可以联系[@yjjiang11](https://github.com/yjjiang11) 寻求帮助

### 2.3 添加 python_api
在测试类中的 setUp 函数中添加 python_api。

1. 当 paddle 中能找到 python 接口并且参数列表和测试中已写的参数一致，可以尝试将 python_api 设置为找到的接口，然后进行测试验证。比如为 tile 算子添加 python_api 样例如下：

```python

# 只摘取部分代码
from eager_op_test import OpTest

@@ -29,6 +29,7 @@
class TestTileOpRank1(OpTest):
    def setUp(self):
        self.op_type = "tile"
        # 添加 python_api = paddle.tile
        # tile 为 paddle 可直接调用的接口
        self.python_api = paddle.tile
        ....

    def test_check_output(self):
        self.check_output()

```
具体可以参考 [PR49877](https://github.com/PaddlePaddle/Paddle/pull/49877)

2. 一般情况下，无法通过为 python_api 设置当前 paddle 接口即可调通测试。主要原因在于当前的测试代码是以静态图算子为基准进行的参数准备，参数列表和 paddle 接口、算子最终态、中间态可能并不一致。如果参数不一致，则需要进行函数适配。现以 normalize 为例

```python
from eager_op_test import OpTest
import paddle.nn.functional as F

def norm_wrapper(x, axis=1, epsilon=1e-12, is_test=False):
    # F.normalize 不需要 is_test 参数
    return F.normalize(x, axis=axis, epsilon=epsilon)

class TestNormTestOp(OpTest):
    def setUp(self):
        self.op_type = "norm"
        # 添加适配函数 norm_wrapper
        self.python_api = norm_wrapper
        self.init_test_case()
        x = np.random.random(self.shape).astype("float64")
        y, norm = l2_norm(x, self.axis, self.epsilon)
        self.inputs = {'X': x}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'axis': int(self.axis),
            'is_test': True,
        }
        # NOTICE：该 normalize 算子为多输出，目前测试框架需要需要添加python_out_sig来进行封装
        self.python_out_sig = ["out"]
```


目前已经做了部分算子迁移，开发者可以参考以下PR：[PR4987](https://github.com/PaddlePaddle/Paddle/pull/49877) [PR49895](https://github.com/PaddlePaddle/Paddle/pull/49895) [PR50061](https://github.com/PaddlePaddle/Paddle/pull/50061) [PR50077](https://github.com/PaddlePaddle/Paddle/pull/50077) [PR50094](https://github.com/PaddlePaddle/Paddle/pull/50093)

补充： 当测试报 `AssertionError: Don't support multi-output with multi-tensor output. (May be you can use set python_out_sig, see test_squeeze2_op as a example.)` 表示已有的测试框架不支持多输出表示，可以在 `setUp` 函数中添加 `self.python_out_sig = ['Out']`


### 2.4 BUG 调试
如果完成以上步骤，发现测试过程中报错，错误主要分为算子正确性问题和是算子本身计算崩溃，则需要仔细分析原因然后修复。




