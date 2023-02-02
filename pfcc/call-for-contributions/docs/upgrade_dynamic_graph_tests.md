# 飞桨老动态图测试迁移至新动态图

> This project will be mentored by [wanghuancoder](https://github.com/wanghuancoder) and [@yjjiang11](https://github.com/yjjiang11)
> 

## 一、概要
### 1.背景
    自2022年7月1日以来，新动态图切换为默认模式以来，在CPU和GPU多场景、多维度经过充分验证，确保了2.4及预后版本的稳定性和安全性，为了进一步降低框架的维护成本和提升Python端的简洁性，已于2022年12月20日正式下线老动态图功能。现需要集中进行部分老动态图测试迁移至新动态图。
### 2.功能目标
    对于 Paddle 现有的[算子单元测试](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests)进行老动态图到新动态迁移，需要确保老动态图能通过的测试新动态图测试也能通过。

### 3.方案要点
测试迁移工作主要分为三个阶段：

#### 3.1 第一阶段：动态图测试接口统一
    此前测试动态图分为老动态图测试和新动态图测试，控制开关分别为 check_dygraph 和 check_eager。随着老动态图下线 ，代表老动态图测试开关语义的 check_dygraph 失效，现将开关进行统一仅保留 check_dygraph 作为新动态图测试开关，默认测试开关打开，去除原来的的老动态图测试代码。
目前这部分工作已完成，见[PR49877](https://github.com/PaddlePaddle/Paddle/pull/49877)

#### 3.2 第二阶段：老动态图测试开迁移至新动态图
    在完成第一阶段的工作后发现有不少算子测试失败，失败原因主要为：
       1.尚未适配新动态图测试，即测试代码中尚未添加 python_api，需要用户写对应的适配代码
       2.已经适配了新动态图测试，即已经添加了 python_api,但此前老动态图支持但新动态图不支持的场景需要修复相关算子
    
    迁移规则：
        1. 测试代码中有check_eager=False，则表示此前不支持新动态图测试，分析原因调通测试
        2. 测试代码中有 check_eager=True,则表示此前已支持新动态图测试，可以直接删除check_eager=True
        3. 测试代码中有check_dygraph=False，则表示此前不支持老动态图测试，分析原因调通测试
        4.测试代码中尚不设置check_eager 和 check_dygraph，则表示此前仅支持测试老动态图，需要添加 python_api，调通测试
    
    迁移样例：
```python
    # 导入方式变化
    # original: from op_test import OpTest
    from eager_op_test import OpTest
    
    # case1: 此前没有新动态图适配，此时测试会报错说没有添加 python_api
    #        Paddle 有可以直接调用的接口
    class TestCase1(OpTest):
        def setUp(self):
            self.op_type = "add"
            # add python_api
            self.python_api = Paddle.add
    
    # case1: 此前没有新动态图适配，此时测试会报错说没有添加 python_api
    #        Paddle 无可以直接调用的接口或者参数列表不一致，需要写适配函数
    class TestCase2(OpTest):
        def setUp(self):
            self.op_type = "case2Op"
            # add python_api
            self.python_api = caseOp_wrapper
    
    # case3: 此前适配了新动态图，添加了 python_api
    #        需要删除 check_eager 调通测试
    class TestCase3(OpTest):
        def setUp(self):
            self.op_type = "case3Op"
            # add python_api
            self.python_api = caseOp_wrapper

        def test_output(self):
            # original:self.check_output(check_eager=False)
            self.check_output()
```
    

## 二、主要工作

目前有如下算子/算子族需要进行迁移（可以在[unitests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests) 中找一下算子测试文件，一般形如test_opName_op.py,比如accuracy，对应的测试文件test_accuracy_op.py)

可以参考[PR4987](https://github.com/PaddlePaddle/Paddle/pull/49877) [PR49895](https://github.com/PaddlePaddle/Paddle/pull/49895) [PR50061](https://github.com/PaddlePaddle/Paddle/pull/50061) [PR50077](https://github.com/PaddlePaddle/Paddle/pull/50077) [PR50094](https://github.com/PaddlePaddle/Paddle/pull/50093)

* accuracy
* activation
* 优化器类：adadelta、adam、adamax、adamw、momentum
* arange
* assign_value
* auc_op、auc_single_pred
* bce_loss、huber_loss、sigmoid_cross_entropy_with_logits
* bilinear_interp、linear_interp_v2、nearest_interp_op、trilinear_interp
* bitwise
* box_coder、box_coder_and_assign
* cast
* channel_shuffle
* cholesky_solve
* chunk_eval
* conv2d、conv2d_op_depthwise_conv、conv2d_transpose、conv2d_transpose_depthwise_conv、conv2d_transpose_part2
* conv3d、conv3d_transpose
* ctc_align
* data_norm
* distribute_fpn_proposals
* detection_map
* dropout_nd
* eig、eigh、einsum
* elementwise_max、elementwise_pow、mul
* expand
* exponential
* fake_quantize
* fc
* fill_any
* generate_proposal_labels、generate_proposal_v2
* group_norm
* gru_rnn、gru_unit、rnn、simple_rnn
* im2sequence
* lamb
* logcumsumexp
* marker
* meshgrid
* mine_hard_example
* minus
* pad、pad2d_op
* pool_max、pool2d、pool3d
* positive_negative_pair
* precision_recall
* prelu
* reduce
* reshapernn
* sequence_conv
* slice_op
* target_assign
* unbind
* unique