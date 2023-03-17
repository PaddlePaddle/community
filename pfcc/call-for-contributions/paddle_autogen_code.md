# 算子定义生成体系建设--静态图算子自动生成
## 问题描述
> This project will be mentored by [@heavyrain-lzy](https://github.com/heavyrain-lzy)，[@zyfncg](https://github.com/zyfncg)

大家好，目前飞桨的算子已十分丰富，能够满足众多用户需求，但另一方面，繁多的算子给框架的维护和开发带来了困难。为了规范静态图算子的定义方式加快算子开发流程，飞桨建立了一套自动代码生成体系。但目前并没有将所有的算子清理完毕，这里筛选出部分简单的算子，欢迎大家一起提交清理。任务目标是将`legacy_ops.yaml` `legacy_backward.yaml`中的OP的配置移动到`ops.yaml` `backward.yaml`，在`op_compat.yaml`进行参数名字映射设置(详见附录),如果有version信息还需要在`op_version.yaml`中配置version信息(详见附录)，**并将原始手写的算子实现进行删除,也就是删除对应的`xxx_op.cc`和`xxx_sig.cc`文件或者文件的一部分**。
为了完成这些静态图算子的清理，建议按照如下步骤进行：
### 预备知识
1. 学习《[Paddle贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)》，着重学习《[开发C++算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html)》中的yaml配置介绍，熟悉yaml配置对应静态图算子的含义。
2. 理清yaml文件对应关系：

    a. `legacy_ops.yaml`和`legacy_backward.yaml`里保存着待清理的算子配置。

    b.`ops.yaml`和`backward.yaml`里保存着普通算子配置，cmake后对应生成`./paddle/fluid/operators/generated_op1.cc~generated_op4.cc`和`generated_sig.cc`

    c.`static_ops.yaml`和`static_backward.yaml`里保存静态图算子独立配置，cmake后对应生成`./paddle/fluid/operators/static_generated_op.cc`和`./paddle/fluid/operators/static_generated_sig.cc`(此次题目中没有涉及这部分配置)

    d.`op_compat.yaml` 为保证算子兼容性增加的参数名映射， `op_version.yaml`里配置算子每次更改的版本信息，二者的配置说明详见附录。
### 建议开发步骤
1. 寻找opreator对应的手写文件。通常情况下一个Op对应一个xxx_op.cc和xxx_sig.cc文件，例如`affine_grid`:
  对应的手写文件：`./paddle/fluid/operators/affine_grid_op.cc`和`./paddle/phi/ops/compat/affine_grid_sig.cc`.当然也有部分算子共享手写文件，例如`argmax`  `argmin`共享`arg_min_op.cc` `arg_max_op.cc`  `arg_min_max_base.h`和`arg_min_max_sig.cc`。注意这里`xxx_op.cc`的相关文件一定是在`./paddle/fluid/operators`目录及其子目录下。

2. 对比`xxx_op.cc xxx_sig.cc`修改yaml文件。将对应的op从`legacy_ops.yaml`和`legacy_backward.yaml`移动到`ops.yaml`和`backward.yaml`,并且在`op_compat.yaml`中配置参数名映射,如果原算子含有version信息，还要在`version.yaml`中配置此信息,如`affine_grid`算子就含有version信息。同时如果`xxx_op.cc xxx_sig.cc`只含有待修改的算子，可以将这两个文件删除；如果`xxx_op.cc xxx_sig.cc`还含有其他算子信息，只需将该次修改的算子相关代码删除即可。

3. 验证结果。代码生成需要执行CMake，建议编译选项开启单元测试`-DWITH_TESTING=ON`。如果CMake执行成功，会生成相应的代码文件`./paddle/fluid/operators/generated_op1.cc~generated_op4.cc`和`./paddle/phi/ops/compat/generated_sig.cc`。
首先人工检查自动生成的算子代码和原始的算子代码**逻辑**是否一致，一般从**参数名、GetExpectedKernelType成员函数、version信息、generated_sig.cc中的参数映射**等方面检查，如果逻辑和原始代码一致，则可以编译Paddle，并在编译成功后在`build`文件夹下执行相关的单元测试，例如，如果修改的是`affine_grid`算子，就执行`ctest -R test_affine_grid -V`.如果单元测试pass，那么就可以提交代码申请PR。

### 注意事项
1. 可以参考的PR

    a. `IntArray` `Scalar`参数的配置[PR48792](https://github.com/PaddlePaddle/Paddle/pull/48792)

    b. version相关的信息配置[PR49413](https://github.com/PaddlePaddle/Paddle/pull/49413)

    c. 算子名映射的相关[PR49772](https://github.com/PaddlePaddle/Paddle/pull/49772)

    d. 有些算子在`legacy_ops.yaml legacy_backward.yaml`中的配置并不完整，通常出现在`kernel`下缺少`data_type`配置，而对应的静态图ops的`GetExpectedKernelType`是含有`data_type`类型推导，所以在你移动yaml配置时，需要补充这个配置，可以参见[PR48977](https://github.com/PaddlePaddle/Paddle/pull/48977)
2. 有些算子的修改可能涉及到API说明文档，如果遇到这种情况，需要将C++端的comment拷贝到Python接口，可以参考[PR50611](https://github.com/PaddlePaddle/Paddle/pull/50611)
3. 建议完成开发任务时，对照已存在的yaml配置，执行CMake，观察yaml配置对应的生成代码，能够快速帮你熟悉yaml配置原则。
4. 这些任务难度不大，可以加深对框架的熟悉程度，增强代码调试能力，欢迎参与
### 任务列表
|任务序号|算子名      |算子注意事项    | 需要修改ops.yaml/backward.yaml | 需要修改static_ops.yaml/static_backward.yaml | 需要修改op_compat.yaml | 需要修改op_version.yaml | 需要删除算子同名xxx_sig.cc |
|----|-------------|-----------------|--------|--------|-----| ---|-----|
|1   |affine_grid|version信息，IntArray参数| ✅           |        | ✅  | ✅  |✅  |   |   |   |
|2   |index_add|常规开发|         ✅   |        | ✅  |   |✅ |   |   |   |
|3   |matrix_rank|常规开发|           ✅ |        | ✅  |   |✅  |   |   |   |
|4   |cumsum|在cum_op.cc中,Scalar参数,version信息|   ✅  |   |✅         |        |  ✅ |   |   |   |
|5   |logcumsumexp|在cum_op.cc中,Scalar参数,version信息|  ✅          |        |  ✅  |   |✅ |   |   |   |
|6   |dropout|Scalar参数|      ✅      |        |✅  |   | ✅ |   |   |  | |
|7   |expand_as|对应expand_as_v2_op.cc,算子名映射，IntArray参数|   ✅         | | ✅  |   |✅  |   |   |   |
|8   |layer_norm|常规开发|  ✅          |        | ✅  |   | ✅ |   |   |   |
|9   |lu|增加inplace参数，关注VarTypeInference|    ✅        |        |✅  |   |  ✅ |   |   |   |
|10  |margin_cross_entropy|常规开发|   ✅         |        | ✅  |   | ✅ |   |   |   |
|11  |matrix_nms|version信息|         ✅   |        | ✅  |  ✅ |✅   |   |   |   |
|12  |spectral_norm|常规开发|         ✅   |        | ✅  |   | ✅ |   |   |   |
|13  |generate_proposals|generate_proposals_v2_op.cc，算子名映射,version信息|        ✅    |        | ✅  | ✅  | ✅ |   |   |   |
|14  |bilinear_tensor_product|version信息|   ✅         |        | ✅  |   | ✅  |   |   |   |
|15  |sigmoid_cross_entropy_with_logits|常规开发|  ✅          |        |  ✅  |   |✅ |   |   |   |
|16  |auc|version信息|         ✅   |        | ✅  | ✅  | ✅ |   |   |   |
|17  |accuracy|version信息|      ✅      |        | ✅  | ✅  | ❌  |   |   |   |
|18  |logical_and|version信息，在同一个文件logical_op.cc| ✅           |        | ✅  | ✅  | ❌ |   |   |   |
|19  |logical_not|version信息，在同一个文件logical_op.cc| ✅           |        | ✅  | ✅  | ❌ |   |   |   |
|20  |logical_or|version信息，在同一个文件logical_op.cc|  ✅          |        | ✅  |  ✅ | ❌ |   |   |   |
|21  |logical_xor|version信息，在同一个文件logical_op.cc| ✅           |        | ✅  | ✅  |❌  |   |   |   |
|22  |mean_all|mean_op.cc，算子名映射|   ✅         |        |✅  |   | mean_sig.cc   |   |   |   |
|23  |rnn|常规开发|       ✅     |        | ✅  |   | ✅ |   |   |   |
|24  |warpctc|常规开发|    ✅        |        |✅  |   | ✅  |   |   |   |
|25  |uniform_inplace|uniform_random_inplace_op.cc，算子名映射| ✅           |        |✅  |   | uniform_random_inplace_sig.cc  |   |   |   |
|26  |merge_selected_rows|常规开发|   ✅         |        |  ✅  |   | ❌ |  |   |   |
|27  |clip_by_norm|含有头文件，注意分析具体实现|   ✅         |        | ✅  |   |✅  |   |   |   |
|28  |reverse|常规开发|      ✅      |        | ✅  |   |✅  |   |   |   |
|29  |squared_l2_norm|常规开发|   ✅         |        |  ✅  |   |✅ |   |   |   |
|30  |temporal_shift|含有头文件，注意分析具体实现|   ✅         |        | ✅  |   | ✅ |   |   |   |
|31  |yolo_box|version信息|    ✅        |        | ✅  |  ✅ |  ✅|   |   |   |
|32  |yolo_loss|yolov3_loss_op.cc，算子名映射|   ✅         |        | ✅  |   | yolov3_loss_sig.cc |   |   |   |
|33  |deformable_conv| 常规开发 |    ✅        |        | ✅  |   | ✅ |   |   |   |
|34  |deformable_conv1|yaml缺少deformable_conv_v1的配置，仿照deformable_conv增加配置|   ✅         |        |  ✅  |   |deformable_conv |   |   |   |
|35  |unpool|IntArray参数|  ✅          |        | ✅  |   | ✅ |   |   |   |
|36  |unpool3d|unpool_op.cc |    ✅        |        |✅  |   |✅   |   |   |   |
|37  |argmax|Scalar参数，要同时关注arg_min_max_base.h文件|   ✅         |        |  ✅  |   |arg_min_max_sig.cc |   |   |   |
|38  |argmin|Scalar参数，要同时关注arg_min_max_base.h文件|  ✅          |        | ✅  |   | arg_min_max_sig.cc |   |   |   |
|39  |class_center_sample|常规开发|    ✅        |        | ✅  |   | ✅ |   |   |   |
|40  |eigvalsh|常规开发|       ✅     |        | ✅  |   | ✅ |   |   |   |
|41  |logsumexp|常规开发|     ✅       |        | ✅  |   |✅  |   |   |   |
|42  |prelu|常规开发|        ✅    |        | ✅  |   | ✅ |   |   |   |
|43  |nms|常规开发|         ✅   |        |  ✅  |   |❌ |   |   |   |
|44  |bce_loss|常规开发|    ✅        |        | ✅  |   | ✅ |   |   |   |
|45  |cumprod|常规开发|       ✅     |        |  ✅  |   |✅ |   |   |   |s
|46  |huber_loss|常规开发|   ✅         |        | ✅  |   | ✅ |   |   |   |
|47  |kldiv_loss|常规开发|   ✅         |        | ✅  |   |✅  |   |   |   |
|48  |max_pool2d_with_index|pool_with_index_op.cc中|   ✅         |        | ✅  |   |pool_sig.cc  |   |   |   |
|49  |max_pool3d_with_index|pool_with_index_op.cc中|   ✅         |        | ✅  |   | pool_sig.cc |   |   |   |
|50|p_norm|version信息|✅||✅| ✅  | ✅ |   |   |
|51|nonzero|where_index_op.cc，算子名映射|✅| |where_index_op.cc| | where_index_sig.cc |  |   |
|52|dirichlet|常规开发|✅|  |❌|  | ✅ |   |  |

### 任务列表说明
1. **算子注意事项**：这列会注明算子的基本信息

    a. **常规开发**：表示该算子难度容易，一般直接移动配置即可

    b. **version信息**：表示含有version信息

    c. **算子名和算子文件不对应**：有些算子对应的文件名和文件名不一致，表格里会注明

    d. **IntArray,Scalar参数**：这些算子含有这两类参数，在`op_compat.yaml`中配置稍微复杂一些

    e. **算子名映射**：这些算子的算子名需要映射到新的名字，需要在`op_compat.yaml`配置

    f. **其他说明**：**增加inplace参数**表示yaml配置缺少对应的`inplace`参数，需要补全；**在同一个文件logical_op.cc**表示多个op在同一个文件中；**含有头文件，注意分析具体实现**表示op不仅有`xxx_op.cc`还有对应的头文件。
2. **`修改ops.yaml/backward.yaml`**:这列表示对应的算子配置需要从`legacy_ops.yaml/legacy_backward.yaml`中移动到`ops.yaml/backward.yaml`,并把原来的配置删除。其中，如果算子有`backward`配置，就需要修改对应的`backward.yaml`文件。

3. **`修改static_ops.yaml/static_backward.yaml`**:这列表示对应的算子配置动态图和静态图不一致，需要从`legacy_ops.yaml/legacy_backward.yaml`中拷贝到`static_ops.yaml/static_backward.yaml`,保留原来`legacy_ops.yaml/legacy_backward.yaml`中的配置，并在依据静态图算子的原始算子定义，在`static_ops.yaml/static_backward.yaml`完善算子配置。

4. **删除算子同名xxx_sig.cc**：有些算子没有对应的**xxx_sig.cc**文件，在这列会说明。

### 附录说明
#### op_version.yaml配置说明
| op_version.yaml | 含义  |    |
|---|---|---|
|- op : version :|固定格式|
|- checkpoint|AddCheckpoint的comment，一个算子可能有多个checkpoint
|action|一个checkpoint对应一个action
|- add_input <br>comment <br>default|对应的NewInput；default按需配置|
|- delete_input<br>comment<br>default|对应的DeleteInput；default按需配置
|- modify_input<br>comment<br>default|对应的ModifyInput；default按需配置
|- add_output<br>comment<br>default|对应的NewOutput；default按需配置
|- delete_output<br>comment<br>default|对应的DeleteOutput；default按需配置
|- modify_output<br>comment<br>default|对应的ModifyOutput；default按需配置
|- add_attr<br>comment<br>default|对应的NewAttr；default按需配置
|- delete_attr<br>comment<br>default|对应的DeleteAttr；default按需配置
|- modify_attr<br>comment<br>default|对应的ModifyAttr；default按需配置

#### op_compat.yaml主要配置说明

|op_version.yaml配置选项|含义|可选性|
|---|---|---|
|- op : abs|固定格式，如果有op_name名映射(对应的sig文件中有<br>PD_REGISTER_BASE_KERNEL_NAME(size, numel);)，需要加上括号，- op : numel(size)|必须
|backward : abs_grad|如果有backward，配置反向名字。同样存在名字映射，如backward : topk_grad (top_k_v2_grad)|按需
|inputs :|inputs参数名字映射，如果有多个用{}，只有在`Maker()`成员函数中参数名和`ops.yaml`不一致时才需要(一般都需要)|按需
|outputs :|outputs参数名字映射，如果有多个用{}，只有在`Maker()`成员函数中参数名和`ops.yaml`不一致时才需要(一般都需要)|按需
|attrs :|attrs参数名字映射，如果有多个用{}，只有在`Maker()`成员函数中参数名和`ops.yaml`不一致时才需要(**一般不需要**)|按需
|extra :|一些硬件相关的配置(无需关心)|按需
|int_array:<br>axis :<br>data_type : int<br>support_tensor : true<br>shape :<br>data_type : int<br>tensor_name : Shape<br>tensors_name : ShapeTensor|配置IntArray特殊参数(axis和shape是对应的两种配置)，可参考[PR48792](https://github.com/PaddlePaddle/Paddle/pull/48792)|按需
|scalar :<br>    rtol :<br>      data_type : std::string<br>      tensor_name : Rtol<br>   axis:<br>      data_type : int<br>      support_tensor : true|配置Scalar特殊参数(rtol和axis是两种不同的配置)，可参考[PR48792](https://github.com/PaddlePaddle/Paddle/pull/48792)|按需