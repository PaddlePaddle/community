# CINN flip 设计文档

| API 名称       | 新增 API 名称                    |
| -------------- | -------------------------------- |
| 提交作者       | Nyakku Shigure（@SigureMo）      |
| 提交时间       | 2022-07-30                       |
| 版本号         | v0.1                             |
| 依赖 CINN 版本 | develop                          |
| 文件名         | 20220730_cinn_api_design_flip.md |

## 一、概述

### 1、相关背景

任务源于 PaddlePaddle Hackathon 第三期任务 [No.71：为神经网络编译器 CINN 增加 flip 算子](https://github.com/PaddlePaddle/Paddle/issues/44069#task71)

### 2、名词解释

<!-- TODO -->

如果有可能发生歧义、或者不同领域使用同一个词，需要辨别本文中的定义的，写至此。例如深度学习框架和深度学习编译器都有前端概念，但前端在二者中有不同含义

### 3、功能目标

该算子可以将输入的某些维度数据反转过来，如 `[1, 2, 3]` 变成 `[3, 2, 1]`

`axis` 为需要翻转的维度，可以是一根或多根轴，比如对于 `[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]`

- 当 `axis` 为 `0` 时，结果为 `[[9, 10, 11, 12], [5, 6, 7, 8], [1, 2, 3, 4]]`
- 当 `axis` 为 `1` 时，结果为 `[[4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]]`
- 当 `axis` 为 `[0, 1]` 时，结果为`[[12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]]`

索引的变换可通过以下 Python 伪码描述：

```python
target_shape: list

def transform_index(target_index: list):
    source_index: list = list(target_index)
    for ax in axis:
        source_index[ax] = target_shape[ax] - 1 - target_index[ax]
    return source_index
```

### 4、意义

增加 flip 算子可以提高 CINN 算子丰富度，使得前端框架对接 CINN 更加方便，也可以为直接通过 CINN 来组网的开发者提供更加灵活的操作

## 二、CINN 现状

CINN 已经有 Reverse 算子，因此不需要再实现一个完全一样的 flip 算子。

## 三、业内方案调研

### 深度学习框架 API 调研

- TensorFlow 并没有 flip API，但是有 [tf.reverse](https://pytorch.org/docs/stable/generated/torch.flip.html?highlight=flip#torch.flip) API 和 tf.reverse_sequence API，tf.reverse 与需要实现的 flip 功能完全一致，而 tf.reverse 可视为是 tf.reverse_sequence 的一个特例
- PyTorch 有 [torch.flip](https://pytorch.org/docs/stable/generated/torch.flip.html?highlight=flip#torch.flip) API，且功能与需实现的 flip 一致
- PaddlePaddle 有 [paddle.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flip_cn.html#flip) API，且功能与需实现的 flip 一致

### 深度学习编译器 API 及算子调研

#### XLA

无论是 [tf2xla](https://github.com/tensorflow/tensorflow/tree/7f26c09cb6b529e6f61e5d202a3419eae303364f/tensorflow/compiler/tf2xla) 中的 [reverse_op](https://github.com/tensorflow/tensorflow/blob/7f26c09cb6b529e6f61e5d202a3419eae303364f/tensorflow/compiler/tf2xla/kernels/reverse_op.cc#L31-L117)还是 [torch_xla](https://github.com/pytorch/xla/tree/10db402a64309daede957462c66d5664de182d48/torch_xla) 中的 [flip](https://github.com/pytorch/xla/blob/10db402a64309daede957462c66d5664de182d48/torch_xla/csrc/ops/flip.cpp#L17-L21)，都是直接调用 `xla::Rev` 实现的，而 [xla::Rev](https://github.com/tensorflow/tensorflow/blob/85371ee5ba5d0a2f0f6bfb39f1fe07e1d1f1c66c/tensorflow/compiler/xla/client/xla_builder.cc#L2151-L2168) 发射了一条 `HloOpcode::kReverse` 指令，对于该指令，具体实现于 [xla/service/elemental_ir_emitter.cc](https://github.com/tensorflow/tensorflow/blob/7f26c09cb6b529e6f61e5d202a3419eae303364f/tensorflow/compiler/xla/service/elemental_ir_emitter.cc#L2523-L2536)，即：

```cpp
    case HloOpcode::kReverse:
      return [this, hlo, &operand_to_generator](
                 const IrArray::Index& target_index) -> StatusOr<llvm::Value*> {
        const HloInstruction* operand = hlo->operand(0);
        std::vector<llvm::Value*> source_multi_index = target_index.multidim();
        for (int64_t dim : hlo->dimensions()) {
          source_multi_index[dim] = Sub(target_index.GetConstantWithIndexType(
                                            hlo->shape().dimensions(dim) - 1),
                                        target_index[dim]);
        }
        llvm_ir::IrArray::Index source_index(
            source_multi_index, operand->shape(), target_index.GetType());
        return operand_to_generator.at(operand)(source_index);
      };
```

其实现与功能目标中 Python 伪码描述一致

#### TVM

TVM 在 [relay](https://tvm.apache.org/docs/reference/api/python/relay/index.html) 中包含了 tvm.relay.reverse、tvm.relay.reverse_sequence API，在 [topi](https://tvm.apache.org/docs/reference/api/python/topi.html) 中包含了 tvm.topi.flip API，其中 tvm.relay.reverse 实现与 tvm.topi.flip 功能上一致，在实现上都是作为 reverse_sequence 一个特例进行实现的。比如 [tvm.relay.reverse](https://github.com/apache/tvm/blob/fb87c21bf8d0fa5edec96a054a57a6d37c11289f/src/relay/op/tensor/transform.cc#L2080-L2086) 实现如下：

```cpp
Array<te::Tensor> ReverseCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const Type& out_type) {
  const ReverseAttrs* param = attrs.as<ReverseAttrs>();
  ICHECK(param != nullptr);
  // pass empty seq_length tensor to reverse_sequence
  return {topi::reverse_sequence(inputs[0], te::Tensor(), param->axis.IntValue())};
}
```

也即 `reverse(x, axis)` 可以看作 `reverse_sequence(x, EmptyTensor, axis)`

reverse_sequence 具体实现于 [include/tvm/topi/transform.h](https://github.com/apache/tvm/blob/fb87c21bf8d0fa5edec96a054a57a6d37c11289f/include/tvm/topi/transform.h#L255-L308)，这里 copy 其中的核心 compute 逻辑：

```cpp
  auto func = [&](const Array<Var>& indices) {
    Array<PrimExpr> real_indices;
    for (size_t i = 0; i < src_tensor_dim; ++i) {
      if (i == static_cast<size_t>(seq_axis)) {
        if (seq_lengths.defined()) {
          auto len = seq_lengths(indices[batch_axis]);
          auto idx = if_then_else(
              len <= 1 || len <= indices[i], indices[i],
              if_then_else(len > x->shape[i], x->shape[i] - 1 - indices[i], len - 1 - indices[i]));
          real_indices.push_back(idx);
        } else {
          real_indices.push_back(x->shape[i] - 1 - indices[i]);
        }
      } else {
        real_indices.push_back(indices[i]);
      }
    }
    return x(real_indices);
  };
```

可以看到其 compute 在 `!seq_lengths.defined()` 的情况时与前面功能目标中的 Python 伪码也是完全一致的

## 四、对比分析

### 不同点

对于 TVM，底层仅仅实现了 reverse_sequence，reverse 算子是作为 reverse_sequence 的一个特例进行实现的

对于 XLA，无论是 TensorFlow 的 reverse 算子还是 PyTorch 的 flip 算子都是通过调用 `xla::Rev` 发射一条 `HloOpcode::kReverse` 指令实现的。针对 `HloOpcode::kReverse` 指令，服务端进一步进行处理，也就是对于 Reverse 这一算子是有单独实现的，而不是作为特例进行实现

### 相同点

具体实现方式一致，与功能目标中伪码描述一致

## 五、设计思路与实现方案

与 XLA 一致，直接实现一个算子而不是作为 reverse_sequence 的一个特例进行实现，具体实现思路与功能目标中伪码描述一致，与 XLA、TVM 思路一致，与 CINN 中现有的 Reverse 算子一致，所以为啥要实现啊

### 命名与参数设计

与 Reverse 一致

### 底层 OP 设计

与 Reverse 一致

### API 实现方案

与 Reverse 一致

## 六、测试和验收的考量

与 Reverse 一致

## 七、可行性分析和排期规划

与 Reverse 一致

## 八、影响面

需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响

## 附件及参考资料

1. [tf.reverse](https://www.tensorflow.org/api_docs/python/tf/reverse)
2. [torch.flip](https://pytorch.org/docs/stable/generated/torch.flip.html?highlight=flip#torch.flip)
3. [paddle.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flip_cn.html#flip)
4. [TVM](https://github.com/apache/tvm)
5. [tf2xla](https://github.com/tensorflow/tensorflow/tree/7f26c09cb6b529e6f61e5d202a3419eae303364f/tensorflow/compiler/tf2xla)
6. [torch_xla](https://github.com/pytorch/xla/tree/10db402a64309daede957462c66d5664de182d48/torch_xla)
