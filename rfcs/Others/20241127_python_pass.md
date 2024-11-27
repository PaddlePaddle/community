|任务名称 | python pass                      | 
|---|-------------------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | 尹帆                                                   | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2024-11-27                                            | 
|版本号 | 1.0                                                   | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 如无特殊情况，都应基于develop版本开发                                | 
|文件名 | 提交的markdown设计文档文件名称，如：20241127_python_pass.md<br> | 

## 前言
这可能看起来不是一个很正常的rfcs，不涉及太多背景相关的知识。更多的是使用说明和接口情况，为了各位了解为什么要设计这个Python Pass的功能。我在此先写一个简单的rfcs。

### why python pass
在说python pass之前，我想先向您说明什么是pass。编译器中的pass是指对程序进行分析和变换的一个处理阶段。每个pass通常完成一个特定的功能，比如代码优化、数据分析等。多个pass按照特定顺序串联执行，共同完成编译器的工作。

深度学习编译器中的pass沿用了这个概念，但处理对象变成了深度学习的计算图。它主要完成以下工作：

1. 图优化：合并操作、消除冗余计算、调整计算顺序等
2. 数据布局优化：调整数据存储方式，提高访问效率
3. 硬件映射：将计算图映射到具体硬件上执行
4. 性能优化：包括内存优化、并行优化等

这些pass共同作用，将高层的深度学习模型转换为能在目标硬件上高效运行的程序。python pass主要针对paddle中pir的一种替换pass展开，在paddle中的名字为DRR。这种pass上子图替换的方式在深度学习领域极其常见，下面我们举例说明一个子图替换的例子。

### pass example
静态图是一种预定义的计算流程，它要求在执行前完整构建整个计算图，一旦定义后图的结构就保持固定不变。静态图的优点是可以提前进行优化从而提高执行效率，且部署方便，适合生产环境，但缺点是缺乏灵活性，难以处理需要动态改变的计算逻辑。TensorFlow 1.x是典型的静态图框架代表。

子图则是完整计算图中的一部分，它具有独立的输入和输出接口。子图在深度学习编译优化中扮演重要角色，常用于图优化过程中的算子融合、设备分配时的任务划分，以及针对特定硬件的编译优化。通过合理划分和处理子图，可以实现更好的性能优化和硬件适配。

在Paddle框架里，支持动态图到静态图的动转静。当拿到静态图之后，还会对静态图进行一系列的处理。这其中有传统编译器也拥有的，类似：

死代码消除（Dead Code Elimination）在传统编译器中，它去除那些计算结果永远不会被使用的代码；在深度学习编译器中，它则移除计算图中那些输出没有被后续节点使用的算子。

常量折叠（Constant Folding）传统编译器中它在编译期计算常量表达式的结果；深度学习编译器中则预先计算那些输入全部为常量的算子。

公共子表达式消除（Common Subexpression Elimination）传统编译器中它避免重复计算相同的表达式；深度学习编译器中它识别并合并计算图中的重复计算模式。

paddle支持这些变换，但有一种变换更常见。也更容易被开发者所使用的就是子图变换。常见来说，一些操作会由细碎的算子组合而成。但过重的算子粒度会增加深度学习加速卡的io瓶颈。对于深度学习来说（或者更进一步对于大模型时代来说）重复的io会造成大量不必要的通信，从而造成浪费。

下图我们举了一个简单而形象的例子，我们看这个情况。输入x先进行矩阵转置之后和y进行矩阵乘法，在这个过程中。有两个算子参与了这个子图，分别是transpose和matmul。但实际上Paddle的matmul其实是可以支持内部专置x or y，从底层说。paddle的matmul其实使用的是cublas的矩阵乘法，在cublas的代码中，它支持以bool类型判断是否要transpose x or y。有了这些信息，我们就可以写出一个pass让transpose+matmul的组合转换成单个matmul。

经过了这个形象的例子，大家或许也明白了。DRR 中 pass完成的功能就是把一种子图转换成另一种子图。也就是用一种新的融合算子或者更少的算子，替换之前多算子的情况。达到减少算子的效果

![画板](https://cdn.nlark.com/yuque/0/2024/jpeg/27816614/1732688926502-15ca9fb9-57c9-4114-8019-f3c1649aa700.jpeg)

## API接口一览
结束了前言，讲完了Python Pass的意义之后。我想向您介绍python pass的接口一览，这能帮助你如何写出一个pass。

### pir.DrrPatternContext
+ 方法简介：这个接口是用作创建上下文的接口，也就是创建一个包含转换之前。转换之后信息的总handle。
+ 传入参数：这个方法不需要传入参数。
+ 传出参数：这个方法会创建一个PatternContext，这个值作为返回进行pass注册。
+ 使用说明：

```python
python_ctx = pir.DrrPatternContext()
```

### SourcePattern
+ 方法简介：这个接口用于创建转换之前的子图，也就是转换之前的图例形式。
+ 传入参数：这个方法不需要传入参数
+ 传出参数：这个方法会创建一个SourcePattern，我们需要在这上面创建Op(算子)和Tensor(张量)。
+ 使用说明：特别说明，SourcePattern需要在创建好的DrrPatternContext对象上进行创建

```python
python_pat = python_ctx.SourcePattern()
```

#### SourcePattern-Op
+ 方法简介：这个方法用于创建一个算子，用于前向计算。
+ 传入参数：算子的名字（注意，这里必须强制对齐pd_op.xxx需要是一个真实存在的算子名称）

   算子所需的其他config（以字典形式传入，第一个变量是config的参数名，第二个变量是以SourcePattern-Attr创建的Attribute）

+ 传出参数：这个方法会创建一个Op，Op会用作前向。
+ 使用说明：

```python
python_pat.Op("name", {"config", python_pat.Attr("config")})
```

#### SourcePattern-Tensor
+ 方法简介：这个方法用于创建一个张量，用于Op的前向计算。
+ 传入参数：张量的名字
+ 传出参数：这个方法会创建一个Tensor，Tensor会用作前向。
+ 使用说明：

```python
python_pat.Tensor("name")
```

#### SourcePattern-Attr
+ 方法简介：这个方法用于创建一个Attribute，用于Op的定义。
+ 传入参数：Attribute的名字
+ 传出参数：这个方法会创建一个Attribute，Attribute会用作Op的定义。
+ 使用说明：

```python
python_pat.Attr("name")
```

#### SourcePattern-AddConstriant
+ 方法简介：这个方法用于添加一个Constriant，用于检测输入是否符合规范。
+ 传入参数：一个python函数
+ 传出参数：这个方法不会返回值。
+ 使用说明：

```python
def cons_example(match_ctx):
    ## do some cons
    pass

python_pat.AddConstriant(cons_example)
```

### match_context
+ 方法简介：这个方法一般不单独出现，用在constrain里或者ComputeAttr里

#### match_context-Tensor
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Tensor。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Tensor
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.Tensor("name")
```

#### match_context-StrAttr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型和创建时相同
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.StrAttr("name")
```

#### match_context-BoolAttr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型和创建时相同
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.BoolAttr("name")
```

#### match_context-Int32Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型和创建时相同
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.Int32Attr("name")
```

#### match_context-Int64Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型和创建时相同
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.Int64Attr("name")
```

#### match_context-Float32Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型和创建时相同
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.Float32Attr("name")
```

#### match_context-DoubleAttr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型和创建时相同
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.DoubleAttr("name")
```

#### match_context-VectorInt32Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型和创建时相同
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.VectorInt32Attr("name")
```

#### match_context-VectorInt64Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型和创建时相同
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.VectorInt64Attr("name")
```

#### match_context-VectorFloat32Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型和创建时相同
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.VectorFloat32Attr("name")
```

#### match_context-DataTypeAttr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型和创建时相同
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.DataTypeAttr("name")
```

#### match_context-PlaceAttr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型和创建时相同
+ 使用说明：

```python
def cons_example(match_ctx):
    match_ctx.PlaceAttr("name")
```

### ResultPattern
+ 方法简介：这个接口用于创建转换之后的子图，也就是转换之后的图例形式。
+ 传入参数：这个方法不需要传入参数
+ 传出参数：这个方法会创建一个ResultPattern，我们需要在这上面创建Op(算子)和Tensor(张量)。
+ 使用说明：特别说明，ResultPattern需要在创建好的SourcePattern对象上进行创建

```python
python_res = python_pat.ResultPattern()
```

#### ResultPattern-Op
+ 方法简介：这个方法用于创建一个算子，用于前向计算。
+ 传入参数：算子的名字（注意，这里必须强制对齐pd_op.xxx需要是一个真实存在的算子名称）

   算子所需的其他config（以字典形式传入，第一个变量是config的参数名，第二个变量是以SourcePattern-Attr创建的Attribute）

+ 传出参数：这个方法会创建一个Op，Op会用作前向。
+ 使用说明：

```python
python_res.Op("name", {"config", python_pat.Attr("config")})
```

#### ResultPattern-Tensor
+ 方法简介：这个方法用于创建一个张量，用于Op的前向计算。
+ 传入参数：张量的名字
+ 传出参数：这个方法会创建一个Tensor，Tensor会用作前向。
+ 使用说明：

```python
python_res.Tensor("name")
```

#### ResultPattern-StrAttr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：

```python
ResultPattern.StrAttr("name")
```

#### ResultPattern-BoolAttr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：

```python
ResultPattern.BoolAttr("name")
```

#### ResultPattern-Int32Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：

```python
ResultPattern.Int32Attr("name")
```

#### ResultPattern-Int64Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：

```python
ResultPattern.Int64Attr("name")
```

#### ResultPattern-Float32Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：

```python
ResultPattern.Float32Attr("name")
```

#### ResultPattern-VectorInt32Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：

```python
ResultPattern.VectorInt32Attr("name")
```

#### ResultPattern-VectorInt64Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：

```python
ResultPattern.VectorInt64Attr("name")
```

#### ResultPattern-VectorFloat32Attr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：

```python
ResultPattern.VectorFloat32Attr("name")
```

#### ResultPattern-DataTypeAttr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：

```python
ResultPattern.DataTypeAttr("name")
```

#### ResultPattern-PlaceAttr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：

```python
ResultPattern.PlaceAttr("name")
```

#### ResultPattern-DataLayoutAttr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个name
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：

```python
ResultPattern.DataLayoutAttr("name")
```

#### ResultPattern-ComputeAttr
+ 方法简介：这个方法用于获得一个在SourcePattern里已创建的Attribute。
+ 传入参数：一个func
+ 传出参数：得到一个和SourcePattern匹配的Attr结果，类型为ComputeAttr
+ 使用说明：（这和其他Attr的区别是，我们这个ComputeAttr可以支持更复杂的计算）

```python
def compute(match_ctx):
    pass

ResultPattern.ComputeAttr(compute)
```

### 其他
#### Op的前向
+ 方法简介：这个方法用于Op的前向。
+ 传入参数：（List[], List[]）,第一个List里放置传入的Tensor。第二个List里放入传出的Tensor。
+ 传出参数：无
+ 使用说明：

```python
matmul_op([input_x, input_y], [output])
```

#### get_shape_from_value
+ 方法简介：这个方法用于获取Tensor的shape。
+ 传入参数：一个Tensor。
+ 传出参数：Tensor的shape
+ 使用说明：

```python
pir.get_shape_from_value(xx.Tensor("a"))
```

#### get_datatype_from_value
+ 方法简介：这个方法用于获取Tensor的dtype。
+ 传入参数：一个Tensor。
+ 传出参数：Tensor的dtype
+ 使用说明：

```python
pir.get_datatype_from_value(xx.Tensor("a"))
```

#### trans_to_phi_datatype
+ 方法简介：这个方法用于把get_datatype_from_value得到的dtype转换成Op可以接受的dtype。
+ 传入参数：一个get_datatype_from_value 输出dtype。
+ 传出参数：phi datatype
+ 使用说明：

```python
trans_to_phi_datatype(pir.get_datatype_from_value(xx.Tensor("a")))
```

#### compare_dtype
+ 方法简介：这个方法用于dtype比较是否相等。
+ 传入参数：一个get_datatype_from_value 输出dtype。一个string
+ 传出参数：bool
+ 使用说明：

```python
compare_dtype(pir.get_datatype_from_value(xx.Tensor("a")), "float16")
```

## 使用用例
### matmul_traspose
```python
def matmul_transpose_fuse_pattern(self):
        def cons_function(match_ctx):
            x_shape = pir.get_shape_from_value(match_ctx.Tensor("a"))
            y_shape = pir.get_shape_from_value(match_ctx.Tensor("b"))
            if len(x_shape) < 2 or len(y_shape) < 2:
                return False
            perm = match_ctx.VectorInt32Attr("perm")
            perm_size = len(perm)
            for i in range(perm_size - 2):
                if perm[i] != i:
                    return False
            if (perm[perm_size - 1] != perm_size - 2) and (
                perm[perm_size - 2] != perm_size - 1
            ):
                return False
            return True

        python_ctx = pir.DrrPatternContext()
        python_pat = python_ctx.SourcePattern()

        matmul_op = python_pat.Op(
            "pd_op.matmul",
            {
                "transpose_x": python_pat.Attr("transpose_x"),
                "transpose_y": python_pat.Attr("transpose_y"),
            },
        )
        transpose_op = python_pat.Op(
            "pd_op.transpose", {"perm": python_pat.Attr("perm")}
        )

        matmul_op(
            [python_pat.Tensor("a"), python_pat.Tensor("b")],
            [python_pat.Tensor("matmul_op_out")],
        )
        transpose_op(
            [python_pat.Tensor("matmul_op_out")],
            [python_pat.Tensor("transpose_op_out")],
        )

        python_pat.AddConstraint(cons_function)

        python_res = python_pat.ResultPattern()

        def res_transpose_x(match_ctx):
            return (not match_ctx.BoolAttr("transpose_x"), "bool")

        transpose_x = python_res.ComputeAttr(res_transpose_x)

        def res_transpose_y(match_ctx):
            return (not match_ctx.BoolAttr("transpose_y"), "bool")

        transpose_y = python_res.ComputeAttr(res_transpose_y)

        fused_matmul_transpose_op = python_res.Op(
            "pd_op.matmul",
            {"transpose_x": transpose_y, "transpose_y": transpose_x},
        )

        fused_matmul_transpose_op(
            [python_res.Tensor("b"), python_res.Tensor("a")],
            [python_res.Tensor("transpose_op_out")],
        )

        return python_ctx
```

### embedding_eltwise_fuse
```python
def fused_2embedding_eltwise_layernorm_pattern(self):
        def cons_function(match_ctx):
            try:
                w1_dtype = pir.get_datatype_from_value(match_ctx.Tensor("w1"))
                w2_dtype = pir.get_datatype_from_value(match_ctx.Tensor("w2"))
                if w1_dtype != w2_dtype or (
                    not pir.compare_dtype(w1_dtype, "float16")
                    and not pir.compare_dtype(w1_dtype, "float32")
                ):
                    return False
                x1_shape = pir.get_shape_from_value(match_ctx.Tensor("x1"))
                x2_shape = pir.get_shape_from_value(match_ctx.Tensor("x2"))
                if len(x1_shape) != len(x2_shape):
                    return False
                for i in range(len(x1_shape)):
                    if x1_shape[i] != x2_shape[i]:
                        return False
                return True
            except Exception as e:
                print(f"Exception in cons_function: {e!s}")
                import traceback

                traceback.print_exc()
                raise

        python_ctx = pir.DrrPatternContext()
        python_pat = python_ctx.SourcePattern()

        embedding1_op = python_pat.Op("pd_op.embedding")
        embedding2_op = python_pat.Op("pd_op.embedding")
        add_op = python_pat.Op("pd_op.add")

        layer_norm_op = python_pat.Op(
            "pd_op.layer_norm", {"epsilon": python_pat.Attr("epsilon")}
        )

        embedding1_op(
            [python_pat.Tensor("x1"), python_pat.Tensor("w1")],
            [python_pat.Tensor("embedding_1_out")],
        )
        embedding2_op(
            [python_pat.Tensor("x2"), python_pat.Tensor("w2")],
            [python_pat.Tensor("embedding_2_out")],
        )

        add_op(
            [
                python_pat.Tensor("embedding_1_out"),
                python_pat.Tensor("embedding_2_out"),
            ],
            [python_pat.Tensor("add_out")],
        )
        layer_norm_op(
            [
                python_pat.Tensor("add_out"),
                python_pat.Tensor("scale"),
                python_pat.Tensor("bias"),
            ],
            [
                python_pat.Tensor("layernorm_out"),
                python_pat.Tensor("layernorm_mean"),
                python_pat.Tensor("layernorm_variance"),
            ],
        )

        python_pat.AddConstraint(cons_function)

        # res pattern
        python_res = python_pat.ResultPattern()

        combine_op_1 = python_res.Op("builtin.combine")
        combine_op_1(
            [python_res.Tensor("x1"), python_res.Tensor("x2")],
            [python_res.Tensor("combine1_out")],
        )

        combine_op_2 = python_res.Op("builtin.combine")
        combine_op_2(
            [python_res.Tensor("w1"), python_res.Tensor("w2")],
            [python_res.Tensor("combine2_out")],
        )

        def compute_dtype(match_ctx):
            w1_dtype = pir.get_datatype_from_value(match_ctx.Tensor("w1"))
            return (pir.trans_to_phi_datatype(w1_dtype), "datatype")

        cast_op_dtype = python_res.ComputeAttr(compute_dtype)

        cast_op_1 = python_res.Op("pd_op.cast", {"dtype": cast_op_dtype})
        cast_op_2 = python_res.Op("pd_op.cast", {"dtype": cast_op_dtype})
        fused_embedding_eltwise_layernorm_op = python_res.Op(
            "pd_op.fused_embedding_eltwise_layernorm",
            {"epsilon": python_pat.Attr("epsilon")},
        )

        # op forward
        cast_op_1(
            [python_res.Tensor("bias")], [python_res.Tensor("casted_bias")]
        )
        cast_op_2(
            [python_res.Tensor("scale")], [python_res.Tensor("casted_scale")]
        )
        fused_embedding_eltwise_layernorm_op(
            [
                python_res.Tensor("combine1_out"),
                python_res.Tensor("combine2_out"),
                python_res.Tensor("casted_bias"),
                python_res.Tensor("casted_scale"),
            ],
            [python_res.Tensor("layernorm_out")],
        )

        return python_ctx
```

## 总结
这就是python pass的rfcs和用法总结

