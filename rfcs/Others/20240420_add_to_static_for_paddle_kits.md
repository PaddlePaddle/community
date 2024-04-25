# 全套件模型接入动转静训练功能

| 任务名称     | 全套件模型接入动转静训练功能              |
| ------------ | ----------------------------------------- |
| 提交作者     | jshh0401                                  |
| 提交时间     | 2024-04-08                                |
| 版本号       | V1.0                                      |
| 依赖飞桨版本 | develop                                   |
| 文件名       | 20240420_add_to_static_for_paddle_kits.md |

# 一、概述

## 1、相关背景

⽬前⻜桨的开源套件如 PaddelClas、PaddleNLP、PaddleSeg、PaddleDetection、Paddle3D、PaddleOCR 等，

都⽀持了动转静训练功能，但是并⾮所有的模型都接⼊了--to_static策略。

随着 PaddleSOT 功能的完善和上线，动转静训练成功率⼤幅度提升，故此任务旨在对开源套件中所有模型进⾏动

转静训练策略推全。

需要对现有的 PaddelClas、PaddleNLP、PaddleSeg、PaddleDetection、Paddle3D、PaddleOCR、

PaddleRec、PaddleGAN、PaddleVideo、PaddleYOLO 套件中的所有模型依次添加 to static 策略，⽀持开启动转

静进⾏训练，且保证对套件模型尽可能少的代码侵⼊。

## 2、功能目标

对Paddle全部套件(PaddleClas、PaddleNLP、PaddleSeg、PaddleDetection、Paddle3D、PaddleOCR、

PaddleRec、PaddleGAN、PaddleVideo、PaddleYOLO)

- 搜集全部模型列表，并调研模型的动转静⽀持情况，产出《待⽀持动转静模型列表⽂档》

- 针对每个待⽀持动转静的模型，对套件代码进⾏修改，以⽀持动转静训练。同时提供开启动转静训练前后前

50 个 step 的 loss ⼀致性截图


## 3、意义

- 动转静 SOT 项目支持动态 Shape

# 二、飞桨现状

> 对Paddle动专静支持的调研

## 动态图 & 静态图
动态图与静态图都属于计算图，是用来描述运算的有向无环图。
### 动态图
动态图意味着计算图的构建与计算同时发生，类似动态语言的解释执行，这种机制能够很容易的得到模型不同层计算的中间结果，使得调试方便，实现简洁。Pytorch就是使用的动态图模式，风格Pythonic，好上手，传播人群广。
### 静态图
静态图意味着计算图的构建和实际计算是分开的，在静态图中，会事先定义好整个计算图，再次运行网络时不需要重新构建计算图，从性能上来说一般比动态图更加高效。但是由于静态图设计的原因，无法像动态图一样随时拿到中间的计算结果，使用复杂，运行效率高，代表框架是TensorFlow。

## 基于AST语法树转写模式

现有的动转静模块主要包括对输入数据的 InputSpec 的处理，对函数调用的递归转写，对 if else、for、while 控制语句的转写，以及 Layer 的 Parameters 和 Buffers 变量的转换。
具体步骤如下:

- AST解析动态图代码
  - 当某个函数被 @to_static 装饰、或用 paddle.jit.to_static() 包裹时，飞桨会隐式地解析动态图的 Python 代码
- AST转写，得到静态图代码
  - 函数转写：递归地对所有函数进行转写，实现用户仅需在最外层函数添加 @to_static。
  - 控制流转写：用户的代码中可能包含依赖 Tensor 的控制流代码，Paddle框架会自动且有选择性地将 if、for、while 转换为静态图对应的控制流。
  - 其他语法处理：包括 break、continue、assert、提前return等语法的处理。
- 生成静态图的 Program 和 Parameters
  - 得到静态图代码后，根据用户指定的 InputSpec 信息（或训练时根据实际输入Tensor隐式创建的 InputSpec）作为输入，执行静态图代码生成 Program。
  - 对于 trainable=True 的 Buffers 变量，动转静会自动识别并将其和 Parameters 一起保存到 .pdiparams 文件中。
- 执行动转静训练
- 使用 paddle.jit.save 保存静态图模型

### 函数转写样例

```python
import paddle
from paddle.jit import to_static

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    @to_static
    def forward(self, x, y):
        out = self.my_fc(x)       # <---- self.other_func
        out = add_two(out, y)     # <---- other plain func
        return out

    def my_fc(self, x):
        out = self.linear(x)
        return out

# 此函数可以在任意文件
def add_two(x, y):
    out = x + y
    return out

net = SimpleNet()
# 查看转写的代码内容
paddle.jit.set_code_level(100)

x = paddle.zeros([2,10], 'float32')
y = paddle.zeros([3], 'float32')

out = net(x, y)

```

转写后的结果如下：

```python
def forward(self, x, y):
    out = paddle.jit.dy2static.convert_call(self.my_fc)(x)
    out = paddle.jit.dy2static.convert_call(add_two)(out, y)
    return out

def my_fc(self, x):
    out = paddle.jit.dy2static.convert_call(self.linear)(x)
    return out

def add_two(x, y):
    out = x + y
    return out

```

可以看到对添加了to_static装饰器的forward函数进行了转写，并对forward中涉及的函数进行了递归转写。统一转写为相同的格式

```python
 out = paddle.jit.dy2static.convert_call( self.my_fc )( x )
  ^                    ^                      ^         ^
  |                    |                      |         |
返回列表           convert_call             原始函数    参数列表

```



### 控制流转写样例

在转写期，动转静模块将控制流语句转写为统一的形式；在执行期，根据控制流是否依赖Tensor来决定是否将控制流转写为相应的cond_op/while_op。



如果if-else控制流涉及 Tensor，那么就会转为cold_op，例如

```python
from paddle.jit import to_static

def depend_tensor_if(x):
    if paddle.mean(x) > 5.:         # <---- Bool Tensor 类型
        out = x - 1
    else:
        out = x + 1
    return out

print(to_static(depend_tensor_if).code)
# 转写后的代码：
"""
def depend_tensor_if(x):
    out = paddle.jit.dy2static.data_layer_not_check(name='out_0', shape=[-1],
        dtype='float32')

    def true_fn_0(x):      # true 分支
        out = x - 1
        return out

    def false_fn_0(x):     # false 分支
        out = x + 1
        return out

    out = paddle.jit.dy2static.convert_ifelse(paddle.mean(x) > 5.0,
        true_fn_0, false_fn_0, (x,), (x,), (out,))

    return out
"""
```

其中 if paddle.mean(x) > 5.0这个判断条件依赖于返回的tensor值，而在convert_ifelse的框架底层实现上

```python
def convert_ifelse(pred, true_fn, false_fn, true_args, false_args, return_vars):

    if isinstance(pred, Variable):  # 触发 cond_op 的转换
        return _run_paddle_cond(pred, true_fn, false_fn, true_args, false_args,
                                return_vars)
    else:                           # 正常的 python if
        return _run_py_ifelse(pred, true_fn, false_fn, true_args, false_args)
```

可以看到如果pred是Variable实例，那么触发 cond_op转换，会运行paddle实现的控制流算子。

For/While循环类似，总的来说，似乎涉及Tensor的情况下，都需要触发转换。



## Paddle SOT孵化项目

### 背景

Paddle 当前的动转静是基于AST 抽象语法树原理实现的。AST 转写方案虽然具有高层级，易于转写的特性，但由于 Python 是一门纯动态语言，以及 Paddle 静态化数据表示能力的有限性，现在的 AST方案存在如下局限性：

- 难以处理动态和静态相互混合的场景。例如numpy和tensor的互相转换；

- 控制流和容器的混合使用时有边界 case。经常出现解析出错或者是无法完全表示的情况；

- 无法支持组合算子场景 `-1` 的消除。例如 mask 类算子，他们的shape不可推导，确定shape输入也会出现不确定shape输出。

  ```python
  import paddle
  
  x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0]])
  mask = paddle.to_tensor([[True, False, False, False],
                           [True, True, False, False],
                           [True, False, False, False]])
  out = paddle.masked_select(x, mask)
  print(out.numpy())
  
  ```

  mask类算子示例如上，可以看出mask类算子是动态的shape，无法一开始确定shape，需要运行时动态推导。

### 设计思路

初步调研看下来，PaddleSOT通过python在 2016 年的 PEP523 提案支持的新机制，将默认的执行器替换为用户自定义的解释函数，相当于在正常的编译执行之中插入了中间步骤。有关代码的信息从python经过编译后产生的代码对象Code Object中获得，它主要包含了Python 字节码及其相关信息，比如常量表、变量名表等。

### Eval Frame模块

Python 在 2016 年的 PEP523 提案支持了自定义回调函数，将默认的执行器替换为用户自定义的解释函数。Paddle SOT项目在 Paddle 的 Pybind 层暴露了 `paddle.fluid.core.set_eval_frame` 接口。

PyFrameObject是python代码执行的运行时栈帧，后续字节码的模拟执行基于PyFrameObject。

整体的设计思想是在字节码层级将用户原来的函数进行分析，抽取静态组网代码（SIR），然后构建成为一个新的等价的 Python 函数。这个新函数通过运行组网代码来实现和原来函数等价的效果。

### 字节码模拟执行OpcodeExecutor

模拟执行的目标是：通过模拟执行这一过程，可以得到一个在接口上与原先目标函数完全一致的函数，不同的情况下，返回的函数不同：

- 若能够`完全静态化`目标函数，则需要返回一个新的可执行函数，该函数能够构建目标函数对应的子图；
- 若只能`部分静态化`目标函数，同样需要返回一个新的可执行函数，该函数将可静态化部分抽取为子图，并将无法静态化的部分抽取为子函数。
- 若完全`无法静态化`目标函数，则返回原本的目标函数，在动态图环境下进行计算。

具体的实现在sot/opcode_translator/executor/opcode_executor.py

OpcodeExecutor继承了OpcodeExecutorBase基类，基类中定义了执行op code的逻辑为了模拟原生的 Python 运行环境，在子图 FallBack 方案里：

1. 实现 Python 字节码对应的操作逻辑，并维护一个运行栈（工作量是收敛的，因为 OpCode 是有限集合）
2. 对模拟执行中出现的所有 Python 实例进行包装，根据其类型，将实例替换为一个对应 Variable对象。
3. 为所有 `Variable` 对象绑定一个 `Tracker`，`Tracker` 用于记录 `Variable` 的“来源”，例如从 `globals` 中加载而来。需要利用 Tracker 来得到 Guard 函数。

OpCode是有限集合，都定义在这里opcode_executor.py



在模拟执行的过程中记录组网信息

子图 FallBack 默认假设所有的组网代码都应该由 `Paddle API` 或者 `Tensor` 的魔法函数触发。

1. 子图 FallBack 记录了所有组网相关的 Paddle API ，并在 `CALL语义` 的字节码逻辑中进行类型检查。
2. 在模拟执行中，所有的原生 `paddle.Tensor` 被包裹为 `TensorVariable` ，通过触发 `TensorVariable` 的魔法函数来记录组网逻辑。

当模拟执行的过程中发现组网逻辑，`OpcodeExecutor` 将相应的信息记录到 `FunctionGraph` 的 `Statement IR` 中。同时，`OpcodeExecutor` 将利用静态图的 `infer_meta` 机制获取输出 `Tensor` 的 `meta` 信息，结合 meta 信息生成新的 `TensorVariable` 并继续推进模拟执行。

当顺利完成模拟执行后，子图 FallBack 将根据保存的 `FunctionGraph` 结合现有的动转静接口生成 Program。

#### 子图Fallback

对于数据依赖的控制流，例如控制流 If、For 依赖 Tensor 的场景，需要打断构图并触发 FallBack。类似上一种方法的控制流转写？最终实现的效果感觉是一样的。

例如

```python
from symbolic_trace import symbolic_trace
import paddle
import dis

def foo(x, y):
   if x > 0:     # <----- 当 x 是一个 Tensor 时
      y += 1
   else:
      y -= 1
   return y

x = paddle.to_tensor([1])
y = paddle.to_tensor([2])

out = symbolic_trace(foo)(x, y)

print(out)
```

`foo` 函数原始的字节码如下所示：

```c
10           0 LOAD_FAST                0 (x)
              2 LOAD_CONST               1 (0)
              4 COMPARE_OP               4 (>)
              6 POP_JUMP_IF_FALSE       18           # <----- JUMP 指令

 11           8 LOAD_FAST                1 (y)
             10 LOAD_CONST               2 (1)
             12 INPLACE_ADD
             14 STORE_FAST               1 (y)
             16 JUMP_FORWARD             8 (to 26)

 13     >>   18 LOAD_FAST                1 (y)
             20 LOAD_CONST               2 (1)
             22 INPLACE_SUBTRACT
             24 STORE_FAST               1 (y)

 14     >>   26 LOAD_FAST                1 (y)
             28 RETURN_VALUE
```

改写后

```c
# foo函数字节码改写后
  9           0 LOAD_GLOBAL              0 (SIR_0)
              2 LOAD_FAST                0 (x)
              4 BUILD_TUPLE              1
              6 CALL_FUNCTION            1
              8 UNPACK_SEQUENCE          1
             10 STORE_FAST               2 (___SIR_out_var_2)
             12 LOAD_FAST                2 (___SIR_out_var_2)
             14 LOAD_FAST                1 (y)
             16 POP_TOP
             18 POP_JUMP_IF_FALSE       28
             20 LOAD_GLOBAL              1 (__resume_fn_0)      # <----- true branch func
             22 LOAD_FAST                1 (y)
             24 CALL_FUNCTION            1
             26 RETURN_VALUE
        >>   28 LOAD_GLOBAL              2 (__resume_fn_1)      # <----- false branch func
             30 LOAD_FAST                1 (y)
             32 CALL_FUNCTION            1
             34 RETURN_VALUE

# __resume_fn_0
  9           0 JUMP_ABSOLUTE           10

 10           2 LOAD_FAST                1 (x)
              4 LOAD_CONST               1 (0)
              6 COMPARE_OP               4 (>)
              8 POP_JUMP_IF_FALSE       20

 11     >>   10 LOAD_FAST                0 (y)
             12 LOAD_CONST               2 (1)
             14 INPLACE_ADD
             16 STORE_FAST               0 (y)
             18 JUMP_FORWARD             8 (to 28)

 13     >>   20 LOAD_FAST                0 (y)
             22 LOAD_CONST               2 (1)
             24 INPLACE_SUBTRACT
             26 STORE_FAST               0 (y)

 14     >>   28 LOAD_FAST                0 (y)
             30 RETURN_VALUE


# __resume_fn_1
  9           0 JUMP_ABSOLUTE           20

 10           2 LOAD_FAST                1 (x)
              4 LOAD_CONST               1 (0)
              6 COMPARE_OP               4 (>)
              8 POP_JUMP_IF_FALSE       20

 11          10 LOAD_FAST                0 (y)
             12 LOAD_CONST               2 (1)
             14 INPLACE_ADD
             16 STORE_FAST               0 (y)
             18 JUMP_FORWARD             8 (to 28)

 13     >>   20 LOAD_FAST                0 (y)
             22 LOAD_CONST               2 (1)
             24 INPLACE_SUBTRACT
             26 STORE_FAST               0 (y)

 14     >>   28 LOAD_FAST                0 (y)
             30 RETURN_VALUE
```

可以看到，对于控制流的转写，于 `if` 和 `else` 分支分别被抽取成了`__resume_fn_0` 和`__resume_fn_1`两个子函数，根据 `if` 判断的结果具体会走`__resume_fn_0`或`__resume_fn_1`之中的一个; 

在字节码的层面`__resume_fn_0`或`__resume_fn_1`两个分支在最上面增加了一行JUMP到指定的Block，其余的字节码是重复的。



此外对于一些无法模拟的函数应用，也需要触发Fall Back，例如print(tensor)。



### `Tracker`、`Guard` 和缓存模块

子图 Fallback 的整体实现可以认为是将用户函数原始字节码转换为新的字节码，为了避免每次传入相同输入都会重新触发开销昂贵的字节码转换操作，需要增加缓存机制来复用之前转写过的代码。字节码的转换是基于 Frame 的初始状态进行模拟执行得到的，也就是说转换后的字节码强依赖于 Frame 的初始状态。当初始状态发生改变，最后转换后的字节码很有可能发生改变。

这里引入了Guard，转换后的字节码，会同时生成一个Guard，对于后续的frame，都会重新计算这个frame是否在cache中，如果不在才触发转写start_translate函数。

### StatementIR模块

**StatementIR 是 Paddle 动转静模块与子图 FallBack的一个『中间桥梁』，它达到了动转静复用的目的。**

`StatementIR` 与 `Program` 类似，都是表征计算的一个结构。在字节码执行过程中，我们需要将所有的组网代码都『临时记录』下来，并最后将他们组网成为一个 `Program` 。这里的组网代码记录的载体就是 `StatementIR` 。在函数结束的时刻，我们会将记录下来的 `StatementIR` 转化为一个函数。与原来的用户代码不同，由 `StatementIR` 转化为的函数可以确保一定可以动转静。这样我们可以复用原来的动转静 `to_static` 函数来实现静态图的执行。

### Paddle SOT使用示例

```python
import paddle
from paddle.vision import resnet50

net = resnet50()

# 开启子图fallback，使用 fallback=True来允许跑多个子图。
net = paddle.jit.to_static(net, fallback=True)

output = net (image)
```



# 三、模型列表

各套件开源模型列表汇总

PaddleClas：

| 模型系列              | 模型名称                    |
| --------------------- | --------------------------- |
| ResNet系列            | ResNet50等                  |
| ResNeXt系列           | ResNeXt50等                 |
| Res2Net 系列          | Res2Net50等                 |
| SENet系列             | SE_ResNet50等               |
| DPN 系列              | DPN68等                     |
| DenseNet系列          | DenseNet169等               |
| HRNet 系列            | HRNet_W40_C等               |
| Inception 系列        | InceptionV3等               |
| EfficientNet系列      | EfficientNetB0等            |
| ResNeSt 系列          | ResNeSt50等                 |
| RegNet 系列           | RegNetX等                   |
| RepVGG 系列           | RepVGG_A0等                 |
| MixNet 系列           | MixNet_M等                  |
| ReXNet 系列           | ReXNet_1_0等                |
| HarDNet 系列          | HarDNet68等                 |
| DLA 系列              | DLA102等                    |
| RedNet 系列           | RedNet50等                  |
| ConvNeXt 系列         | ConvNeXt_tiny               |
| VGG 系列              | VGG16等                     |
| 轻量级移动端模型      | MobileNetV1、ShuffleNetV2等 |
| ViT 系列              | ViT_base等                  |
| DeiT 系列             | DeiT_base等                 |
| SwinTransformer 系列  | SwinTransformer_base等      |
| CSWinTransformer 系列 | CSWinTransformer_base等     |
| PVTV2 系列            | PVT_V2_B0等                 |
| LeViT 系列            | LeViT_256等                 |
| MobileViT 系列        | MobileViT_XS等              |

PaddleNLP：

| 预训练模型   |
| ------------ |
| electra      |
| ernie-1.0    |
| ernie-3.0    |
| ernie-doc    |
| ernie-gen    |
| ernie-health |
| ernie-layout |
| ernie-m      |
| ernie-tiny   |
| uie          |
| gpt          |
| gpt-3        |
| glm          |
| chatglm      |
| llama        |
| opt          |
| qwen         |
| bert         |
| albert       |

PaddleSeg：

| 模型类型       | 模型名称         |
| -------------- | ---------------- |
| 语义分割       | PP-LiteSeg       |
|                | PP-MobileSeg     |
|                | DeepLabV3P       |
|                | OCRNet           |
|                | MobileSeg        |
|                | ANN              |
|                | Att U-Net        |
|                | BiSegNet         |
|                | CCNet            |
|                | DANet            |
|                | DDRNet           |
|                | DeepLabV3        |
|                | DMNet            |
|                | DNLNet           |
|                | ENet             |
|                | ESPNet           |
|                | FastFCN          |
|                | Fast-SCNN        |
|                | UNet             |
|                | MaskFormer       |
|                | SegFormer        |
|                | SegNeXt          |
| 交互式分割模型 | EISeg            |
| 图像抠图模型   | PP-Matting       |
|                | DIM              |
|                | PP-HumanMatting  |
|                | RVM              |
| 全景分割       | Mask2Former      |
|                | Panoptic-DeepLab |

PaddleDetection：

| 任务类型              | 模型名称     |
| --------------------- | ------------ |
| 2D Detection          | Faster RCNN  |
|                       | FPN          |
|                       | Cascade-RCNN |
|                       | PSS-Det      |
|                       | YOLO系列     |
|                       | RTMDet       |
|                       | PP-YOLO系列  |
|                       | PP-PicoDet   |
|                       | SSD          |
|                       | CenterNet    |
|                       | FCOS         |
|                       | FCOSR        |
|                       | TTFNet       |
|                       | TOOD         |
|                       | GFL          |
|                       | DETR         |
|                       | Sparse RCNN  |
| Multi Object Tracking | JDE          |
|                       | FairMOT      |
|                       | DeepSORT     |
|                       | ByteTrack    |
|                       | OC-SORT      |
|                       | CenterTrack  |
| KeyPoint Detection    | HRNet        |
|                       | HigherHRNet  |
|                       | Lite-HRNet   |
|                       | PP-TinyPose  |
| Others                | MaskRCNN     |
|                       | PETR等       |

Paddle3D：

| 任务类型                | 模型名称         |
| ----------------------- | ---------------- |
| 单目3D感知-检测         | CaDDN            |
|                         | SMOKE            |
|                         | DD3D             |
| 激光雷达3D感知-检测     | PointPillars     |
|                         | CenterPoint      |
|                         | IA-SSD           |
|                         | PV-RCNN          |
|                         | Voxel-RCNN       |
|                         | PAConv           |
| 激光雷达3D感知-分割     | SqueezeSegV3     |
| 多相机3D感知-BEV-Camera | PETR             |
|                         | BEVFormer        |
|                         | CAPE             |
| 多相机3D感知-BEV-Fusion | BEVFusion(ADLab) |

PaddleOCR：

| 任务类型         | 模型名称       |
| ---------------- | -------------- |
| 文本检测算法     | DB、DB++       |
|                  | EAST           |
|                  | SAST           |
|                  | PSENet         |
|                  | PCENet         |
|                  | DRRG           |
|                  | CT             |
| 文本识别算法     | CRNN           |
|                  | Rosetta        |
|                  | STAR-Net       |
|                  | RARE           |
|                  | SRN            |
|                  | NRTR           |
|                  | SAR            |
|                  | SEED           |
|                  | SVTR           |
|                  | ViTSTR         |
|                  | ABINet         |
|                  | VisionLAN      |
|                  | SPIN           |
|                  | RobustScanner  |
|                  | RFL            |
|                  | ParseQ         |
|                  | CPPD           |
|                  | SATRN          |
| 文本超分辨率算法 | Text Gestalt   |
|                  | Text Telescope |
| 公式识别算法     | CAN            |
| 表格识别算法     | TableMaster    |
| 关键信息抽取算法 | LayoutLM       |
|                  | VI-LayoutXLM   |
| Paddle独家       | PP-OCR-v2等    |

PaddleRec：

| 任务类型 | 模型名称            |
| -------- | ------------------- |
| 内容理解 | TextCNN             |
|          | TagSpace            |
| 匹配     | DSSM                |
|          | Match-Pyramid       |
|          | MultiView-Simnet    |
|          | KIM                 |
| 召回     | TDM                 |
|          | FastText            |
|          | MIND                |
|          | Word2Vec            |
|          | DeepWalk            |
|          | SSR                 |
|          | Gru4Rec             |
|          | NCF                 |
|          | TiSAS               |
|          | ENSFM               |
|          | MHCN                |
|          | GNN                 |
|          | RALM                |
| 排序     | Logistic Regression |
|          | DNN                 |
|          | FM                  |
|          | BERT4REC            |
|          | FAT_DeepFFM         |
|          | FFM                 |
|          | FNN                 |
|          | Deep Crossing       |
|          | PNN                 |
|          | DCN                 |
|          | NFM                 |
|          | AFM                 |
|          | DMR                 |
|          | DeepFM              |
| 多任务   | AITM                |
|          | PLE                 |
|          | ESMM                |
| 重排序   | Listwise            |

PaddleGAN：

| 任务类型                  | 模型名称                 |
| ------------------------- | ------------------------ |
| 图像翻译-风格迁移         | Pixel2Pixel              |
| 图像翻译-风格迁移         | CycleGAN                 |
| 图像翻译-图像风格艺术转换 | LapStyle                 |
| 图像翻译-人脸换妆         | PSGAN                    |
| 图像翻译-照片动漫化       | AnimeGANv2               |
| 图像翻译-人像动漫化       | U-GAT-IT                 |
| 图像翻译-人像卡通化       | Photo2Cartoon            |
| 图像翻译-多种风格迁移     | StarGANv2                |
| 动作迁移-人脸表情迁移     | First Order Motion Model |
| 动作迁移-唇形合成         | Wav2Lip                  |
| 基础GAN                   | DCGAN                    |
|                           | WGAN                     |
| 人脸生成                  | StyleGAN                 |
|                           | FaceParsing              |
| 图片超分辨率              | SISR                     |
| 视频超分                  | VSR                      |
|                           | PP-MSVSR                 |
| 图像去模糊去噪去雨        | MPR Net                  |
|                           | SwinIR                   |
|                           | InvDN                    |
| 视频去模糊                | EDVR                     |
| 图像补全                  | AOT-GAN                  |

PaddleVideo：

| 任务类型             | 模型名称       |
| -------------------- | -------------- |
| 行为识别             | PP-TSM         |
|                      | PP-TSN         |
|                      | PP-TimeSformer |
|                      | SlowFast       |
|                      | MoViNet        |
|                      | VideoSwin      |
| 基于骨骼点的行为识别 | ST-GCN         |
|                      | AGCN           |
| 视频时序分割         | MS-TCN         |
|                      | ASRF           |
| 时序动作检测         | BMN            |
| 视频目标分割         | CFBI           |
|                      | MA-Net         |
| 单目深度估计         | ADDS           |
| 多模态               | ActBERT        |
|                      | T2VLAD         |

PaddleYOLO：

| 任务类型 | 模型名称           |
| -------- | ------------------ |
| 目标检测 | YOLOv3             |
|          | YOLOv5             |
|          | YOLOv6             |
|          | YOLOv7             |
|          | YOLOv8             |
|          | PP-YOLOv1/v2       |
|          | PP-YOLO-Tiny       |
|          | PP-YOLOE/PP-YOLOE+ |
|          | YOLOX              |
|          | RTMDet             |

# 五、实现方案

阶段一：

		1. 针对各个开源套件，构建Docker镜像，注意到不同开源套件对Paddle版本的依赖有所不同
		1. 测试列表列出的所有模型对动转静的支持情况，包括基于AST方式动转静/PaddleSOT方式动转静
		1. 产出待支持动转静模型列表文档

阶段二：

1. 优先修改，同一任务及数据集下有多个开源模型的套件，如PaddleClas、PaddleYOLO、PaddleOCR。
2. 为不支持动转静的模型依次添加to_static策略，参考[样例PR](https://github.com/PaddlePaddle/PaddleNLP/pull/1290/files)，包括修改config配置文件，添加to_static脚本，支持根据配置文件开启转静态模型。
3. 在开源数据集上进行训练实验，主要验证开启动转静训练后，前50个epoch的loss一致并提供截图，检查开启前后梯度计算是否一致。

# 六、测试和验收的考量

保证对套件模型尽可能少的代码侵⼊

# 七、影响面

对用户影响为：可通过配置文件选择是否开启动转静训练，转静成功后可以有效提高计算效率。

# 八、排期规划

1. 产出《**待⽀持动转静模型列表⽂档**》：搜集各套件开源模型列表，并对所有模型的动转静⽀持情况进⾏调研，

包括PaddleClas、PaddleNLP、PaddleSeg、PaddleDetection、Paddle3D、PaddleOCR、PaddleRec、

PaddleGAN、PaddleVideo、PaddleYOLO。（4月）

2. 逐步修改各个开源套件待⽀持动转静模型的代码，⽀持动转静训练，提供开启动转静训练前后前 50 个 step 

的 loss ⼀致性截图（5～6月）

# 附件及参考资料

1. https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/symbolic_opcode_translator

2. https://github.com/PaddlePaddle/PaddleSOT

3. https://github.com/PaddlePaddle/PaddleNLP/pull/7576
4. https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/jit/index_cn.html
5. https://github.com/PaddlePaddle/Paddle