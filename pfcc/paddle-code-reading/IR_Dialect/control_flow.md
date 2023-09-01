> 版本，作者，时间

| 版本 | 作者   | 时间       | 主要更新           |
| ---- | ------ | ---------- | ------------------ |
| v1.0 | 王明冬 | 2023.08.10 | 初版               |
| v1.1 | 王明冬 | 2023.08.22 | 添加评审时会谈纪要 |

# 一、概要

## 1、相关背景

IR升级的核心功能目标是设计一套能够由多方共用且优于当前的IR基础设施。

本文是IR升级《控制流模块》的设计评审文档，是在[类型系统](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/5K6Iojo8fU/hklvoOpOf9amJs)、[模型结构](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/6UmOO2EkH2/7a4ab3fcfc0e42)的基础上继续设计的。

本文的功能目标是完成控制流算子的抽象。

控制流能够设定特定的顺序执行计算任务，帮助构建更加灵活和复杂的模型。在模型中引入控制流后可以让计算图中某些节点循环执行任意次数，也可以根据条件判断选择某些节点不执行。

许多深度学习模型依赖控制流进行训练和推理，基于递归神经网络和强化学习的模型就依赖于循环递归关系和依据输入数据状态条件执行计算。

```cpp
def cond(i, ten):
    return i < ten

def body(i, ten):
    i = i + 1
    return [i, ten]

i = paddle.full(shape=[1], fill_value=0, dtype='int64')     # loop counter
ten = paddle.full(shape=[1], fill_value=10, dtype='int64')  # loop length
i, ten = paddle.static.nn.while_loop(cond, body, [i, ten])
```

以该while循环为例，目前paddle框架对其在计算图中的描述如下：

```cpp
{ // block 0
    var fill_constant_1.tmp_0 : LOD_TENSOR.shape(1,).dtype(int64).stop_gradient(True)
    var fill_constant_3.tmp_0 : LOD_TENSOR.shape(1,).dtype(int64).stop_gradient(True)
    var tmp_0 : LOD_TENSOR.shape(1,).dtype(bool).stop_gradient(True)
    var _generated_var_0 : STEP_SCOPES)

    {Out=['fill_constant_1.tmp_0']} = fill_constant(inputs={ShapeTensor=[], ShapeTensorList=[], ValueTensor=[]}, dtype = 3, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], place_type = -1, shape = [1], str_value = 0, value = 0.0, with_quant_attr = False)
    {Out=['fill_constant_3.tmp_0']} = fill_constant(inputs={ShapeTensor=[], ShapeTensorList=[], ValueTensor=[]}, dtype = 3, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], place_type = -1, shape = [1], str_value = 10, value = 10.0, with_quant_attr = False)
    {Out=['tmp_0']} = less_than(inputs={X=['fill_constant_1.tmp_0'], Y=['fill_constant_3.tmp_0']}, axis = -1, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
    {Out=['tmp_0', 'fill_constant_1.tmp_0'], StepScopes=['_generated_var_0']} = while(inputs={Condition=['tmp_0'], X=['fill_constant_3.tmp_0', 'fill_constant_1.tmp_0']}, is_test = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], sub_block = block[1], with_quant_attr = False)
}
{ // block 1
    var tmp_1 : LOD_TENSOR.shape(1,).dtype(int64).stop_gradient(True)
    var tmp_2 : LOD_TENSOR.shape(1,).dtype(bool).stop_gradient(True)

    {Out=['tmp_1']} = scale(inputs={ScaleTensor=[], X=['fill_constant_1.tmp_0']}, bias = 1.0, bias_after_scale = True, op_device = , op_namescope = /, op_role = 0, op_role_var = [], scale = 1.0, with_quant_attr = False)
    {Out=['tmp_2']} = less_than(inputs={X=['tmp_1'], Y=['fill_constant_3.tmp_0']}, axis = -1, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
    {Out=['fill_constant_1.tmp_0']} = assign(inputs={X=['tmp_1']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
    {Out=['tmp_0']} = assign(inputs={X=['tmp_2']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
}
```

这存在两个问题：

1. while算子的部分输入和也是输出，意味着计算图在主block里面存在有向环。[关于“有向环问题”总结](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/hPr_u_N_Lx/LNKjesZGk_qCD5)  这种情况对控制流相关分析优化显然是不太友好的。
2. cond 函数体在主block和子block中同时存在，相应的输入输出变量也会出现两份。目前是因此cond 函数的函数体比较简单，所以影响不大，但cond函数体变复杂以后，这种实现显然是不合理的。

新IR设计借鉴了MLIR的设计思想，坚持SSA原则。显然跟当前paddle的这种实现方式是相悖的。因此，本方案在不违反IR整体设计原则的情况下，综合考虑组合算子、自动微分、动态shape等各模块的需求，对控制流算子进行设计实现。

## 2、功能目标

- 完成新IR体系中控制流算子（If、While）的IR表示.
- 在不破坏IR设计原则的前提下，描述控制流算子的反向IR实现。

# 二、意义



# 三、竞品对照

目前，MLIR、TorchMLIR等都不涉及到控制流的反向描述，而控制流设计的最复杂之处就在于反向设计。

和普通算子不同，控制流反向的实现需要用到前向的子block中的局部变量。如果直接引用，显然是跟作用域原则相悖的。

因此，竞品的参考意义不大。本章只是对MLIR的前向控制流算子的设计进行简要描述。

## 1、MLIR的结构化控制流算子定义

MLIR对控制流算子的描述是通过scf方言来描述的。

### 1.1 WhileOp

~~~cpp
//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : SCF_Op<"while",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     RecursiveMemoryEffects]> {
  let summary = "a generic 'while' loop";
  let description = [{
    This operation represents a generic "while"/"do-while" loop that keeps
    iterating as long as a condition is satisfied. There is no restriction on
    the complexity of the condition. It consists of two regions (with single
    block each): "before" region and "after" region. The names of regions
    indicates whether they execute before or after the condition check.
    Therefore, if the main loop payload is located in the "before" region, the
    operation is a "do-while" loop. Otherwise, it is a "while" loop.

    The "before" region terminates with a special operation, `scf.condition`,
    that accepts as its first operand an `i1` value indicating whether to
    proceed to the "after" region (value is `true`) or not. The two regions
    communicate by means of region arguments. Initially, the "before" region
    accepts as arguments the operands of the `scf.while` operation and uses them
    to evaluate the condition. It forwards the trailing, non-condition operands
    of the `scf.condition` terminator either to the "after" region if the
    control flow is transferred there or to results of the `scf.while` operation
    otherwise. The "after" region takes as arguments the values produced by the
    "before" region and uses `scf.yield` to supply new arguments for the
    "before" region, into which it transfers the control flow unconditionally.

    A simple "while" loop can be represented as follows.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> f32 {
      // "Before" region.
      // In a "while" loop, this region computes the condition.
      %condition = call @evaluate_condition(%arg1) : (f32) -> i1

      // Forward the argument (as result or "after" region argument).
      scf.condition(%condition) %arg1 : f32

    } do {
    ^bb0(%arg2: f32):
      // "After" region.
      // In a "while" loop, this region is the loop body.
      %next = call @payload(%arg2) : (f32) -> f32

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %next : f32
    }
    ```

    A simple "do-while" loop can be represented by reducing the "after" block
    to a simple forwarder.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> f32 {
      // "Before" region.
      // In a "do-while" loop, this region contains the loop body.
      %next = call @payload(%arg1) : (f32) -> f32

      // And also evaluates the condition.
      %condition = call @evaluate_condition(%arg1) : (f32) -> i1

      // Loop through the "after" region.
      scf.condition(%condition) %next : f32

    } do {
    ^bb0(%arg2: f32):
      // "After" region.
      // Forwards the values back to "before" region unmodified.
      scf.yield %arg2 : f32
    }
    ```

    Note that the types of region arguments need not to match with each other.
    The op expects the operand types to match with argument types of the
    "before" region; the result types to match with the trailing operand types
    of the terminator of the "before" region, and with the argument types of the
    "after" region. The following scheme can be used to share the results of
    some operations executed in the "before" region with the "after" region,
    avoiding the need to recompute them.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> i64 {
      // One can perform some computations, e.g., necessary to evaluate the
      // condition, in the "before" region and forward their results to the
      // "after" region.
      %shared = call @shared_compute(%arg1) : (f32) -> i64

      // Evaluate the condition.
      %condition = call @evaluate_condition(%arg1, %shared) : (f32, i64) -> i1

      // Forward the result of the shared computation to the "after" region.
      // The types must match the arguments of the "after" region as well as
      // those of the `scf.while` results.
      scf.condition(%condition) %shared : i64

    } do {
    ^bb0(%arg2: i64) {
      // Use the partial result to compute the rest of the payload in the
      // "after" region.
      %res = call @payload(%arg2) : (i64) -> f32

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %res : f32
    }
    ```

    The custom syntax for this operation is as follows.

    ```
    op ::= `scf.while` assignments `:` function-type region `do` region
           `attributes` attribute-dict
    initializer ::= /* empty */ | `(` assignment-list `)`
    assignment-list ::= assignment | assignment `,` assignment-list
    assignment ::= ssa-value `=` ssa-value
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$before, SizedRegion<1>:$after);

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$operands,
      "function_ref<void(OpBuilder &, Location, ValueRange)>":$beforeBuilder,
      "function_ref<void(OpBuilder &, Location, ValueRange)>":$afterBuilder)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, ValueRange)>;

    OperandRange getSuccessorEntryOperands(std::optional<unsigned> index);
    ConditionOp getConditionOp();
    YieldOp getYieldOp();
    Block::BlockArgListType getBeforeArguments();
    Block::BlockArgListType getAfterArguments();
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
~~~

如上所述，while op包含两个region， 称为before region和after region。 before region以scf.conditon算子结尾，如果conditon的输入条件为true， 就会把控制流传递到after region。否则，将控制流返回到父op， 表示执行结束。

after region以scf.yield算子结尾，表示将控制流传递到before region。

显然，当主要循环开销在before region时， while op等价于c++的 "do-while"语句。当主要循环开销在after region时，while op等价于c++的while语句。

### 1.2  IfOp

~~~cpp
//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : SCF_Op<"if", [DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getNumRegionInvocations", "getRegionInvocationBounds"]>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    SingleBlockImplicitTerminator<"scf::YieldOp">, RecursiveMemoryEffects,
    NoRegionArguments]> {
  let summary = "if-then-else operation";
  let description = [{
    The `scf.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    scf.if %b  {
      ...
    } else {
      ...
    }
    ```

    `scf.if` may also produce results. Which values are returned depends on
    which execution path is taken.

    Example:

    ```mlir
    %x, %y = scf.if %b -> (f32, f32) {
      %x_true = ...
      %y_true = ...
      scf.yield %x_true, %y_true : f32, f32
    } else {
      %x_false = ...
      %y_false = ...
      scf.yield %x_false, %y_false : f32, f32
    }
    ```

    The "then" region has exactly 1 block. The "else" region may have 0 or 1
    block. In case the `scf.if` produces results, the "else" region must also
    have exactly 1 block.

    The blocks are always terminated with `scf.yield`. If `scf.if` defines no
    values, the `scf.yield` can be left out, and will be inserted implicitly.
    Otherwise, it must be explicit.

    Example:

    ```mlir
    scf.if %b  {
      ...
    }
    ```

    The types of the yielded values must match the result types of the
    `scf.if`.
  }];
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond,
      "bool":$addThenBlock, "bool":$addElseBlock)>,
    OpBuilder<(ins "Value":$cond, "bool":$withElseRegion)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond,
      "bool":$withElseRegion)>,
    OpBuilder<(ins "Value":$cond,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "buildTerminatedBody">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>,
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(0);
      return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                                  : OpBuilder::atBlockEnd(body, listener);
    }
    OpBuilder getElseBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(1);
      return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                                  : OpBuilder::atBlockEnd(body, listener);
    }
    Block* thenBlock();
    YieldOp thenYield();
    Block* elseBlock();
    YieldOp elseYield();
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
~~~

如上所示，if op包含两个region， 称为then region和else region。if op只有一个输入conditon，如果输入为true，执行then region， 否则，执行else region。

then region和else region的输出都跟if op的输出匹配。如果if op没有输出，那么else region可以为空。

# 四、设计思路与实现方案

新IR通过Operation、Region、Block三者的循环嵌套来表示结构化控制流。

一个Op会包含0个或多个Region,   一个Region会包含0个或多个Block, 一个Block里面包含了0个或多个Operation。  三者循环嵌套包含，用来描述复杂的模型结构。

## 1、 基础组件

### 1.1 Block

- **Block**

新IR的Block等价于基本块， 里面包含了一个算子列表（std::list<Operaiton*>)， 用来表示该基本块的计算语意。

```cpp
^block:
  %a  = "pd.feed" () ...
  %b  = "pd.feed" () ...
  %c = pd.add(%a, %b) ...
  pd.fetch(%c) ....
```

样例1就是一个简单的block样例。

- **BlockArgument**

Block可以包含一个形参列表(std::vector<BlockArgument>), 来表示执行该Block所需要的参数数量和类型。

```cpp
^block (%a ：tensor<...>, %b:tensor<...>):
  %c = pd.add(%a, %b) ...
  pd.fetch(%c) ....
```

样例2是一个简单的带BlockArgument的block样例。 它将样例1的通过feed算子来获取的两个变量%a和%b通过BlockArgument来描述。这意味着，控制流在进入该block之前，必须给%a和%b绑定变量。

- **BlockOperand**

Block可以被封装为BlockOperand(类似Value和OpOperand的关系)。作为终止符一种特殊的操作数， 称为后继块(successor)。终止符算子是指一类有特殊语意的算子，他们可以作为基本块的最后一个op。比如： return、fetch、branch等等。

```cpp
#reigon
{
    ^condition_block (%cond):
        %1 = pd.constant(1)
        %2 = pd.constant(2)
        pd.condition_branch %cond, ^then_block(%1), else_block(%2)
    ^then_block(%val_1):
        pd.return %val_1 
    ^else_block(%val_2):
        pd.return %val_2 
}
```

样例3是一个Block作为终止符算子的操作数的一个例子。 样例中，pd.condition_branch接受三个操作数：%cond、%1、%2的同时，接受两个blockOperand: then_block和else_block，它的语意时，如果%cond的值为True， 就将控制流传递到then_block, 同时将%1作为参数传递给then_block的BlockArgument。否则，就将控制流传递到else_block, 同时将%2作为参数传递给else_block的BlockArgument。

注: 在控制流之前， 一个operation由它的输入、输出、属性以及类型信息构成。  加入控制流以后，一个operation的内容包含：它的输入(OpOperand)、输出（OpResult）、属性(AttributeMap)、后继块（BlockOperand）、region组成。 新增了后继块和region。

**当Block的最后一个算子执行结束时，根据块内最后一个算子(终止符算子)的语意，控制流会有两种去处：**

1. **进入同Region的另外一个Block, 该Block一定是终止符算子的后继块。**
2. **返回该Block的父Region, 表示该Region的一次执行的结束。**

### 1.2 Region

Region里面包含了一个Block列表（std::vector<Block>)， 第一个Block(如果存在的话)，称为该Region的入口块。

**与基本块不同，Region存在一个最显著的约束是：Region内定义的Value只能在该Region内部使用，Region的外面不允许使用****。**这个约束与我们执行期的scope比较类似。

当控制流进入一个region， 相当于创建了一个新的子scope， 当控制流退出该region时，该子scope中定义的所有变量都可以回收。

**控制流进入Region, 一定会首先进入该Region的入口块。**因此，Region的参数用入口块参数即可描述，不需要额外处理。

**当Region的一次执行结束，控制流由子Block返回到该Region时，控制流会有两种去处：**

1. **进入同Op的某一个Region(可能是自己)。**
2. **返回该Region的父Op，表示该Op的一次执行的结束。**

具体去处由该Region的父Op的语意决定。

## 2、 控制流算子

### 2.1 辅助工具

为了实现分支语意以及反向函数，需要预定义一些特殊的类型和算子，用来辅助描述控制流算子。

这些辅助类型和算子目前先定义在cf（control flow） dialect中。后续有必要的话，可以将部分类型下沉到builtin dialect中。

------

考虑到WhileOp的反向需要访问前向的局部变量，而且反向的循环迭代与前向是相反的，特提供以下类型和算子：

- **cf.StackType**

StackType表示一个支持先进后出的栈类型。 该类型不需要参数。

```cpp
class IR_API StackType : public Type {
    DECLARE_TYPE_UTILITY_FUNCTOR(StackType, TypeStorage);   
    .......
}
```

- **cf.CreateStackOp**

CreateStackOp算子的语意是创建一个空栈。 该算子没有输入、 没有属性、 输出一个类型为 StackType的value。

```cpp
// %0是一个stack类型变量
%0 = cf.create_stack() {} : ()->cf.stack
```

- **cf.PushBackOp**

PushBackOp算子的语意是将一个变量进栈。该算子接受两个输入，第一个为StackType的Value，表示栈， 第二个输入为要被进栈的变量。没有属性，没有输出。

```cpp
// %1对应的变量被压栈到了%0中
cf.push_back(%0, %1){}: (cf.stack, tensor<...>)->()
```

- **cf.PopBackOp**

PopBackOp算子的语意是将栈末尾的变量弹出来。该算子接受一个类型为StackType的输入。 没有属性，有一个输出，表示栈中被弹出的变量。

```cpp
// 从%0对应的栈中pop_back出一个变量，记为%2
%2 = cf.pop_back(%0) {}: cf.stack -> tensor<...>
```

- **cf.IsEmptyOp**

IsEmptyOp算子的语意是判断栈是否为空。该算子接受一个类型为StackType的输入。没有属性，有一个bool类型的输出，表示栈是否为空。

```cpp
// 判断栈是否为空，返回bool变量
%cond = cf.is_empty(%0) {}:cf.stack -> bool
```

------

同时，我们新增一些终止符算子作为控制流中block的最后一个算子：

- **cf.YieldOp**

Yield算子接受可变输入，没有输出和属性。它的语意是将控制流和输入变量传递给父region。类似于函数的return语句。

```cpp
// %1，%2等表示该region执行的返回值。
cf.yield(%1, %2, ....)
```

在pd.CondOp的子block中，会用cf.yield算子来描述返回值。

- **cf.CondYieldOp**

CondYield算子比Yield算子多了一个bool类型的条件变量。它的父region会根据该变量，进行一次分支决策。

```cpp
// 将控制流传递到父region. 父region会根据%cond的值，进行分支。 
// 对于while op的body region而言，如果%cond为True， 他会将控制流传递给body_region， %0、%1...会被传递给body_region当参数。否则，将控制流返回while_op, %0、%1...会被当作while_op的输出。
cf.cond_yield (%cond, %0, %1,  ...)
```

### 2.2 IfOp(CondOp)

IfOp应该包含两个或三个Region；

两个region：then_region和else_region对应无反向场景；

如果存在反向，会额外增加一个init region，同时会增加一个表示变量栈的stack输出。(这种场景会在2.3.1:IfOp的反向实现中进行描述)

IfOp只有一个输入condition。 输出是可变的。 （**这是因为子block可以直接访问父block的变量，CondOp的内部block的前驱也是唯一的，因此没必要设置参数，在用的地方直接访问原变量即可**）

如果IfOp的输出为空，那么else_region可以为空。

否则，else_region和then_region一样，都必须包含一个不带参数的Block， 分别表示then和else的分支。这两个block都必须以cf.yield算子结尾(如果输出为空，cf.yield可以省略)。

cf.yield算子接收可变输入，没有输出。 语意是将它的输入转发给父Op当作输出，它的输入个数与IfOp的输出匹配。

IfOp的Verify函数功能包括：

1. 如果包含两个region：
   1.  then_region只包含一个block， 该block
      1. 参数为空
      2. 以cf.yield算子结尾，且cf.yield算子的输入跟IfOp的输出的数量和类型相匹配。
   2.  如果IfOp的输入为空，那么else_region也可以为空。否则else_region一定只包含一个block，该bock
      1. 参数为空
      2. 以cf.yield算子结尾，且cf.yield算子的输入跟IfOp的输出的数量和类型相匹配。
2. 如果包含三个region:
   1. init_region只包含一个block， 该block只包含两个算子： cf.create_stack 和 cf.yield.
   2.  then_region只包含一个block， 该block
      1. 只有一个stack类型的block_argument。
      2. 以cf.yield算子结尾，且cf.yield算子的输入跟CondOp的输出的数量和类型相匹配。
   3. 如果else_region为空，那么CondOp的非stack输出也一定为空。否则else_region也一定包含一个block，该bock
      1. 只有一个stack类型的block_argument。
      2. 以cf.yield算子结尾，且cf.yield算子的输入跟CondOp的输出的数量和类型相匹配。

```cpp
#
# pseudocode:
# if 0.1 < 0.23:
#     return 1, True
# else:
#     return 3, 2
#
def true_func():
    a = paddle.full(shape=[1, 2], dtype='int32',fill_value=1)
    b = paddle.full(shape=[2, 3], dtype='bool', fill_value=True)
    return a, b


def false_func():
    a = paddle.full(shape=[3, 4], dtype='float32',fill_value=3)
    b = paddle.full(shape=[4, 5], dtype='int64', fill_value=2)
    return  a, b

x = paddle.full(shape=[1], dtype='float32', fill_value=0.1)
y = paddle.full(shape=[1], dtype='float32', fill_value=0.23)
pred = paddle.less_than(x=x, y=y, name=None)
ret = paddle.static.nn.cond(pred, true_func, false_func)
```

对应的新IR下的算子描述为：

```cpp
%x = pd.full(....)
%y = pd.full(....)
%cond = pd.less_than(x, y)
%ret1, %ret2 = pd.if(%cond) {
      %1 = pd.full(....)
      %2 = pd.full(...)
      cf.yield(%1, %2)
    } else {
      %1 = pd.full(....)
      %2 = pd.full(...)
      cf.yield(%1, %2)
    }
```

当前paddle框架对应的描述为：

```cpp
{ // block 0
    var fill_constant_1.tmp_0 : LOD_TENSOR.shape(1,).dtype(float32).stop_gradient(True)
    var fill_constant_3.tmp_0 : LOD_TENSOR.shape(1,).dtype(float32).stop_gradient(True)
    var less_than_0.tmp_0 : LOD_TENSOR.shape(1,).dtype(bool).stop_gradient(True)
    var _generated_var_0 : LOD_TENSOR.shape(1, 2).dtype(int32).stop_gradient(True)
    var _generated_var_1 : LOD_TENSOR.shape(2, 3).dtype(bool).stop_gradient(True)
    var _generated_var_2 : STEP_SCOPES)
    var logical_not_0.tmp_0 : LOD_TENSOR.shape(1,).dtype(bool).stop_gradient(True)
    var _generated_var_3 : LOD_TENSOR.shape(3, 4).dtype(float32).stop_gradient(True)
    var _generated_var_4 : LOD_TENSOR.shape(4, 5).dtype(int64).stop_gradient(True)
    var _generated_var_5 : STEP_SCOPES)
    var cast_0.tmp_0 : LOD_TENSOR.shape(1,).dtype(int32).stop_gradient(True)
    var _generated_var_6 : LOD_TENSOR.shape(-1, -1).dtype(int32).stop_gradient(True)
    var _generated_var_7 : LOD_TENSOR.shape(-1, -1).dtype(bool).stop_gradient(True)

    {Out=['fill_constant_1.tmp_0']} = fill_constant(inputs={ShapeTensor=[], ShapeTensorList=[], ValueTensor=[]}, dtype = 5, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], place_type = -1, shape = [1], str_value = 0.1, value = 0.10000000149011612, with_quant_attr = False)
    {Out=['fill_constant_3.tmp_0']} = fill_constant(inputs={ShapeTensor=[], ShapeTensorList=[], ValueTensor=[]}, dtype = 5, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], place_type = -1, shape = [1], str_value = 0.23, value = 0.23000000417232513, with_quant_attr = False)
    {Out=['less_than_0.tmp_0']} = less_than(inputs={X=['fill_constant_1.tmp_0'], Y=['fill_constant_3.tmp_0']}, axis = -1, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
    {Out=['_generated_var_1', '_generated_var_0'], Scope=['_generated_var_2']} = conditional_block(inputs={Cond=['less_than_0.tmp_0'], Input=[]}, is_scalar_condition = True, op_device = , op_namescope = /, op_role = 0, op_role_var = [], sub_block = block[1], with_quant_attr = False)
    {Out=['logical_not_0.tmp_0']} = logical_not(inputs={X=['less_than_0.tmp_0']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
    {Out=['_generated_var_4', '_generated_var_3'], Scope=['_generated_var_5']} = conditional_block(inputs={Cond=['logical_not_0.tmp_0'], Input=[]}, is_scalar_condition = True, op_device = , op_namescope = /, op_role = 0, op_role_var = [], sub_block = block[2], with_quant_attr = False)
    {Out=['cast_0.tmp_0']} = cast(inputs={X=['less_than_0.tmp_0']}, in_dtype = 0, op_device = , op_namescope = /, op_role = 0, op_role_var = [], out_dtype = 2, use_mkldnn = False, with_quant_attr = False)
    {Out=['_generated_var_6']} = select_input(inputs={Mask=['cast_0.tmp_0'], X=['_generated_var_3', '_generated_var_0']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
    {Out=['_generated_var_7']} = select_input(inputs={Mask=['cast_0.tmp_0'], X=['_generated_var_4', '_generated_var_1']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
}
{ // block 1
    var fill_constant_5.tmp_0 : LOD_TENSOR.shape(1, 2).dtype(int32).stop_gradient(True)
    var fill_constant_7.tmp_0 : LOD_TENSOR.shape(2, 3).dtype(bool).stop_gradient(True)

    {Out=['fill_constant_5.tmp_0']} = fill_constant(inputs={ShapeTensor=[], ShapeTensorList=[], ValueTensor=[]}, dtype = 2, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], place_type = -1, shape = [1, 2], str_value = 1, value = 1.0, with_quant_attr = False)
    {Out=['fill_constant_7.tmp_0']} = fill_constant(inputs={ShapeTensor=[], ShapeTensorList=[], ValueTensor=[]}, dtype = 0, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], place_type = -1, shape = [2, 3], str_value = 1.0, value = 1.0, with_quant_attr = False)
    {Out=['_generated_var_0']} = assign(inputs={X=['fill_constant_5.tmp_0']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
    {Out=['_generated_var_1']} = assign(inputs={X=['fill_constant_7.tmp_0']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
}
{ // block 2
    var fill_constant_9.tmp_0 : LOD_TENSOR.shape(3, 4).dtype(float32).stop_gradient(True)
    var fill_constant_11.tmp_0 : LOD_TENSOR.shape(4, 5).dtype(int64).stop_gradient(True)

    {Out=['fill_constant_9.tmp_0']} = fill_constant(inputs={ShapeTensor=[], ShapeTensorList=[], ValueTensor=[]}, dtype = 5, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], place_type = -1, shape = [3, 4], str_value = 3.0, value = 3.0, with_quant_attr = False)
    {Out=['fill_constant_11.tmp_0']} = fill_constant(inputs={ShapeTensor=[], ShapeTensorList=[], ValueTensor=[]}, dtype = 3, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], place_type = -1, shape = [4, 5], str_value = 2, value = 2.0, with_quant_attr = False)
    {Out=['_generated_var_3']} = assign(inputs={X=['fill_constant_9.tmp_0']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
    {Out=['_generated_var_4']} = assign(inputs={X=['fill_constant_11.tmp_0']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
}
```

显然，paddle框架会将控制流的ture分支和false分支分别插入一个condition_block op。再通过select_input op对两个condition_block op的输出进行选择。这是因为在当前框架，一个Op只能包含一个Block,所以遇见IfOp这种算子，必须拆分两个Op。

### 2.3 WhileOp

WhileOp包含两个或三个region。

如果是两个region，那就是cond_region和body_region。

如果是三个region,  那就是init_region、condition_region、body_region。

init_region只做一件事：创建一个stack，将其和输入参数一起， 转发给cond_region，该stack会在循环中压栈一些局部变量，并作为输出传递到更高层次的作用域，提供给反向算子中使用。如果不考虑反向，那么stack输出可以省略，相应的，init_region也可以省略。 WhileOp直接拿cond_region作为入口执行也是可以的。

------

以下是一个只考虑前向的WhileOp的样例：

```cpp
def cond(i, ten):
    return i < ten

def body(i, ten):
    i = i + 1
    return [i, ten]

i = paddle.full(shape=[1], fill_value=0, dtype='int64')     # loop counter
ten = paddle.full(shape=[1], fill_value=10, dtype='int64')  # loop length
i, ten = paddle.static.nn.while_loop(cond, body, [i, ten])
```

对应的新IR下，只观翻译后的初始版本描述为：

```cpp
%i = pd.full(...)
%ten = pd.full(...)
%i_2, %ten2 = pd.while(%i, %ten) {
    // cond region
    ^bb0 (%arg1, %arg2):
      %cond = pd.less_than(%arg1, %arg2)
      cf.cond_yield (%cond, %arg1, %arg2)
   } do {
    // body region
    ^bb1(%arg1, %arg2):
      %1 = pd.const(1)
      %i_3 = pd.add(%arg1, %1)
      cf.yield (%i_3, %arg2)
  }
```

可以通过数据流分析发现，cond_region和body_region的第二个块参数始终绑定的是%ten, 因此可以进一步优化为：

```cpp
%i = pd.full(...)
%ten = pd.full(...)
%i_2 = pd.while(%i) {
   // cond region
   ^bb0(%arg1):
     %cond = pd.less_than(%arg1, %ten)
     cf.cond_yield (%cond, %arg1)
  } do {
   // body region
   ^bb1(%arg1):
     %1 = pd.const(1)
     %i_3 = pd.add(%arg1, %1)
     cf.yield (%i_3)
  }
```

再进一步，可以发现body_region里面的%1其实每一轮都是一样的，因此，可以将其提升到循环外：

```cpp
%i = pd.full(...)
%1 = pd.const(1)
%ten = pd.full(...)
%i_2 = pd.while(%i) {
    // cond_region
    ^bb0(%arg1):
     %cond = pd.less_than(%arg1, %ten)
     cf.cond_yield (%cond, %arg1)
  } do {
   // body_region
    ^bb1(%arg2):
     %i_3 = pd.add(%arg2, %1)
     cf.yield (%i_3)
  }
```

当前paddle框架对应的描述为：

```cpp
{ // block 0
    var fill_constant_1.tmp_0 : LOD_TENSOR.shape(1,).dtype(int64).stop_gradient(True)
    var fill_constant_3.tmp_0 : LOD_TENSOR.shape(1,).dtype(int64).stop_gradient(True)
    var tmp_0 : LOD_TENSOR.shape(1,).dtype(bool).stop_gradient(True)
    var _generated_var_0 : STEP_SCOPES)

    {Out=['fill_constant_1.tmp_0']} = fill_constant(inputs={ShapeTensor=[], ShapeTensorList=[], ValueTensor=[]}, dtype = 3, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], place_type = -1, shape = [1], str_value = 0, value = 0.0, with_quant_attr = False)
    {Out=['fill_constant_3.tmp_0']} = fill_constant(inputs={ShapeTensor=[], ShapeTensorList=[], ValueTensor=[]}, dtype = 3, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], place_type = -1, shape = [1], str_value = 10, value = 10.0, with_quant_attr = False)
    {Out=['tmp_0']} = less_than(inputs={X=['fill_constant_1.tmp_0'], Y=['fill_constant_3.tmp_0']}, axis = -1, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
    {Out=['tmp_0', 'fill_constant_1.tmp_0'], StepScopes=['_generated_var_0']} = while(inputs={Condition=['tmp_0'], X=['fill_constant_3.tmp_0', 'fill_constant_1.tmp_0']}, is_test = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], sub_block = block[1], with_quant_attr = False)
}
{ // block 1
    var tmp_1 : LOD_TENSOR.shape(1,).dtype(int64).stop_gradient(True)
    var tmp_2 : LOD_TENSOR.shape(1,).dtype(bool).stop_gradient(True)

    {Out=['tmp_1']} = scale(inputs={ScaleTensor=[], X=['fill_constant_1.tmp_0']}, bias = 1.0, bias_after_scale = True, op_device = , op_namescope = /, op_role = 0, op_role_var = [], scale = 1.0, with_quant_attr = False)
    {Out=['tmp_2']} = less_than(inputs={X=['tmp_1'], Y=['fill_constant_3.tmp_0']}, axis = -1, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
    {Out=['fill_constant_1.tmp_0']} = assign(inputs={X=['tmp_1']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
    {Out=['tmp_0']} = assign(inputs={X=['tmp_2']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], with_quant_attr = False)
}
```

显然，当前版本由于只支持一个子block, 因此会将cond_block的代码复制一遍，一份放在主block，一份放在子block。

------

### 2.3 对backward的支持

下文对于backward的设计支持，基于以下思路：

1. 问：前向算子和反向算子是否应该处于同一个block？如何描述他们的嵌套关系？  

答：对于最顶层block中的算子，它的前反向处于同一个block。但是对于子作用域中的算子，前反向应该处于不同的block中，但前反向block的辈分(嵌套层级)一定是相同的。 

------

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=59cc5d6b27ad4d3db713dd31159d03f2&docGuid=e81ffae8658a48)

如图1所示，假设program包含了while_op_1,  while_op_1包含了while_op_2,  while_op_2嵌套包含了while_op_3, .......嵌套包含了while_op_n.......

则在经过了backward pass之后，program会包含while_op_1和while_op_1_grad,  while_op_1_grad嵌套包含了while_op_2_grad,  while_op_2_grad嵌套包含了while_op_3_grad,  .......嵌套包含了while_op_n_grad........。

while_op_n和while_op_n_grad位于不同的block， 但二者的辈分(离program的嵌套层数)是相同的。

------

1. 问：如果前反向算子不在同一个block, 当反向算子需要访问前向的输入输出时，如何在不破坏作用域原则(父作用域不允许直接访问子作用域变量)的前提下，构造计算图的拓扑关系？

答：新IR严格遵守作用域原则：当反向block需要访问前向block中定义的变量时，必须先以返回值传递的方式，将该变量沿着前向嵌套作用层级，逐层向上传递到他们的公共祖先作用域；再以参数传递的方式，沿着反向block嵌套作用层级，逐层向下传递到反向block。

通过我们在2.1节中描述的栈类型变量、push_back算子、pop_back算子，我们只需要传递一个栈变量，在前向block中，按照定义顺序，将反向block中需要用到的局部变量依次压栈，在反向block中，依次出栈即可。因为反向block的执行顺序和前向block的执行顺序是互逆的，因此，这儿使用先进后出的栈来进行值传递。

backward pass 或者每个op的反向创建的接口：需要保证，在每个前向block中压栈变量的数量和顺序和反向block中出栈变量的数量和顺序是匹配的。

------

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=ff386536474f4ef9b2fd7c0e846ee1e0&docGuid=e81ffae8658a48)

因为子作用域可以访问父作用域中定义的变量， 图2所举的例子中，while_3_op的子block可以访问的变量范围是： program的主block、while_1_op的子block、while_2_op的子block。

在构建while_3_op_grad算子的子block时，program的主block本来就是它的顶层作用域，可以直接访问。

while_1_op的子block中的变量都通过入栈出栈的方式，对偶到了while_1_op_grad的子block中，而while_1_op_grad的子block也是while_3_op_grad的祖先block, 可以直接访问。

类似的，while_2_op的子block中变量也被对偶到了while_2_op_grad的子block中，显然，这是while_3_op_grad父block， 因此可以直接访问。

综上，while_3_op_grad所在的位置可以访问到while_3_op所能访问的所有变量(可能是对偶变量)，因此，while_3_op_grad可以在不违反作用域原则的情况下，成功构建。

------

1. 问：反向block需要访问前向block中的局部变量，为了实现该目的，我们设计了压栈出栈的实现方式。当添加完反向，训练结束以后，如何进行推理部署呢？或者说如何移除其中的压栈算子呢？

答：只需要在裁剪反向的pass的最后，追加一个特殊的类似DCE的Pass。比如while_op， 在裁剪了反向以后，我们就会发现，while_op的代表局部变量栈的输出变量已经没有消费者了。这个输出本来就是optional的，所以可以直接将该输出移除即可。 相对应的，该while_op里面的子block的终止符算子也需要移除相应的输入。推往前推，相应的push_back算子、create_stack算子也可以被移除。经过这个pass, 计算图会被变换得和裁剪前一致。

更通用的情况，可以提供backward_pass和逆backward_pass。验证二者的可逆性。

#### 2.3.1 IfOp的反向实现

本方案将IfOp的反向构建流程从逻辑上分为三步：（后面whileOp也类似）

第一步：对前向IfOp进行改造，增加一个region，创建变量栈，将所有子block中的局部变量压栈，并将该变量栈新增为IfOp的输出。  

If包含两个或三个region。 如果包含了三个region，说明已经求了一次反向，这种情况我们在后文2.3.3中进行描述。这儿直接假设遇见的IfOp一定只包含了两个region。

1. 在then_region的前方，插入一个init region，该init region只包含一个block， 承担两个功能：
   1. 创建一个stack。（一个create_stack算子）
   2. 将该栈变量转发给其它region。（一个cf.yield算子）
2. 在then region和else region的输入输出中都新增stack变量。并将所有该region中定义的局部变量压栈该stack中。

```cpp
%x = pd.full(....)
%y = pd.full(....)
%cond = pd.less_than(x, y)
%ret1, %ret2 = pd.if(%cond) {
      %1 = pd.full(....)
      %2 = pd.full(...)
      cf.yield(%1, %2)
    } else {
      %1 = pd.full(....)
      %2 = pd.full(...)
      cf.yield(%1, %2)
    }
```

改造后的if op为：

```cpp
%x = pd.full(....)
%y = pd.full(....)
%cond = pd.less_than(x, y)
%ret1, %ret2, %stack = pd.if(%cond) 
   init {
      %stack = cf.create_stack()
      cf.yield(%stack)
   }
   then(%arg_stack)
   {
      %1 = pd.full(....)
      %2 = pd.full(...)
      cf.push_back(%arg_stack, %1)
      cf.push_back(%arg_stack, %2)
      cf.yield(%1, %2, %arg_stack)
    } 
    else (%arg_stack)
    {
      %1 = pd.full(....)
      %2 = pd.full(...)
      cf.push_back(%arg_stack, %1)
      cf.push_back(%arg_stack, %2)
      cf.yield(%1, %2, %arg_stack)
    }
```

第二步：构造反向if_grad op。（反向if_grad op其实也是一个if_op，只是将其命名为if_grad，实现和cond一致）

创建一个if_grad op, 它包含then_region和else_region。 它的输入和前向cond_op的输入完全一致，只包含一个%cond变量即可。

**（正常来说，反向op的输入应该包含了前向op的输出变量的反向，但是我们之前在CondOp的设计中已经描述过了，子block可以直接访问父block变量，因此，不需要作为输入）**

if_grad op 也没有输出。这是因为if_op是在子block中，直接对父block中的变量进行引用，那相应的，在子block中，如果涉及到对父block中变量的使用，我们之间在原地进行梯度累加即可。

在then_region和else_region中将前向中的压栈的所有局部变量都出栈。然后按照bakcward的正常逻辑，依次给后向region中添加前向region的反向算子。

```cpp
%x = pd.full(....)
%y = pd.full(....)
%cond = pd.less_than(x, y)
%ret1, %ret2, %stack = pd.if(%cond) 
   init {
      %stack = cf.create_stack()
      cf.yield(%stack)
   }
   then(%arg_stack)
   {
      %1 = pd.full(....)
      %2 = pd.full(...)
      cf.push_back(%arg_stack, %1)
      cf.push_back(%arg_stack, %2)
      cf.yield(%1, %2, %arg_stack)
    } else (%arg_stack)
    {
      %1 = pd.full(....)
      %2 = pd.full(...)
      cf.push_back(%arg_stack, %1)
      cf.push_back(%arg_stack, %2)
      cf.yield(%1, %2, %arg_stack)
    }
 
 pd.if_grad(%cond) 
   then{
      %1 = cf.pop_back(%stack)
      %2 = cf.pop_back(%stack)
   }
   else {
      %1 = cf.pop_back(%stack)
      %2 = cf.pop_back(%stack)
   }
```

第三步：优化剪枝，将反向算子中用不到的栈变量，移除它的压栈和出栈算子。

为了保证第二步中给每个子block中的算子添加反向可以正常进行。我们在第一步和第二步中对所有的局部变量都进行了压栈出栈操作。

但实际上，并不是所有的局部变量在反向计算中都会被使用。因此。这一步主要就是移除无意义的压栈出栈操作。

如果发现栈里的局部变量完全用不到的话，可以将if_op的init_region也直接删除。

#### 2.3.2 WhileOp的反向实现

本方案将WhileOp的反向构建流程从逻辑上也分为三步：

第一步：对前向WhileOp进行改造，增加一个region，创建变量栈，将所有子block中的局部变量压栈，并将该变量栈新增为while_op的输出。  

WhileOp包含两个或三个region。 如果包含了三个region，说明已经求了一次反向，这种情况我们在下一节2.3.3中进行描述。这儿直接假设预计的WhileOp一定只包含了两个region。

1. 在cond_region的前方，插入一个init region，该init region只包含一个block， 承担两个功能：
   1. 创建一个stack。（一个create_stack算子）
   2. 将输入参数原样转发给condition region。（一个cf.yield算子, 跳转到condition region）
2. 在condition region和body region的输入输出中都新增stack变量。并将所有该region中定义的局部变量压栈该stack中。

```cpp
%i = pd.full(...)
%1 = pd.const(1)
%ten = pd.full(...)

%i_2 = pd.while(%i) 
   cond(%arg1) {
     // cond_region
     %cond = pd.less_than(%arg1, %ten)
     cf.cond_yield (%cond, %arg1)
   }
   body(%arg2){
     %i_3 = pd.add(%arg2, %1)
     cf.yield (%i_3)
   }
```

改造后的WhileOp：

```cpp
%i = pd.full(...)
%1 = pd.const(1)
%ten = pd.full(...)

%i_2， %stack = pd.while(%i) 
   init(%arg) {
     %stack = cf.create_stack()
     cf.yield(%arg, %stack)
   }
   cond(%arg1, %stack){
       %cond = pd.less_than(%arg1, %ten)
       cf.push_back(%stack, %arg1)
       cf.push_back(%stack, %cond)
       cf.cond_yield(%cond, %arg1, %stack)
    }
    body(%arg2, %stack){
       %i_3 = pd.add(%arg2, %1)
       cf.push_back(%stack, %arg2)
       cf.push_back(%stack, %i_3)
       pd.yield(%i_3, %stack)
    }
```

第二步：构造反向while_grad op。（反向while_grad op其实也是一个while_op，只是将其命名为while_grad，实现和while一致）

创建一个while_grad op, 它包含condition_region和body_region。 它的输入包含前向while_op的输出的所有变量的梯度以及while_op输出的容器栈。在condition_region和body_region中将前向中的压栈的所有局部变量都出栈。然后按照bakcward的正常逻辑，依次给后向region中添加前向region的反向算子。

```cpp
%i = pd.full(...)
%1 = pd.const(1)
%ten = pd.full(...)

%i_2， %stack = pd.while(%i) 
  init(%arg) {
       %stack = cf.create_stack()
       cf.yield(%arg, %stack)
   }
   cond (%arg1, %stack){
       %cond = pd.less_than(%arg1, %ten)
       cf.push_back(%stack, %arg1)
       cf.push_back(%stack, %cond)
       cf.cond_yield(%cond, %arg1, %stack)
    }
    body(%arg2, %stack){
       %i_3 = pd.add(%arg2, %1)
       cf.push_back(%stack, %arg2)
       cf.push_back(%stack, %i_3)
       pd.yield(%i_3, %stack)
    }
........
%i_grad = pd.while_grad(%i_2_grad, %stack) 
    cond(%arg1_grad, %stack) {
        %cond = cf.pop_back(%stack)
        %arg1 = cf.pop_back(%stack)
        // less_than的输入是：%arg1, %ten, 输出是%cond， 正常来说，
        // 反向的输出应该是%arg1_grad, %ten_grad.
        // 通过反向接口发现，less_than的反向算子是空的，也就是说less_than算子对%arg1_grad和%ten_grad没有贡献。
        // 此处直接跳过less_than的反向。
        %new_cond = cf.is_empty(%stack)
        cf.cond_yield(%new_cond, %arg1_grad)
      }
      body(%arg2_grad)){
           %arg2 = cf.pop_back(%stack)
           %i_3 = cf.pop_back(%stack)
   
           // add的输入是%arg2, %i1, 输出是%i_3
           // 所以反向的输出应该是%arg2_grad, %i1_grad 
           // 这儿需要注意的一点是，反向变量的定义域一定要和前向变量的定义域对偶。如果%i1_grad没有定义，那我们应该上溯到前向的对偶定义域中去定义一个初始为0的梯度向量，在这儿对其进行累加。
           %tmp_arg2_grad, %tmp_1_grad = pd.add_grad(%arg2, %1, %i_3, %arg_2_grad)
           //在构建pd.add_grad的时候，可以发现目前已经存在一个arg2_grad 和 1_grad。 因此我们需要将pd.add_grad的输出累加到之前的变量上。
           pd.inplace_add(%arg2_arg, %tmp_arg2_grad)
           pd.inplace_add(%1_grad, %temp_1_grad)
           pd.yield(%arg2_grad)
      }
```

第三步：优化剪枝，将反向算子中用不到的栈变量，移除它的压栈和出栈算子。

实际上，在前两步中，我们通过压栈出栈操作，给反向block中的子算子提供了前向算子的所有输入输出。 但其实反向算子不一定会用到前向的所有输出输入，或者有些算子的反向算子直接就是空的。这个时候就会出现很多用不到的局部变量。在这一步中对这些无用的变量和算子进行剪枝。

```cpp
%i = pd.full(...)
%1 = pd.const(1)
%ten = pd.full(...)

%i_2， %stack = pd.while(%i) 
    init(%arg){
        %stack = cf.create_stack()
        cf.yield(%arg, %stack)
     }
     cond(%arg1, %stack) {
        %cond = pd.less_than(%arg1, %ten)
        //cf.push_back(%stack, %arg1)
        //cf.push_back(%stack, %cond)
        cf.cond_yield(%cond, %arg1, %stack)
    }
    body(%arg2, %stack){
        %i_3 = pd.add(%arg2, %1)
        cf.push_back(%stack, %arg2)
        cf.push_back(%stack, %i_3)
        pd.yield(%i_3, %stack)
    }
........
^%i_grad = pd.while_grad(%i_2_grad, %stack) 
    cond(%arg1_grad, %stack){
       // 发现%cond, %arg1都没有被使用，所以这两行的pop_back可以删除。
       / 删除的时候，需要同步删除前向中的push_back算子。
       //%cond = cf.pop_back(%stack)
       //%arg1 = cf.pop_back(%stack)
       %new_cond = cf.is_empty(%stack)
       cf.cond_yield(%new_cond, %arg1_grad)
     }
     body(%arg2_grad){
        %arg2 = cf.pop_back(%stack)
        %i_3 = cf.pop_back(%stack)
        %tmp_arg2_grad, %tmp_1_grad = pd.add_grad(%arg2, %1, %i_3, %arg_2_grad)
        pd.inplace_add(%arg2_arg, %tmp_arg2_grad)
        pd.inplace_add(%1_grad, %temp_1_grad)
        pd.yield(%arg2_grad)
     }
```

#### 2.3.3 更近一步的考虑

我们在上一节(2.3.1)的最开始假设了backward pass中遇见的while op一定是包含了两个region。但实际上，在高阶微分场景，会对一个算子多次求反向，如果此时模型包含了while op, 在第二次及以后的backward pass中，遇见的while_op就会包含三个region。

目前paddle没有支持控制流高阶微分的需求。但在本节中我们可以提供一些简单的思路，方便后续有需求的时候，去进行相关实现。主要是为了论述新IR设计的完备性，证明后续对控制流高阶微分的实现只需要做增量开发。

当我们遇见while_op包含了三个region之时，那它一定已经包含了stack输出。（这是因为init region的存在就是为了创建stack输出，如果没有stack输出，那说明init region可以被移除，这就转变为了2.3.1中描述的两个region的情况）。而while_op包含了stack输出，那么一定存在while_grad_op使用了该stack。（如果不存在，说明该stack可以移除，因而init_region也可以被移除，再次转变为了2.3.1中描述的两个region的情况）。

一种简单的做法是，直接将根据stack变量获得的while_grad op直接复制一份，修改一下输入的反向变量即可。这样做唯一的问题在于 stack里面的变量被压栈了一次，却出栈了两次，而这个问题，我们只要将stack变量进行一次拷贝即可解决。

这种方式会有一个性能问题，那就是模型中存在两个只有输入和输出不同的while_grad op。 这个可以通过定义函数算子来进行解决。 将while_grad可以封装成一个函数。调用调用两次即可。

