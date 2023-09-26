# IR 底层基础类型系统设计文档
> 本文文档作者： @winter-wang


- [IR 底层基础类型系统设计文档](#ir-底层基础类型系统设计文档)
  - [一、IR 升级概要](#一ir-升级概要)
    - [1. 相关背景](#1-相关背景)
      - [1.1 分布式方向](#11-分布式方向)
      - [1.2 编译器方向](#12-编译器方向)
      - [1.3 推理方向](#13-推理方向)
      - [1.4 单机框架方向](#14-单机框架方向)
    - [2. 功能目标](#2-功能目标)
    - [3. 设计概要](#3-设计概要)
    - [4. Paddle 现状](#4-paddle-现状)
  - [二、设计思路与实现方案](#二设计思路与实现方案)
    - [2.1 Type 类](#21-type-类)
    - [2.2 Type 对象](#22-type-对象)
    - [2.3 扩展工具](#23-扩展工具)
    - [2.4 使用方式](#24-使用方式)
- [附录](#附录)
  - [一、竞品对照](#一竞品对照)
    - [1.1 MLIR 类型系统](#11-mlir-类型系统)
    - [1.2 TVM 类型系统](#12-tvm-类型系统)
    - [1.3 Mindspore 类型系统](#13-mindspore-类型系统)
    - [1.4 竞品类型系统比对](#14-竞品类型系统比对)

## 一、IR 升级概要

### 1. 相关背景

近些年来，越来越多竞品和研究者将编译器技术引入到深度学习的神经网络模型优化中，通过一种IR（`Intermediate Representation`，中间表示形式）来表示计算图，并在此基础上借助编译器的理念、技术和工具对神经网络进行自动优化和代码生成。常见的深度学习 IR 包括 MLIR，TVM relay，MindIR，XLA hlo，TorchScript等。

历史上，飞桨先后实现了两套 IR，一套是 `Program` ，一套是 `Graph` 。但是，这两套 IR 在早期设计的时候并没有考虑到发展到今天这么复杂的优化需求和使用场景，之前统计了各个方向(分布式、编译器、推理、单机框架)对IR的重点需求如下：

#### 1.1 分布式方向
分布式方向的重点需求是能够在算子或者变量中添加一些只在分布式中有意义的属性。

这个需求目前是通过在 `OpDesc` 和 `VarDesc` 里面添加的分布式专用的一些成员（`id`、`TensorDistAttr`、`OperatorDistAttr`)来实现的。另一方面，从短期来说，希望 IR 能够合理地支持 batch 维度和动态维度的识别，这些目前是通过 trick 的方式实现的（ `var_desc` 里面加分布式属性，分布式属性里面包含了一个 bool 列表，标识哪一个维度是动态 shape, 以及一个额外的整数，标识第几个维度是 batch 维度）。但是从长期来说，希望 IR 能够对动态 shape 语义有合理的抽象，能够支持动态shape的运算以及优化，这些是目前的IR无法通过 trick 方式支持的。

#### 1.2 编译器方向
主要需求是能够复用主框架现有 `Pass` 。要做到这一点，核心是需要统一编译器和主框架的IR数据结构（类型、算子、模型）。

#### 1.3 推理方向
主要是希望模型结构能够跟业界对齐（无环）。在此基础上，让 `Pass` 更加的完备、简洁。

要做到这一点，一是对现有模型数据结构升级改造，从数据结构上保证无环。同时，对算子定义进行完善升级，保证类似 `inplace` 算子的合理语义表示。在做到这些的基础上，进一步对 `Pass` 进行升级，保证完备性、稳定性等等。


#### 1.4 单机框架方向
主要是希望IR表达能力更加强大可扩展，

+ 能够支持`List`、`dict` 等复杂的类型数据结构，以及更近一步的类型嵌套表示。
+ 能够做到一些更复杂的 IR 分析：包括控制流分析、依赖分析等等。

因此，为了适应深度学习框架的发展和各方向的需求，飞桨的IR需要进行重新设计和升级。除此之外，也有一些大家公认的痛点，也希望能在本次 IR 升级中进行改善。

+ C++ 的 `desc` 数据结构( `ProgramDesc`、`BlockDesc`、`OpDesc`、`VarDesc`)里面内嵌了 `proto` 的 desc。二者经常需要同步，影响效率。
+ `Graph` 数据结构里面内嵌了 `ProgramDesc` , 二者经常不同步，容易出错。

### 2. 功能目标
根据前期的调研，我们计划将 IR 体系升级重点分为四个模块：类型系统、模型结构、高阶特性、Pass升级，并分别进行设计和评审

相应的功能目标如下：
1. 构建可扩展的 `Type` 类型系统
   + 能够支持List、Dict等复杂容器
   + 支持类型的嵌套递归定义

2. 构建符合 SSA 的 `Program` 结构表示
   + 模型结构无环
   + `Program` & `Graph` 合二为一

3. 支持模型的高阶特性（复杂算子和算子分层分类）
   + 支持通过 `Dialect` 对算子进行分层、分类以及扩展
   + 支持控制流算子、inplac e算子、编译器算子等复杂算子的合理定义
   + 合理支持多层控制流嵌套

4. 构建可复用的 `Pass` 设施
  a. 开发能够兼顾推理、分布式、编译器的 `Pass` 基础设施
  b. `Pass` 更加完备、稳定、易开发


### 3. 设计概要

1.  坚持SSA(静态单赋值)原则，模型等价于一个有向无环图。 `Operation` 为节点，`Value` 为边

2.  算子分层，分类。通过将算子和类型定义在不同的 `Dialect` 来实现算子和类型的分层分类
    * 顶层算子与现有的 Op 定义等价，输入&输入&属性类似。
    * 比如顶层 relu 算子包含一个输入，一个输出

3. 底层算子与算子库 Api 对齐
   * 比如底层 relu 算子接受两个输入，一个类型是 `Tensor`, 另一个类型是`Tensor*`. 没有输出。
   * inplace 语义通过不同的算子名进行区分。
   * 以 relu 为例，底层的 `relu_inplace` 只有一个 `Tensor*`类型的输入. 没有输出。

4. 通过`yaml + python`脚本的形式，生成算子的 C++ 定义
   * 算子的原始信息与目前类似，用yaml格式定义。
   * 以python脚本的形式生成算子定义。后续算子定义有改动，只需改动脚本。

IR 框架作为一个最底层的动态库，提供算子、类型等的基础设施和注册接口，由上层按需使用。同时，IR 库提供内置 `Dialect`, 注册一些最常见的类型和算子。

类型的本质是对内存的一种抽象。不同的类型，会有不同的内存布局和内存分配的策略。 而类型系统。则是程序设计语言中类型的集合，以及使用类型来规定程序行为的规则。典型现代语言的类型系统有四个主要组件：一组基础类型（或内置类型）、根据现存类型构建新类型的规则、用于确定两种类型是否等价或兼容的方法、用于为每个源语言表达式推断类型的规则。《编译器设计(第二版)》。

对于深度学习框架而言，在最初始阶段，通过枚举值对涉及到的类型进行简单的列举就可以满足需求。但是随着后续的发展，涉及到的类型种类越来越多，涉及到的优化也越来越细，对类型安全和类型推导的需求也越来越高，框架也开始向编译器方向靠拢。简单的枚举值类型定义已经逐步地无法深度学习框架的需求。

因此，本文对提出一种 Paddle 的类型系统升级方案。

### 4. Paddle 现状
在目前的 Paddle 主框架中，涉及到类型的地方有三处。

第一处是通过 `proto` 定义的 `var_desc` 里面的类型。下面代码段是目前 Paddle 的 `frame.proto` 中对 `VarType` 的定义。可以看出，对于大部分复杂类型，该 `proto` 都需要在字段中添加额外的 `optional Desc` 字段。
```cpp
message VarType {
  enum Type {
    // Pod Types
    BOOL = 0;
    INT16 = 1;
    INT32 = 2;
    INT64 = 3;
    FP16 = 4;
    FP32 = 5;
    FP64 = 6;
    // Tensor<size_t> is used in C++.
    SIZE_T = 19;
    UINT8 = 20;
    INT8 = 21;

    // Other types that may need additional descriptions
    LOD_TENSOR = 7;
    SELECTED_ROWS = 8;
    FEED_MINIBATCH = 9;
    FETCH_LIST = 10;
    STEP_SCOPES = 11;
    LOD_RANK_TABLE = 12;
    LOD_TENSOR_ARRAY = 13;
    PLACE_LIST = 14;
    READER = 15;
    // Any runtime decided variable type is raw
    // raw variables should manage their own allocations
    // in operators like nccl_op
    RAW = 17;
    TUPLE = 18;
  }

  required Type type = 1;

  message TensorDesc {
    // Should only be PODType. Is enforced in C++
    required Type data_type = 1;
    repeated int64 dims = 2; // [UNK, 640, 480] is saved as [-1, 640, 480]
  }
  optional TensorDesc selected_rows = 2;

  message LoDTensorDesc {
    required TensorDesc tensor = 1;
    optional int32 lod_level = 2 [ default = 0 ];
  }
  optional LoDTensorDesc lod_tensor = 3;

  message LoDTensorArrayDesc {
    required TensorDesc tensor = 1;
    optional int32 lod_level = 2 [ default = 0 ];
  }
  optional LoDTensorArrayDesc tensor_array = 4;

  message ReaderDesc { repeated LoDTensorDesc lod_tensor = 1; }
  optional ReaderDesc reader = 5;

  message Tuple { repeated Type element_type = 1; }
  optional Tuple tuple = 7;
}
```


第二处是通过 c++ 模版索引定义的 `var` 里面的类型（见 `var_type_trait.h`）。`var` 里面的 `type` 是对 `proto` 里面 `type` 的补充和新增，但本质上，还是通过整形常量进行标识。

```cpp
using VarTypeRegistry = detail::VarTypeRegistryImpl<
    phi::DenseTensor,
    phi::SelectedRows,
    phi::SparseCooTensor,
    phi::SparseCsrTensor,
    std::vector<Scope *>,
    LoDRankTable,
    Strings,
    LoDTensorArray,
    platform::PlaceList,
    ReaderHolder,
    String,
    Scope *,
    operators::reader::LoDTensorBlockingQueueHolder,
    FetchList,
    FeedList,
    operators::reader::OrderedMultiDeviceLoDTensorBlockingQueueHolder,
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    ncclUniqueId,
    platform::Communicator,
    platform::NCCLCommunicator,
#endif
    operators::CudnnRNNCache,
#endif
#if defined(PADDLE_WITH_ASCEND_CL)
    HcclRootInfo,
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
    BKCLUniqueId,
    platform::BKCLCommunicator,
#endif
#if defined(PADDLE_WITH_CNCL)
    cnclCliqueId,
#endif
    std::vector<std::unique_ptr<operators::CUDAGraphWithInOuts>>,
    int,
    float,
    Vocab,
    std::vector<int>,
    std::vector<float>>;
template <typename T>
struct VarTypeTrait {
  static_assert(VarTypeRegistry::IsRegistered<T>(), "Must be registered type");
  using Type = T;
  /**
   * Unique VarType Id generation.
   *
   * The auto-generated id should not be the same as any protobuf id defined in
   * framework.proto. Therefore, we generate id by adding the type pos and
   * maximum protobuf id (i.e., proto::VarType::TUPLE).
   *
   * However, we may need more protobuf id in the future.
   * To avoid changing this auto id generation algorithm frequently, we
   * generate id by adding the type pos and twice of maximum protobuf id (i.e.,
   * proto::VarType::TUPLE).
   */
  static constexpr int kId = VarTypeRegistry::TypePos<T>() +
                             static_cast<int>(proto::VarType::TUPLE) * 2;
};

```

第三处是 phi 算子库中的类型，还是通过枚举进行定义，见`common/data_type.h`

```cpp
// The enum value are consistent with jit/property.proto
enum class DataType {
  UNDEFINED = 0,

  BOOL,

  UINT8,  // BYte
  INT8,   // Char
  UINT16,
  INT16,
  UINT32,
  INT32,
  UINT64,
  INT64,

  FLOAT32,
  FLOAT64,

  COMPLEX64,
  COMPLEX128,

  // In Paddle 2.3, we add a new type of Tensor, StringTensor, which is designed
  // for string data management. We design the dtype of StringTensor, pstring.
  // In order to express a unique data dtype of StringTensor, we add
  // DataType::PSTRING.
  PSTRING,

  // IEEE754 half-precision floating-point format (16 bits wide).
  // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  FLOAT16,

  // Non-IEEE floating-point format based on IEEE754 single-precision
  // floating-point number truncated to 16 bits.
  // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
  BFLOAT16,

  NUM_DATA_TYPES,
  // See Note [ Why we need ALL in baisc kernel key member? ]
  ALL_DTYPE = UNDEFINED,
};
```

这三处类型的转换，也都是通过 `switch` 语句，进行枚举值的转换，如下是 `ProtoType` 与 `VarType` 之间的转换：

```cpp
inline proto::VarType::Type ToVarType(int type) {
  switch (type) {
    case proto::VarType::LOD_TENSOR:
    case proto::VarType::SELECTED_ROWS:
    case proto::VarType::SPARSE_COO:
    case proto::VarType::LOD_RANK_TABLE:
    case proto::VarType::LOD_TENSOR_ARRAY:
    case proto::VarType::FETCH_LIST:
    case proto::VarType::READER:
      return static_cast<proto::VarType::Type>(type);
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "ToVarType method Unsupported type %d.", type));
  }
}

template <typename Visitor>
inline void VisitVarType(const framework::Variable& var, Visitor visitor) {
  switch (var.Type()) {
    case proto::VarType::LOD_TENSOR:
      visitor(var.Get<phi::DenseTensor>());
      return;
    case proto::VarType::LOD_RANK_TABLE:
      visitor(var.Get<LoDRankTable>());
      return;
    case proto::VarType::LOD_TENSOR_ARRAY:
      visitor(var.Get<LoDTensorArray>());
      return;
    case proto::VarType::SELECTED_ROWS:
      visitor(var.Get<phi::SelectedRows>());
      return;
    case proto::VarType::SPARSE_COO:
      visitor(var.Get<phi::SparseCooTensor>());
      return;
    case proto::VarType::READER:
      visitor(var.Get<ReaderHolder>());
      return;
    case proto::VarType::FETCH_LIST:
      visitor(var.Get<FetchList>());
      return;
    default:
      PADDLE_THROW(platform::errors::Unavailable("Not supported visit type %s.",
                                                 ToTypeName(var.Type())));
  }
}
```

如下是 VarType 与 phi Type 之间的转换：

```cpp
// 转换通过先判断typeid，以C++作为桥梁进行判断
template <typename T>
inline bool IsType(const std::type_index& type) {
  return type == typeid(T);
}

#define PD_FOR_EACH_DATA_TYPE(_)      \
  _(bool, DataType::BOOL)             \
  _(int8_t, DataType::INT8)           \
  _(uint8_t, DataType::UINT8)         \
  _(int16_t, DataType::INT16)         \
  _(uint16_t, DataType::UINT16)       \
  _(int32_t, DataType::INT32)         \
  _(uint32_t, DataType::UINT32)       \
  _(int64_t, DataType::INT64)         \
  _(uint64_t, DataType::UINT64)       \
  _(bfloat16, DataType::BFLOAT16)     \
  _(float16, DataType::FLOAT16)       \
  _(float, DataType::FLOAT32)         \
  _(double, DataType::FLOAT64)        \
  _(complex64, DataType::COMPLEX64)   \
  _(complex128, DataType::COMPLEX128) \
  _(pstring, DataType::PSTRING)

```

总而言之，目前 Paddle类型：

+ 从表达能力上来说， 没有抽象出具体的 `Type` 数据结构，因此无法支持类型的组合嵌套。 支持不了` List`、`Dict`等模版类型。
+ 从扩展性上来说，一处类型的扩展，其它两处，以及对应的转换函数，都需要进行扩展修改。




## 二、设计思路与实现方案

C++ 认为万事万物皆可为对象（ `Object` ），对象上有其属性和行为。 具有相同性质的对象，可以抽象为类( `Class` )。类似的，我们可以抽象出 `Type` 类和 `Type` 对象的概念。

`Float32Type`、`DenseTensorType`、`FunctionType`等数据结构的定义代表着 `Type` 类，里面会定义该 `Type` 类对应的接口及成员。而描述计算图中一个输入&输出的类型，需要的是一个 `Type` 对象。一个 `Type` 对象一定属于某个 `Type` 类。

`Type` 类和 `Type` 对象是一对一或者一对多的关系。

有些 `Type` 类只对应唯一个 `Type` 对象，比如，`Float32Type`、`Uint8Type`等等。这类 `Type` 对象构造时不需要额外参数描述。
有些 `Type` 类则对应多个 `Type` 对象，比如`FunctionType`、`DenseTensorType`等等。这类 `Type` 对象构造时需要额外参数描述，比如 `DenseTensorType` 需要传递 `shape`、`lod_level`、`data_type`作为参数。 参数不同，构造出的 `Type` 对象也不同。

### 2.1 Type 类

如下是 `TypeID` 的简单定义。可以通过 `TypeID` 对象对每一种 `Type` 类进行唯一标识。对任意一种 `Type` 类，比如 `DenseTensorType` , 可以通过 `TypeID::get<DenseTensorType>()` 来获得其对应的 `TypeID` 。 `TypeID相` 等，表示 `Type` 类相同。

```cpp
class TypeID {
     struct Storage {};
 public:
    template<typename T> static TypeID get() {
        static Storage instance;
        return TypeID(&instance);
    }
  private:
    TypeID(const Storage *storage) : storage(storage) {}
    const Storage *storage_;
}
```
我们将不同 `Type` 类的属性和行为，抽象为 `AbstractType` 对象。目前来说，暂时可能就只有 `name` 和 `type_id` 。但是后续看 `Pass` 设计时的需求，可能会增加对特征( `Trait` )和接口( `Interface` )的抽象。特征用来做类型做横向划分，比如在 IR 变换中，某些时候，我们可能不会在意具体类型是什么，只在意该类型有没有某种性质（比如 `shape` ）。特征和接口会在下个阶段的算子定义设计中详细讲述，此处不再展开。

```cpp
class AbstractType {
    const TypeID type_id_; // 表明该对象对应的Type的TypeID.
    const char* name_;//表明该Type的name.
    // const Dialect& dialect_; //表明该对象对应的Type注册在哪个dialect里面。目前暂时不用。
}
```

在 `IRContext` 中，可以维持一个以 `TypeID` 为 key 的散列表。 记录框架支持的所有 `Type` 类,以及相应的性质。
```cpp
class IRContext {
    static IRContex* instance();

private:
    IRContext();
    // 在静态变量初始化的时候初始化一次，中途不再改变。
    std::unordered_map<TypeID, AbstractType*> register_types;

    /// 用来做类型的内存管理。
    StorageUniquer type_storage;
};
```

### 2.2 Type 对象

在 `Type` 的数据结构定义中，`impl` 指针指向由 `IRContext` 管理的，包含了该 `Type` 对象的所有知识的存储对象。两个相同的 `Type` 对象，它们的存储对象也一定是相同的，在 `IRContext` 中共用同一份内存。 基于此，`impl`指针的是否相等，与 `Type` 对象是否相同，完全等价。
```cpp
class Type {
  //其它接口省略
  TypeStorage* impl{nullptr};
};
```

下文讨论如何在 `IRContext` 中，对不同 `Type` 对象的存储对象进行管理。显然，同一个 `Type` 类对应的存储对象的类型也一定是相同的，但是不同的 `Type` 类对应的存储对象类型则不一定相同。理所应当地，我们定义数据结构 `TypeStorage` 作为存储对象的基类，并构建存储对象的派生体系。

`TypeStorage` 里面存储所有 `Type` 对象都需要的信息。就目前而言，首先需要存储的是它的 `Type` 类信息。因此，如下所示， `TypeStorage` 里面包含一个 `AbstractType` 指针，这个指针和 `IRContext` 中的 `register_types` 的哈希对象共享底层。

```cpp
//如果需要统一属性和类型，还可以在TypeStorage和AttribtueStorage的基础上，进一步抽象出公共的StorageBase。
class TypeStroraage{
   /// The abstract description for this type.
   AbstractType *abstract_type{nullptr};
}
```

对于无参 `Type` 类（单例类型），这种 `Type` 类对应唯一的 `Type` 对象。比如 `Float32Type` 、 `Float64Type` 等类型，用 `TypeStorage` 作为存储对象即可。
对于有参 `Type` 类，这种 `Type` 类一般对应多个 `Type` 对象。需要在存储对象中存储参数值，因此，必须对 `TypeStorage` 进行派生。
比如 `DenseTensorType` , 需要额外参数。因此，我们基于 `TypeStorage` 派生 `DenseTensorTypeStorage` 类型. 新增了 `data_type` 、 `dims` 和 `lod_level` 这三个成员变量存储参数。
此时， `DenseTensorType` 类对应的 `Type` 对象的 `impl` 指针，指向的是 `TypeStorage` 的真实类型是 `DenseTensorTypeStorage` 。

```cpp
class DenseTensorTypeStorage{
  public:
    // 定义hash_key， 用以在TypeContext中用哈希表进行存储
    using KeyTy = std::tuple<Type, std::vector<int64_t>, int32_t>;
    KeyTy getAsKey() const {
       return KeyTy(data_type, dims, lod_level);
    }
    // 哈希函数，用来存储
    static std::size_t hashFunc(const KeyTy &tblgenKey) {
       return hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
    }

    //判断是否相等,保证互斥性
    bool operator==(const KeyTy &tblgenKey) const {
      return (data_type == std::get<0>(tblgenKey)) && (dims == std::get<1>(tblgenKey)) && (lod_level == std::get<2>(tblgenKey));
    }

    //用来在TypeContext中构造类型对象。 后续可以给这个函数额外添加分配器参数。
    static DenseTensorTypeStorage *construct(const KeyTy &tblgenKey) {
        auto data_type = std::get<0>(tblgenKey);
        auto dims = std::get<1>(tblgenKey);
        auto lod_level = std::get<2>(tblgenKey);
        return new DenseTensorTypeStorage(data_type, dims, lod_level);
    }
    Type data_type;
    std::vector<int64_t> dims;
    int32_t lod_level;

};
```

### 2.3 扩展工具

所有 `Type` 类的派生，只派生接口，不派生成员。比如 `DenseTensorType` 、 `Float32Type` 等都是 `Type` 的派生类,但不会新增任何成员变量。
只派生接口，不派生成员可以保证，从子类到父类的类型转换，不会丢失任何信息。当定义一种具体类型（`ConcreteType`）时，需要考虑它的基类（`BaseType`）以及相应的内存类型（`StorageType`)。 我们通过提供一个 `TypeBase` 的模版工具类将这三者关联起来。

```cpp
// TypeBase用来将ConcreteType, BaseType, StorageType关联在一起
template <typename ConcreteT, typename BaseT, typename StorageT>
class TypeBase: public BaseT {
  public:
    using ImplType = StorageT;
    /// Utility for easy access to the storage instance.
    ImplType *getImpl() const { return static_cast<ImplType *>(this->impl); }
}
```

比如我们定义 Paddle 中对应 `Float32Type` 类型时,  `Float32Type` 不需要额外的内存，因此，直接用 `TypeStorage` 作为它的存储类型。

```cpp
class Float32Type : public TypeBase<Float32Type, Type, TypeStorage> {
  public:
  // 该函数会返回一个Float32Type对象，由于集成关系，可以直接转变为Type对象。
  // 该Type对象的impl指针是恒定且唯一的。
  static Float32Type get(TypeContext *context);
}

// 将该Type对应的TypeID注册到TypeContext中。
REGISTER_TYPE_ID(Float32Type)
```


再比如定义 Paddle 中对应 `DenseTensorType` , 注意到 `DenseTensorType` 中需要 `data_type` 、`dims`、`lod_level`。因此，需要定义相应的`DenseTensorTypeStorage`

```cpp
class DenseTensorType: public TypeBase<DenseTensorType, Type, DenseTensorTypeStorage> {
    static DenseTensorType get(TypeContext *context, Type data_type, std::vector<int64_t> dims, int32_t lod_level = 0) {
            ....在context中查找是否已经构造，如果没有，则进行构造。并以构造的 DenseTensorTypeStorage指针构造DenseTensorType。
    }
    std::vector<int64_t> getDims() const {
        return getImpl()->dims;
    }
    Type getDataType() const {
        return getImpl()->data_type;
    }
    int32_t getLoDlevel() const {
       return getImpl()->lod_level;
    }
}
REGISTER_TYPE_ID(DenseTensorType)
```

### 2.4 使用方式

利用静态 `get` 接口，初始化 `type` 对象。

该接口的使用，主要是在类型的构造阶段。包括 `pass` 中，直接设置类型。以及在 python api 中，对类型初始化的使用。
比如目前的 python 中组网时， `type=core.VarDesc.VarType.FP32` 会被替换为 `type = ir.Type.FP32`。而对应的 FP32 的底层 C++ 实现就会调用以上接口。
```cpp
// 初始化一个Float32Type对象
Type fp32_type = Float32Type::get(type_context);

// 初始化一个shape为[1，1], data_type为 Float32Type， lod为默认值0的LoDTensorType对象
Type lod_tensor_type = LoDTensorType::get(type_context, fp32_type, {1,1});
```

判断两个类型相等， 直接用相等运算符。这个主要用来做类型验证，比如 Op 定义中约束了算子的输入类型。可通过该接口判定类型是否满足约束。

```cpp
Type. type1, type2;
........
if(type1 == type2) {
......}

if(type1 != type2) {
.......}
```

判断是否是某种类型, 用 `isa` 接口(关于 `isa` 接口和下文的 `dyn_cast` 接口的实现，在后文的具体实现中会讲到)
```cpp
Type type1;
.....

// type的impl指针里面存储了AbstractType*指针，里面有TypeID对象，所以只需要判断该TypeID和 LoDTensorType的TypeID是否一致即可实现isa接口。
if(type1.isa<LoDTensorType>()) {
   .....
}
```

类型转化使用 `dyn_cast` 接口:
```cpp
// 初始化一个Float32Type对象
Type fp32_type = Float32Type::get(type_context);

Float32Type fp32_type_1 = fp32_type.dyn_cast<Float32TensorType >();

....
```


# 附录
## 一、竞品对照

### 1.1 MLIR 类型系统

在 MLIR 中，类型、属性、算子是 `Dialect` 的三大成员。用户通过ODS形式进行类型定制扩展，如下是 `Shape Dialect` 中 `Shape_Type` 的定义：

```cpp
class Shape_Type<string name, string typeMnemonic> : TypeDef<ShapeDialect, name> {
  let mnemonic = typeMnemonic;
}

def Shape_ShapeType : Shape_Type<"Shape", "shape"> {
  let description = [{
    `shape.shape` represents either an unranked shape, a ranked shape with
    possibly unknown dimensions or an invalid shape. The rank is of type
    `shape.size` and, if rank is known, the extent is a 1D tensor of type
    `shape.size`.

    Shape is printed:
    * `[*]` if it is an unranked shape
    * `[?, 2]` if a rank 2 tensor with one unknown dimension
    * `[3, 4]` is a rank 2 static tensor
    * `[]` is a scalar
    * `[1]` is a rank 1 tensor with 1 element
    * `[invalid]` for an invalid shape
  }];
}
```

通过相应的 `tablegen` 编译命令，该定义会生成相应的C++类型定义。并自动将该类型注册到 `ShapeDialect` 里面。

```cpp
class ShapeType : public ::mlir::Type::TypeBase<ShapeType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shape"};
  }
};
```

MLIR 在` BuiltinTypes.td` 里面通过 ODS 形式在内置 Dialect 定义了以下这些常见类型。
```cpp
class BFloat16Type;
class ComplexType;
class Float128Type;
class Float16Type;
class Float32Type;
class Float64Type;
class Float80Type;
class Float8E5M2Type;
class FunctionType;
class IndexType;
class IntegerType;
class MemRefType;
class NoneType;
class OpaqueType;
class RankedTensorType;
class TupleType;
class UnrankedMemRefType;
class UnrankedTensorType;
class VectorType;
```

MLIR 通过 `Type` 对象对描述类型。每种类型对应的存储对象是唯一的，它包含一个不变的标志符和一个可选的可变组件。`Type` 里面封装了一个指针，指针指向由 `MLIRContext` 管理的存储对象。因为， `Type` 可以通过值传递进行拷贝。

一些类型是 `“primitives”` ，意味着该类型不需要任何参数，比如 `Float32Type` 。参数类型有着额外的信息，用来和同 class 的其它类型区分，比如 `DenseTensor` 类型需要 `data_type` 参数。类型参数也是唯一不变标志符的一部分。可变组件是一个类型在创建以后，仍然可以修改的部分，但是这个修改不能影响到类型的唯一不变标志符。

类型通过 `'detail::TypeUniquer'` 类型进行构造和实现唯一。类型存储对象派生自 `TypeStorage` 包含以下三项：
1. 定义类型的 Dialect
2. 类型的参数
3. 可选的可变组件。

对于无参类型，MLIR 提供了默认的 `TypeStorage` 。参数类型的存储对象必须派生自 `TypeStorage` 并且符合以下要求：
1. 定义一个类型别名`KeyTy`，该类型的对象可以唯一标识类型对象。
   + key 类型必须可以通过传递给 `detail::TypeUniquer::get` 的参数值进行构造
   + 如果 key 类型没用被 `llvm::DenseMapInfo` 特例化，那么在存储了里面必须定义相关的哈希函数。
2. 必须提供判等函数，` bool operator==(const KeyTy &) const`
3. 提供一个静态构造方法。`DerivedStorage *construct(TypeStorageAllocator &, const KeyTy &key)`
4. 如果包含了可变组件，可变组件不可以成为 key 的一部分。

### 1.2 TVM 类型系统

与 MLIR 类似，TVM 中也定义了 `Type` 类型作为所有类型的基类。

MLIR 中用 `TypeStorage` 表示类型存储对象， `Type` 仅含了一个由 `MLIRContext` 管理的 `TypeStorage` ，来进行类型的区分。 TVM 中定义了 `Object `ObjectRef` 作为工具基类， `Object` 里面定义了引用计数、类型索引等成员， `ObjectRef` 里面仅包含了一个Object指针，通过该指针，实现类似指针指针的浅拷贝功能。

```cpp

class TVM_DLL Object {
 .......
 protected:
  // The fields of the base object cell.
  /*! \brief Type index(tag) that indicates the type of the object. */
  uint32_t type_index_{0};
  /*! \brief The internal reference counter */
  RefCounterType ref_counter_{0};
  /*!
   * \brief deleter of this object to enable customized allocation.
   * If the deleter is nullptr, no deletion will be performed.
   * The creator of the object must always set the deleter field properly.
   */
  FDeleter deleter_ = nullptr;
};

/*!
 * \brief A custom smart pointer for Object.
 * \tparam T the content data type.
 * \sa make_object
 */
template <typename T>
class ObjectPtr {
 .......
 private:
  /*! \brief internal pointer field */
  Object* data_{nullptr};
};

/*! \brief Base class of all object reference */
class ObjectRef {
 protected:
  /*! \brief Internal pointer that backs the reference. */
  ObjectPtr<Object> data_;
};
```

对类型系统而言， `TypeNode` 表示类型存储对象，派生自 `Object`。  `Type` 表示类型，派生自 `OjectRef`。二者通过`TVM_DEFINE_OBJECT_REF_METHODS` 宏进行关联。
```cpp
class TypeNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  static constexpr const char* _type_key = "Type";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 14;
  TVM_DECLARE_BASE_OBJECT_INFO(TypeNode, Object);
};

/*!
 * \brief Managed reference to TypeNode.
 * \sa TypeNode
 */
class Type : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Type, ObjectRef, TypeNode);
};
```

对于新类型的定义，先定义相应的 `TypeNode` 类型，然后定义 `Type` 类型。如下是 `TensorType` 的定义。 `显然，TensorTypeNode` 新增了 `shape` 和 `dtype` 作为成员变量。 `TensorType` 在派生自 `Type` 的基础上，通过 `TVM_DEFINE_OBJECT_REF_METHODS` 宏声明的它底层的Object指针的真正类型为 `TensorTypeNode` 。

```cpp
class BaseTensorTypeNode : public TypeNode {
 public:
  static constexpr const char* _type_key = "relay.BaseTensorType";
  static constexpr const uint32_t _type_child_slots = 1;
  TVM_DECLARE_BASE_OBJECT_INFO(BaseTensorTypeNode, TypeNode);
};

class TensorTypeNode : public BaseTensorTypeNode {
 public:
  /*!
   * \brief The shape of the tensor,
   *  represented by PrimExpr(tvm::Expr).
   */
  Array<PrimExpr> shape;
  /*! \brief The content data type */
  DataType dtype;
  .......
};

/*!
 * \brief Managed reference to TensorTypeNode.
 * \sa TensorTypeNode.
 */
class TensorType : public Type {
 public:
  /*!
   * \brief Constructor.
   * \param shape The shape of the tensor.
   * \param dtype The runtime dtype of the tensor's elements.
   */
  TVM_DLL TensorType(Array<PrimExpr> shape, DataType dtype);

  /*!
   * \brief Construct an scalar containing elements of dtype.
   * \param dtype The runtime dtype of the tensor's elements.
   * \return THe constructed type.
   */
  TVM_DLL static TensorType Scalar(DataType dtype);

  TVM_DEFINE_OBJECT_REF_METHODS(TensorType, Type, TensorTypeNode);
};
```

### 1.3 Mindspore 类型系统
MIndspore 首先通过枚举类型 `TypeId` 对所有类型进行了分类。 从大的分类，分为 `MetaType` 、 `ObjectType` 、 `Number Types` 、 `Monad Types` 、 `Sparse Types` 。继续细化出了如下所列的 65 种 `TypeID` 。

```cpp
enum TypeId : int {
  kTypeUnknown = 0,
  //
  // Meta types.
  //
  kMetaTypeBegin = kTypeUnknown,
  kMetaTypeType,  // Type
  kMetaTypeAnything,
  kMetaTypeObject,
  kMetaTypeTypeType,  // TypeType
  kMetaTypeProblem,
  kMetaTypeExternal,
  kMetaTypeNone,
  kMetaTypeNull,
  kMetaTypeEllipsis,
  kMetaTypeEnd,
  //
  // Object types
  //
  kObjectTypeBegin = kMetaTypeEnd,
  kObjectTypeNumber,
  kObjectTypeString,
  kObjectTypeList,
  kObjectTypeTuple,
  kObjectTypeSlice,
  kObjectTypeKeyword,
  kObjectTypeTensorType,
  kObjectTypeRowTensorType,
  kObjectTypeCOOTensorType,
  kObjectTypeUndeterminedType,
  kObjectTypeClass,
  kObjectTypeDictionary,
  kObjectTypeFunction,
  kObjectTypeJTagged,
  kObjectTypeSymbolicKeyType,
  kObjectTypeEnvType,
  kObjectTypeRefKey,
  kObjectTypeRef,
  kObjectTypeEnd,
  //
  // Number Types
  //
  kNumberTypeBegin = kObjectTypeEnd,
  kNumberTypeBool,
  kNumberTypeInt,
  kNumberTypeInt8,
  kNumberTypeInt16,
  kNumberTypeInt32,
  kNumberTypeInt64,
  kNumberTypeUInt,
  kNumberTypeUInt8,
  kNumberTypeUInt16,
  kNumberTypeUInt32,
  kNumberTypeUInt64,
  kNumberTypeFloat,
  kNumberTypeFloat16,
  kNumberTypeFloat32,
  kNumberTypeFloat64,
  kNumberTypeDouble,
  kNumberTypeComplex,
  kNumberTypeComplex64,
  kNumberTypeComplex128,
  kNumberTypeInt4,
  kNumberTypeGLUInt,
  kNumberTypeEnd,
  //
  // Monad Types
  //
  kMonadTypeBegin = kNumberTypeEnd,
  kObjectTypeMonad,
  kObjectTypeUMonad,
  kObjectTypeIOMonad,
  kMonadTypeEnd,
  //
  // Sparse Types
  //
  kSparseTypeBegin = kMonadTypeEnd,
  kObjectTypeCSRTensorType,
  kObjectTypeSparseTensorType,
  kObjectTypeMapTensorType,
  kSparseTypeEnd,
  // New types should placed at the end of enum,
  // in order to keep fit with the type of existing model on the lite side.
};
```

然后定义了 `Type` 数据结构作为所有类型的基类。 `Type` 包含两个成员，一个是 `TypeId` ，表示类型 id。 另一个 `is_generic_` ，设为 true ,表示该 `TypeId` 的默认类型。

```cpp
/// \brief Type defines an Value class for type.
class MS_CORE_API Type : public Value {
 public:
  /// \brief Default constructor for Type.
  Type() : meta_type_(kMetaTypeType), is_generic_(true) {}

  /// \brief Constructor for Type.
  ///
  /// \param[in] t Define TypeId for Type object.
  /// \param[in] is_generic Define whether the Type object is generic.
  explicit Type(TypeId t, bool is_generic = true) : meta_type_(t), is_generic_(is_generic) {}

  /// \brief Destructor of Type.
  ~Type() override = default;
  MS_DECLARE_PARENT(Type, Value)

  bool operator==(const Value &other) const override;

  /// \brief Show the meta type of the Type object.
  ///
  /// \return The meta type of the Type object.
  TypeId meta_type() const { return meta_type_; }

  /// \brief Show the type id of the Type object.
  ///
  /// \return The type id of the Type object.
  virtual TypeId type_id() const { return meta_type_; }

 private:
  TypeId meta_type_;
  bool is_generic_;
};
```

以 `Type` 为基类， `MindSpore` 构建了自己的类型派生体系。
常见的 `Float` 、 `Int` 等类型，对应的 `TypeID` 的 `kObjectTypeNumber` ，属于 `ObjectType` 中。
比如 `Float` 类型其对应的派生路径为： `Float --> Number --> Object --> Type` 。

```cpp
// Float
/// \brief Float defines a Number class whose type is float.
class MS_CORE_API Float : public Number {
 public:
  /// \brief Default constructor for Float.
  Float() : Number(kNumberTypeFloat, 0) {}

  /// \brief Constructor for Float.
  ///
  /// \param nbits Define the bit length of Float object.
  explicit Float(const int nbits);

  /// \brief Destructor of Float.
  ~Float() override {}
  MS_DECLARE_PARENT(Float, Number)

  TypeId generic_type_id() const override { return kNumberTypeFloat; }
  TypePtr DeepCopy() const override {
    if (nbits() == 0) {
      return std::make_shared<Float>();
    }
    return std::make_shared<Float>(nbits());
  }

  std::string ToString() const override { return GetTypeName("Float"); }
  std::string ToReprString() const override { return nbits() == 0 ? "float_" : GetTypeName("float"); }
  std::string DumpText() const override {
    return nbits() == 0 ? std::string("Float") : std::string("F") + std::to_string(nbits());
  }
};

```

再比如 `Tensor` 类型，其对应的 `TypeID` 的 `kObjectTypeTensorType` ，也属于 `ObjectType` 中。
对应的派生路径为： `TensorType --> Object --> Type` 。

```cpp
/// \brief TensorType defines interface for tensor data type.
class MS_CORE_API TensorType : public Object {
 public:
  /// \brief Default constructor for TensorType.
  TensorType() : Object(kObjectTypeTensorType, kObjectTypeUndeterminedType) {}

  /// \brief Constructor for TensorType.
  ///
  /// \param[in] ele The element of TensorType.
  explicit TensorType(const TypePtr &ele)
      : Object(kObjectTypeTensorType, kObjectTypeUndeterminedType, false), element_type_(ele) {}

  /// \brief Destructor of TensorType.
  ~TensorType() override = default;
  MS_DECLARE_PARENT(TensorType, Object)

  TypeId generic_type_id() const override { return kObjectTypeTensorType; }

  /// \brief Get the element of TensorType object.
  ///
  /// \return The element of TensorType object.
  const TypePtr element() const { return element_type_; }

  /// \brief Set the element of TensorType object.
  ///
  /// \param[in] element_type Define the element type to be set.
  void set_element(const TypePtr &element_type) { element_type_ = element_type; }

  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string ToReprString() const override;
  std::string DumpText() const override;
  bool operator==(const Type &other) const override;
  size_t hash() const override;

 private:
  TypePtr element_type_;
};

```
Mindspore 对类型的使用一般时通过智能指针进行使用的。

```cpp
using TypePtr = std::shared_ptr<Type>;
```

当需要使用类型作为成员的地方，一般会包含一个 `TypePtr` 。比如前面所列的 `TensorType` 的定义。 `TensorType` 里面会包含一个成员 `TypePrt element_type_` 来表示元素的类型;

当比较两个类型对象是否相等，因为 `operator== `是虚函数，因为会直接利用虚函数机制跳转到左边真实类型的成员函数中，在判等成员函数中，一般会先根据右边对象的成员 `TypeID` 是否匹配，将其转换为真实类型，进行相等判断。
在相等判断时，涉及到 `TypePtr` ，可以直接判断是否时同一个指针，如果指针不同，再去访问底层对象，判断是否相等。

Mindspore 针对常见的类型对象，定义了静态对象 `TypePtr` 。方便在不同的地方，可以直接通过指针来快速判定相等。

```cpp
#define GVAR_DEF(type, name, value) MS_CORE_API inline const type name = value;
GVAR_DEF(TypePtr, kBool, std::make_shared<Bool>());
GVAR_DEF(TypePtr, kInt8, std::make_shared<Int>(static_cast<int>(BitsNum::eBits8)));
GVAR_DEF(TypePtr, kInt16, std::make_shared<Int>(static_cast<int>(BitsNum::eBits16)));
GVAR_DEF(TypePtr, kInt32, std::make_shared<Int>(static_cast<int>(BitsNum::eBits32)));
GVAR_DEF(TypePtr, kInt64, std::make_shared<Int>(static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kUInt8, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits8)));
GVAR_DEF(TypePtr, kUInt16, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits16)));
GVAR_DEF(TypePtr, kUInt32, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits32)));
GVAR_DEF(TypePtr, kUInt64, std::make_shared<UInt>(static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kFloat16, std::make_shared<Float>(static_cast<int>(BitsNum::eBits16)));
GVAR_DEF(TypePtr, kFloat32, std::make_shared<Float>(static_cast<int>(BitsNum::eBits32)));
GVAR_DEF(TypePtr, kFloat64, std::make_shared<Float>(static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kInt, std::make_shared<Int>());
GVAR_DEF(TypePtr, kUInt, std::make_shared<UInt>());
GVAR_DEF(TypePtr, kFloat, std::make_shared<Float>());
GVAR_DEF(TypePtr, kNumber, std::make_shared<Number>());
GVAR_DEF(TypePtr, kComplex64, std::make_shared<Complex>(static_cast<int>(BitsNum::eBits64)));
GVAR_DEF(TypePtr, kComplex128, std::make_shared<Complex>(static_cast<int>(BitsNum::eBits128)));
```

### 1.4 竞品类型系统比对
首先，MLIR、TVM、MIndspore 都有自己的类型系统。都定义了 `Type` 数据结构来表示类型。 每种类型，都会在 `Type` 的基础上派生自己的类型。
从复杂类型对象的成员内存管理上来说，MLIR 是构建了 `MLIRContext` 数据结构，类型的内存管理由 `MLIRContex` 去负责。 `Type` 对象里面永远只包含一个唯一的指向内存对象的 `impl` 指针。判断相等时，只需要判断指针相等即可得到结论。
TVM 和 Mindspore 没有 `context` 的概念，因此，存储对象的管理是通过智能指针去管理的。TVM 使用的自己定义的类似智能指针的数据结构， MIndspore 使用的标准库的 `share_ptr`.

从复杂性上来说，MLIR 更加复杂，存储对象的派生体系， `context` 对存储对象的内存管理，都需要详细的设计实现。
但从效率上来说，MLIR 更加高效。MLIR 的 `Type` 对象只包含了一个 `impl` 指针。而 TVM 和 Mindspore 中的 `Type` 对象包含的是智能指针，显然，在空间和时间复杂性、以及空间和时间局部性方面，MLIR的方案都优于 TVM 和 `Mindspore` .

因此，我们采用类似 MLIR 的实现方式来设计实现 Paddle 的类型系统。
