# ProgramTranslator设计文档
> 本文文档作者： @吕永康

## 一、新旧 IR 在翻译时的显著区别

1. 类型系统，包括变量的类型和 Op 的函数类型
2. SSA 约束


## 二、旧 IR 在翻译时的其他问题

1. OpDesc 没有严格约束
    - a. 对于可选输入 / 输出， OpDesc 的 inputs 或者 outputs 可能将其包含，也可能不包含
    - b. 前向 Op 会有 OpProto 和 OpProtoChecker ，反向则没有，进而反向完全没有 checker
 
        i. 目前新 IR 的反向也不做 verify ，因为反向会根据 no_grad_set 做裁剪，因而输出个数会不一致

    - c. 对于 VectorType 的判断比较 trick ，并且同时存在 TensorArray 作为 VectorType ，个人认为比较
混乱
2. Op 动静定义不一致，历史问题，处理起来很麻烦


## 三、 ProgramTranslator 的结构

### 3.1 对外API

`paddle/fluid/ir_adaptor/translator/translate.h`

```c++
std::unique_ptr<Program> TranslateLegacyProgramToProgram(
    const LegacyProgramDesc& legacy_program) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<dialect::PaddleDialect>();
  auto program = std::make_unique<Program>(ctx);
  translator::ProgramTranslator program_translator(&legacy_program,
                                                   program.get());
  program_translator.Translate();

  return program;
}
```
如果需要翻译的中间结果怎么办？

理论上可以调用 ProgramTranslator ，但是真的有需要中间结果的场景吗？翻译后的结果应有完备的信息，执行器也应该具备只拥有 ir::Program 就能执行的能力。

### 3.2 ProgramTranslator

设计目标：负责 Block/Region 级别的变换， Op 级别的变换由 OpTranslator 负责。

当前解决的主要问题： Parameter 相关设置，以确保 program 的 SSA 性质

#### 3.2.1 Parameter 与 Get/SetParameterOp

我们在每个模型中，维护一个哈希表： `hash_map<StrAttribute, Variable*>` 来表示该模型对应的权重
值。 用户可以通过接口在此哈希表中插入、删除、访问、修改相应的 Variable 。

Variable 类似于 paddle 中的 Varibale ， 它包含：

1. `Type type_` ：表明 Variable 的类型；
2. `void* data_`: 指向具体的数据；
3. `bool is_mutable_`: 表明数据是否会在模型的执行当中被改变；
4. 数据的大小、对齐等等其他性质。


对于模型中的对权重的使用，我们定义 GetParameterOp 、 SetParameterOp 。分别从相应模型的哈希表中 , 获取、设置相应的权重内容。

其中， GetParameterOp 接受一个字符串作为属性，意义是从该模型的哈希表中加载该字符串对应的属性，并将其转换为输出。

SetParameterOp 接受一个字符串作为属性，一个张量类型的输入，没有输出。 表示用该属性和张量组成的键值对更新模型权重哈希表。

相应的，在模型组网的时候，我们需要在 startup program 中插入相应的 SetParameterOp, 而在 main program 中插入相应的 GetParameterOp 。 我们通过将 starpup program 执行完得到的参数哈希表移动给 main program ，来实现两个 program 的通信。

对于模型的任何参数（比如学习率等），只要我们想要在权重文件中存储该值，那就应该在相应的位置插入 Get/SetParameterOp 。

后期如果有必要，我们也可以定义 Get/SetCombineParameterOp 等，一次性加载 & 存储大批量权重。

当模型导出的时候，会将模型中的哈希表存储为权重文件，算子列表存储为模型文件。


当从文件中初始化模型的时候，会将所有的 Variable 的 isMutable 设为 False ， 然后遍历模型中的所有算子，遇见 SetParameterOp 的时候，就将相应的 isMutable 设为 True 。 对于 Pass 而言，对于权重，可以通过访问相应的 isMutable 来判定是否可以将该 Parameter 当作常量进行变换。

Program 中 parameter 的存储方式如下：

`paddle/ir/core/parameter.h`

```c++
namespace ir {
///
/// \brief Parameter represents the weight in the calculation graph.
///
class IR_API Parameter {
 public:

  ~Parameter() { free(data_); }

  Type type() const { return type_; }

  void* data() const { return data_; }

  bool is_mutable() const { return is_mutable_; }

  void set_mutable() { is_mutable_ = true; }

 private:
  void* data_; // 注意这个字段，代表它持有一块内存地址

  ///
  /// \brief Number of bytes held in data_.
  ///
  size_t size_;

  Type type_;

  bool is_mutable_ = false;
};

}  // namespace ir
```

GetParameterOp 的定义与作用：
```
(%0) = "builtin.get_parameter" () {parameter_name:conv2d_0.w_0} : () -> tensor<64x3x7x7xf32> 
......
(%431) = "pd.conv2d" (%430, %0)
{data_format : NCHW, 
 groups : 1, 
 padding_algorithm : EXPLICIT, 
 dilations : array[1, 1],
 paddings : array[3, 3],
 strides : array[2, 2]} : (tensor<-1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<-1x64x112x112xf32>
```

可以看到，执行 GetParameterOp 的作用是读取 Program 的某个权重，作为后续 Op 的输入，一般出现在 main_program 翻译后的结果中。 main_program 中的可训练参数原本作为 Block 的 Var 出现，但是并没有 Op 定义，因此需要通过插入 GetParameterOp 来满足新 IR 下 SSA 的要求。

SetParameterOp 的定义与作用：

```
(%2) = "pd.full" () 
{place: Place(undefined: 0), 
 dtype:float32,
 value:1,
 shape:IntArray[64]} : () -> tensor<64xf32>
 
"builtin.set_parameter" (%2) {parameter_name:batch_norm_0.w_0} : (tensor<64xf32>) ->
```

执行 SetParameterOp 的作用相反，它得到某个 Op 的输入并将其保存到权重中去 ( 注意到 Program 中 parameter 会持有一块内存地址 ) 。一般出现在 startup_program 的翻译结果中，在执行完权重初始化过程中，需要将其保存到 program 中以供下一次执行时使用。


#### 3.2.2 插入 GetParameterOp 的逻辑

主要是对于可训练参数 ProgramDesc 内部未定义的 Var 插入

```c++
void ProgramTranslator::GetParameterForSingleBlock(const BlockDesc& block) {
  for (auto& var : block.AllVars()) {
    if (!var->Persistable()) continue;
    if (param_map_.count(var->Name()) != 0) continue;
    if (no_cast_var_names.count(var->Name()) != 0) continue;

    parameter_name_mappings_[var->Name()] = var;
  }

  std::unordered_set<std::string> inner_defining_variables;

  for (auto op_desc : block.AllOps()) {
    for (const auto& n : op_desc->Inputs()) {
      const auto& input_var_names = n.second;
      for (const auto& var_name : input_var_names) {
        if (no_cast_var_names.count(var_name) != 0) continue;
        VarDesc* var_desc = nullptr;

        bool is_parameter = (parameter_name_mappings_.find(var_name) !=
                             parameter_name_mappings_.end());
        is_parameter &= (parameter_visited_.count(var_name) == 0);
        if (is_parameter) {
          var_desc = parameter_name_mappings_[var_name];
        }
        bool is_unseen_variable =
            (inner_defining_variables.count(var_name) == 0);
        if (is_unseen_variable) {
          var_desc = block.FindVarRecursive(var_name);
        }

        bool need_get_parameter_op = is_parameter && is_unseen_variable;
        if (need_get_parameter_op) {
          ir::Operation* op = InsertGetParamaterOp(ctx_, var_desc);
          program_->block()->push_back(op);
          param_map_[var_name] = VariableDefiningInfo(op->result(0));
          VLOG(10) << "[op translated][get parameter]" << var_name;

          program_->SetParameter(var_name, nullptr);
          parameter_visited_.insert(var_name);
          inner_defining_variables.insert(var_name);
        }
      }
    }
    for (const auto& n : op_desc->Outputs()) {
      const auto& output_var_names = n.second;
      for (const auto& var_name : output_var_names) {
        inner_defining_variables.insert(var_name);
      }
    }
  }
}
```

#### 3.2.3 插入 SetParameterOp 的逻辑

针对没有插入过 get_parameter 并且是可训练参数的 Var

```c++
void ProgramTranslator::SetParameterFromSingleBlock(const BlockDesc& block) {
  const auto& ops = block.AllOps();
  for (auto op_desc = ops.rbegin(); op_desc != ops.rend(); op_desc++) {
    if ((*op_desc)->Type() == "data") {
      continue;
    }

    const auto& input_var_names = (*op_desc)->InputArgumentNames();
    std::unordered_set<std::string> set_input_var_names(input_var_names.begin(),
                                                        input_var_names.end());

    for (const auto& n : (*op_desc)->Outputs()) {
      const auto& output_var_names = n.second;
      for (const auto& var_name : output_var_names) {
        bool need_set_parameter_op = (parameter_name_mappings_.find(var_name) !=
                                      parameter_name_mappings_.end());
        need_set_parameter_op &= (parameter_visited_.count(var_name) == 0);
        need_set_parameter_op &= (param_map_.count(var_name) != 0);
        need_set_parameter_op &= (!set_input_var_names.count(var_name));
        if (need_set_parameter_op) {
          ir::OpResult defining_op_result = param_map_[var_name].value;
          if (!defining_op_result) {
            continue;
          }

          if (param_map_[var_name].generated_by_vector) {
            InsertSliceOperationForTarget(
                ctx_, &param_map_, program_, param_map_[var_name], var_name);
            defining_op_result = param_map_.at(var_name).value;
          }

          ir::Operation* op = InsertSetParamaterOp(
              ctx_, defining_op_result, parameter_name_mappings_[var_name]);

          ir::Block* block = program_->block();
          ir::Block::iterator insert_pos = std::find(
              block->begin(), block->end(), defining_op_result.owner());

          IR_ENFORCE(
              insert_pos != block->end(),
              "Parameter %s must have corresponding its defining operation",
              var_name);
          insert_pos++;

          block->insert(insert_pos, op);
          VLOG(10) << "[op translated][set parameter]" << var_name;

          program_->SetParameter(var_name, nullptr);
          parameter_visited_.insert(var_name);
        }
      }
    }
  }
}

```


#### 3.2.4 整体逻辑

```c++
void ProgramTranslator::Translate() {
  PADDLE_ENFORCE_EQ(
      legacy_program_->Size(),
      1u,
      platform::errors::PreconditionNotMet(
          "Not support multi block ProgramDesc translated, now has %d blocks",
          legacy_program_->Size()));
  for (size_t block_idx = 0; block_idx < legacy_program_->Size(); block_idx++) {
    const BlockDesc& block = legacy_program_->Block(block_idx);
    GetParameterForSingleBlock(block);
  }

  for (size_t block_idx = 0; block_idx < legacy_program_->Size(); block_idx++) {
    const BlockDesc& block = legacy_program_->Block(block_idx);
    InsertOperationToSingleBlock(block);
  }

  for (size_t block_idx = 0; block_idx < legacy_program_->Size(); block_idx++) {
    const BlockDesc& block = legacy_program_->Block(block_idx);
    SetParameterFromSingleBlock(block);
  }

  for (size_t block_idx = 0; block_idx < legacy_program_->Size(); block_idx++) {
    const BlockDesc& block = legacy_program_->Block(block_idx);
    SetStopGradientAttributeForAllValue(block);
  }

  for (size_t block_idx = 0; block_idx < legacy_program_->Size(); block_idx++) {
    const BlockDesc& block = legacy_program_->Block(block_idx);
    SetIsPersisableAttributeForAllValue(block);
  }
}

```

### 3.3 OpTranslator

设计目标：确保将理论上可以转换的 OpDesc 转换为新 IR 下的 Operation

#### 3.3.1 调用方式与接口


调用方式

```c++
void ProgramTranslator::InsertOperationToSingleBlock(const BlockDesc& block) {
  auto& op_translator = OpTranslator::instance();
  for (auto op : block.AllOps()) {
    OpTranslateFn& fn = op_translator[op->Type()];
    if (op->Type() == "shadow_output") {
      if (!param_map_.count(op->Input("x")[0])) {
        continue;
      }
    }
    ir::Operation* operation = fn(ctx_, &param_map_, *op, program_);
    VLOG(10) << "[op translated][special]" << operation;
  }
}
```

```c++
class OpTranslator {
 public:
  using ResultIdx = size_t;
  using OpDesc = paddle::framework::OpDesc;
  using BlockDesc = paddle::framework::BlockDesc;
  using VarDesc = paddle::framework::VarDesc;
  using OpTranslateFn = std::function<ir::Operation*(
      ir::IrContext*, TranslationContext*, const OpDesc&, ir::Program*)>;

 private:
  OpTranslator();  // Disallow instantiation outside of the class.
  std::unordered_map<std::string, OpTranslateFn> special_handlers;
  OpTranslateFn general_handler;

 public:
  OpTranslator(const OpTranslator&) = delete;
  OpTranslator& operator=(const OpTranslator&) = delete;
  OpTranslator(OpTranslator&&) = delete;
  OpTranslator& operator=(OpTranslator&&) = delete;

  static auto& instance() {
    static OpTranslator OpTranslator;
    return OpTranslator;
  }

  OpTranslateFn& operator[](const std::string& op_type) {
    if (special_handlers.count(op_type) == 0) {
      return general_handler;
    } else {
      return special_handlers[op_type];
    }
  }
};

using OpTranslateFn = OpTranslator::OpTranslateFn;
```

可以看到，当翻译一个 Op 时，首先通过 OpTranslator 得到翻译该类的方式 (一个函数指针) , 接着通过函数指针将 OpDesc 翻译到 Operation 。这样做的目的是尽最大可能完成 OpDesc 到 Operation 的转换，在这一体系下，如果不能通过某种通用的规则进行转换，那么就尝试通过对某类 Op 定义特殊的转换规则进行翻译。

现有的特殊转换规则

```c++
OpTranslator::OpTranslator() {
  general_handler = OpTranscriber();
  special_handlers["add_n"] = AddNOpTranscriber();
  special_handlers["assign_value"] = AssignValueOpTranscriber();
  special_handlers["cast"] = CastOpTranscriber();
  special_handlers["feed"] = FeedOpTranscriber();
  special_handlers["data"] = DataOpTranscriber();
  special_handlers["fetch_v2"] = FetchOpTranscriber();
  special_handlers["grad_add"] = GradAddOpTranscriber();
  special_handlers["increment"] = IncrementOpTranscriber();
  special_handlers["lookup_table_v2"] = EmbeddingOpTranscriber();
  special_handlers["lookup_table_v2_grad"] = EmbeddingGradOpTranscriber();
  special_handlers["one_hot_v2"] = OneHotTranscriber();
  special_handlers["reduce_all"] = ReduceOpTranscriber();
  special_handlers["reduce_any"] = ReduceOpTranscriber();
  special_handlers["rnn"] = RnnOpTranscriber();
  special_handlers["shadow_output"] = ShadowOutputOpTranscriber();
  special_handlers["split"] = SplitOpTranscriber();
  special_handlers["sum"] = AddNOpTranscriber();
  special_handlers["tril_triu"] = TrilAndTriuOpTranscriber();

  // special handler for elementwise ops with axis != -1
  // note(lyk): maybe we should do this by a pass, which seems more reasonable
  special_handlers["elementwise_add"] = ElementwiseTranscriber();
  special_handlers["elementwise_sub"] = ElementwiseTranscriber();
  special_handlers["elementwise_mul"] = ElementwiseTranscriber();
  special_handlers["elementwise_div"] = ElementwiseTranscriber();
  special_handlers["elementwise_max"] = ElementwiseTranscriber();
  special_handlers["elementwise_min"] = ElementwiseTranscriber();
  special_handlers["elementwise_mod"] = ElementwiseTranscriber();
  special_handlers["elementwise_floordiv"] = ElementwiseTranscriber();
  special_handlers["elementwise_add_grad"] = ElementwiseGradTranscriber();
  special_handlers["elementwise_sub_grad"] = ElementwiseGradTranscriber();
  special_handlers["elementwise_mul_grad"] = ElementwiseGradTranscriber();
  special_handlers["elementwise_div_grad"] = ElementwiseGradTranscriber();
  special_handlers["elementwise_max_grad"] = ElementwiseGradTranscriber();
  special_handlers["elementwise_min_grad"] = ElementwiseGradTranscriber();
  special_handlers["elementwise_mod_grad"] = ElementwiseGradTranscriber();
  special_handlers["elementwise_floordiv_grad"] = ElementwiseGradTranscriber();
}
```

#### 3.3.2 OpTranscriber

```c++
struct OpTranscriber {
 public:
  virtual ~OpTranscriber() = default;

 public:
  // OpTranscriber 是一个Functor ，重载了 Operator() ，等价于一个函数
  // 这个函数的签名和 OpTranslateFn 一致
  virtual ir::Operation* operator()(ir::IrContext* ctx,
                                    TranslationContext* param_map, // 用于记录 Value 和 Var 间的映射关系
                                    const OpDesc& op_desc,
                                    ir::Program* program);
```

#### 3.3.3 Operation 的组成元素

```c++
  static Operation *Create(const std::vector<ir::OpResult> &inputs,
                           const AttributeMap &attributes,
                           const std::vector<ir::Type> &output_types,
                           ir::OpInfo op_info,
                           size_t num_regions = 0);
  static Operation *Create(OperationArgument &&op_argument);
```
一个 Operation( 在不考虑控制流的情形下 ) ，包含四个部分：

1. inputs ，从 OpDesc 的 Inputs 得到
2. attributes ，从 OpDesc 的 Attributes 得到
3. output_types ，从 OpDesc 的 Outputs 得到
4. op_info ，所有的 ir::OpInfo 都注册在 ir::Context 中，可以通过 opname 获取


#### 3.3.4 如何翻译一个 OpDesc ？
```c++
ir::Operation* OpTranscriber::operator()(ir::IrContext* ctx,
                                         TranslationContext* param_map,
                                         const OpDesc& op_desc,
                                         ir::Program* program) {
  auto op_info = this->LoopkUpOpInfo(ctx, op_desc);
  auto* op_info_concept =
      op_info.GetInterfaceImpl<dialect::OpYamlInfoInterface>();

  OpInputInfoList input_infos;
  OpAttributeInfoList attr_infos;
  OpOutputInfoList output_infos;
  std::tie(input_infos, attr_infos, output_infos, std::ignore) =
      op_info_concept->get_op_info_();

  this->InsertSliceOperationForInput(
      ctx, param_map, op_desc, input_infos, program);

  auto op_inputs = this->GenerateOperationInput(
      ctx, param_map, op_desc, op_info.name(), input_infos, program);

  OpOutputMapping arg_to_idx;
  OpOutputTypeList op_output_types;
  std::tie(op_output_types, arg_to_idx) =
      this->GenerateOperationOutput(ctx, op_desc, output_infos);

  auto attribute_map =
      this->TranslateOpAttribute(ctx, op_info.name(), attr_infos, op_desc);
  VLOG(4) << "[general op][" << op_desc.Type() << "] preparation end.";

  ir::Operation* operation =
      ir::Operation::Create(op_inputs, attribute_map, op_output_types, op_info);
  VLOG(4) << "[general op][" << op_desc.Type() << "] opearation creation end.";
  program->block()->push_back(operation);

  VLOG(4) << "[general op][" << op_desc.Type() << "] opearation insertion end.";
  this->RecordOpResultMapping(ctx, param_map, op_desc, operation, arg_to_idx);

  return operation;
}
```

#### 3.3.5 如何为某个 Op 特殊定义转换规则？
```c++
struct CastOpTranscriber : public OpTranscriber {
  ir::AttributeMap TranslateOpAttribute(
      ir::IrContext*,
      const std::string& normalized_op_name,
      const OpAttributeInfoList& op_attr_infos,
      const OpDesc& op_desc) override {
    auto& attribute_translator = AttributeTranslator::instance();
    ir::AttributeMap attribute_map = {};
    const OpAttributeInfo info = op_attr_infos[0];

    std::string legacy_attr_name("out_dtype");

    paddle::framework::Attribute legacy_attr;
    if (op_desc.HasAttr(legacy_attr_name)) {
      legacy_attr = op_desc.GetAttr(legacy_attr_name);
    }
    VLOG(10) << "attribute in " << op_desc.Type()
             << " name: " << legacy_attr_name << " " << legacy_attr.index();
    ir::Attribute new_attr = attribute_translator(info.type_name, legacy_attr);
    attribute_map[info.name] = new_attr;

    return attribute_map;
  }
};
```

虽然我们需要为某些 Op 定义特殊的转换规则，但是并不是所有的转换逻辑都是特殊的，比如说，有些时候我们只需要针对属性进行特殊处理，那么就没有再把其他部分的转换规则重复一遍。

因此我们通过继承与成员函数的重载，允许只自定义转换流程中某一部分的转换规则。可以这样理解，一个 Op 的转换函数 OpTranslateFn 实际上是由若干个函数指针组成的，如果需要为某个 Op 定义特殊规则，一般只需要更改其中的一个或几个函数指针即可。

目前，我们支持重载的模块如下：


```c++
  virtual ir::OpInfo LoopkUpOpInfo(ir::IrContext* ctx, const OpDesc& op_desc);
  virtual std::vector<ir::OpResult> GenerateOperationInput(
      ir::IrContext* ctx,
      TranslationContext* param_map,
      const OpDesc& op_desc,
      const std::string& normalized_op_name,
      const OpInputInfoList& input_infos,
      ir::Program* program);
  virtual std::tuple<OpOutputTypeList, OpOutputMapping> GenerateOperationOutput(
      ir::IrContext* ctx,
      const OpDesc& op_desc,
      const OpOutputInfoList& output_infos);
  virtual void HandleNonexistentAttribute(ir::IrContext*,
                                          ir::AttributeMap* attribute_map,
                                          const OpAttributeInfo& info) {
    auto& attribute_translator = AttributeTranslator::instance();
    (*attribute_map)[info.name] =
        attribute_translator(info.type_name, paddle::framework::Attribute());
  }
  virtual ir::AttributeMap TranslateOpAttribute(
      ir::IrContext* ctx,
      const std::string& normalized_op_name,
      const OpAttributeInfoList& op_attr_infos,
      const OpDesc& op_desc);

  virtual void RecordOpResultMapping(ir::IrContext* ctx,
                                     TranslationContext* param_map,
                                     const OpDesc& op_desc,
                                     ir::Operation* operation,
                                     const OpOutputMapping& arg_to_idx);

 public:
  virtual InputHandlerFn GetSpecialInputHandlers(
      const std::string& input_name) {
    return nullptr;
  }
```

问题 3.3.1 为什么基于 OpDesc 翻译而非 OpProto ？

问题 3.3.2 对可变 attribute 的处理 

1. 在新 IR 下，所有可变 attribute 都应该是输入 
2. OpDesc 中的可变 attribute 是如何表示的？存在新旧两种方式，可以参考可变 attribute 支持 
3. 对于新的方式，如果 attribute 是 VarDesc ，那么按照输入处理即可，否则需要插入 full op 作为新的 Input
4. 对于旧的方式，也就是通过额外输入表示的，检索对应的输入是否存在，并进行相应处理


问题 3.3.3 如何支持 Vector<Type>

1. 新增 combineOp 和 sliceOp ， combineOp 将多个 Type 合成为 Vector<Type> ， sliceOp 从 Vector<Type> 中得到 Type
2. 在转换时，首先判断某个 Var 是不是从 Vector<Tensor> 中产生的，如果是的话，需要插入 sliceOp 获取该 Var 对应的 Value
3. 如果某个输入是 Vector<Tensor> ，那么它一般对应多个 Var ，找到这些 Var 对应的 value，然后通过 combineOp 获得该输入对应的 Value

```
{relu_33.tmp_0@GRAD@RENAME@block0@0} = f()
{relu_33.tmp_0@GRAD@RENAME@block0@1} = f()
{Out=['relu_33.tmp_0@GRAD']} =
sum(inputs={X=['relu_33.tmp_0@GRAD@RENAME@block0@0',
'relu_33.tmp_0@GRAD@RENAME@block0@1']})

(%938) = f() -> tensor<-1x2048x7x7xf32>
(%954) = f() -> tensor<-1x2048x7x7xf32>

(%956) = "builtin.combine" (%938, %954) {} : (tensor<-1x2048x7x7xf32>,
tensor<-1x2048x7x7xf32>) ->
vec[tensor<-1x2048x7x7xf32>,tensor<-1x2048x7x7xf32>]

(%957) = "pd.add_n" (%956) {} :
(vec[tensor<-1x2048x7x7xf32>, tensor<-1x2048x7x7xf32>]) ->
tensor<-1x2048x7x7xf32>
```


### 3.4 OpCompatInfo

OpCompatInfo 用于处理动静定义不一致的问题，通过扫描 op_compat.yaml 生成，其接口如下：

```c++
  std::string operator[](const std::string& op_type) {
    if (op_name_mappings.find(op_type) == op_name_mappings.end()) {
      return op_type;
    }
    return op_name_mappings.at(op_type);
  }

  bool HasMutableAttribute(const std::string& op_type) {
    return (op_mutable_attributes.find(op_type) != op_mutable_attributes.end());
  }

  const std::unordered_set<std::string>* GetMutableAttributes(
      const std::string& op_type) {
    if (!HasMutableAttribute(op_type)) return nullptr;
    return &op_mutable_attributes.at(op_type);
  }

  const MutableAttributeInfo& GetMutableAttributeInfos(
      const std::string& op_type, const std::string& arg_name) {
    return op_mutable_attribute_infos.at(op_type).at(arg_name);
  }

  std::string GetLegacyArgName(const std::string& op_type,
                               const std::string& arg_name);

  std::string GetLegacyAttrName(const std::string& op_type,
                                const std::string& arg_name);
```

#### 3.5.1 调用方式与接口

```c++
  // `DropoutState` is a tensor
  VarDesc* dropout_state =
      op_desc.Block()->FindVarRecursive(legacy_output_vars[0]);
  if (dropout_state == nullptr) {
    IR_THROW("Unexpected: Rnn Op should have a non-empty DropoutState");
  }
  auto& type_translator = TypeTranslator::instance();
  ir::Type translated_var_type =
      type_translator[dropout_state->GetType()](ctx, *dropout_state);
```


```c++
class TypeTranslator {
 public:
  using VarType = paddle::framework::proto::VarType;

 private:
  TypeTranslator();  // Disallow instantiation outside of the class.
  std::unordered_map<VarType::Type, TypeTranslateFn> handlers;

 public:
  TypeTranslator(const TypeTranslator&) = delete;
  TypeTranslator& operator=(const TypeTranslator&) = delete;
  TypeTranslator(TypeTranslator&&) = delete;
  TypeTranslator& operator=(TypeTranslator&&) = delete;

  static auto& instance() {
    static TypeTranslator TypeTranslator;
    return TypeTranslator;
  }

  TypeTranslateFn& operator[](VarType::Type type) {
    PADDLE_ENFORCE_NE(
        handlers.count(type),
        0,
        platform::errors::PreconditionNotMet(
            "ProtoType %d has no corresponding translator", type));

    return handlers[type];
  }
};
```


#### 3.5.2 实现

```c++
TypeTranslator::TypeTranslator() {
  handlers = {
      {VarType::BOOL,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::BoolType::get(ctx);
       }},
      {VarType::UINT8,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::UInt8Type::get(ctx);
       }},
      {VarType::INT8,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Int8Type::get(ctx);
       }},
      {VarType::INT16,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Int16Type::get(ctx);
       }},
      {VarType::INT32,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Int32Type::get(ctx);
       }},
      {VarType::INT64,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Int64Type::get(ctx);
       }},
      {VarType::FP16,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Float16Type::get(ctx);
       }},
      {VarType::FP32,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Float32Type::get(ctx);
       }},
      {VarType::FP64,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Float64Type::get(ctx);
       }},
      {VarType::BF16,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::BFloat16Type::get(ctx);
       }},
      {VarType::COMPLEX64,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Complex64Type::get(ctx);
       }},
      {VarType::COMPLEX128,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         return ir::Complex128Type::get(ctx);
       }},
      {VarType::LOD_TENSOR,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         VLOG(10) << "[vartype translating]"
                  << "[" << var_desc.Name() << "] from LOD_TENSOR";

         ir::Type dtype =
             this->operator[](var_desc.GetDataType())(ctx, var_desc);
         DenseTensorTypeStorage::Dim dim = phi::make_ddim(var_desc.GetShape());
         DenseTensorTypeStorage::DataLayout layout =
             DenseTensorTypeStorage::DataLayout::UNDEFINED;
         DenseTensorTypeStorage::LoD lod = {};
         size_t offset = 0;
         return DenseTensorType::get(ctx, dtype, dim, layout, lod, offset);
       }},
      {VarType::LOD_TENSOR_ARRAY,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         VLOG(10) << "[vartype translating]"
                  << "[" << var_desc.Name() << "] from LOD_TENSOR_ARRAY";

         return ir::VectorType::get(ctx, std::vector<ir::Type>{});
       }},
      {VarType::SELECTED_ROWS,
       [&](ir::IrContext* ctx, const VarDesc& var_desc) -> ir::Type {
         VLOG(10) << "[vartype translating]"
                  << "[" << var_desc.Name() << "] from SELECTED_ROWS";

         ir::Type dtype =
             this->operator[](var_desc.GetDataType())(ctx, var_desc);

         SelectedRowsTypeStorage::Dim dim = phi::make_ddim(var_desc.GetShape());
         SelectedRowsTypeStorage::DataLayout layout =
             SelectedRowsTypeStorage::DataLayout::UNDEFINED;
         SelectedRowsTypeStorage::LoD lod = {};
         size_t offset = 0;
         ir::Type SelectedRows =
             SelectedRowsType::get(ctx, dtype, dim, layout, lod, offset);
         return SelectedRows;
       }},
  };
}
```

### 3.6 AttributeTranslator

#### 3.6.1 调用方式与接口


```c++
class AttributeVisitor;

class AttributeTranslator {
 private:
  AttributeTranslator();
  AttributeVisitor* general_visitor;
  std::unordered_map<std::string, AttributeVisitor*> special_visitors;

 public:
  AttributeTranslator(const AttributeTranslator&) = delete;
  AttributeTranslator& operator=(const AttributeTranslator&) = delete;
  AttributeTranslator(AttributeTranslator&&) = delete;
  AttributeTranslator& operator=(AttributeTranslator&&) = delete;

  static auto& instance() {
    static AttributeTranslator attribute_translator;
    return attribute_translator;
  }

  ir::Attribute operator()(const framework::Attribute& attr);
  ir::Attribute operator()(const std::string& target_type,
                           const framework::Attribute& attr);
};
```


```c++
struct CastOpTranscriber : public OpTranscriber {
  ir::AttributeMap TranslateOpAttribute(
      ir::IrContext*,
      const std::string& normalized_op_name,
      const OpAttributeInfoList& op_attr_infos,
      const OpDesc& op_desc) override {
    auto& attribute_translator = AttributeTranslator::instance();
    ir::AttributeMap attribute_map = {};
    const OpAttributeInfo info = op_attr_infos[0];

    std::string legacy_attr_name("out_dtype");

    paddle::framework::Attribute legacy_attr;
    if (op_desc.HasAttr(legacy_attr_name)) {
      legacy_attr = op_desc.GetAttr(legacy_attr_name);
    }
    VLOG(10) << "attribute in " << op_desc.Type()
             << " name: " << legacy_attr_name << " " << legacy_attr.index();
    ir::Attribute new_attr = attribute_translator(info.type_name, legacy_attr);
    attribute_map[info.name] = new_attr;

    return attribute_map;
  }
};
```

#### 3.6.2 实现

```c++
class Int64ArrayAttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;

  ir::Attribute operator()(const std::vector<int>& is) override {
    VLOG(10) << "translating vector<int64>";
    std::vector<ir::Attribute> attrs;
    attrs.reserve(is.size());
    for (const auto& v : is) {
      attrs.push_back(ir::Int64Attribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }
};
```

问题 3.6.1 为什么需要 special_visitors? 为了解决动静定义不一致问题，在动态图下，某个类型可能是 vec<i64> ，也可能是 vec<i32> ，但在 opdesc 中，都是 vec<i32>

## 四、Q&A

