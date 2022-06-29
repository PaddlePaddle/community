# Paddle.iinfo设计文档

| API名称                                                      | 新增API名称                      |
| ------------------------------------------------------------ | -------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | 林旭(isLinXu)                    |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-04-12                       |
| 版本号                                                       | V2.0                             |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                          |
| 文件名                                                       | 20220330_api-design_for_iinfo.md |

# 一、概述

## 1、相关背景

为提升飞桨API接口丰富度，支持数值计算、科学计算相关领域API，因此针对Paddle进行扩充`paddle.iinfo`。
任务认领：[链接](https://github.com/PaddlePaddle/Paddle/issues/40333)

## 2、功能目标


详细描述： finfo计算浮点数类型的数值限制，输入参数为Paddle浮点数类型(paddle.float16/paddle.float32/paddle.float64/paddle.complex64/paddle.complex128)，返回包含如下属性对象:

| 属性 | 类型 | 描述      |
| ---- | ---- | --------- |
| bits | int  | 占用bit数 |
| max  | int  | 最大数    |
| min  | int  | 最小数    |


## 3、意义

本次升级的意义在于为其他计算API的数值计算提升精度与准确率。

# 二、飞桨现状

Paddle目前不支持该API功能。
实现该API，若绕过对Paddlee的API进行开发，可以导入第三方库keras与Numpy组合实现。
但事实上并不建议这么做，因为Paddle与TensorFlow、Keras的许多结构和类型不一致，反倒事倍功半。

# 三、业内方案调研

## 1、 Pytorch

Pytorch目前已实现该API功能。

torch.finfo提供以下属性：

A [`torch.iinfo`](https://pytorch.org/docs/stable/type_info.html?highlight=finfo#torch.torch.iinfo) provides the following attributes:

| Name | Type | Description                              |
| ---- | ---- | ---------------------------------------- |
| bits | int  | The number of bits occupied by the type. |
| max  | int  | The largest representable number.        |
| min  | int  | The smallest representable number.       |

Pytorch实现

```cpp
PyObject* THPFInfo_New(const at::ScalarType& type) {
  auto finfo = (PyTypeObject*)&THPFInfoType;
  auto self = THPObjectPtr{finfo->tp_alloc(finfo, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPDTypeInfo*>(self.get());
  self_->type = c10::toRealValueType(type);
  return self.release();
}

PyObject* THPIInfo_New(const at::ScalarType& type) {
  auto iinfo = (PyTypeObject*)&THPIInfoType;
  auto self = THPObjectPtr{iinfo->tp_alloc(iinfo, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPDTypeInfo*>(self.get());
  self_->type = type;
  return self.release();
}

PyObject* THPFInfo_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
      "finfo(ScalarType type)",
      "finfo()",
  });

  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  TORCH_CHECK(r.idx < 2, "Not a type");
  at::ScalarType scalar_type;
  if (r.idx == 1) {
    scalar_type = torch::tensors::get_default_scalar_type();
    // The default tensor type can only be set to a floating point type/
    AT_ASSERT(at::isFloatingType(scalar_type));
  } else {
    scalar_type = r.scalartype(0);
    if (!at::isFloatingType(scalar_type) && !at::isComplexType(scalar_type)) {
      return PyErr_Format(
          PyExc_TypeError,
          "torch.finfo() requires a floating point input type. Use torch.iinfo to handle '%s'",
          type->tp_name);
    }
  }
  return THPFInfo_New(scalar_type);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIInfo_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
      "iinfo(ScalarType type)",
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  TORCH_CHECK(r.idx == 0, "Not a type");

  at::ScalarType scalar_type = r.scalartype(0);
  if (scalar_type == at::ScalarType::Bool) {
    return PyErr_Format(
        PyExc_TypeError,
        "torch.bool is not supported by torch.iinfo");
  }
  if (!at::isIntegralType(scalar_type, /*includeBool=*/false) && !at::isQIntType(scalar_type)) {
    return PyErr_Format(
        PyExc_TypeError,
        "torch.iinfo() requires an integer input type. Use torch.finfo to handle '%s'",
        type->tp_name);
  }
  return THPIInfo_New(scalar_type);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDTypeInfo_compare(THPDTypeInfo* a, THPDTypeInfo* b, int op) {
  switch (op) {
    case Py_EQ:
      if (a->type == b->type) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    case Py_NE:
      if (a->type != b->type) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
  }
  return Py_INCREF(Py_NotImplemented), Py_NotImplemented;
}

static PyObject* THPDTypeInfo_bits(THPDTypeInfo* self, void*) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers)
  int bits = elementSize(self->type) * 8;
  return THPUtils_packInt64(bits);
}

static PyObject* THPFInfo_eps(THPFInfo* self, void*) {
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::kHalf, at::ScalarType::BFloat16,
      self->type, "epsilon", [] {
        return PyFloat_FromDouble(
            std::numeric_limits<
                at::scalar_value_type<scalar_t>::type>::epsilon());
      });
}

static PyObject* THPFInfo_max(THPFInfo* self, void*) {
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::kHalf, at::ScalarType::BFloat16, self->type, "max", [] {
    return PyFloat_FromDouble(
        std::numeric_limits<at::scalar_value_type<scalar_t>::type>::max());
  });
}

static PyObject* THPFInfo_min(THPFInfo* self, void*) {
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::kHalf, at::ScalarType::BFloat16, self->type, "lowest", [] {
    return PyFloat_FromDouble(
        std::numeric_limits<at::scalar_value_type<scalar_t>::type>::lowest());
  });
}

static PyObject* THPIInfo_max(THPIInfo* self, void*) {
  if (at::isIntegralType(self->type, /*includeBool=*/false)) {
    return AT_DISPATCH_INTEGRAL_TYPES(self->type, "max", [] {
      return THPUtils_packInt64(std::numeric_limits<scalar_t>::max());
    });
  }
  // Quantized Type
  return AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(self->type, "max", [] {
      return THPUtils_packInt64(std::numeric_limits<underlying_t>::max());
  });
}

static PyObject* THPIInfo_min(THPIInfo* self, void*) {
  if (at::isIntegralType(self->type, /*includeBool=*/false)) {
    return AT_DISPATCH_INTEGRAL_TYPES(self->type, "min", [] {
      return THPUtils_packInt64(std::numeric_limits<scalar_t>::lowest());
    });
  }
  // Quantized Type
  return AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(self->type, "min", [] {
      return THPUtils_packInt64(std::numeric_limits<underlying_t>::lowest());
  });
}

static PyObject* THPIInfo_dtype(THPIInfo* self, void*) {
  std::string primary_name, legacy_name;
  std::tie(primary_name, legacy_name) = torch::utils::getDtypeNames(self->type);
  // NOLINTNEXTLINE(clang-diagnostic-unused-local-typedef)
  return AT_DISPATCH_INTEGRAL_TYPES(self->type, "dtype", [primary_name] {
    return PyUnicode_FromString((char*)primary_name.data());
  });
}

static PyObject* THPFInfo_tiny(THPFInfo* self, void*) {
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::kHalf, at::ScalarType::BFloat16, self->type, "min", [] {
    return PyFloat_FromDouble(
        std::numeric_limits<at::scalar_value_type<scalar_t>::type>::min());
  });
}

static PyObject* THPFInfo_resolution(THPFInfo* self, void*) {
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::kHalf, at::ScalarType::BFloat16, self->type, "digits10", [] {
    return PyFloat_FromDouble(
        std::pow(10, -std::numeric_limits<at::scalar_value_type<scalar_t>::type>::digits10));
  });
}

static PyObject* THPFInfo_dtype(THPFInfo* self, void*) {
  std::string primary_name, legacy_name;
  std::tie(primary_name, legacy_name) = torch::utils::getDtypeNames(self->type);
  // NOLINTNEXTLINE(clang-diagnostic-unused-local-typedef)
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::kHalf, at::ScalarType::BFloat16, self->type, "dtype", [primary_name] {
    return PyUnicode_FromString((char*)primary_name.data());
  });
}

PyObject* THPFInfo_str(THPFInfo* self) {
  std::ostringstream oss;
  oss << "finfo(resolution=" << PyFloat_AsDouble(THPFInfo_resolution(self, nullptr));
  oss << ", min=" << PyFloat_AsDouble(THPFInfo_min(self, nullptr));
  oss << ", max=" << PyFloat_AsDouble(THPFInfo_max(self, nullptr));
  oss << ", eps=" << PyFloat_AsDouble(THPFInfo_eps(self, nullptr));
  oss << ", tiny=" << PyFloat_AsDouble(THPFInfo_tiny(self, nullptr));
  oss << ", dtype=" << PyUnicode_AsUTF8(THPFInfo_dtype(self, nullptr)) << ")";

  return THPUtils_packString(oss.str().c_str());
}

PyObject* THPIInfo_str(THPIInfo* self) {
  auto type = self->type;
  std::string primary_name, legacy_name;
  std::tie(primary_name, legacy_name) = torch::utils::getDtypeNames(type);
  std::ostringstream oss;

  oss << "iinfo(min=" << PyFloat_AsDouble(THPIInfo_min(self, nullptr));
  oss << ", max=" << PyFloat_AsDouble(THPIInfo_max(self, nullptr));
  oss << ", dtype=" << PyUnicode_AsUTF8(THPIInfo_dtype(self, nullptr)) << ")";

  return THPUtils_packString(oss.str().c_str());
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-avoid-c-arrays)
static struct PyGetSetDef THPFInfo_properties[] = {
    {"bits", (getter)THPDTypeInfo_bits, nullptr, nullptr, nullptr},
    {"eps", (getter)THPFInfo_eps, nullptr, nullptr, nullptr},
    {"max", (getter)THPFInfo_max, nullptr, nullptr, nullptr},
    {"min", (getter)THPFInfo_min, nullptr, nullptr, nullptr},
    {"tiny", (getter)THPFInfo_tiny, nullptr, nullptr, nullptr},
    {"resolution", (getter)THPFInfo_resolution, nullptr, nullptr, nullptr},
    {"dtype", (getter)THPFInfo_dtype, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-avoid-c-arrays)
static PyMethodDef THPFInfo_methods[] = {
    {nullptr} /* Sentinel */
};
```



## 2、 TensorFlow

TensorFlow同样具有.iinfo的API功能，但与Pytorch不同的是，它的实现方式仅仅只是通过改写变体来转发到Numpy的同名函数进行处理。

NumPy 的 TensorFlow 变体`iinfo`。

```python
tf.experimental.numpy.iinfo(int_type)
```

```python
import math
import tensorflow as tf
int_value = math.pi
print(tf.experimental.numpy.iinfo(int(int_value)))
```

```python
Machine parameters for int64
---------------------------------------------------------------
min = -9223372036854775808
max = 9223372036854775807
---------------------------------------------------------------
```

## 3、Numpy

前面两者都是基于Numpy来进行参考实现，那么Numpy自然是具有`numpy.finfo`API函数的。

参考Numpyv1.22版本的文档，其具有以下属性：

Attributes

- **bits**int

  The number of bits occupied by the type.

- [`min`](https://numpy.org/doc/stable/reference/generated/numpy.iinfo.min.html#numpy.iinfo.min)int

  Minimum value of given dtype.

- [`max`](https://numpy.org/doc/stable/reference/generated/numpy.iinfo.max.html#numpy.iinfo.max)int

  Maximum value of given dtype.

# 四、对比分析

下面对第三部分调研的方案进行对比**评价**和**对比分析**，论述各种方案的优劣势。

| 方案名称                   | 方案思路                                        | 优势                                         | 劣势                                                    |
| -------------------------- | ----------------------------------------------- | -------------------------------------------- | ------------------------------------------------------- |
| 一：TensorFlow转发实现方案 | 直接在框架下改写变体实现Numpy同名转发API接口    | 便于快速开发实现                             | 可读性差，去Paddle其他接口交互不友好                    |
| 二：Pytorch集成实现方案    | 对齐Numpy类型与Paddle数据类型，进行深度集成实现 | 接口命名方式与Paddle接近，适配性和兼容性好。 | 与Pytorch的实现方案过于接近，会影响其他相关的接口开发。 |
| 三：重写Numpy实现方案      | 通过重写Numpy下的功能函数与接口实现该API        | 与Numpy原生代码接近，可读性和健壮性更好。    | 开发难度大，不易于实现                                  |



# 五、设计思路与实现方案

## 命名与参数设计

API设计为`paddle.iinfo(dtype)`，根据选择计算方法(比如eps、max、min、tiny)的不同，输出不同的结果。

参数类型要求：

- input：dtype，dtype包含uint8、int8、int16、int32、int64、float32、float64、bfloat16、complex64、complex128、bool等数据类型

其他说明：

- 使用时，可只进行参数类型指定，例如dtype=float
- 根据计算需要进行方法选择，那么得出的结果类型也不同。





## 底层OP设计

由于API本身并不涉及计算逻辑，为了保证API返回值类型与numpy一致，同时本着敏捷开发的角度，因此这里不直接通过OP/Kernel方式来进行设计和实现。



## API实现方案

通过设计实现与API对应的Class，并通过pybind将相应的成员函数绑定到python，从而实现该API。

- `.h`头文件定义声明

```cpp
namespace paddle {
namespace pybind {
    void BindFinfoVarDsec(pybind11::module *m);
    void BindIinfoVarDsec(pybind11::module *m);
}
}
```

- `.cc`绑定实现设计

```cpp
void BindFInfoVarDsec(pybind11::module *m){
	pybind11::class_<pd::VarDesc> finfo_var_desc(*m, "VarDesc", "");
    finfo_var_desc.def(pybind11::init<const std::string &>())
    .def("bits", &pd::Tinfo::Bits)
    .def("min", &pd::Tinfo::Min)
   	.def("max", &pd::Tinfo::Max)
}
```

实现思路：

- 从调研Torch的实现方案来看，它并没有使用OP或者重写Kernel来进行实现，而是通过设计实现一个Class来进行返回API结果。

- 因此要实现该API，需要如上抽象出一个符合要求的Class，同时并声明定义类下的成员函数来分别实现功能

- 通过类的成员函数分别来实现eps、min、max等函数，通过Pybind11来进行接口与参数的绑定

## API实现方案

在paddle/fluid/framework/Info.h与Info.cc下新增实现函数
定义class为`Tinfo`(借鉴Torch的结构设计，将finfo与iinfo合并为一个类进行实现)

```c
class Tinfo {
public:
	int Bits(const at::ScalarType& type)
    float Min(const at::ScalarType& type)
    float Max(const at::ScalarType& type)
}
```

`.cc`实现

```cpp
int Tinfo::Bits(const at::ScalarType& type){
	int bits = elementSize(self->type) * 8;
  	return THPUtils_packInt64(bits);
}

float Tinfo::Min(const at::ScalarType& type){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::lowest());
}

float Tinfo::Max(const at::ScalarType& type){
    return std::numeric_limits<at::scalar_value_type<scalar_t>::type>::max());
}
```





# 六、测试和验收的考量

测试考虑的case如下所示：

- 保证与torch.iinfo各个属性计算结果的对齐
- 保证与调用接口时计算其他模块或函数时的与numpy的结果对齐
- 输入输出的容错性与错误提示信息
- 输出Dtype错误或不兼容时抛出异常
- 保证调用属性时是可以被正常找到的

# 七、可行性分析和排期规划

时间和开发排期规划，主要milestone

暂定。


# 八、影响面

需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响

对其他模块暂无影响。


# 名词解释

暂无。



# 附件及参考资料

## 1、参考材料：

1.`numpy.iinfo`文档：[链接](https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html)
2.`torch.iinfo`文档：[链接](https://pytorch.org/docs/stable/type_info.html?highlight=finfo#torch.torch.finfo)
3.`tf.experimental.numpy.iinfo`文档：[链接](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/iinfo)

## 2、附件

暂无。

