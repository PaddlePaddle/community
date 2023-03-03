# Paddle.finfo设计文档

| API名称                                                      | paddle.finfo                     |
| ------------------------------------------------------------ | -------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">    | lisamhy，林旭(isLinXu)                      |
| 提交时间<input type="checkbox" class="rowselector hidden">    | 2022-04-12                        |
| 版本号                                                        | V2.0                             |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                          |
| 文件名                                                        | 20220330_api-design_for_finfo.md |

# 一、概述

## 1、相关背景

为提升飞桨API接口丰富度，支持数值计算、科学计算相关领域API，因此针对Paddle进行扩充`paddle.finfo`。
任务认领：[链接](https://github.com/PaddlePaddle/Paddle/issues/40334)

## 2、功能目标


详细描述： finfo计算浮点数类型的数值限制，输入参数为Paddle浮点数类型(paddle.float16/paddle.float32/paddle.float64/paddle.complex64/paddle.complex128)，返回包含如下属性对象:

| 属性       | 类型  | 描述                                                         |
| ---------- | ----- | ------------------------------------------------------------ |
| bits       | int   | 该类型占用的bit数                                            |
| eps        | float | 该类型所能表示的epsilon值，即满足1.0 + eps != 1.0的最小值，参考https://numpy.org/doc/stable/reference/generated/numpy.finfo.html |
| min        | float | 该类型能表示的最小值                                         |
| max        | float | 该类型能表示最大值                                           |
| tiny       | float | 该类型所能表示的最小正数                                     |
| resolution | float | 该类型十进制形式精度 `10**-precision`. 其中precision为IEEE754标准中该类型有效数字位数 |



## 3、意义

本次升级的意义在于为其他计算API的数值计算提升精度与准确率，也更加方便Paddle在复现其他框架代码计算loss、softmax等的数值对齐等。

例如在下面这样的用法中，finfo的功能就可以起到限制数值溢出的作用。

```python
from torch import finfo
def get_loss(self, y_pred, y_true, *args, **kwargs):
        if isinstance(self.criterion_, torch.nn.NLLLoss):
            eps = torch.finfo(y_pred.dtype).eps
            y_pred = torch.log(y_pred + eps)
        return super().get_loss(y_pred, y_true, *args, **kwargs)

    # pylint: disable=signature-differs 
```



# 二、飞桨现状

Paddle目前不支持该API功能。
实现该API，若绕过对Paddlee的API进行开发，可以导入第三方库keras与Numpy组合实现。
但事实上并不建议这么做，因为Paddle与TensorFlow、Keras的许多结构和类型不一致，反倒事倍功半。

# 三、业内方案调研

## 1、 Pytorch

Pytorch目前已实现该API功能。

```python
import torch
print(torch.finfo(torch.float32).eps) #1.1920928955078125e-07
print(torch.finfo(torch.float64).eps) #2.220446049250313e-16
print(torch.finfo(torch.double).eps)  #2.220446049250313e-16 
```

Torch.finfo是表示浮点Torch.dtype的数字属性的对象，（即Torch.float32，Torch.float64，Torch.float16和Torch.bfloat16）。这类似于numpy.finfo。

torch.finfo提供以下属性：

| Name       | Type  | Description                                                  |
| ---------- | ----- | ------------------------------------------------------------ |
| bits       | int   | The number of bits occupied by the type.                     |
| eps        | float | The smallest representable number such that `1.0 + eps != 1.0`. |
| max        | float | The largest representable number.                            |
| min        | float | The smallest representable number (typically `-max`).        |
| tiny       | float | The smallest positive normal number. See notes.              |
| resolution | float | The approximate decimal resolution of this type, i.e., `10**-precision`. |


可以无参数调用torch.finfo的构造函数，在这种情况下，在这种情况下，将为pytorch默认数据类型创建类（由torch.get_default_dtype()返回）

```python
井号后为返回值
max_neg_value0 = torch.finfo(torch.float16).tiny #6.103515625e-05
max_neg_value1 = torch.finfo(torch.float16).max  #65504.0
max_neg_value2 = torch.finfo(torch.float32).max  #3.4028234663852886e+38
max_neg_value3 = torch.finfo(torch.float64).max  #1.7976931348623157e+308
```

Pytorch实现方案

```C++
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

TensorFlow同样具有.finfo的API功能，但与Pytorch不同的是，它的实现方式仅仅只是通过改写变体来转发到Numpy的同名函数进行处理。

NumPy 的 TensorFlow 变体`finfo`。

```python
tf.experimental.numpy.finfo(dtype)
```

```python
import tensorflow as tf
print(tf.experimental.numpy.finfo(tf.as_type()))
```

```python
Machine parameters for float32
---------------------------------------------------------------
precision =   6   resolution = 1.0000000e-06
machep =    -23   eps =        1.1920929e-07
negep =     -24   epsneg =     5.9604645e-08
minexp =   -126   tiny =       1.1754944e-38
maxexp =    128   max =        3.4028235e+38
nexp =        8   min =        -max
---------------------------------------------------------------
```



## 3、Numpy

前面两者都是基于Numpy来进行参考实现，那么Numpy自然是具有`numpy.finfo`API函数的。

参考Numpyv1.22版本的文档，其具有以下属性：

Attributes

- **bits** int

  The number of bits occupied by the type.

- **eps** float

  The difference between 1.0 and the next smallest representable float larger than 1.0. For example, for 64-bit binary floats in the IEEE-754 standard, `eps = 2**-52`, approximately 2.22e-16.

- **epsneg** float

  The difference between 1.0 and the next smallest representable float less than 1.0. For example, for 64-bit binary floats in the IEEE-754 standard, `epsneg = 2**-53`, approximately 1.11e-16.

- **iexp** int

  The number of bits in the exponent portion of the floating point representation.

- [`machar`](https://numpy.org/doc/stable/reference/generated/numpy.finfo.machar.html#numpy.finfo.machar)MachAr

  The object which calculated these parameters and holds more detailed information.

- **machep** int

  The exponent that yields *eps*.

- **max**floating point number of the appropriate type

  The largest representable number.

- **maxexp** int

  The smallest positive power of the base (2) that causes overflow.

- **min** floating point number of the appropriate type

  The smallest representable number, typically `-max`.

- **minexp** int

  The most negative power of the base (2) consistent with there being no leading 0’s in the mantissa.

- **negep** int

  The exponent that yields *epsneg*.

- **nexp** int

  The number of bits in the exponent including its sign and bias.

- **nmant** int

  The number of bits in the mantissa.

- **precision** int

  The approximate number of decimal digits to which this kind of float is precise.

- **resolution** floating point number of the appropriate type

  The approximate decimal resolution of this type, i.e., `10**-precision`.

- [`tiny`](https://numpy.org/doc/stable/reference/generated/numpy.finfo.tiny.html#numpy.finfo.tiny) float

  Return the value for tiny, alias of smallest_normal.

- [`smallest_normal`](https://numpy.org/doc/stable/reference/generated/numpy.finfo.smallest_normal.html#numpy.finfo.smallest_normal) float

  Return the value for the smallest normal.

- **smallest_subnormal** float

  The smallest positive floating point number with 0 as leading bit in the mantissa following IEEE-754.

**接口调用测试**

```python
"""
np.finfo使用方法
    eps是一个很小的非负数
    除法的分母不能为0的,不然会直接跳出显示错误。
    使用eps将可能出现的零用eps来替换，这样不会报错。
"""
import numpy as np
 
x = np.array([1, 2, 3], dtype=float)
eps = np.finfo(x.dtype).eps  # eps = 2.220446049250313e-16 type = <class 'numpy.float64'>
print(eps, type(eps))
height = np.array([0, 2, 3], dtype=float)
height = np.maximum(height, eps) #一旦height中出现0，就用eps进行替换
print(height)   #[2.22044605e-16 2.00000000e+00 3.00000000e+00]
dy = x / height
print(dy)   #[4.50359963e+15 1.00000000e+00 1.00000000e+00]
```

# 四、对比分析

下面对第三部分调研的方案进行对比**评价**和**对比分析**，论述各种方案的优劣势。

| 方案名称                   | 方案思路                                        | 优势                                         | 劣势                                                    |
| -------------------------- | ----------------------------------------------- | -------------------------------------------- | ------------------------------------------------------- |
| 一：TensorFlow转发实现方案 | 直接在框架下改写变体实现Numpy同名转发API接口    | 便于快速开发实现                             | 可读性差，去Paddle其他接口交互不友好                    |
| 二：Pytorch集成实现方案    | 对齐Numpy类型与Paddle数据类型，进行深度集成实现 | 接口命名方式与Paddle接近，适配性和兼容性好。 | 与Pytorch的实现方案过于接近，会影响其他相关的接口开发。 |
| 三：重写Numpy实现方案      | 通过重写Numpy下的功能函数与接口实现该API        | 与Numpy原生代码接近，可读性和健壮性更好。    | 开发难度大，不易于实现                                  |



# 五、设计思路与实现方案

## 命名与参数设计

API设计为`paddle.finfo(dtype)`，根据选择计算方法(比如eps、max、min、tiny)的不同，输出不同的结果。

参数类型要求：

- input：dtype，dtype包含 float16、float32、float64、bfloat16、complex64 和 complex128 数据类型

其他说明：

- 使用时，可只进行参数类型指定，例如dtype=float
- 根据计算需要进行方法选择，那么得出的结果类型也不同。



## 底层OP设计

由于API本身并不涉及计算逻辑，为了保证API返回值类型与numpy一致，同时本着敏捷开发的角度，因此这里不直接通过OP/Kernel方式来进行设计和实现。


## API实现方案

通过设计实现与API对应的Class，并通过pybind将相应的成员函数绑定到python，从而实现该API。

- `pybind.cc` finfo class 实现

```cpp
struct finfo {
  int64_t bits;
  double eps;
  double min;  // lowest()
  double max;
  double tiny;
  double smallest_normal;  // min()
  double resolution;
  std::string dtype;

  explicit finfo(const framework::proto::VarType::Type &type) {
    switch (type) {
      case framework::proto::VarType::FP16:
        eps = std::numeric_limits<paddle::platform::float16>::epsilon();
        min = std::numeric_limits<paddle::platform::float16>::lowest();
        max = std::numeric_limits<paddle::platform::float16>::max();
        smallest_normal = std::numeric_limits<paddle::platform::float16>::min();
        tiny = smallest_normal;
        resolution = std::pow(
            10, -std::numeric_limits<paddle::platform::float16>::digits10);
        bits = 16;
        dtype = "float16";
        break;
      case framework::proto::VarType::FP32:
      case framework::proto::VarType::COMPLEX64:
        eps = std::numeric_limits<float>::epsilon();
        min = std::numeric_limits<float>::lowest();
        max = std::numeric_limits<float>::max();
        smallest_normal = std::numeric_limits<float>::min();
        tiny = smallest_normal;
        resolution = std::pow(10, -std::numeric_limits<float>::digits10);
        bits = 32;
        dtype = "float32";
        break;
      case framework::proto::VarType::FP64:
      case framework::proto::VarType::COMPLEX128:
        eps = std::numeric_limits<double>::epsilon();
        min = std::numeric_limits<double>::lowest();
        max = std::numeric_limits<double>::max();
        smallest_normal = std::numeric_limits<double>::min();
        tiny = smallest_normal;
        resolution = std::pow(10, -std::numeric_limits<double>::digits10);
        bits = 64;
        dtype = "float64";
        break;
      case framework::proto::VarType::BF16:
        eps = std::numeric_limits<paddle::platform::bfloat16>::epsilon();
        min = std::numeric_limits<paddle::platform::bfloat16>::lowest();
        max = std::numeric_limits<paddle::platform::bfloat16>::max();
        smallest_normal =
            std::numeric_limits<paddle::platform::bfloat16>::min();
        tiny = smallest_normal;
        resolution = std::pow(
            10, -std::numeric_limits<paddle::platform::bfloat16>::digits10);
        bits = 16;
        dtype = "bfloat16";
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "the argument of paddle.finfo can only be paddle.float32, "
            "paddle.float64, paddle.float16, paddle.bfloat16"
            "paddle.complex64, or paddle.complex128"));
        break;
    }
  }
};
```

- `pybind.cc` finfo 绑定实现

```cpp
  py::class_<finfo>(m, "finfo")
      .def(py::init<const framework::proto::VarType::Type &>())
      .def_readonly("min", &finfo::min)
      .def_readonly("max", &finfo::max)
      .def_readonly("bits", &finfo::bits)
      .def_readonly("eps", &finfo::eps)
      .def_readonly("resolution", &finfo::resolution)
      .def_readonly("smallest_normal", &finfo::smallest_normal)
      .def_readonly("tiny", &finfo::tiny)
      .def_readonly("dtype", &finfo::dtype)
      .def("__repr__", [](const finfo &a) {
        std::ostringstream oss;
        oss << "paddle.finfo(min=" << a.min;
        oss << ", max=" << a.max;
        oss << ", eps=" << a.eps;
        oss << ", resolution=" << a.resolution;
        oss << ", smallest_normal=" << a.smallest_normal;
        oss << ", tiny=" << a.tiny;
        oss << ", bits=" << a.bits;
        oss << ", dtype=" << a.dtype << ")";
        return oss.str();
      });
```

- `dtype.py` python 暴露 finfo API

```python
from ..fluid.core import finfo as core_finfo

def finfo(dtype):
    return core_finfo(dtype)
```

实现思路：

- 从调研Torch的实现方案来看，它并没有使用OP或者重写Kernel来进行实现，而是通过设计实现一个Class来进行返回API结果。

- 因此要实现该API，需要如上抽象出一个符合要求的Class，同时并声明定义类下的成员函数来分别实现功能

- 通过类的成员函数分别来实现 eps、min、max、bits、resolution、tiny、smallest_normal、dtype 等函数，通过Pybind11来进行接口与参数的绑定



# 六、测试和验收的考量

测试考虑的case如下所示：

- 保证与torch.finfo各个属性计算结果的对齐
- 保证与调用接口时计算其他模块或函数时的与numpy的结果对齐
- 输入输出的容错性与错误提示信息
- 输出Dtype错误或不兼容时抛出异常
- 保证调用属性时是可以被正常找到的

# 七、可行性分析和排期规划

暂定。


# 八、影响面

需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响

对其他模块暂无影响。






# 名词解释

暂无。



# 附件及参考资料

## 1、参考材料：

1.`numpy.finfo`文档：[链接](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)
2.`torch.finfo`文档：[链接](https://pytorch.org/docs/stable/type_info.html?highlight=finfo#torch.torch.finfo)
3.`tf.experimental.numpy.finfo`文档：[链接](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/finfo)

## 2、附件

暂无。

