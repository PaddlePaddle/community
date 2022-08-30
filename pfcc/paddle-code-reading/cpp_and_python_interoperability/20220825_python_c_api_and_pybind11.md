# Python & C++ 交互简单介绍

| 分享内容     | Python & C++ 交互简单介绍                          |
| ------------ | ----------------------------------------------- |
| 提交作者     | Wang Huan(@wanghuancoder)  |
| 提交时间     | 2022-08-25                                      |
| 版本号       | v1.0                                            |
| 依赖飞桨版本 | develop                                         |
| 文件名       | 20220825_python_c_api_and_pybind11.md |

## 1. Python & C++ 交互，在 Paddle 中的位置
![image](https://user-images.githubusercontent.com/6836917/187344211-9cb409e5-2c66-48d8-9bea-b94d1b19e434.png)

Paddle的核心逻辑主要是用C/C++开发的，如算子、动态图框架、静态图框架，最终我们将内核代码编译成如`core_avx.so`的动态库，并向外暴露Python可访问的API。
再通过Python端的封装，向用户提供开发接口。本质上，单纯使用`core_avx.so`即可完成简单的开发：
```python
import core_avx	# 加载core_avx.so
import time
import numpy as np

core_avx.init_devices()
tracer = core_avx.Tracer()
place = core_avx.CUDAPlace(0)
tracer._expected_place = place
core_avx._switch_tracer(tracer)
x_data = np.random.random([1]).astype(np.float32)
y_data = np.random.random([1]).astype(np.float32)
x = core_avx.eager.Tensor(x_data, place, False, False, "wh", True)
y = core_avx.eager.Tensor(y_data, place, False, False, "wh2", True)
transpose_X = False
transpose_Y = False
for i in range(100):
    out = core_avx.eager.ops.matmul(x, y, transpose_X, transpose_Y)
```
## 2. Python/C API & pybind11 的技术选型
- Python/C API是Python官方提供的用于编写扩展模块或将 Python解释器嵌入其应用程序中的官方API。但由于过于底层，导致学习成本高，开发难度大，官网不推荐直接使用：
![image](https://user-images.githubusercontent.com/6836917/187344269-d1edf148-aef3-4982-8f4a-ac8f1895a547.png)
- Pybind11 是一个轻量级的 C++ 库，用于将你的 C++ 代码暴露给 Python 调用。在业界广受好评。

相较于Python/C API，pybind11非常易用，举例说明：

 - 例子1，暴露一个函数
 
```python
// 使用Python/C API
#include <Python.h>
static PyObject* Add(PyObject*self,PyObject*args)
{
       int x = 0 ;
       int y = 0;
       if(!PyArg_ParseTuple(args,"i|i", &x, &y))   // 使用Python/C API需要做类型转换，如果是自定义类型，编码更复杂
             returnNULL;
       int z=x +y;
       returnPy_BuildValue("i",z);   // 使用Python/C API需要将返回的result“转意”成PyObject*
}

staticPyMethodDefPyExtMethods[]=
{
       {"Add", Add,METH_VARARGS,"Addtwo number - edit by magictong."},
       {NULL,NULL, 0,NULL}
};

 PyMODINIT_FUNCinitPyExt()
{
       Py_InitModule("PyExt",PyExtMethods);
}


// 使用pybind11
#include <pybind11/pybind11.h>
namespace py = pybind11;
int add(int i, int j)
{
    return i + j;
}
 
PYBIND11_MODULE(example, m)
{
    m.def("add", &add, "A function which adds two numbers");
}
```

 - 例子2，暴露一个类型
```python
// 使用Python/C API
void BindEager(pybind11::module* module) {
  auto m = module->def_submodule("eager");

  auto heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  heap_type->ht_name = ToPyObject("Tensor");
  heap_type->ht_qualname = ToPyObject("Tensor");
  auto type = &heap_type->ht_type;
  type->tp_name = "Tensor";
  type->tp_basicsize = sizeof(TensorObject);
  type->tp_dealloc = (destructor)TensorDealloc;
  type->tp_as_number = &number_methods;
  type->tp_as_sequence = &sequence_methods;
  type->tp_as_mapping = &mapping_methods;
  type->tp_methods = variable_methods;
  type->tp_getset = variable_properties;
  type->tp_init = TensorInit;
  type->tp_new = TensorNew;
  type->tp_weaklistoffset = offsetof(TensorObject, weakrefs);
  Py_INCREF(&PyBaseObject_Type);
  type->tp_base = reinterpret_cast<PyTypeObject*>(&PyBaseObject_Type);
  type->tp_flags |=
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
#if PY_VERSION_HEX >= 0x03050000
  type->tp_as_async = &heap_type->as_async;
#endif
  p_tensor_type = type;

  if (PyType_Ready(type) < 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle error in BindEager(PyType_Ready)."));
    return;
  }

  Py_INCREF(type);
  if (PyModule_AddObject(m.ptr(), "Tensor", reinterpret_cast<PyObject*>(type)) <
      0) {
    Py_DECREF(type);
    Py_DECREF(m.ptr());
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle error in BindEager(PyModule_AddObject)."));
    return;
  }

  BindFunctions(m.ptr());
  BindEagerPyLayer(m.ptr());
  BindEagerOpFunctions(&m);
}

// 使用pybind11
void BindTensor(pybind11::module &m) {
  py::class_<framework::Tensor> framework_tensor(m, "Tensor", py::buffer_protocol());
  framework_tensor.def("__array__", [](framework::Tensor &self) { return TensorToPyArray(self); })
                  .def("_slice", &framework::Tensor::Slice);
}
```
早期 Paddle 主推静态图，因此选用 pybind11 作为 Paddle 内核向 Python 暴露 API 的第三方工具。
但随着 Paddle 支持了动态图，渐渐的，我们发现 pybind11 成为了动态图 Python & C++ 交互的性能瓶颈。
因此将高频交互的 API 改用了 Python/C API。而其余低频使用的 API ，仍使用 pybind11 暴露。
高频使用的 API 绝大多数是算子类 API ，低频使用的往往是环境配置、初始化类的 API 。

## 3. pybind11 在 Paddle 中的使用

无论是 pybind11 还是 Python/C API相关的代码，都在`paddle/fluid/pybind`路径下。

- pybind11 的入口位置在`pybind.cc`中:
```cpp
#ifdef PADDLE_WITH_AVX
PYBIND11_MODULE(core_avx, m) {
#else
PYBIND11_MODULE(core_noavx, m) {
#endif
```
- 将普通函数 API 暴露到 Python 端的代码如下：
```cpp
#ifdef PADDLE_WITH_AVX
PYBIND11_MODULE(core_avx, m) {
#else
PYBIND11_MODULE(core_noavx, m) {
#endif

  m.def("set_num_threads", &platform::SetNumThreads);
  
  m.def("clear_gradients",
        [](std::vector<std::shared_ptr<imperative::VarBase>> param_list,
           bool set_to_zero) {
          for (auto param : param_list) {
            param->ClearGradient(set_to_zero);
          }
        });
}
```
- 将1个 C++ 类暴露到 Python 作为1个类型的代码实例如下：
```cpp
#ifdef PADDLE_WITH_AVX
PYBIND11_MODULE(core_avx, m) {
#else
PYBIND11_MODULE(core_noavx, m) {
#endif

	py::class_<framework::Tensor> framework_tensor(m, "Tensor", py::buffer_protocol());
	framework_tensor.def("__array__", [](framework::Tensor &self) { return TensorToPyArray(self); })
                    .def("_slice", &framework::Tensor::Slice);
}
```
- 在 core_avx 下新建一个 ops 模块的实例如下：
```cpp
#ifdef PADDLE_WITH_AVX
PYBIND11_MODULE(core_avx, m) {
#else
PYBIND11_MODULE(core_noavx, m) {
#endif

	auto submodule = m.def_submodule("ops");
	submodule.def(....
}
```
- 参数中使用 key args 用法：
```cpp
#ifdef PADDLE_WITH_AVX
PYBIND11_MODULE(core_avx, m) {
#else
PYBIND11_MODULE(core_noavx, m) {
#endif

  m.def("set_printoptions", [](const py::kwargs &kwargs) {
    auto &print_opt = framework::PrintOptions::Instance();
    if (kwargs.contains("precision")) {
      print_opt.precision = kwargs["precision"].cast<int>();
    }
    if (kwargs.contains("threshold")) {
      print_opt.threshold = kwargs["threshold"].cast<int>();
    }
  }
  m.def(
      "run_cmd",
      [](const std::string &cmd,
         int time_out = -1,
         int sleep_inter = -1) -> const std::string {
        return paddle::framework::shell_get_command_output(
            cmd, time_out, sleep_inter);
      },
      py::arg("cmd"),
      py::arg("time_out") = -1,
      py::arg("sleep_inter") = -1);
}
```
- 使用 pybind 需要注意，在 C++ 需要运行很长时间的 API 中加`pybind11::gil_scoped_release release;`
这是释放 Python 的 GIL 锁的 guard 。当释放 GIL 锁后，Python 端其它多线程程序可并行运行，否则将与其它 Python 多线程互斥。

## 4. Python/C API 在 Paddle 中的使用
pybind11 虽然让程序员开发更容易，但相较于 Python/C API 性能差一些。
而动态图对调度性能非常敏感，微乎其微的 CPU 性能浪费都可能让模型训练速度大打折扣。
为提升性能，Paddle 动态图将高频使用的 API 改用 Python/C API 对外暴露。
使用 Python/C API的主要源码文件包括：
```cpp
paddle/fluid/pybind/eager.cc	//动态图Tensor对外暴露
paddle/fluid/pybind/eager_method.cc	//Tensor的成员函数对外暴露
paddle/fluid/pybind/eager_properties.cc	//Tensor的成员变量对外暴露
paddle/fluid/pybind/eager_functions.cc	//动态图普通API对外暴露
paddle/fluid/pybind/eager_op_function.cc	//动态图的算子类API对外暴露（这些代码是自动生成的）
paddle/fluid/pybind/eager_legacy_op_function.cc	//动态图兼容态的算子类API对外暴露（这些代码是自动生成的）
paddle/fluid/pybind/eager_py_layer.cc	//PyLayer相关业务API对外暴露
paddle/fluid/pybind/eager_utils.cc	//使用Python/C API常用基础函数的封装
```
- 如何让 pybind11 和 Python/C API 共存？
  - pybind11 其实就是对 Python/C API 的封装，pybind11 的底层就是 Python/C API。
  基本 pybind11 的所有类均继承自 handle 类。handle 类的成员变量`PyObject *m_ptr`就是 Python/C API 的基础 Object。我们可以通过成员函数`ptr`获取到。
- Eager 的 Tensor 对外暴露
```cpp
// eager.cc中的代码
void BindEager(pybind11::module* module) {
  auto m = module->def_submodule("eager");

  auto heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  heap_type->ht_name = ToPyObject("Tensor");
  heap_type->ht_qualname = ToPyObject("Tensor");
  auto type = &heap_type->ht_type;
  type->tp_name = "Tensor";
  type->tp_basicsize = sizeof(TensorObject);
  type->tp_dealloc = (destructor)TensorDealloc;
  type->tp_as_number = &number_methods;
  type->tp_as_sequence = &sequence_methods;
  type->tp_as_mapping = &mapping_methods;
  type->tp_methods = variable_methods;
  type->tp_getset = variable_properties;
  type->tp_init = TensorInit;
  type->tp_new = TensorNew;
  type->tp_weaklistoffset = offsetof(TensorObject, weakrefs);
  Py_INCREF(&PyBaseObject_Type);
  type->tp_base = reinterpret_cast<PyTypeObject*>(&PyBaseObject_Type);
  type->tp_flags |= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
  p_tensor_type = type;

  if (PyType_Ready(type) < 0) {
    return;
  }
  Py_INCREF(type);
  if (PyModule_AddObject(m.ptr(), "Tensor", reinterpret_cast<PyObject*>(type)) <
      0) {
    Py_DECREF(type);
    Py_DECREF(m.ptr());
    return;
  }
}
```
- 如何为 Tensor 添加成员函数
```cpp
// eager_method.cc的代码
static PyObject* tensor_method__copy_to(TensorObject* self,
                                        PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 0), 0);
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 1), 1);
  auto cp_tensor = self->tensor.copy_to(place, blocking);
  if (!blocking) {
    IncreaseTensorReferenceCountUntilCopyComplete(self->tensor, place);
  }
  egr::EagerUtils::autograd_meta(&cp_tensor)->SetStopGradient(true);
  egr::EagerUtils::autograd_meta(&cp_tensor)
      ->SetPersistable(
          egr::EagerUtils::autograd_meta(&(self->tensor))->Persistable());
  return ToPyObject(cp_tensor);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}


PyMethodDef variable_methods[] = {
{"_copy_to", (PyCFunction)(void (*)(void))tensor_method__copy_to, METH_VARARGS | METH_KEYWORDS, NULL},
{NULL, NULL, 0, NULL}};
```
- 常用的参数解析和返回值封装函数在`eager_utils.h`中
```cpp
bool PyObject_CheckLongOrConvertToLong(PyObject** obj);
bool PyObject_CheckFloatOrConvertToFloat(PyObject** obj);
bool PyObject_CheckStr(PyObject* obj);
bool CastPyArg2AttrBoolean(PyObject* obj, ssize_t arg_pos);
int CastPyArg2AttrInt(PyObject* obj, ssize_t arg_pos);
int64_t CastPyArg2AttrLong(PyObject* obj, ssize_t arg_pos);
size_t CastPyArg2AttrSize_t(PyObject* obj, ssize_t arg_pos);
float CastPyArg2AttrFloat(PyObject* obj, ssize_t arg_pos);
std::string CastPyArg2AttrString(PyObject* obj, ssize_t arg_pos);
paddle::CustomOpKernelContext CastPyArg2CustomOpKernelContext(PyObject* obj, ssize_t arg_pos);
paddle::experimental::Tensor CastPyArg2Tensor(PyObject* obj, ssize_t arg_pos);
std::shared_ptr<imperative::VarBase> CastPyArg2VarBase(PyObject* obj, ssize_t arg_pos);
std::vector<paddle::experimental::Tensor> CastPyArg2VectorOfTensor( PyObject* obj, ssize_t arg_pos);
platform::Place CastPyArg2Place(PyObject* obj, ssize_t arg_pos);
framework::Tensor CastPyArg2FrameworkTensor(PyObject* obj, ssize_t arg_pos);
std::vector<framework::LoDTensor> CastPyArg2VectorOfTensorBase(PyObject* obj, ssize_t arg_pos);
std::vector<int> CastPyArg2VectorOfInt(PyObject* obj, size_t arg_pos);
std::vector<size_t> CastPyArg2VectorOfSize_t(PyObject* obj, size_t arg_pos);
std::vector<std::vector<size_t>> CastPyArg2VectorOfVectorOfSize_t( PyObject* obj, size_t arg_pos);
framework::proto::VarType::Type CastPyArg2ProtoType(PyObject* obj, ssize_t arg_pos);
std::unordered_map<std::wstring, int> CastPyArg2Vocab(PyObject* obj, ssize_t arg_pos);
std::vector<std::string> CastPyArg2Strings(PyObject* obj, ssize_t arg_pos);
std::shared_ptr<jit::Function> CastPyArg2JitFunction(PyObject* obj, ssize_t arg_pos);

PyObject* ToPyObject(int value);
PyObject* ToPyObject(uint32_t value);
PyObject* ToPyObject(bool value);
PyObject* ToPyObject(int64_t value);
PyObject* ToPyObject(size_t value);
PyObject* ToPyObject(float value);
PyObject* ToPyObject(double value);
PyObject* ToPyObject(const char* value);
PyObject* ToPyObject(const std::string& value);
PyObject* ToPyObject(const paddle::experimental::Tensor& value, bool return_py_none_if_not_initialize = false);
PyObject* ToPyObject(const paddle::experimental::Tensor& value, PyObject* args, const std::map<ssize_t, ssize_t>& inplace_var_idx_map);
PyObject* ToPyObject(PyObject* args, ssize_t arg_idx);
PyObject* ToPyObject(const std::vector<bool>& value);
PyObject* ToPyObject(const std::vector<int>& value);
PyObject* ToPyObject(const std::vector<int64_t>& value);
PyObject* ToPyObject(const std::vector<size_t>& value);
PyObject* ToPyObject(const std::vector<float>& value);
PyObject* ToPyObject(const std::vector<double>& value);
PyObject* ToPyObject(const std::vector<std::vector<size_t>>& value);
PyObject* ToPyObject(const std::vector<paddle::experimental::Tensor>& value, bool return_py_none_if_not_initialize = false);
PyObject* ToPyObject(const platform::Place& value);
PyObject* ToPyObject(const framework::LoDTensor* value);
PyObject* ToPyObject(const phi::SelectedRows* value);
PyObject* ToPyObject(const paddle::framework::proto::VarType::Type& dtype);
PyObject* ToPyObject(const paddle::framework::proto::VarType& type);
PyObject* ToPyObject(const void* value);
PyObject* ToPyObject( const std::unordered_map<std::string, std::vector<std::string>>& value);
PyObject* ToPyObject(const std::unordered_map<std::wstring, int>& value);
```
- 如何为 Tensor 添加成员变量
```cpp
PyObject* tensor_properties_get_name(TensorObject* self, void* closure) {
  EAGER_TRY
  // NOTE(dev): [why not use egr::Controller::Instance::GernerateUniqueName()?]
  // Beacause Controller must holder a tracer, but 'tensor.name' maybe called
  // everywhere such as static mode in @to_static, which means tracer is None.
  static egr::UniqueNameGenerator name_generator;
  if (self->tensor.name().empty()) {
    self->tensor.set_name(name_generator.Generate());
  }
  return ToPyObject(self->tensor.name());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

int tensor_properties_set_name(TensorObject* self,
                               PyObject* value,
                               void* closure) {
  EAGER_TRY
  self->tensor.set_name(CastPyArg2AttrString(value, 0));
  return 0;
  EAGER_CATCH_AND_THROW_RETURN_NEG
}

struct PyGetSetDef variable_properties[] = {
    {"name", (getter)tensor_properties_get_name, (setter)tensor_properties_set_name, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};
```
- 如何使用 Python/C API 暴露常规的 API ？
```cpp
static PyObject* eager_api_run_backward(PyObject* self,
                                        PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  auto tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0);
  auto grad_tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 1), 1);
  {
    eager_gil_scoped_release guard;
    egr::Backward(tensors,
                  grad_tensors,
                  CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2));
  }
  RETURN_PY_NONE
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef variable_functions[] = {
    {"run_backward", (PyCFunction)(void (*)(void))eager_api_run_backward, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};
```
- 算子类 API 是如何生成的?
```cpp
// eager_op_function.cc中生成的代码
static PyObject * eager_api_add(PyObject *self, PyObject *args, PyObject *kwargs) {
  paddle::platform::RecordEvent pythonc_record_event("add pybind_imperative_func", paddle::platform::TracerEventType::UserDefined, 1);
  PyThreadState *tstate = nullptr;
  try {
    auto x = GetTensorFromArgs("add", "x", args, 0, false);
    auto y = GetTensorFromArgs("add", "y", args, 1, false);
    tstate = PyEval_SaveThread();
    decltype(::add_dygraph_function(x,y)) out = ::add_dygraph_function(x,y);
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return ToPyObject(out);
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}
```
以上代码是`paddle/fluid/eager/auto_code_generator/generator/python_c_gen.py`根据`paddle/phi/api/yaml/legacy_api.yaml`自动生成的，感兴趣的同学可以自行阅读。
```yaml
- api : add
  args : (Tensor x, Tensor y)
  output : Tensor(out)
  infer_meta :
    func : ElementwiseInferMeta
  kernel :
    func : add
  inplace : (x -> out)
  backward : add_grad
```
## 5. 提问环节
参考 [Paddle Frawework Contributor Club 第九次会议纪要的提问环节](https://github.com/PaddlePaddle/community/blob/master/pfcc/2022-08-25-meeting-minutes.md#%E6%8F%90%E9%97%AE%E7%8E%AF%E8%8A%82)。
