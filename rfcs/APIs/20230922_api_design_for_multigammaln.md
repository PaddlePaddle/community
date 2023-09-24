# 标题 paddle.multigammaln 设计文档

|API名称 | paddle.multigammaln | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 吴俊([bapijun] (https://github.com/bapijun)) | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-01 | 
|版本号 | 此设计文档的版本号，如V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 如无特殊情况，都应基于develop版本开发 | 
|文件名 | 20230923_api_design_for_multigammaln.md | 


# 一、概述
## 1、相关背景
`multigammaln` 函数返回多元 gamma 函数的对数，有时也称为广义 gamma 函数。$d$ 维实数 $a$ 的多元 gamma 函数定义为：

$$\Gamma_d(a) = \int_{A > 0} {e^{-{tr}(A)}|A|^{a - (d + 1) / 2}} dA $$

其中，$a > (d - 1) / 2$ 且 $A > 0$ 为正定矩阵。上式可写为更加友好的形式：

$$\Gamma_d(a) = \pi^{d(d - 1) / 4} \prod_{i = 1}^d \Gamma(a - (i - 1) / 2)$$

对上式取对数：

$$\log \Gamma_d(a) = \frac{d(d - 1)}{4} \log \pi + \sum_{i = 1}^d \log \Gamma(a - (i - 1) / 2)$$

许多概率分布是用伽马函数定义的——如：伽马分布、贝塔分布、狄利克雷分布(Dirichlet distribution)、卡方分布(Chi-squared distribution)、学生t-分布(Student’s t-distribution)等。对数据科学家、机器学习工程师、科研人员来说，伽马函数可能是应用最广泛的函数之一。它的多元对数形式，同样对于在多元统计中非常有用。而通过取对数的形式，产生的多元对数伽马函数则是其最为有用的等价形式。

## 2、功能目标

实现多元对数伽马函数的计算：

paddle.multigammaln 作为独立的函数调用，非 inplace
paddle.multigammaln_ 作为独立的函数，inplace 地修改输入；
Tensor.multigammaln(input, other) 做为 Tensor 的方法使用，非 inplace;
Tensor.multigammaln_(input, other) 做为 Tensor 的方法使用， inplace 修改输入；

## 3、意义

为 Paddle 新增 `paddle.multigammaln` API，提供多元 gamma 函数的对数计算功能。

# 二、飞桨现状

目前飞桨框架并不存在对应的api，可以通过其他的代码实现

```Python
import paddle
import numpy as np

a = paddle.to_tensor(1.0)
d = paddle.to_tensor(2)
pi = paddle.to_tensor(np.pi, dtype="float32")

out = (
    d * (d - 1) / 4 * paddle.log(pi)
    + paddle.lgamma(a - 0.5 * paddle.arange(0, d, dtype="float32")).sum()
)

print(out)
```


# 三、业内方案调研

### 1. Pytorch

在 Pytorch 中使用的 API 格式如下：

`torch.special.multigammaln(input, p, *, out=None)`

- `input` 为 输入计算多元 gamma 函数的Tensor。其中内的值必须大于(p-1)/2，否则结果未知（behavior is undefiend）。
- `p` 为 `int` 类型，输入的维度。

其实现的代码如下

```cpp
Tensor mvlgamma(const Tensor& self, int64_t p) {
  mvlgamma_check(self, p);
  auto dtype = c10::scalarTypeToTypeMeta(self.scalar_type());
  if (at::isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
    // int -> float promotion
    dtype = c10::get_default_dtype();
  }
  Tensor args = native::arange(
      -p * HALF + HALF,
      HALF,
      HALF,
      optTypeMetaToScalarType(dtype),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt());
  args = args.add(self.unsqueeze(-1));
  const auto p2_sub_p = static_cast<double>(p * (p - 1));
  return args.lgamma_().sum(-1).add_(p2_sub_p * std::log(c10::pi<double>) * QUARTER);
}

Tensor& mvlgamma_(Tensor& self, int64_t p) {
  mvlgamma_check(self, p);
  Tensor args = native::arange(
      -p *HALF  + HALF,
      HALF,
      HALF,
      optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt());
  args = args.add(self.unsqueeze(-1));
  const auto p2_sub_p = static_cast<double>(p * (p - 1));
  return self.copy_(args.lgamma_().sum(-1).add_(p2_sub_p * std::log(c10::pi<double>) * QUARTER));
}
```

### 2. TensorFlow

没有找到对应的api

### 3. MindSpore

在 MindSpore 中使用的 API 格式如下：

`mindspore.ops.mvlgamma(input, p)`

- input (Tensor) - 多元对数伽马函数的输入Tensor，支持数据类型为float32和float64。其shape为 (N, *)，其中*为任意数量的额外维度。 input 中每个元素的值必须大于(p-1)/2。

- p (int) - 进行计算的维度，必须大于等于1。

其实现的代码如下
在CPU端前向
```cpp
uint32_t MvlgammaCpuKernel::MvlgammaCheck(CpuKernelContext &ctx) {
  // check input, output and attr not null
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("p"), KERNEL_STATUS_PARAM_INVALID, "Get attr failed.")
  NormalCheck(ctx, 1, 1, {"p"});

  // check input and output datatype as the same
  DataType input_datatype = ctx.Input(0)->GetDataType();
  DataType output_datatype = ctx.Output(0)->GetDataType();
  KERNEL_CHECK_FALSE((input_datatype == output_datatype), KERNEL_STATUS_PARAM_INVALID,
                     "Input data type[%d] must be the same as Output data type[%d].", input_datatype, output_datatype)

  auto attr_value = ctx.GetAttr("p")->GetInt();
  KERNEL_CHECK_FALSE((attr_value >= 1), KERNEL_STATUS_PARAM_INVALID, "p has to be greater than or equal to 1[%lld]",
                     attr_value)  // 已经用GetAttr获取

  KERNEL_LOG_INFO("MvlgammaCpuKernel[%s], input: size[%llu], output: size[%llu].", ctx.GetOpType().c_str(),
                  ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
T MvlgammaCpuKernel::MvlgammaSingle(T &x, const int &p, bool &error) {
  if (!(x > HALF * (p - 1))) {
    error = true;
    KERNEL_LOG_ERROR("All elements of `x` must be greater than (p-1)/2");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const auto p2_sub_p = static_cast<T>(p * (p - 1));
  T output = p2_sub_p * std::log(M_PI) * QUARTER;
  for (int i = 0; i < p; i++) {
    output += Lgamma(x - HALF * i);
  }
  return output;
}

template <typename T>
uint32_t MvlgammaCpuKernel::MvlgammaCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto attr_p = ctx.GetAttr("p")->GetInt();

  auto input0_shape = ctx.Input(0)->GetTensorShape();
  int64_t data_num = input0_shape->NumElements();
  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  if (max_core_num > data_num) {
    max_core_num = data_num;
  }

  bool error = false;
  auto shard_mvlgamma = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      *(output_y + i) = MvlgammaSingle<T>(*(input_x + i), attr_p, error);
    }
  };

  if (max_core_num == 0) {
    KERNEL_LOG_ERROR("max_core_num could not be 0,");
  }
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_mvlgamma),
                      "Mvlgamma Compute failed.");
  if (error == true) {
    return KERNEL_STATUS_PARAM_INVALID;
  } else {
    return KERNEL_STATUS_OK;
  }
}

```

在CPU端后向

```cpp
uint32_t MvlgammaGradCpuKernel::MvlgammaGradCheck(CpuKernelContext &ctx) {
  // check input, output and attr not null
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Input(1)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("p"), KERNEL_STATUS_PARAM_INVALID, "Get attr failed.")
  NormalCheck(ctx, 2, 1, {"p"});

  // check input and output datatype as the same
  DataType input0_type = ctx.Input(0)->GetDataType();
  DataType input1_type = ctx.Input(1)->GetDataType();
  DataType output_type = ctx.Output(0)->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%d] need be same with "
                     "input1 [%d].",
                     input0_type, input1_type)
  KERNEL_CHECK_FALSE((input0_type == output_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%d] need be same with "
                     "output [%d].",
                     input0_type, output_type)

  auto attr_value = ctx.GetAttr("p")->GetInt();
  KERNEL_CHECK_FALSE((attr_value >= 1), KERNEL_STATUS_PARAM_INVALID, "p has to be greater than or equal to 1[%lld]",
                     attr_value)  // 已经用GetAttr获取

  KERNEL_LOG_INFO(
    "MvlgammaGradCpuKernel[%s], input0: size[%llu];"
    "input1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Input(1)->GetDataSize(), ctx.Output(0)->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
T MvlgammaGradCpuKernel::MvlgammaGradSingle(T &y_grad, T &x, const int &p) {
  T output = 0;
  for (int i = 0; i < p; i++) {
    output += Digamma(x - HALF * i);
  }
  output *= y_grad;
  return output;
}

template <typename T>
uint32_t MvlgammaGradCpuKernel::MvlgammaGradCompute(CpuKernelContext &ctx) {
  auto input_y_grad = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_x = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output_x_grad = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto attr_p = ctx.GetAttr("p")->GetInt();

  auto input0_shape = ctx.Input(0)->GetTensorShape();
  int64_t data_num = input0_shape->NumElements();
  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  if (max_core_num > data_num) {
    max_core_num = data_num;
  }

  auto shard_mvlgammagrad = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      *(output_x_grad + i) = MvlgammaGradSingle<T>(*(input_y_grad + i), *(input_x + i), attr_p);
    }
  };

  if (max_core_num == 0) {
    KERNEL_LOG_ERROR("max_core_num could not be 0,");
  }
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_mvlgammagrad),
                      "MvlgammaGrad Compute failed.");
  return KERNEL_STATUS_OK;
}

```
在gpu端
前向
```cu
template <typename T>
__global__ void Mvlgamma(const size_t size, const T *input, const int p, T *output, int *valid) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T input_val = input[pos];
    
    if (input_val <= (0.5 * (p - 1))) {
      *valid = static_cast<int>(pos);
      return;
    }
    T temp = 0;
    for (int i = 1; i <= p; i++) {
      temp += lgamma(input_val - static_cast<T>((i - 1) * 0.5));
    }
    output[pos] = temp + static_cast<T>(p * (p - 1) * 0.25 * log(M_PI));
  }
  return;
}

template <typename T>
int CalMvlgamma(int *valid, const size_t size, const T *input, const int p, T *output, const uint32_t &device_id,
                cudaStream_t cuda_stream) {
  int host_valid = -1;
  int thread_num = size > 256 ? 256 : size;
  cudaMemsetAsync(valid, -1, sizeof(int), cuda_stream);

  Mvlgamma<<<CUDA_BLOCKS_CAL(device_id, size, thread_num), thread_num, 0, cuda_stream>>>(size, input, p, output, valid);
  cudaMemcpyAsync(&host_valid, valid, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
  cudaStreamSynchronize(cuda_stream);
  return host_valid;
}

```
后向
```cu
__constant__ double kLanczosCoefficientsd[8] = {
  676.520368121885098567009190444019, -1259.13921672240287047156078755283,
  771.3234287776530788486528258894,   -176.61502916214059906584551354,
  12.507343278686904814458936853,     -0.13857109526572011689554707,
  9.984369578019570859563e-6,         1.50563273514931155834e-7};
template <typename T>
__device__ __forceinline__ T CalNumDivDenom(T x) {
  T num = 0;
  T denom = 0.99999999999980993227684700473478;
  for (int j = 0; j < 8; ++j) {
    num -= kLanczosCoefficientsd[j] / ((x + j + 1) * (x + j + 1));
    denom += kLanczosCoefficientsd[j] / (x + j + 1);
  }
  return num / denom;
}
template <typename T>
__global__ void MvlgammaGrad(const size_t size, const T *y_grad, const T *x, const int p, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T kLanczosGamma = 7;
    T log_lanczos_gamma_plus_one_half = log(7.5);
    T temp = 0;
    T cur_input = 0;
    T num_div_denom = 0;
    for (int i = 0; i < p; i++) {
      cur_input = x[pos] - 0.5 * i;
      if (cur_input < 0 && cur_input == floor(cur_input)) {
        temp += std::numeric_limits<T>::quiet_NaN();
        break;
      }
      if (cur_input < 0.5) {
        num_div_denom = CalNumDivDenom(-cur_input);
        temp += (log_lanczos_gamma_plus_one_half + log1pf((-cur_input) / (kLanczosGamma + 0.5))) + num_div_denom -
                kLanczosGamma / (kLanczosGamma + 0.5 - cur_input);
        temp -= PI / tan(PI * (cur_input + abs(floor(cur_input + 0.5))));
      } else {
        num_div_denom = CalNumDivDenom(cur_input - 1);
        temp += (log_lanczos_gamma_plus_one_half + log1pf((cur_input - 1) / (kLanczosGamma + 0.5))) + num_div_denom
                - kLanczosGamma / (kLanczosGamma + 0.5 + cur_input - 1);
      }
    }
    output[pos] = temp * y_grad[pos];
  }
}

template <typename T>
void CalMvlgammaGrad(const size_t size, const T *y_grad, const T *x, const int p, T *output, const uint32_t &device_id,
                     cudaStream_t cuda_stream) {
  int thread_num = 256 < size ? 256 : size;
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(static_cast<int>(((size - 1) / thread_num) + 1), max_blocks);
  MvlgammaGrad<<<block_num, thread_num, 0, cuda_stream>>>(size, y_grad, x, p, output);
  return;
}

template
CUDA_LIB_EXPORT void CalMvlgammaGrad<float>(const size_t size, const float *y_grad, const float *x, const int p,
                                            float *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalMvlgammaGrad<double>(const size_t size, const double *y_grad, const double *x, const int p,
                                             double *output, const uint32_t &device_id, cudaStream_t cuda_stream);
```

# 四、对比分析

目前在pytorch中采用native::arange的方式去实现的话，没有在paddle中找到对应的实现。比较难以实现。
参考MindSpore的算子开发方式的话，可以参考对应的算子开发编写对应的cpu和gpu算子，编写对应的前向和后向算子。

# 五、设计思路与实现方案

## 命名与参数设计

API设计为 `paddle.multigammaln(x, p)`。其中，`x` 为 `Tensor` 类型，是多元 gamma 函数的变量，其中内的值必须大于(p-1)/2，否则结果未知（behavior is undefiend）。支持float32、float64。
`p` 为 `int` 类型，是多元 gamma 函数的积分空间的维度，p≥1。

`paddle.multigammaln_(x, p)` 为 inplace 版本。`Tensor.multigammaln(p)` 为 Tensor 的方法版本。`Tensor.multigammaln_(p)` 为 Tensor 的 方法 inplace 版本。

## 底层OP设计

Kernel部分CPU实现添加在 `paddle/phi/kernels/cpu/multigammaln_kernel.cc` 和 `paddle/phi/kernels/cpu/multigammaln_grad_kernel.cc`。
Kernel部分GPU实现添加在 `paddle/phi/kernels/gpu/multigammaln_kernel.cu` 和 `paddle/phi/kernels/gpu/multigammaln_grad_kernel.cu`。
实现上可以参考MindSpore,但是在实践上，MindSpore的digamma和lgamma都是采用自行编写的函数来实现，而paddle目前有的算子中采用的都是用Eigen::numext的的函数，根据[paddle中lgamma的情况](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/lgamma_cn.html)应该也只能支持到float32、float64。

## API实现方案
根据要求需要实现multigammaln的inplace版本和非inplace版本，并且需要将 API 绑定为 Tensor 的。

# 六、测试和验收的考量
参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

可考虑一下场景：

- 输出数值结果的一致性和数据类型是否正确，使用 scipy 作为参考标准
- 对不同 dtype 的输入数据 `input` 和 `p` 进行计算精度检验，与PyTorch保持一致。并对空张量进行考察
- 输入输出的容错性与错误提示信息
- 输出 Dtype 错误或不兼容时抛出异常
- 保证调用属性时是可以被正常找到的
- 覆盖静态图和动态图测试场景
  
# 七、可行性分析和排期规划
方案主要根据相关数学原理并参考 PyTorch 的工程实现方法，工期上可以满足在当前版本周期内开发完成。

# 八、影响面
由于采用独立的模块开发，对其他模块是否有影响。

# 名词解释
无
# 附件及参考资料
无