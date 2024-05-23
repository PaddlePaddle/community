# paddle.normal / paddle.distribution.Normal / paddle.nn.initializer.Normal 支持复数正态分布设计文档

|API名称 | paddle.normal / paddle.distribution.Normal / paddle.nn.initializer.Normal | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | NKNaN | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2024-05-06 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20240506_api_design_for_complex_gaussian_distribution.md<br> | 


# 一、概述
## 1、相关背景
复值随机变量在许多科学领域都有应用，如信号处理、无线电工程和大气物理学，因此希望对 paddle 目前已有的正态分布 API 添加复数支持。详见：[NO.31 paddle Normal 分布支持复数](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/【Hackathon%206th】开源贡献个人挑战赛框架开发任务合集.md#no31-paddle-normal-分布支持复数)

复正态分布随机变量的数学表示：
若 $Z = X + iY$ ，且 $X \sim N(\mu, \sigma^2)$ , $Y \sim N(\mu, \sigma^2)$ ， $X$ 与 $Y$ 相互独立，则 $Z \sim CN(\mu+i\mu, 2\sigma^2)$  
记 $\mu+i\mu = \mu_z \in \mathbb{C}$ ， $2\sigma^2 = \sigma_z^2 \in \mathbb{R}$ ，则其概率密度函数为：

$$ p_Z(z) = \frac{1}{\pi\sigma_z^2}\exp[-\frac{(z - \mu_z)^2}{\sigma_z^2}] $$

## 2、功能目标
针对涉及正态分布采样方法的 API - paddle.normal，paddle.nn.initializer.Normal 等，即调用底层 _C_ops.gaussian 的 API，需要统一对采样方法增加复数支持；
针对 paddle.distribution.Normal 的 mean、varaince、probs、log_probs、entropy、kl_divergence 方法也需要添加对应支持。

## 3、意义
支持复正态分布，丰富 Paddle 能够提供的分布类型，进一步完善 Paddle 框架。

# 二、飞桨现状
Paddle 框架内定义了复数类型，也有基本的实变量正态分布相关 API。

# 三、业内方案调研

Numpy 或 PyTorch 暂无直接生成复正态分布随机数的 API，可以通过一定的数量关系使用 view 方法生成复正态分布随机数，案例如下：

- 以 PyTorch 为例，若需要生成 10 个相互独立的复正态分布随机数，且均值为 0，方差为 1：

```python
n = 10
mu = torch.tensor([0.]*n*2).reshape([n,2])
std = torch.tensor([math.sqrt(2)/2]*n*2).reshape([n,2])
a = torch.normal(mu,std)
print(a)
# tensor([[ 0.6172, -0.3429],
#       [ 0.8868, -0.2534],
#       [ 0.5567, -0.1783],
#       [-0.0548, -0.8767],
#       [-0.4670,  0.4014],
#       [-0.0194, -0.6066],
#       [-0.8244, -0.1279],
#       [-0.2216, -0.9890],
#       [ 0.1335, -0.2122],
#       [-0.3304,  0.0459]])
print(a.view(torch.complex64))
# tensor([[ 0.6172-0.3429j],
#         [ 0.8868-0.2534j],
#         [ 0.5567-0.1783j],
#         [-0.0548-0.8767j],
#         [-0.4670+0.4014j],
#         [-0.0194-0.6066j],
#         [-0.8244-0.1279j],
#         [-0.2216-0.9890j],
#         [ 0.1335-0.2122j],
#         [-0.3304+0.0459j]])
```

- 检测方差是否为1：

```python
n = 10000
mu = torch.tensor([0.]*n*2).reshape([n,2])
std = torch.tensor([math.sqrt(2)/2]*n*2).reshape([n,2])
a = torch.normal(mu,std)
a = a.view(torch.complex64)
print(torch.var(a))
# tensor(0.9985)
```


# 四、对比分析

由于复正态分布随机数的实部与虚部有一定的数量关系，都可以通过实正态分布的采样得到，所以可以利用组合现有 API 进行实现。

# 五、设计思路与实现方案

## 命名与参数设计

不涉及新增 API

复正态分布随机数采样相关 API 的修改部分：
1. paddle.gaussian 的 `mean` 参数添加支持 complex 类型，`dtype` 参数添加支持 complex64 和 complex128
2. paddle.stand_normal `dtype` 参数添加支持 complex64 和 complex128，当 `dtype` 为复数类型时调用 paddle.gaussian(dtype=复数类型)
3. paddle.normal 的 `mean` 参数添加支持 complex、dtype 为 complex64 或 complex128 的 Tensor，当 `mean` 为复数类型时调用 paddle.gaussian(dtype=复数类型) 或 paddle.stand_normal(dtype=复数类型)
4. paddle.randn 的 `dtype` 参数添加支持 complex64 和 complex128，当 `dtype` 为复数类型时调用 paddle.stand_normal(dtype=复数类型)
5. paddle.normal_(paddle.gaussian_) 单独修改，修改方法和 paddle.gaussian 类似
6. paddle.nn.initializer.NormalInitializer 的 `loc` 参数添加支持 complex 类型，当 `loc` 为 complex 类型时修改 `forward` 方法以支持复数类型
7. paddle.distribution.Normal 的 `loc` 参数添加支持 complex、dtype 为 complex64 或 complex128 的 Tensor，当 `loc` 为 complex 类型时修改 sample、rsample、mean、varaince、probs、log_probs、entropy、kl_divergence 方法

## 底层OP设计

增加 GaussianKernel、GaussianInplaceKernel 以及对应反向算子的复数类型特化：

以 GaussianKernel 为例：
```cpp
#include "paddle/phi/common/complex.h"

// If T is complex
template <
    typename T,
    typename Context,
    std::enable_if_t<std::is_same<T, phi::dtype::complex<float>>::value ||
                         std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
void GaussianKernel(const Context& dev_ctx,
                    const IntArray& shape,
                    std::complex<float> mean,
                    float std,
                    int seed,
                    DataType dtype,
                    DenseTensor* out) {
  auto tensor = out;

  tensor->Resize(common::make_ddim(shape.GetData()));
  int64_t size = tensor->numel();
  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }

  float mean_real = mean.real();
  std = std::sqrt(std::pow(std, 2)/2);
  std::normal_distribution<T> dist(mean_real, std);
  
  T* data = dev_ctx.template Alloc<T>(tensor);
  for (int64_t i = 0; i < size; ++i) {
    phi::dtype::Real<T> x = dist(*engine);
    phi::dtype::Real<T> y = dist(*engine);
    data[i] = T(x, y);
  }

}
```

## API实现方案

paddle.gaussian 支持复数类型：
```python
def gaussian(shape, mean=0.0, std=1.0, seed=0, dtype=None, name=None):
    op_type_for_check = 'gaussian/standard_normal/randn/normal'
    supported_dtypes = ['float32', 'float64', 'float16', 'uint16', 'bfloat16', 'complex64', 'complex128']
    complex_gaussian = False

    if dtype is None:
        dtype = paddle.framework.get_default_dtype()
        if dtype not in supported_dtypes:
            raise TypeError(
                f"{op_type_for_check} only supports {supported_dtypes}, but the default dtype is {dtype}"
            )
    if not isinstance(dtype, (core.VarDesc.VarType, core.DataType)):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if dtype in ['complex64', 'complex128']:
      if (not isinstance(mean, complex)) or (mean.real != mean.imag):
          raise ValueError(
            f"if dtype is {dtype}, mean should be a complex number with real 
            part equal imag part, but got {mean}"
          )
      complex_gaussian = True
      mean = mean.real
      dtype = 'float32' if dtype == 'complex64' else 'float64'

    if in_dynamic_or_pir_mode():
        shape = paddle.utils.convert_shape_to_list(shape)
        place = _current_expected_place()
        return _C_ops.gaussian(
            shape, float(mean), float(std), seed, dtype, complex_gaussian, place
        )
    else:
        check_shape(shape, op_type_for_check)
        check_dtype(dtype, 'dtype', supported_dtypes, op_type_for_check)
        inputs = {}
        attrs = {
            'mean': mean,
            'std': std,
            'seed': seed,
            'dtype': dtype,
            'complex_gaussian': complex_gaussian,
        }
        paddle.utils.get_shape_tensor_inputs(
            inputs=inputs, attrs=attrs, shape=shape, op_type=op_type_for_check
        )

        helper = LayerHelper('gaussian', **locals())
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='gaussian_random',
            inputs=inputs,
            outputs={'Out': out},
            attrs=attrs,
        )
        out.stop_gradient = True
        return out
        
```

paddle.nn.initializer.NormalInitializer 支持复数类型：
```python
class NormalInitializer(Initializer):
    def __init__(self, loc=0.0, scale=1.0, seed=0):
        assert loc is not None
        assert scale is not None
        assert seed is not None
        super().__init__()
        self._mean = loc
        self._std_dev = scale
        self._seed = seed
        self._complex_gaussian = False
        if isinstance(self._mean, complex):
            if self._mean.real != self._mean.imag:
                raise ValueError(
                  f"if mean is a complex number, its real 
                  part should equal imag part, but got {self._mean}"
                )
            self._complex_gaussian = True
            self._mean = self._mean.real

    def forward(self, var, block=None):
        assert not (
            isinstance(var, framework.EagerParamBase) and var.is_dist()
        ), "Currently, normal initializer not support lazy init for dist param."
        block = self._check_block(block)

        assert isinstance(block, (framework.Block, pir.Block))

        check_variable_and_dtype(
            var,
            "Out",
            ["uint16", "float16", "float32", "float64", "complex64", "complex128"],
            "guassian_random",
        )

        if self._seed == 0:
            self._seed = block.program.random_seed

        if var.dtype == "complex64":
            dtype = "float32"
        elif var.dtype == "complex128":
            dtype = "float64"
        else:
            dtype = var.dtype
        if in_dygraph_mode():
            place = _current_expected_place()
            out_var = _C_ops.gaussian(
                var.shape,
                self._mean,
                self._std_dev,
                self._seed,
                dtype,
                self._complex_gaussian,
                place,
            )
            return None
        elif in_pir_mode():
            place = _current_expected_place()
            out_var = _C_ops.gaussian(
                var.shape,
                self._mean,
                self._std_dev,
                self._seed,
                dtype,
                self._complex_gaussian,
                place,
            )
            return out_var
        else:
            op = block.append_op(
                type="gaussian_random",
                outputs={"Out": var},
                attrs={
                    "shape": var.shape,
                    "dtype": dtype,
                    "mean": self._mean,
                    "std": self._std_dev,
                    "seed": self._seed,
                    "complex_gaussian": self._complex_gaussian,
                },
                stop_gradient=True,
            )
            var.op = op
            return op
```

paddle.distribution.Normal 支持复数类型：
```python
def __init__(self, loc, scale, name=None):
    ...
    self._complex_gaussian = False
    if isinstance(loc, complex) or (isinstance(loc, np.adarray) and loc.dtype in [np.complex64, np.complex128]) or (isinstance(loc, paddle.Tensor) and loc.dtype in [np.complex64, np.complex128]):
        if loc.real != loc.imag:
          raise ValueError(
            f"if loc is a complex number, its real 
            part should equal imag part, but got {loc}""
          )
        self.dtype = 'float32' if dtype == 'complex64' else 'float64'
        self._complex_gaussian = True


def sample(self, shape=(), seed=0):
    if not isinstance(shape, Iterable):
        raise TypeError('sample shape must be Iterable object.')

    if not in_dynamic_mode():
        check_type(seed, 'seed', (int), 'sample')

    shape = list(shape)
    batch_shape = list((self.loc + self.scale).shape)
    name = self.name + '_sample'
    if -1 in batch_shape:
        output_shape = shape + batch_shape
        fill_shape = list(batch_shape + shape)
        fill_shape[0] = paddle.shape(self.loc + self.scale)[0].item()
        zero_tmp = paddle.full(fill_shape, 0.0, self.dtype)
        zero_tmp_reshape = paddle.reshape(zero_tmp, output_shape)

        zero_tmp_shape = paddle.shape(zero_tmp_reshape)
        normal_random_tmp = random.gaussian(
            zero_tmp_shape,
            mean=(0.0+0.0j) if self._complex_gaussian else 0.0,
            std=1.0,
            seed=seed,
            dtype=self.dtype
        )
        output = normal_random_tmp * (zero_tmp_reshape + self.scale)
        output = paddle.add(output, self.loc, name=name)
        return output
    else:
        output_shape = shape + batch_shape
        output = random.gaussian(
            output_shape,
            mean=(0.0+0.0j) if self._complex_gaussian else 0.0,
            std=1.0,
            seed=seed,
            dtype=self.dtype
        ) * (paddle.zeros(output_shape, dtype=self.dtype) + self.scale)
        output = paddle.add(output, self.loc, name=name)
        if self.all_arg_is_float:
            return paddle.reshape(output, shape, name=name)
        else:
            return output

def rsample(self, shape=()):
    if not isinstance(shape, Iterable):
        raise TypeError('sample shape must be Iterable object.')

    shape = self._extend_shape(tuple(shape))
    eps = paddle.normal(
        mean=(0.0+0.0j) if self._complex_gaussian else 0.0,
        shape=shape
    )
    return self.loc + eps * self.scale

def entropy(self):
        r"""Shannon entropy in nats.

        If non-complex, the entropy is

        .. math::

            entropy(\sigma) = 0.5 \log (2 \pi e \sigma^2) + 0.5

        If complex gaussian:

        .. math::

            entropy(\sigma) = \log (\pi e \sigma^2) + 1

        In the above equation:

        * :math:`scale = \sigma`: is the std.

        Returns:
            Tensor, Shannon entropy of normal distribution.The data type is float32.

        """
        name = self.name + '_entropy'
        batch_shape = list((self.loc + self.scale).shape)
        if -1 in batch_shape:
            fill_shape = list(batch_shape)
            fill_shape[0] = paddle.shape(self.loc + self.scale)[0].item()
            fill_dtype = (self.loc + self.scale).dtype
            zero_tmp = paddle.full(fill_shape, 0.0, fill_dtype)
        else:
            zero_tmp = paddle.full(batch_shape, 0.0, self.dtype)
        if self._complex_gaussian:
            return paddle.add(
                1.0 + zero_tmp,
                math.log(math.pi) + 2.0 * paddle.log(self.scale + zero_tmp),
                name=name,
            )
        else:
            return paddle.add(
                0.5 + zero_tmp,
                0.5 * math.log(2 * math.pi) + paddle.log(self.scale + zero_tmp),
                name=name,
            )
```
entropy:

$$ \begin{aligned}
H(Z) & = - \int_z p(z)\log p(z) dz \\
& = - \mathbb{E}[\log p(z)] \\
& = -  \mathbb{E}[-\log(\pi \sigma_z^2)-\frac{1}{\sigma_z^2}|z-\mu_z|^2] \\
& = \log(\pi \sigma_z^2) + \frac{1}{\sigma_z^2}\mathbb{E}[|z-\mu_z|^2] \\
& = \log(\pi \sigma_z^2) + 1
\end{aligned}
$$

```python
def log_prob(self, value):
    name = self.name + '_log_prob'
    value = self._check_values_dtype_in_probs(self.loc, value)

    var = self.scale * self.scale
    log_scale = paddle.log(self.scale)
    if self._complex_gaussian:
        return paddle.subtract(
            -1.0 * ((value - self.loc).conj() * (value - self.loc)) / (var),
            2.0 * log_scale + math.log(math.pi),
            name=name,
        )
    else:
        return paddle.subtract(
            -1.0 * ((value - self.loc) * (value - self.loc)) / (2.0 * var),
            log_scale + math.log(math.sqrt(2.0 * math.pi)),
            name=name,
        )

def probs(self, value):
    name = self.name + '_probs'
    value = self._check_values_dtype_in_probs(self.loc, value)

    var = self.scale * self.scale
    if self._complex_gaussian:
        return paddle.divide(
            paddle.exp(
                -1.0 * ((value - self.loc).conj() * (value - self.loc)) / (var)
            ),
            (math.pi * var),
            name=name,
        )
    else:
        return paddle.divide(
            paddle.exp(
                -1.0 * ((value - self.loc) * (value - self.loc)) / (2.0 * var)
            ),
            (math.sqrt(2 * math.pi) * self.scale),
            name=name,
        )

def kl_divergence(self, other):
    r"""The KL-divergence between two normal distributions.

    If non-complex, the KL-divergence is

    .. math::

        KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = 0.5 (ratio^2 + (\frac{diff}{\sigma_1})^2 - 1 - 2 \ln {ratio})

    If complex gaussian:

    .. math::

        KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = ratio^2 + (\frac{diff}{\sigma_1})^2 - 1 - 2 \ln {ratio}

    .. math::

        ratio = \frac{\sigma_0}{\sigma_1}

    .. math::

        diff = \mu_1 - \mu_0

    In the above equations:

    * :math:`loc = \mu_0`: is the mean of current Normal distribution.
    * :math:`scale = \sigma_0`: is the std of current Normal distribution.
    * :math:`loc = \mu_1`: is the mean of other Normal distribution.
    * :math:`scale = \sigma_1`: is the std of other Normal distribution.
    * :math:`ratio`: is the ratio of scales.
    * :math:`diff`: is the difference between means.

    Args:
        other (Normal): instance of Normal.

    Returns:
        Tensor, kl-divergence between two normal distributions.The data type is float32.

    """
    if not in_dynamic_mode():
        check_type(other, 'other', Normal, 'kl_divergence')

    name = self.name + '_kl_divergence'
    var_ratio = self.scale / other.scale
    var_ratio = var_ratio * var_ratio
    t1 = (self.loc - other.loc) / other.scale
    t1 = t1 * t1
    if self._complex_gaussian:
        return paddle.add(
            var_ratio, (t1 - 1.0 - paddle.log(var_ratio)), name=name
        )
    else:
        return paddle.add(
            0.5 * var_ratio, 0.5 * (t1 - 1.0 - paddle.log(var_ratio)), name=name
        )
```
KL-divergence:

$$ \begin{aligned}
\mathcal{D}\_{KL}(p || q) & = \int_z p(z)\log \frac{p(z)}{q(z)} dz \\
& = \mathbb{E}\_p\[\log \frac{p(z)}{q(z)}\] \\
& = \mathbb{E}\_p\[\log\[\frac{\sigma_q^2}{\sigma_p^2} \exp(-\frac{|z-\mu_p|^2}{\sigma_p^2}+\frac{|z-\mu_q|^2}{\sigma_q^2})\]\] \\
& = -\log \frac{\sigma_p^2}{\sigma_q^2} - \frac{1}{\sigma_p^2}\mathbb{E}\_p(z-\mu_p)^2 + \frac{1}{\sigma_q^2}\mathbb{E}\_p(z-\mu_q)^2\\
& = -2\log \frac{\sigma_p}{\sigma_q} - 1 + \frac{1}{\sigma_q^2}\mathbb{E}\_p(z^cz -\mu_q^cz-z^c\mu_q+\mu_q^c\mu_q) \\
& = -2\log \frac{\sigma_p}{\sigma_q} - 1 + \frac{1}{\sigma_q^2}(\sigma_p^2+\mu_p^c\mu_p -\mu_q^c\mu_p-\mu_p^c\mu_q+\mu_q^c\mu_q) \\
& = -2\log \frac{\sigma_p}{\sigma_q} - 1 + \frac{1}{\sigma_q^2}(\sigma_p^2+|\mu_p-\mu_q|^2) \\
& = \frac{\sigma_p^2}{\sigma_q^2} - 1 + \frac{|\mu_p-\mu_q|^2}{\sigma_q^2}-2\log \frac{\sigma_p}{\sigma_q}
\end{aligned} $$

# 六、测试和验收的考量

1. 针对 paddle.gaussian、paddle.nn.initializer.NormalInitializer 和 paddle.distribution.Normal 的采样方法，生成5000个样本，测试这些这样的均值和方差是否正确：例如设置采样的复高斯分布均值为 0.0+0.0j，方差为 1.0，则需要测试得到样本的实部均值是否接近 0.0，虚部均值是否接近 0.0，实部方差是否接近 0.5，虚部方差是否接近 0.5，以及实部虚部总体方差是否接近 1.0。 

2. 由于numpy或scipy没有实现复高斯分布，paddle.distribution.Normal 的 `kl_divergence`、`entropy`、`prob`、`log_prob` 方法在修改支持复高斯分布后，需要分别用 numpy 重写对应公式进行验证。


# 七、可行性分析和排期规划
- 排期规划

1. 修改 c++ gaussian kernel
2. 修改 paddle.gaussian、paddle.stand_normal
3. 修改 paddle.normal、paddle.randn
4. 修改 paddle.normal_(paddle.gaussian_)
5. 修改 paddle.nn.initializer.NormalInitializer
6. 修改 paddle.distribution.Normal

# 八、影响面
除以上提及的API外不影响其他模块

# 名词解释

# 附件及参考资料
1. [复高斯分布wiki](https://en.wikipedia.org/wiki/Complex_normal_distribution)
