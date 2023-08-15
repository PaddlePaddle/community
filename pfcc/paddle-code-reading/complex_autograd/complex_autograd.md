
本文档主要介绍如何计算复数梯度，详细推导证明可见[附录 Wertinger Calculus](./Wertinger_Calculus.md)

## 实函数微分
实函数(Real Function)指定义域(domain)和值域(range)或对应域(co-domain)均为实数集的子集的函数。是函数的特性之一是可以在坐标平面上画出图形。
对于一个函数 $f$ , 其定义域和值域都是实数集的子集，如果 $f(x)$ 在 $x_0$ 点的左极限和右极限都存在，且相等，那么函数在 $f$ 在 $x_0$ 可微。函数 $f$ 在该点的导数定义为：

$$
\lim_{h \rightarrow 0}\frac{f(x_{0} + h) - f(x_{0})}{h}
$$

对于实变函数，解析函数(Analytic Function) 是一个比可微更强的条件。对于定义实变函数 $f: D \rightarrow \mathbb{R}$ , 在 $x_{0}$ 点的邻域实解析 (real analytic) 的定义是 $f$ 在这个领域内可以表示为下述的收敛幂级数。

$$
f(x) = \sum_{n=0}^{\infty} a_{n}(x-x_{0})^{n}
$$

其中 $a_{i}, i=0, 1, 2,...$ 均为实数
这同时意味着它无穷可微（这个特性又称为：光滑 Smooth）。但值得注意的是光滑函数不一定是解析函数

## 复函数微分
一般意义的复函数，指定义域和值域均为复数集的子集的函数。
对于一个复函数 $f$ , 其定义与和值域都是复数集的子集。直观的理解复函数的可微性：如果 $f$ 在 $z_0$ 点的一个领域内，从任何方向，以任意方式趋近 $z_0$, 下述极限都存在，且都相等。

$$
\lim_{h \rightarrow 0}\frac{f(z_{0} + h) - f(z_{0})}{h}
$$

如果以上的条件满足，那么就认为 $f$ 在 $z_{0}$ 的邻域可微 (differentiable)。这虽然是实数情形的直接扩展，也符合“变化率之比的极限”这一本质，但是复函数的可微是远比实函数的可微更强的条件。
对于复函数来说，可微同时也意味着复解析 (complex analytic), 全纯 (holomorphic) 或者正则 (regular).复解析 (complex analytic)：函数 $f:U \rightarrow \mathbb{C}$  在定义域上的某个点 $z_{0}$ 的领域上可以表示为下述收敛幂级数，则在这一点复解析。

$$
f(z) = \sum_{n=0}^{\infty}c_{n}(z - z_{n})^n
$$

其中 $c_{i}, i=0, 1, 2, ...$ 均为复数, $U$ 为复平面的一个开子集。
在整个复平面都全纯的函数，被称为整函数 (Entire function)。复函数可以被拆分为实部和虚部两个函数，它们都是 z 的实部(x)和虚部(y) 的函数。
亦即可以把一个 $\mathbb{C} \rightarrow \mathbb{C}$ 的函数表示为 $\mathbb{R}^{2} \rightarrow \mathbb{R}^{2}$

$f(z) = f(x+jy) \triangleq u(x, y) + jv(x, y), z = x + jy$

可以证明，如果复函数 $f$ 需要满足以上的 holomorphic 条件，则需要满足柯西-黎曼方程（Cauchy-Riemann differential equations) ,这是一个更严格的条件。

$$\frac{\partial u(x, y)}{\partial x} = \frac{\partial v(x, y)}{\partial y}$$

$$\frac{\partial v(x, y)}{\partial x} = -\frac{\partial u(x, y)}{\partial y}$$

非全纯函数，从数学的角度来说就是不可微分了。但这种条件下仍然有值得研究的问题。首先我们从定义在复数域上的实函数开始，因为一般神经网络优化的目标也是一个实数。

$$w = f(z) = u(x, y) , w \in \mathbb{R}, z \in \mathbb{C}, z = x + jy$$

因为 $v(x,y) \equiv 0$ 所以 Cauchy-Riemann 等式一般不成立，因此这些函数一般不是 holomorphic. 只有一种平凡的条件下实函数才能是 holomorphic 的。

$$\frac{\partial u(x, y)}{\partial x} \equiv 0 \\ \frac{\partial u(x, y)}{\partial y} \equiv 0$$

而这种情况下 $f$ 的值只能是一个固定的实数，这种平凡的情形并没有什么研究的价值。对于优化来说，优化的目标总是可以用实数来衡量的，而实函数一般又不全纯(holomorphic), 那么要怎么优化它呢？

## Wertinger Calculus
即使 $f(z)$ 不是全纯的，可以将其重写为二变量函数 $f(z, z^{\*})$ ,总是全纯的。这是因为实部和虚部的组成部分 $z$ 可以表示为 $z$ 和 $z^*$

$$Re(z) = \frac{z + z^*}{2}$$

$$Im(z) = \frac{z - z^*}{2j}$$

可以研究 $f(z, z^{\*})$ ,因为如果 $f$ 是真实可微的,那么这个函数具有偏导数

$$
\frac{\partial }{\partial x}  =  \frac{\partial z}{\partial x} * \frac{\partial }{\partial z} + \frac{\partial z^* }{\partial x} * \frac{\partial }{\partial z^* }
$$

得出

$$
\frac{\partial }{\partial x} = \frac{\partial }{\partial z} + \frac{\partial }{\partial z^*}
$$

$$
\frac{\partial }{\partial y}  =  \frac{\partial z}{\partial y} * \frac{\partial }{\partial z}+ \frac{\partial z^* }{\partial y} * \frac{\partial }{\partial z^* }
$$

得出

$$
\frac{\partial }{\partial y}= j * (\frac{\partial }{\partial z} - \frac{\partial }{\partial z^*} )
$$

通过上面的公式，我们可以得到

$$
\frac{\partial }{\partial z}  =  \frac{1}{2} * (\frac{\partial }{\partial x} - j * \frac{\partial }{\partial y})
$$

$$
\frac{\partial }{\partial z^*} = \frac{1}{2} * (\frac{\partial }{\partial x} + j * \frac{\partial }{\partial y})
$$

这是在wiki上关于Wirtinger calculus的经典定义

## 如何优化？
通常在实数领域优化的计算公式为：

$$x_{n+1} = x_n - \alpha * \frac{\partial L}{\partial x} $$

其中 $\alpha$ 为步长， $L$ 为loss，如何将这个推广到复数空间呢
根据上面Wirtinger calculus的定义，推广到复数空间：

$$
z_{n+1} = x_n - \alpha * \frac{\partial L}{\partial x} + 1j * (y_n - \alpha * \frac{\partial L}{\partial y}) =>
$$

$$
z_{n+1} = z_n - 2 \alpha * \frac{\partial L}{\partial z^*}
$$

可以发现，复数空间优化可以简化为仅用共轭的Wirtinger导数进行优化。
所以pytorch与tensorflow 复数导数结果均为共轭的Wirtinger 导数，而jax为本身的导数，但优化时提供共轭导数

## 共轭Wirtinger导数推导
假设有函数 $s = f(z) = f(x+jy) \triangleq u(x, y) + jv(x, y), z = x + jy$ $\mathbb{C} \rightarrow \mathbb{C}$

$L$ 为最终的loss， $s$ 为 $f(z)$ 的输出,所以这里我们的目标是去计算 $\frac{\partial L}{\partial z^*}$

$$
\frac{\partial L}{\partial z^* } = \frac{\partial L}{\partial u} * \frac{\partial u}{\partial z^* } + \frac{\partial L}{\partial v} * \frac{\partial v}{\partial z^* }
$$

根据上述Wirtinger calculus的经典定义:

$$
\frac{\partial L}{\partial s} = \frac{1}{2} * (\frac{\partial L}{\partial u} - j* \frac{\partial L}{\partial v}) 
$$

$$
\frac{\partial L}{\partial s^* } = \frac{1}{2} * (\frac{\partial L}{\partial u} + j* \frac{\partial L}{\partial v}) 
$$

由于这里的 $L$ 为实数，且 $u$ 和 $v$ 均是实函数,不难发现，他们互为共轭:

$$
(\frac{\partial L}{\partial s})^* = \frac{\partial L}{\partial s^* }
$$

且 $\frac{\partial L}{\partial s^* }$ 为我们反向计算时输入的梯度:

通过上述等式，我们可以得到：

$$
\frac{\partial L}{\partial u} = \frac{\partial L}{\partial s} + \frac{\partial L}{\partial s^* }
$$

$$
\frac{\partial L}{\partial v} = -1j * (\frac{\partial L}{\partial s} - \frac{\partial L}{\partial s^* })
$$

综合上述可得：

$$
\frac{\partial L}{\partial z^* } = (\frac{\partial L}{\partial s} + \frac{\partial L}{\partial s^* }) * \frac{\partial u}{\partial z^* } - 1j * (\frac{\partial L}{\partial s} - \frac{\partial L}{\partial s^* })* \frac{\partial v}{\partial z^* } =>
$$

$$
\frac{\partial L}{\partial z^* } = \frac{\partial L}{\partial s} * (\frac{\partial u}{\partial z^* } + j * \frac{\partial v}{\partial z^* }) + \frac{\partial L}{\partial s^* } * (\frac{\partial u}{\partial z^* } - j * \frac{\partial u}{\partial z^* })=>
$$

$$
\frac{\partial L}{\partial z^* } = \frac{\partial L}{\partial s} * \frac{\partial (u+vj)}{\partial z^* }  + \frac{\partial L}{\partial s^* } * \frac{\partial (u+vj)^* }{\partial z^* } =>
$$

$$
\frac{\partial L}{\partial z^* } = \frac{\partial L}{\partial s} * \frac{\partial s}{\partial z^* }  + \frac{\partial L}{\partial s^* } * \frac{\partial s^* }{\partial z^* } =>
$$

最终我们可以得到:

$$\frac{\partial L}{\partial z^* } = (\frac{\partial L}{\partial s^* })^* * \frac{\partial s}{\partial z^* }  + \frac{\partial L}{\partial s^* } * (\frac{\partial s}{\partial z})^* =(output\_{grad})^* * \frac{\partial s}{\partial z^* } + output\_{grad} * (\frac{\partial s}{\partial z})^*
$$

这个就是我们要计算的梯度。

## 工程实现
假设有函数 $f(z=x+yj) = cz = c(x+yj) = cx+cyj$ $c \in \mathbb{R}$
1. 利用Wirtinger实现

$$\frac{\partial s}{\partial z} = \frac{1}{2} * (\frac{\partial s}{\partial x} - j * \frac{\partial s}{\partial y}) = \frac{1}{2} * (c - (c*1j)*1j) = c$$

$$\frac{\partial s}{\partial z^* } = \frac{1}{2} * (\frac{\partial s}{\partial x} + j * \frac{\partial s}{\partial y}) = \frac{1}{2} * (c + (c*1j)*1j) = 0$$

$$\frac{\partial L}{\partial z^* } = 1 * 0 + 1*c = c $$

2. 但是我们可以注意到如果将 $z ，z^*$ 当作独立的两个变量，那么就很像二元函数的性质，以二元函数的性质计算,证明见:[Wertinger Calculus](./Wertinger_Calculus.md)

   
$$\frac{\partial s}{\partial z} = \frac{\partial (c * z )}{\partial z} =c$$

$$\frac{\partial s}{\partial z^* } = \frac{\partial (c * z )}{\partial z^* } =0$$

1. 类似fft相关的方法，会产生 $\mathbb{C} \rightarrow \mathbb{R}$ 以及 $\mathbb{R} \rightarrow \mathbb{C}$ 的场景
1) 针对 $\mathbb{C} \rightarrow \mathbb{R}$ 的场景，输出变量为实数，则共轭即为本身 $s^* = s$
   
$$\frac{\partial L}{\partial z^* } = 2 * (output\_{grad}) * \frac{\partial s}{\partial z^* }$$

2) 针对 $\mathbb{R} \rightarrow \mathbb{C}$ 的场景，输入变量为实数
   
$$\frac{\partial L}{\partial z^* } = 2 * Re(output\_{grad}^* * \frac{\partial s}{\partial z^* })$$


## 参考文献
1. [The Complex Gradient Operator and the CR-Calculus](https://arxiv.org/abs/0906.4835)
2. [On the Computation of Complex-valued Gradients with Application to Statistically Optimum Beamforming](http://arxiv.org/abs/1701.00392)
3. [Wertinger Calculus](https://onlinelibrary.wiley.com/doi/pdf/10.1002/0471439002.app1)
