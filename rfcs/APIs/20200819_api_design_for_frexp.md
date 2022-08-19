# paddle.Tensor.quantile 设计文档

|API名称 | paddle.frexp                        | 
|---|-------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | 郑必城                                 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-08-19                          | 
|版本号 | V1.0                                | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | v2.2.0                              | 
|文件名 | 20200819_api_design_for_frexp.md<br> | 

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，支持科学计算领域API，Paddle需要扩充API`paddle.frexp`

## 2、功能目标

增加API`paddle.frexp`，实现将输入分解为尾数张量和指数张量。返回(尾数张量， 指数张量)，其中 x = 尾数 * 2**
指数。尾数位于开区间 (-1, 1) 中，而二进制指数是有符号整数。

## 3、意义

Paddle 将可以使用 `paddle.frexp` 进行浮点数分解，丰富了API。

# 二、飞桨现状

- 目前paddle缺少相关功能实现，因此可以使用c，或者几个飞桨api算子结合的办法来进行实现。

# 三、业内方案调研

## Pytorch

在 PyTorch 中存在类似api，名字为`torch.frexp(input, *, out=None)`，该api通过调用底层c++实现。看开发任务，不涉及c++，因此不参考pytorch的实现

## python

python自带了frexp函数，在math包下，也是通过调用底层api接口实现

## numpy

Numpy 中也有对应的 API `numpy.frexp()`，一样通过调用底层c++接口实现

### 实现方法

Pytorch，python，numpy为了追了更快的速度都使用调用c++的方案来实现frexp函数，也找不到相关代码。上google搜了一下类似的实现代码，有一段代码是java实现
大致过程是对输入数据转换成进行位移操作，减去阶码，判断正负最后输出。但是这一点也不面向对象，有悖python语言的初衷，因此我根据飞桨的api涉及了新的算法。

```text
public class FrexpDemo {
 
	public static FRexpResult frexp(double value) {
		final FRexpResult result = new FRexpResult();
		long bits = Double.doubleToLongBits(value);
		double realMant = 1.;
 
		// Test for NaN, infinity, and zero.
		if (Double.isNaN(value) || value + value == value || Double.isInfinite(value)) {
			result.exponent = 0;
			result.mantissa = value;
		} else {
 
			boolean neg = (bits < 0);
			int exponent = (int) ((bits >> 52) & 0x7ffL);
			long mantissa = bits & 0xfffffffffffffL;
 
			if (exponent == 0) {
				exponent++;
			} else {
				mantissa = mantissa | (1L << 52);
			}
 
			// bias the exponent - actually biased by 1023.
			// we are treating the mantissa as m.0 instead of 0.m
			// so subtract another 52.
			exponent -= 1075;
			realMant = mantissa;
 
			// normalize
			while (realMant > 1.0) {
				mantissa >>= 1;
				realMant /= 2.;
				exponent++;
			}
 
			if (neg) {
				realMant = realMant * -1;
			}
 
			result.exponent = exponent;
			result.mantissa = realMant;
		}
		return result;
	}
 
	public static void main(String[] args) {
		FRexpResult r = frexp(18);
		System.out.println(r);
	}
}
 
class FRexpResult {
	public int exponent = 0;
	public double mantissa = 0.;
	@Override
	public String toString() {
		return String.format("mantissa=%f,exponent=%d", mantissa,exponent);
	}
}


```

# 四、对比分析

- 使用场景与功能：pytorch和Numpy都支持多维，这里模仿他们的实现逻辑，使用paddle.frexp(x)，返回尾数张量，指数张量。

# 五、方案设计

## 命名与参数设计

### 函数命名

API设计为`paddle.frexp(x)`，返回尾数张量位于开区间 (-1, 1)，返回二进制指数是有符号整数张量。

x为输入张量，y为输出张量



## 底层OP设计

不涉及底层op

## API实现方案

**思路来源**
参考[paddle炼丹师提交的提案](https://github.com/PaddlePaddle/community/pull/180/files/96ff9847d01a28e16fa455c40aad450f2bffb511#diff-a1cb961065ef85e96f4f68364a77eedc2066171fb04574de1cb2e1cceb424564)
中tizhou86同学的建议，使用paddle log以及devide组合的方式来实现对应的功能，面向对象，python NO1。

**实现细节**
* 大致方法: 使用对输入数据取log2，利用devide广播除对应的数字。
* 注意点: 输入值含有0元素时，警告，并且将返回值中该元素对应位置设置为0

# 六、测试和验收的考量

和numpy结果的数值的一致性

# 七、可行性分析及规划排期

实现阶段：实现方案较为简单，可以快速实现
书写中英文文档：英文文档需要花费更多时间，但仍在合理范围之内

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
