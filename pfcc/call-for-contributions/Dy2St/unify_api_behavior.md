# 飞桨 API 动静行为统一

> This project will be mentored by [@2742195759](https://github.com/2742195759) and [@Aurelius84](https://github.com/Aurelius84)
> 

## 一、概要
### 1.背景
飞桨自2.0版本后正式切换到以动态图为默认执行范式，但提供了paddle.enable_static()接口用户一键切换到静态图下执行，得益于飞桨API的「动静行为统一」设计理念。

由于飞桨框架包含众多API和功能，尚存在着少数的API仅支持动态图，影响飞桨框架「动静行为统一」的用户使用体验，亟需对此类API进行升级优化。

### 2.功能目标
升级部分飞桨API功能，扩展支持静态图，提升API的动静行为统一的用户使用体验。

### 3.方案要点
为了确保API行为动静统一，方案要点主要包括：

+ **明确API对静态图支持现状**：需首先梳理和明确动静行为不统一的API列表，典型 Case 为：某个API仅支持动态图，缺少静态图分支
+ **扩展添加静态图逻辑分支**：给相关API添加静态图分支逻辑，以扩展支持静态图行为

## 二、主要工作

### 1. API 列表梳理
从[官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html)来看，paddle的核心API主要包括：paddle根目录、paddle.linalg、paddle.nn、paddle.metric、paddle.vision、paddle.text。因此开发者目前基于当前目录进行分析，筛选出缺少静态图分支的API列表。

如下是一个典型的「**动静行为统一**」的 API 实现：
```python
def matmul(x, y, transpose_x=False, transpose_y=False, name=None):
    if in_dygraph_mode():
        return _C_ops.matmul(x, y, transpose_x, transpose_y)  # <--- 动态图分支

    attrs = {
        'trans_x': transpose_x,
        'trans_y': transpose_y,
    }

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_variable_and_dtype(
                val,
                name,
                ['float16', 'float32', 'float64', 'complex64', 'complex128'],
                'matmul',
            )

    __check_input(x, y)

    helper = LayerHelper('matmul_v2', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(    # <--- 静态图分支
        type='matmul_v2',
        inputs={'X': x, 'Y': y},
        outputs={'Out': out},
        attrs=attrs,
    )
    return out
```

如下是一个典型的「**动静行为不统一**」的 API 实现：
```python
@dygraph_only
def abs(x, name=None):  # paddle.sparse.abs
    """
    Calculate elementwise absolute value of x, requiring x to be a SparseCooTensor or SparseCsrTensor.

    Examples:
        .. code-block:: python

            import paddle

            dense_x = paddle.to_tensor([-2, 0, 3], dtype='float32')
            sparse_x = dense_x.to_sparse_coo(1)
            out = paddle.sparse.abs(sparse_x)

    """
    return _C_ops.sparse_abs(x)
    
 class RandomHorizontalFlip(BaseTransform):  # <--- paddle.vision.transforms.RandomHorizontalFlip
    """Horizontally flip the input data randomly with a given probability.

    Examples:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from paddle.vision.transforms import RandomHorizontalFlip

            transform = RandomHorizontalFlip(0.5)

            fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

            fake_img = transform(fake_img)
            print(fake_img.size)
    """

    def __init__(self, prob=0.5, keys=None):
        super().__init__(keys)
        assert 0 <= prob <= 1, "probability must be between 0 and 1"
        self.prob = prob

    def _apply_image(self, img):
        if random.random() < self.prob:   # <---- 此仅支持动态图，静态图下行为不一致
            return F.hflip(img)
        return img
```
 

### 2. API静态图扩展
在明确了待升级的API列表后，可以针对此部分API分别新增静态图分支，主要逻辑为：

+ 熟悉 API 动态图的逻辑和原理
+ 借助 `if in_dygraph_mode()` 分流动态图和静态图分支逻辑
+ 确定 `paddle.enable_static()` 下，API 的动态图逻辑可借助静态图append一个或多个 OP 来组合等价实现
+ 添加必要的单测，确保 API 在动、静态图下执行的结果是一致的




## 三、执行步骤建议
开发者可以借助如下步骤来有节奏地开发和贡献 Pull Requests:

+ 以`paddle.vision.transforms`下的 API 入手，熟悉「动静行为统一」概念
* 统一 `paddle.vision.transforms` 的 API 动静行为，帮助厘清静态分支逻辑
* 扩展到其他飞桨模块 API，梳理和升级相关 API

## 四、总结

飞桨API动静行为统一是飞桨框架以及动转静功能的重要基石，扩展和对齐 API 动静行为，对于飞桨的API生态和用户体验具有非常大的意义。