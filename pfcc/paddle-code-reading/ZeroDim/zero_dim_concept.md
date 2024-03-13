## 一、概念背景

Paddle框架曾经使用shape为[1]的1-D Tensor来充当0-D Tensor，但是这从数学上来看是不合理的。为了这个不合理的替代，使得很多本来应该支持0-D Tensor的API不支持0-D Tensor，而且会在出现0-D Tensor的地方加入一些特殊的操作处理，使得其以shape为[1]的1-D Tensor运行，这与业界通用规范形成了大面积的diff。

对于0-D Tensor，先从数学角度上明确其概念：
- 0-D Tensor表示的是一个标量Tensor，其对应的是Numpy的0-D array，可以表示为np.array(10.)，其shape为[]，维度为0，size为1。
- 1-D Tensor表示的是一个向量Tensor，其对应的是Numpy的1-D array，如果只有1个元素，可以表示为np.array([10.])，其shape为[1]，维度为1，size为1。

**标量Tensor** 与仅有1个元素的 **向量Tensor** 非常容易混淆，其元素个数相同，但是在数学定义是完全不同的，如果强行用shape=[1]来表示shape=[]，则无法区分标量Tensor与向量Tensor，与既有的 数学语义、业界通用计算规则 出现严重diff，在很多场景中会导致模型出现非预期错误，并增加模型的开发调试成本。例如以下代码：
```python
# Pytorch写法1行
out = torch.stack([x[0], x[0]]) # 预期 out.ndim == x.ndim，不会出现升维

# Paddle写法需4行：需要额外判断x是否为1D，是就需要补squeeze来矫正结果以对齐pytorch，不是就可和Pytorch相同
if len(x.shape) == 1:
    # 因为用shape=[1]的1维替代0维，导致x[0]还是1维，stack结果为2维，出现异常升维，需要补squeeze来校正维度
    out = paddle.stack([x[0], x[0]]).squeeze()
else:
    out = paddle.stack([x[0], x[0]])
```

当x为1维时，会出现out维度异常升高，用户必须加入一些额外判断和操作使之符合预期，并且这种非预期的维度错误会在网络中不断传播，增大调试成本。类似于此的API问题还有很多，由此也产生了大量用户反馈issue。


## 二、历史问题

用户的历史问题记录如下：

| 序号 |	问题链接 |	问题描述 |	业务收益 |
| ----| ----       | ----     | ----    |
| 1 | https://github.com/PaddlePaddle/Paddle/issues/45627 (EinOps支持paddle需求提出)、https://github.com/arogozhnikov/einops/pull/122 （EinOps支持Paddle的PR）、https://github.com/PaddlePaddle/Paddle/issues/34220 （持Paddle的0D API报错）	| EinOps（一个用户量较大的爱因斯坦求和库）计划支持PaddlePaddle后端，为与其他框架（MxNet、TF、Pytorch等）保持统一结构，需要使用0维Tensor，然而发现Paddle有些API不支持0维Tensor，目前只能暂停对PaddlePaddle的适配。https://github.com/arogozhnikov/einops |	EinOps第三方库（5.5k start）支持Paddle为后端，拓展Paddle生态  |
| 2	| https://console.cloud.baidu-int.com/devops/icafe/issue/DLTP-33495/show |	CV模型需求：国内1名开发者需要在CV-3D模型构建上使用0维Tensor，因为部分点云数据可能需要0维Tensor的支持，目前也在考虑发表论文，希望Paddle可以高优支持，后期将会贡献在AgentMaker的3D点云套件中。https://github.com/AgentMaker/PAPC |	支持3D点云模型，对0维Tensor整体有较大需求，目前无法支持|
| 3	| https://github.com/PaddlePaddle/Paddle/issues/42757 |	有用户希望在deepxde(https://github.com/lululxvi/deepxde)中将Paddle作为后端适配，需要保持与其他框架的通用规则一致，但是paddle不支持0维使得适配过程存在一些问题	| deepxde第三方库（1.2k start）更好的适配Paddle为后端，拓展Paddle生态 |
| 4	| https://github.com/PaddlePaddle/Paddle/issues/41247 |	用户了tensorflow、pytorch、mindspore、paddle四个框架的max、min这类的归约函数，发现只有paddle不符合预期。需要加一些reshape操作才能符合预期 |	reduce不符合预期 |
| 5	| https://github.com/PaddlePaddle/Paddle/issues/39825 |	用户对比了Paddle、Pytorch、numpy的Tensor.getitem后，发现paddle不符合预期。同时列举了stack+切片的例子，认为需要先判断维度，然后将不符合预期的修正回去，要增加很多代码 | 	索引不符合预期	| 
| 6	| https://github.com/PaddlePaddle/Paddle/issues/29951 |	paddle和pytorch在计算mean时输出的形状不同，希望能输出正确的shape | reduce不符合预期	 |
| 7	| https://github.com/PaddlePaddle/Paddle/issues/16507 |	reduce_sum如何只输出常量，不要带维度  |	reduce不符合预期 |
| 8	| https://github.com/PaddlePaddle/Paddle/issues/35891、https://github.com/PaddlePaddle/Paddle/issues/35762、https://console.cloud.baidu-int.com/devops/icafe/issue/DLTP-41007/show?source=drawer-header	 | 用户发现paddle在使用shape为[1]的Tensor进行索引时，会导致降维，少一根轴，实际上用shape为[]的标量Tensor才应该降维	| 索引不符合预期 |
| 9	| https://console.cloud.baidu-int.com/devops/icafe/issue/DLTP-41007/show?source=drawer-header	| 3D项目，发现用shape为[1]的Tensor作为索引，就会导致降维，但理论上用shape为[]的标量Tensor才应该降维。所以需要把这一维重新reshape回来才使后面OP不报错	 | 索引不符合预期，导致模型代码挂了，需要逐个的去hack修复。如果支持后，业务在做模型迁移时就会快很多。|


## 三、竞品对比
Pytorch对0D Tensor的支持符合业界习惯，支持完整的0D Tensor语义。对其中输入、输出涉及torch.Tensor的API数量进行大致统计，其支持0D Tensor的API 约有500个。

Numpy对0D array的支持同样符合业界习惯，支持完整的0D array 含义。对输入输出涉及 np.ndarray 的numpy API进行统计，其支持0D array的API约有300个。

因此，当前主流竞品均支持完整的0D语义，对Paddle中所有API进行了分析，我们梳理了共333个API应支持0-D Tensor，详见文件：`judge_zero_dim.md` 。
