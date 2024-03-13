## API需支持0维的判断标准

一、应支持0-D的API用法场景
1) elementwise计算类，应支持输入0-D
对于所有的逐元素操作，都需要支持0-D输入、输出，包括：
* elementwise一元API，可参考 tanh、relu等
* elementwise二元API，同时应支持0-D广播，可参考add、sub、multiply、divide等（注意：相应的组合API则也应支持0-D，例如Linear相当于matmul、add的组合，则也应支持0-D）

2) 导致升维、降维等操作类API，应支持输入0-D或输出0-D
* 升维API，可参考unsqueeze、reshape、expand、broadcast_to、stack、as_real等，应支持输入0-D
* 降维API，可参考squeeze、as_complex、reshape等，应支持输入0-D

3) 创建类API，应支持输出0-D
* 不指定shape，可参考to_tensor将Python标量转换为Tensor
* 指定shape=[]，可参考data、zeros、ones、full、rand、randn等
* 拷贝Tensor应保持其原始shape，可参考 assign、copy、fetch、numpy()等

4) 轴向reduce操作，应支持输入和输出0-D
对Tensor进行reduce操作时，会减少axis个维度，这里的降维操作可以触发一些0-D的输出：
* 当axis=None时，对所有维度进行降维，N-D可以降为0-D
* 当axis=int时，仅降1维，输出维度 = 输入维度-1，1-D可以降为0-D

5) 索引切片，支持输入和输出0-D
* 标量索引：输入0-D时，与int标量索引的效果一致，具有降维效果
* 标量输出：根据索引个数来降维，例如3-D Tensor取[0, 0, 0]，降3维应输出0D
* 同理，gather、scatter等类似功能API应具有相同效果

6) 输入Tensor具有标量语义，应输入0-D
一般为OP-Attribute，具有单个元素的语义
* shape：如果为list时，应为int list或者0-D Tensor的list
* start、end、stride、stop、step等：应为int或0-D Tensor
* cond：应支持0-D Tensor

7) 返回Tensor为标量语义，应输出0-D
* 向量点积返回标量
* 返回值本身表示某个标量，可参考rank:秩，norm:范数，size:数量

二、不应支持0-D的API用法场景
1）API具有axis属性且不可为空list
* 如果需要指定axis沿着某个轴运算，由于0-D没有轴，则不应支持0-D。可参考 concat、split等
> 注：如果可指定axis=[]，此时相当于不指定轴，则仍能支持0-D。例如transpose、sum虽然有axis，但paddle.sum(x, axis=[])不指定轴运算时，输入0-D也是合法的

2）API具有特定维度的要求
* 例如卷积类/池化类等大量组网API具有特定维度的要求。可参考 Conv1D、Conv2D、Conv3D等
> 注：在深度学习中有相当数量的API功能限定了维度，以上作示例，需要根据该API具体的功能与数学公式来定
