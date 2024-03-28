## Paddle框架的0-D设计全表

根据 **API需支持0维的判断标准**，当前Paddle框架中所有支持0-D的API如下表所示，其中主体设计与Numpy/Pytorch一致，但部分API考虑到应用场景与数学语言，Paddle进行了更合理的设计调整：

- Paddle 比 Pytorch 额外支持0-D，例如 `paddle.upsample` 的scale_factor系数支持float/Tensor，而竞品仅支持float，所以Paddle相比竞品额外多支持0-D Tensor
- Pytorch的所有loss函数均支持输入input、label为0-D，但由于input、label一般情况下都是有维度的，没应用场景，因此Paddle不支持

整体全表为：

| API名称 | 小类 | 0D语义定义 | output-tensor1 | output-tensor2 | input-tensor1 | input-tensor2 | input-tensor3 | input-tensor4 | input-tensor5 | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| paddle.nn.functional.elu | activate | element-wise unary | T | T | 
| paddle.nn.functional.gelu | activate | element-wise unary | T | T | 
| paddle.nn.functional.hardshrink | activate | element-wise unary | T | T | 
| paddle.nn.functional.hardsigmoid | activate | element-wise unary | T | T | 
| paddle.nn.functional.hardswish | activate | element-wise unary | T | T | 
| paddle.nn.functional.hardtanh | activate | element-wise unary | T | T | 
| paddle.nn.functional.leaky_relu | activate | element-wise unary | T | T | 
| paddle.nn.functional.log_sigmoid | activate | element-wise unary | T | T | 
| paddle.nn.functional.prelu | activate | element-wise unary, weight可以支持[], [1], [in_channel] | T | T | T | 
| paddle.nn.functional.relu | activate | element-wise unary | T | T | 
| paddle.nn.functional.relu6 | activate | element-wise unary | T | T | 
| paddle.nn.functional.selu | activate | element-wise unary | T | T | 
| paddle.nn.functional.sigmoid | activate | element-wise unary | T | T | 
| paddle.nn.functional.softplus | activate | element-wise unary | T | T | 
| paddle.nn.functional.softshrink | activate | element-wise unary | T | T | 
| paddle.nn.functional.softsign | activate | element-wise unary | T | T | 
| paddle.nn.functional.swish | activate | element-wise unary（Paddle特有API） | T | T | 
| paddle.nn.functional.tanhshrink | activate | element-wise unary | T | T | 
| paddle.nn.functional.thresholded_relu | activate | element-wise unary（Paddle特有API） | T | T | 
| paddle.stanh | activate | element-wise unary（Paddle特有API） | T | T | 
| paddle.static.nn.prelu | activate | element-wise unary | T | T | 
| paddle.nn.functional.celu | activate | element-wise unary | T | T | 
| paddle.nn.functional.mish | activate | element-wise unary | T | T | 
| paddle.nn.functional.relu_ | activate | element-wise unary | T | T | 
| paddle.nn.functional.silu | activate | element-wise unary | T | T | 
| paddle.nn.functional.tanh | activate | element-wise unary | T | T | 
| paddle.nn.functional.tanh_ | activate | element-wise unary | T | T | 
| paddle.imag | pointwise | element-wise unary | T | T | 
| paddle.real | pointwise | element-wise unary | T | T | 
| paddle.cast | pointwise | element-wise unary | T | T | 
| paddle.cosh | pointwise | element-wise unary | T | T | 
| paddle.sinh | pointwise | element-wise unary | T | T | 
| paddle.abs | pointwise | element-wise unary | T | T | 
| paddle.acos | pointwise | element-wise unary | T | T | 
| paddle.asin | pointwise | element-wise unary | T | T | 
| paddle.atan | pointwise | element-wise unary | T | T | 
| paddle.ceil | pointwise | element-wise unary | T | T | 
| paddle.clip | pointwise | element-wise unary | T | T | 
| paddle.cos | pointwise | element-wise unary | T | T | 
| paddle.erf | pointwise | element-wise unary | T | T | 
| paddle.exp | pointwise | element-wise unary | T | T | 
| paddle.floor | pointwise | element-wise unary | T | T | 
| paddle.increment | pointwise | element-wise unary | T | T | 
| paddle.log | pointwise | element-wise unary | T | T | 
| paddle.log1p | pointwise | element-wise unary | T | T | 
| paddle.reciprocal | pointwise | element-wise unary | T | T | 
| paddle.round | pointwise | element-wise unary | T | T | 
| paddle.rsqrt | pointwise | element-wise unary | T | T | 
| paddle.sign | pointwise | element-wise unary | T | T | 
| paddle.sin | pointwise | element-wise unary | T | T | 
| paddle.sqrt | pointwise | element-wise unary | T | T | 
| paddle.square | pointwise | element-wise unary | T | T | 
| paddle.tanh | pointwise | element-wise unary | T | T | 
| paddle.acosh | pointwise | element-wise unary | T | T | 
| paddle.angle | pointwise | element-wise unary | T | T | 
| paddle.asinh | pointwise | element-wise unary | T | T | 
| paddle.atanh | pointwise | element-wise unary | T | T | 
| paddle.conj | pointwise | element-wise unary | T | T | 
| paddle.deg2rad | pointwise | element-wise unary | T | T | 
| paddle.erfinv | pointwise | element-wise unary | T | T | 
| paddle.expm1 | pointwise | element-wise unary | T | T | 
| paddle.log10 | pointwise | element-wise unary | T | T | 
| paddle.log2 | pointwise | element-wise unary | T | T | 
| paddle.logit | pointwise | element-wise unary | T | T | 
| paddle.neg | pointwise | element-wise unary | T | T | 
| paddle.rad2deg | pointwise | element-wise unary | T | T | 
| paddle.scale | pointwise | element-wise unary | T | T | 
| paddle.tan | pointwise | element-wise unary | T | T | 
| paddle.tanh_ | pointwise | element-wise unary | T | T | 
| paddle.trunc | pointwise | element-wise unary | T | T | 
| paddle.digamma | elementiwsie unary | T | T | 
| paddle.lgamma | elementiwsie unary | T | T | 
| paddle.nn.functional.alpha_dropout | pointwise | 可视作elementiwsie unary | T | T | 
| paddle.nn.functional.dropout | pointwise | 可视作elementiwsie unary，此时axis必须为None | T | T | 
| paddle.nn.functional.softmax | pointwise | 可视作elementiwsie unary，此时axis只能为0/-1 | T | T | 
| paddle.nn.functional.log_softmax | pointwise | 可视作elementiwsie unary，此时axis只能为0/-1 | T | T | 
| paddle.nn.functional.gumbel_softmax | pointwise | 可视作elementiwsie unary，此时axis只能为0/-1 | T | T | 
| paddle.poisson | 可视作elementiwsie unary | T | T | 
| paddle.bernoulli | random | 可视作element-wise unary | T | T | 
| paddle.add | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.add_n | pointwise | element-wise multiary，同时应支持0D输入的广播 | T | T | T | T | 
| paddle.subtract | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.multiply | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.divide | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.floor_divide | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.floor_mod | pointwise | 与mod同 , element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.mod | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.remainder | pointwise | 与mod同，element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.equal | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.not_equal | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.greater_equal | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.greater_than | pointwise | element-wise binary，同时应支持1D输入的广播 | T | T | T | 
| paddle.less_equal | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.less_than | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.allclose | compare/reduction | 1) element-wise binary；2）返回应为0D，目前暂不要修改返回 | T | T | T | 
| paddle.equal_all | compare/reduction | 1) element-wise binary；2）返回应为0D，目前暂要不修改返回 | T | T | T | 
| paddle.logical_and | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.logical_not | pointwise | element-wise unary | T | T | 
| paddle.logical_or | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.logical_xor | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.bitwise_and | bitwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.bitwise_not | bitwise | element-wise unary | T | T | 
| paddle.bitwise_or | bitwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.bitwise_xor | bitwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.pow | pointwise | 如果y为float|int，则为element-wise unary; 如果y为Tensor，则为element-wise binary；两种情况都需支持0D输入 | T | T | T | 
| paddle.atan2 | pointwise | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.complex | pointwise | elementwise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.maximum | compare | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.minimum | compare | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.fmax | compare | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.fmin | compare | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.lcm | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.gcd | element-wise binary，同时应支持0D输入的广播 | T | T | T | 
| paddle.lerp | element-wise binary; weigth系数与x，y同维，也可为0D | T | T | T | T | 
| paddle.nn.functional.embedding | layer | 输入tensor x为查表id，可以支持0维与多维，输入weight是2维 shape为（num_embeddings, embedding_dim），输出tensor 为x维度+1 | F | T | F | 
| paddle.nn.functional.one_hot | layer | 输入tensor x可以支持0维及多维，将x每个元素转化为one_hot向量，输出tensor 为x维度+1 | F | T | 
| paddle.static.nn.embedding | layer | 与nn.functional.embedding同 | F | T | 
| paddle.nn.functional.linear | layer | 输入x >= 2维，输入weight是2维，输入bias应支持0维，并broadcast成预期的shape | F | F | F | T | 
| paddle.nn.functional.interpolate | cv | size可以是0D的list，也可以是0D，视为用于每个维度；scale_factor可以是0D的list，也可以是0D，视为用于每个维度 | F | F | T | T | 
| paddle.nn.functional.upsample | cv | size可以是0D的list，也可以是0D，视为用于每个维度；scale_factor可以是0D的list，也可以是0D，视为用于每个维度 | F | F | T | T | 
| paddle.static.nn.sequence_pad | lod | lod 的输入x都不支持 0d tensor，但pad_value可以支持0D与1D | F | F | T | 
| paddle.nn.functional.sigmoid_focal_loss | loss | normalizer系数可考虑支持0D | T | F | F | T | 
| paddle.distribution.AbsTransform | transform | 正逆变换应支持0D输入、输出 | T | T | 
| paddle.distribution.AffineTransform | transform | 正逆变换应支持0D输入、输出 | T | T | 
| paddle.distribution.ChainTransform | transform | 正逆变换应支持0D输入、输出 | T | T | 
| paddle.distribution.ExpTransform | transform | 正逆变换应支持0D输入、输出 | T | T | 
| paddle.distribution.IndependentTransform | transform | 将batch_shape挪到event_shape，因此输入shape不能为[]，不支持0D | 
| paddle.distribution.PowerTransform | transform | 正逆变换应支持0D输入、输出 | T | T | 
| paddle.distribution.ReshapeTransform | transform | 正逆变换应支持0D输入、输出 | T | T | 
| paddle.distribution.SigmoidTransform | transform | 正逆变换应支持0D输入、输出 | T | T | 
| paddle.distribution.SoftmaxTransform | transform | 最少1D，softmax不应支持0D | 
| paddle.distribution.StackTransform | transform | 将一系列变换沿着某个轴作用于输入Tensor上，输入输出维度相等，因此不支持 | 
| paddle.distribution.StickBreakingTransform | transform | 输入最少1D | 
| paddle.distribution.TanhTransform | transform | 正逆变换应支持0D输入、输出 | T | T | 
| paddle.distribution.Transform | transform | 正逆变换应支持0D输入、输出 | T | T | 
| paddle.distribution.TransformedDistribution | transform | 正逆变换应支持0D输入、输出 | T | T | 
| paddle.distributed.send | distributed | 发送，应该支持0D Tensor | T | 
| paddle.distributed.recv | distributed | 接收其它卡/机器传输过来的tensor，需要支持0维tensor | T | 
| paddle.distributed.all_reduce | distributed | 将多卡的数据按某个操作（例如求和）后放到全部卡，可支持0D Tensor输入输出 | 输出在输入里 | T | 
| paddle.distributed.reduce | distributed | 将多卡的数据按某个操作（例如求和）后放到指定卡，可支持0D Tensor输入输出 | 输出在输入里 | T | 
| paddle.distributed.broadcast | distributed | 可支持将0D Tensor广播到其他卡 | T | 
| paddle.distributed.all_gather | distributed | 多卡聚合，输入单个Tensor，输出Tensor list，可支持输入、输出0D | 输出在输入里 | T(输出的多个tensor，为list) | T(输入的单个Tensor) | 
| paddle.distributed.scatter | distributed | 多卡分发，输入Tensor list，输出为单个Tensor，可支持输入、输出0D | 输出在输入里 | T(输出的单个tensor) | T(输入的多个tensor, 为list) | 
| paddle.static.nn.case | control-flow | 输入条件pred_fn_pairs中的pred需支持0D bool Tensor | T(fn返回0D) | T | 
| paddle.static.nn.cond | control-flow | 输入条件pred支持0D bool Tensor | T(fn返回0D) | T | 
| paddle.static.nn.switch_case | control-flow | 输入条件branch_index支持0D bool Tensor | T(fn返回0D) | T | 
| paddle.static.nn.while_loop | control-flow | loop_vars需要支持0D tensor | T(body返回0D) | T | 
| paddle.where | search | condition、x、y、out均可以为0D | T | T | T | T | 
| paddle.Tensor.backward | backward | 需要支持0D Tensor求动态图反向 | T | T | 
| paddle.static.append_backward | backward | 输入loss支持0D，输出[param, grad]支持0D | T | T | 
| paddle.static.gradients | backward | outputs,inputs, output_grads 都是 list of tensors, 需要支持 0d | T | T | T | T | 
| paddle.autograd.backward | backward | 输入的tensors和grad_tensors 是tensor list，其中的元素应该支持0维tensor(比如loss) | T | T | 
| paddle.grad | backward | outputs,inputs, output_grads 都是 list of tensors, 需要支持 0d | T | T | T | T | 
| paddle.expand | manipulate | 0D可以expand为0D或更高维 | T | T | 
| paddle.expand_as | manipulate | 0D可以expand为0D或更高维 | T | T | T | 
| paddle.flip | manipulate | shape=[]表示不翻转，此时输入输出可为0D | T | T | 
| paddle.reshape_ | manipulate | []、[1]、[[1]]…可以互相reshape | T | T | 
| paddle.reshape | manipulate | 输入0D，保持numel不变即可，[]可以reshape为[1]、[1, 1].. | T | T | 
| paddle.stack | manipulate | stack 多个 0D 得到一个 1D | F | T | 
| paddle.tile | manipulate | 1）复制repeat_times次数据，x与repeat_times的维度拼接，连接处相乘，因此对于0D 输入，由repeate_times决定维度，0D + []->0D，0D+[1]->[1]，0D+[2, 3]->[2, 3]；2）repeat_times表示多个int标量，应支持0D Tensor/int的list/tuple | T | T | T | 
| paddle.unsqueeze | manipulate | 指定位置插入尺寸为1的维度，输入可为0D，输出>=0D | F | T | 
| paddle.unsqueeze_ | manipulate | 与unsqueeze同 | F | T | 
| paddle.as_real | manipulate | 将1个复数拆成2个实数，增加1维，支持输入0D | F | T | 
| paddle.moveaxis | manipulate | 移动轴的位置，src_dim=dst_dim=[]时可支持0D输入，输出亦为0D | T | T | 
| paddle.unique_consecutive | manipulate | numel=1没有意义，可以支持 | T | T | 
| paddle.flatten | manipulate | 0D可以flatten为1D，start_axis与end_axis必须在[0, -1]范围 | F | T | 
| paddle.repeat_interleave | manipulate | 0D先flatten为1D，再repeat，因此输入可为0D，输出为1D，此时axis必须为None | F | T | F | 
| paddle.t | manipulate | 0D转置仍为自身，输入0D，输出0D | T | T | 
| paddle.reverse | manipulate | 0D反转仍为自身，输入0D，输出0D | T | T | 
| paddle.transpose | manipulate | 0D转置仍为自身，输入0D，输出0D | T | T | 
| paddle.gather | manipulate | 沿着某个轴来切片，默认第1个维度，因此x>=1D，当index为1D时，输出不降维，index为0D时，输出降1维；因此当 x为1D, index为0D, 会输出0D | T | F | T | 
| paddle.scatter | manipulate | 使用update来更新gather出来的内容，参考gather，可支持 x为1D, index为0D, updates=0D的情形 | F | F | T | T | 
| paddle.scatter_ | manipulate | 与scatter同 | T | F | T | T | 
| paddle.scatter_nd | manipulate | 参考gather_nd，存在updates为0D的情形 | F | F | F | T | 
| paddle.is_complex | predicate | 一元判断类，输入支持0D，返回python bool值 | T | 
| paddle.is_floating_point | predicate | 一元判断类，输入支持0D，返回python bool值 | T | 
| paddle.is_integer | predicate | 一元判断类，输入支持0D，返回python bool值 | T | 
| paddle.is_tensor | predicate | 一元判断类，输入支持0D，返回python bool值 | T | 
| paddle.is_empty | predicate | 一元判断类，输入支持0D，返回bool Tensor应为0D，暂不修改输出 | T | T | 
| paddle.isfinite | predicate | 可视作elementiwsie unary | T | T | 
| paddle.isinf | predicate | 可视作elementiwsie unary | T | T | 
| paddle.isnan | predicate | 可视作elementiwsie unary | T | T | 
| paddle.isclose | predicate | element-wise binary, 同时应支持0D输入的广播 | T | T | T | 
| paddle.slice | manipulate | starts、ends表示多个int标量，应该为int，或者0D Tensor/shape为[1]/int的list/tuple；输出与输入同维，[end-start]，例如对于3D Tensor可能出现[0,0,0], [1,1,1]，这里勿混淆0-sized与0D的场景 | F | F | T | T | 
| paddle.strided_slice | manipulate | starts、ends、strides表示多个int标量，可以为0D Tensor，或者0D Tensor/int的list/tuple | T | F | T | T | T | 
| paddle.linspace | creation | start, stop, num均可以为0D，返回1D的list | F | T | T | 
| paddle.arange | creation | start, end, step均可以为0D，返回1D的list | F | T | T | T | 
| paddle.normal | random | 1)shape可为[]或[1] Tensor的list；2）mean/std可为0D Tensor，此时也返回0D | T | T(mean) | T(std) | T(shape) | 
| paddle.rand | random | 1)shape可为[]或[1] Tensor的list | T | T | 
| paddle.randint | random | 1)shape可为[]或[1] Tensor的list | T | T | 
| paddle.randn | random | 1)shape可为[]或[1] Tensor的list | T | T | 
| paddle.standard_normal | random | 1)shape可为[]或[1] Tensor的list | T | T | 
| paddle.uniform | random | 1)shape可为[]或[1] Tensor的list | T | T | 
| paddle.empty | creation | 1)shape可为[]或[1] Tensor的list | T | 
| paddle.full | creation | 1)shape可为[]或[1] Tensor的list | T | 
| paddle.ones | creation | 1)shape可为[]或[1] Tensor的list | T | 
| paddle.zeros | creation | 1)shape可为[]或[1] Tensor的list | T | 
| paddle.empty_like | creation | 可以传入0D Tensor作为like对象 | T | T | 
| paddle.full_like | creation | 可以传入0D Tensor作为like对象 | T | T | 
| paddle.ones_like | creation | 可以传入0D Tensor作为like对象 | T | T | 
| paddle.zeros_like | creation | 可以传入0D Tensor作为like对象 | T | T | 
| paddle.randint_like | creation | 可以传入0D Tensor作为like对象 | T | T | 
| paddle.all | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.any | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.logsumexp | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.max | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.mean | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.min | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.prod | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.sum | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.amax | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.amin | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.nanmean | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.nansum | reduction | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.einsum | reduction | 1) 输入可为0D；2) 指定为reduce操作时，输出可为0D | T | T | T | (N元) | 
| paddle.quantile | stat | 1) reduce操作，输入可为0D，此时axis=None或[]，返回也为0D | T | T | 
| paddle.std | stat | 1) reduce操作，输入可为0D，此时axis=None或[]，返回也为0D | T | T | 
| paddle.var | stat | 1) reduce操作，输入可为0D，此时axis=None或[]，返回也为0D | T | T | 
| paddle.median | stat | 1) reduce操作，输入可为0D，此时axis=None或[]，返回也为0D | T | T | 
| paddle.kthvalue | stat | 1) reduce操作，输入可为0D，此时axis=None，返回也为0D | T | T | 
| paddle.mode | stat | 1) reduce操作，输入可为0D，此时axis=-1，返回也为0D | T | T | 
| paddle.cumsum | scan | 1) reduce操作，输入可为0D，此时axis=None，返回也为0D | T | T | 
| paddle.cumprod | scan | 1) reduce操作，输入可为0D，此时axis=None，返回也为0D | T | T | 
| paddle.argmax | search | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.argmin | search | 1) reduce操作，输入可为0D，此时axis=[]，返回也为0D | T | T | 
| paddle.argsort | search | 可视作elementiwsie unary | T | T | 
| paddle.masked_select | search | x与mask可为0D，此时输出的shape为[1]（输出shape恒为1D） | T | T | T | 
| paddle.sort | search | 输入为0D，此时axis=None，排序结果为0D（与原来一致） | T | T | 
| paddle.topk | search | 输入为0D，此时axis=None，k=1，查找结果为0D（与原来一致） | T | T | 
| paddle.unique | search | 输入输出可均为0D | T | T | 
| paddle.searchsorted | search | 查找元素和索引。第一个输入x至少1D，第二个输入values和输出应支持0d | T | F | T | 
| paddle.clone | copy | 拷贝Tensor，应支持 | T | T | 
| paddle.assign | copy | 拷贝Tensor，应支持 | T | T | 
| paddle.Tensor.item | io | 输入0D Tensor，返回python scalar | T | 
| paddle.tolist | io | 1D及以上Tensor转为Python list，0D Tensory应转为Python scalar | T | 
| paddle.numel | meta-data | numel为标量，输入可支持0D | T | T | 
| paddle.rank | meta-data | rank为标量，输入可支持0D。注：输出已经支持0D | T | T | 
| paddle.shape | meta-data | 输入应支持0D，0D的shape返回空list Tensor | T | 
| paddle.broadcast_tensors | broadcast | 输入可以为零维 | T | T | 
| paddle.broadcast_to | broadcast | 输入可以为零维，支持将0D broadcast为0D或更高维 | T | T | 
| paddle.kron | 输入x, y均可为0D，此时输出0D | T | T | T | 
| paddle.diagflat | linalg | 展平x作为对角线元素生成方阵，输入>=0D，输出2D | F | T | 
| paddle.histogram | 输入tensor x因为会被展平，所以应该支持任意维度(包括0维)，输出为直方图的结果和边界值，必为1维 | F | T | 
| paddle.linalg.cond | linalg | 输入tensor x要求是2维或3维，输出tensor是条件数为0维(输入2维) 或 1维(输入3维) | T(matrix) | F | 
| paddle.linalg.cov | linalg | 输入tensor x要求是1维或2维，fweights和aweights是1维，输出tensor是协方差，只有一行或一列数据时返回标量 | T(一行或一列) | F | F | F | 
| paddle.linalg.det | linalg | 输入tensor x要求是2维或3维，输出tensor是行列式的值为0维(输入2维) 或 1维(输入3维) | T(matrix) | F | 
| paddle.linalg.matrix_rank | linalg | 输入tensor x要求是2维或3维，输出tensor 矩阵的秩为0维(输入2维) 或 1维(输入3维) | T(输入为2维时) | F | 
| paddle.linalg.multi_dot | linalg | 向量内积输出为0D，输入为1D*2D*1D | T | F | 
| paddle.linalg.norm | linalg | 计算向量或矩阵的范数，则输入tensor x要求是1维向量或2维矩阵，输出tensor范数应为0维 | T | F | 
| paddle.linalg.slogdet | linalg | 输入tensor x要求是2维或3维，输出tensor 行列式符号值/行列式自然对数值 为0维(输入2维) 或 1维(输入3维) | T(matrix) | T(matrix) | F | 
| paddle.dist | linalg | 输入任意维度，输出范数为0D | T | F | F | 
| paddle.trace | linalg | 输入2D Tensor，输出0D Tensor | T | F | 
| paddle.allclose | reduction | 1)输入应支持0D; 2)输出为0D bool Tensor | T | T | T | 
| paddle.equal_all | reduction | 1)输入应支持0D; 2)输出为0D bool Tensor | T | T | T | 
| paddle.all | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.any | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.logsumexp | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.max | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.mean | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.min | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.nanmean | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.nansum | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.prod | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.sum | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.amax | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.amin | reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.quantile | stat/reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.std | stat/reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.var | stat/reduction | 轴向reduction操作。1) 输入可为0D，axis限定为[-1, 0, None, [] ]，输出也为0D；2) 输出可降为0D | T | T | 
| paddle.argmax | search/reduction | 1) reduce操作，输入可为0D，axis限定为[-1, 0, None]，输出也为0D；2) 输入ND可降为0D | T | T | 
| paddle.argmin | search/reduction | 1) reduce操作，输入可为0D，axis限定为[-1, 0, None]，输出也为0D；2) 输入ND可降为0D | T | T | 
| paddle.median | stat/reduction | 1) reduce操作，输入可为0D，axis限定为[-1, 0, None]，输出也为0D；2) 输入ND可降为0D | T | T | 
| paddle.kthvalue | stat/reduction | 1) reduce操作，输入可为0D，axis限定为-1，输出也为0D；2) 输入1D可降为0D（不可reduce_all） | T | T | 
| paddle.mode | stat/reduction | 1) reduce操作，输入可为0D，axis限定为-1，输出也为0D；2) 输入1D可降为0D（不可reduce_all） | T | T | 
| paddle.nn.functional.binary_cross_entropy | loss | 有reduce 时输出是 0d | T | F | F | F | 
| paddle.nn.functional.binary_cross_entropy_with_logits | loss | 有reduce 时输出是 0d | T | F | F | F | 
| paddle.nn.functional.cross_entropy | loss | 有reduce 时输出是 0d | T | F | F | 
| paddle.nn.functional.softmax_with_cross_entropy | loss | 有reduce 时输出是 0d | T | F | F | F | 
| paddle.nn.functional.nll_loss | loss | 有reduce 时输出是 0d | T | F | F | F | 
| paddle.nn.functional.ctc_loss | loss | 有reduce 时输出是 0d | T | F | F | F | F | 
| paddle.nn.functional.kl_div | loss | 有reduce 时输出是 0d | T | F | F | 
| paddle.nn.functional.l1_loss | loss | 有reduce 时输出是 0d | T | F | F | 
| paddle.nn.functional.margin_ranking_loss | loss | 有reduce 时输出是 0d | T | F | F | F | 
| paddle.nn.functional.mse_loss | loss | 有reduce 时输出是 0d | T | F | F | 
| paddle.nn.functional.npair_loss | loss | 输出为paddle.mean的结果，自然为0d | T | F | F | F | 
| paddle.nn.functional.sigmoid_focal_loss | loss | 有reduction输出为0D | T | F | F | F | 
| paddle.nn.functional.smooth_l1_loss | loss | 有reduce 时输出 0D | T | F | F | 
| paddle.create_parameter | creation-variable | 应支持创建0D Tensor，shape=[] | T | 
| paddle.static.create_global_var | creation-variable | 应支持创建0D Tensor，shape=[] | T | 
| paddle.static.create_parameter | creation-variable | 应支持创建0D Tensor，shape=[] | T | 
| paddle.static.data | creation-variable | 应支持创建0D Tensor，shape=[] | T | 
| paddle.to_tensor | creation | 应支持创建0D Tensor，传入一个Python标量 | T | 
| paddle.empty | creation | 1)可以传入shape=[]，动态图已支持; 2)shape可为0D/1D Tensor的list | T | 
| paddle.full | creation | 1)可以传入shape=[]，动态图已支持; 2)shape可为0D/1D Tensor的list | T | 
| paddle.ones | creation | 1)可以传入shape=[]，动态图已支持; 2)shape可为0D/1D Tensor的list | T | 
| paddle.zeros | creation | 1)可以传入shape=[]，动态图已支持; 2)shape可为0D/1D Tensor的list | T | 
| paddle.normal | creation | 1)可以传入shape=[]，动态图已支持; 2)shape可为0D/1D Tensor的list；3）mean/std可为0D Tensor，此时也返回0D | T | T(mean) | T(std) | T(shape) | 
| paddle.rand | creation | 1)可以传入shape=[]，动态图已支持; 2)shape可为0D/1D Tensor的list | T | T | 
| paddle.randint | creation | 1)可以传入shape=[]，动态图已支持; 2)shape可为0D/1D Tensor的list | T | T | 
| paddle.randn | creation | 1)可以传入shape=[]，动态图已支持; 2)shape可为0D/1D Tensor的list | T | T | 
| paddle.standard_normal | creation | 1)可以传入shape=[]，动态图已支持; 2)shape可为0D/1D Tensor的list | T | T | 
| paddle.uniform | creation | 1)可以传入shape=[]，动态图已支持; 2)shape可为0D/1D Tensor的list | T | T | 
| paddle.dot | contraction | 输入必须为1D，输出向量内积，为0D | T | F | 
| paddle.einsum | contraction | 指定einsum的缩约操作时，输出可以为0D | T | F | 
| paddle.inner | contraction | 对于1D向量内积应返回0D，对于2D多个向量目前正确 | T | F | F | 
| paddle.tensordot | contraction | 缩约，输出可以为零维 | T | F | F | 
| paddle.gather | manipulate | 沿着某个轴gather，x>=1D，index为1D时，输出与输入维度一致；index为0D时，输出维度=x维度-1，此时可能输出0D | T | F | T | 
| paddle.gather_nd | manipulate | 沿index最后一维gather，x>=1D，index>=1D，如x.shape=[m, n, k], index.shape=[a, b, c], 则out.shape=[a, b, ..(x被c个元素切片)..]，如c等于x.rank，且a/b均无，输出为0D | T | F | F | 
| paddle.squeeze | manipulate | 压缩维度中的1，输入输出都可以为0D，需根据具体的压缩维度而定 | T | T | 
| paddle.squeeze_ | manipulate | 与sequeeze同 | T | F | 
| paddle.unbind | manipulate | 分割成多个子Tensor，输出=x维度-1，输入1D，输出应为0D | T | F | 
| paddle.unstack | manipulate | 分割成多个子Tensor，输出=x维度-1，输入1D，输出应为0D | T | F | 
| paddle.as_complex | manipulate | 将最后一维每2个实数组合成1个复数，降低1维，支持输出0D；动态图已支持 | T | F | 
| paddle.numel | meta-data | 输入任意维度，输出应为0D，表示标量 | T | T | 
| paddle.rank | meta-data | rank为标量，输出应为0D，输入可支持0D。注：输出已经支持0D | T | T | 
| paddle.is_empty | predicate | 一元判断类，输入支持0D，返回bool 0D Tensor | T | T | 
| paddle.static.accuracy | accuracy | 返回准确率应为0D | T | F | F | 
| paddle.static.auc | accuracy | 返回准确率应为0D | T | F | F | 
| paddle.metric.accuracy | accuracy | 返回准确率应为0D | T | F | F | 
| Tensor.__getitem__/__setitem__ | manipulate | 1)索引切片可能导致降维，例如x[0, 1]为x维度-2，则2D输入，输出0D; 2)索引的index需区分0D与1D，目前index的shape为[1]也会错误降维，实际应在index.shape为[]时降维，而index.shape为[1]不降维; 3)0D不允许作为输入，被切片 | T | T | 
| paddle.distribution.Beta | distribution | 输出=sample_shape+batch_shape+event_shape，输出可为0D | T (sample的输出) | 
| paddle.distribution.Categorical | distribution | 输出=sample_shape+batch_shape+event_shape，输出可为0D | T (sample的输出) | 
| paddle.distribution.Dirichlet | distribution | 输入 | T (sample的输出) | 
| paddle.distribution.Multinomial | distribution | 输出=sample_shape+batch_shape+event_shape，输出可为0D | T (sample的输出) | 
| paddle.distribution.Normal | distribution | 输出=sample_shape+batch_shape+event_shape，输出可为0D | T (sample的输出) | 
| paddle.distribution.Uniform | distribution | 输出=sample_shape+batch_shape+event_shape，输出可为0D | T (sample的输出) | 
| paddle.distribution.kl_divergence | distribution | 输出散度应该为0D | T (sample的输出) | 
| paddle.distribution.kl.kl_divergence | distribution | 输出散度应该为0D | T (sample的输出) | 
| paddle.distribution.TransformedDistribution | distribution | 输出=sample_shape+batch_shape+event_shape，输出可为0D | T (sample的输出) | 
| paddle.distribution.Laplace | distribution | 输出=sample_shape+batch_shape+event_shape，输出可为0D | T (sample的输出) | 
| paddle.distribution.LogNormal | distribution | 输出=sample_shape+batch_shape+event_shape，输出可为0D | T (sample的输出) | 
| paddle.distribution.Gumbel | distribution | 输出=sample_shape+batch_shape+event_shape，输出可为0D | T (sample的输出) | 
