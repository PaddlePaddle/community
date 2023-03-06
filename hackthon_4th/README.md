## 任务目录

- <a href='#paddlepaddle'>赛道一：核心框架开源贡献</a>
  - <a href='#api'>API开发</a>
  - <a href='#op'>算子性能优化</a>
  - <a href='#data'>数据类型扩展</a>
  - <a href='#phi'>PHI算子库独立编译</a>
  - <a href='#tensor'>TensorRT开发</a>
  - <a href='#cinn'>CINN开发</a>
  - <a href='#insight'>开源社区洞察</a>
  - <a href='#other'>其他</a>
- <a href='#paddlefamily'>赛道二：模型套件开源贡献</a>
  - <a href='#ocr'>文字识别开发套件</a>
  - <a href='#detection'>目标检测开发套件</a>
  - <a href='#seg'>图像分割开发套件</a>
  - <a href='#clas'>图像分类开发套件</a>
  - <a href='#nlp'>自然语言处理模型库</a>
  - <a href='#speech'>语音模型库</a>
  - <a href='#3d'>3D任务开发套件</a>
  - <a href='#rs'>遥感影像智能解译开发套件</a>
  - <a href='#fastdeploy'>全场景高性能AI部署工具</a>
  - <a href='#science'>飞桨科学计算套件</a>
- <a href='#paddlefriend'>赛道三：生态伙伴开源贡献</a>
  - <a href='#openvino'>OpenVINO项目</a>
  - <a href='#arm'>ARM项目</a>
  - <a href='#jina'>Jina AI项目</a>
  - <a href='#tvm'>TVM项目</a>
- <a href='#paddleindustry'>赛道四：产业合作开源贡献</a>
  - <a href='#paddleindustry'>基于PaddleOCR的工频场强计读数识别</a>
  - <a href='#paddleindustry'>基于PaddleNLP的跨模态文档信息抽取</a>
  - <a href='#paddleindustry'>基于PaddleClas的中草药识别</a>
  - <a href='#paddleindustry'>基于PaddleDetection的无人机航拍图像检测</a>


- **重要通知**：任务描述链接失效，请直接打开具体任务对应的markdown文件。
- **重要通知**：任务描述链接失效，请直接打开具体任务对应的markdown文件。
- **重要通知**：任务描述链接失效，请直接打开具体任务对应的markdown文件。


大家好，非常高兴地告诉大家，第四期 [PaddlePaddle Hackathon](https://www.paddlepaddle.org.cn/PaddlePaddleHackathon-2023-2?fr=paddleg) 开始了。PaddlePaddle Hackathon 是面向全球开发者的深度学习领域编程活动，鼓励开发者了解与参与 PaddlePaddle 开源社区。本次共有四个赛道：核心框架开源贡献、模型套件开源贡献、生态伙伴开源贡献、产业合作开源贡献，二十四个大方向：API开发、算子性能优化、数据类型扩展、图像分类开发套件、3D任务开发套件、飞桨科学计算套件、OpenVINO项目、ARM项目、数字电表示数识别等，共计 200+个任务供大家认领。详细信息可以参考 [PaddlePaddle Hackathon 说明](https://www.paddlepaddle.org.cn/contributionguide?docPath=hackathon_cn)。大家是否已经迫不及待了呢~
为了帮助大家更好地了解每个任务的内容和进度，并找到一起合作完成的小伙伴（我们强烈推荐大家组队完成，组队完成提交后，将额外获得飞桨黑客松定制周边），本次活动将在此 ISSUE 中汇总信息，如果你报名/提交提案/提交作品，请**参考格式**在此 ISSUE 下回复，并向 paddle-hack@baidu.com 发送[邮件](https://aistudio.baidu.com/aistudio/competition/detail/777/0/task-definition?previewCode=8aa43848-d2c3-4dd0-9cdb-5b9b17d1d759)发起评审。我们每天会汇总所有的信息，并更新在下表中，你可以通过下表，了解到每一个任务的相关信息。

回复的格式为：

【队名】：你队伍的名称

【序号】：你队伍想要完成的任务序号

【状态】：报名/提交提案/提交作品

【链接】：你 fork 的 repo 链接(报名时) / PR 链接（提交时）

如：

【队名】：百度飞桨

【序号】：100

【状态】：提交作品

【链接】：https://github.com/PaddlePaddle/Paddle/pull/36034

**请注意：**
1、一次只能认领/提交一个任务。
2、提交PR时，PR 标题参考 “【Hackathon + No.任务编号】 + PR 内容 ” ，如 “【Hackathon No.1】 新增 finfo API” 以便我们快速 review PR。

3、每支队伍最多**仅能获取3万任务奖金**，建议你选择自己最感兴趣的任务进行开发。





**赛道一：核心框架开源贡献**<a name='paddlepaddle'></a>

**API开发**<a name='api'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 1    | ⭐️    | [为 Paddle 新增 finfo API](https://github.com/PaddlePaddle/Paddle/issues/50630#task1) |                  |          |
| 2    | ⭐️    | [为 Paddle 新增 cdist API](https://github.com/PaddlePaddle/Paddle/issues/50630#task2) |                  |          |
| 3    | ⭐️    | [为 Paddle 新增 trapezoid API](https://github.com/PaddlePaddle/Paddle/issues/50630#task3) |                  |          |
| 4    | ⭐️    | [为 Paddle 新增 cumulative_trapezoid  API](https://github.com/PaddlePaddle/Paddle/issues/50630#task4) |                  |          |
| 5    | ⭐️    | [为 Paddle 新增 nextafter API](https://github.com/PaddlePaddle/Paddle/issues/50630#task5) |                  |          |
| 6    | ⭐️    | [为 Paddle 新增 ldexp API](https://github.com/PaddlePaddle/Paddle/issues/50630#task6) |                  |          |
| 7    | ⭐️    | [为 Paddle 新增 Unflatten API](https://github.com/PaddlePaddle/Paddle/issues/50630#task7) |                  |          |
| 8    | ⭐️    | [为 Paddle 新增 xlogy API](https://github.com/PaddlePaddle/Paddle/issues/50630#task8) |                  |          |
| 9    | ⭐️    | [为 Paddle 新增 pca_lowrank API](https://github.com/PaddlePaddle/Paddle/issues/50630#task9) |                  |          |
| 10   | ⭐️    | [为 Paddle 新增 copysign API](https://github.com/PaddlePaddle/Paddle/issues/50630#task10) |                  |          |
| 11   | ⭐️    | [为 Paddle 新增 Geometric API](https://github.com/PaddlePaddle/Paddle/issues/50630#task11) |                  |          |
| 12   | ⭐️    | [为 Paddle 新增 Cauchy API](https://github.com/PaddlePaddle/Paddle/issues/50630#task12) |                  |          |
| 13   | ⭐️    | [为 Paddle 新增 LogNormal API](https://github.com/PaddlePaddle/Paddle/issues/50630#task13) |                  |          |
| 14   | ⭐️    | [为 Paddle 新增 polar  API](https://github.com/PaddlePaddle/Paddle/issues/50630#task14) |                  |          |
| 15   | ⭐️    | [为 Paddle 新增 GaussianNLLLoss API](https://github.com/PaddlePaddle/Paddle/issues/50630#task16) |                  |          |
| 16   | ⭐️    | [为 Paddle 新增 PoissonNLLLoss API](https://github.com/PaddlePaddle/Paddle/issues/50630#task16) |                  |          |
| 17   | ⭐️⭐️   | [为 Paddle 新增 cummax / cummin API](https://github.com/PaddlePaddle/Paddle/issues/50630#task17) |                  |          |
| 18   | ⭐️⭐️   | [为 Paddle 新增 matrix_exp API](https://github.com/PaddlePaddle/Paddle/issues/50630#task18) |                  |          |
| 19   | ⭐️⭐️   | [为 Paddle 新增 polygamma API](https://github.com/PaddlePaddle/Paddle/issues/50630#task19) |                  |          |
| 20   | ⭐️⭐️   | [为 Paddle 新增 i0 / i0e API](https://github.com/PaddlePaddle/Paddle/issues/50630#task20) |                  |          |
| 21   | ⭐️⭐️   | [为 Paddle 新增 i1/ i1e API](https://github.com/PaddlePaddle/Paddle/issues/50630#task21) |                  |          |
| 22   | ⭐️⭐️   | [为 Paddle 新增 lu_solve API](https://github.com/PaddlePaddle/Paddle/issues/50630#task22) |                  |          |
| 23   | ⭐️⭐️   | [为 Paddle 新增 vander API](https://github.com/PaddlePaddle/Paddle/issues/50630#task23) |                  |          |
| 24   | ⭐    | [为 Paddle 新增 paddle.sparse.is_nan 稀疏 API](https://github.com/PaddlePaddle/Paddle/issues/50630#task24) |                  |          |
| 25   | ⭐️    | [为 Paddle 新增 paddle.sparse.any 稀疏 API](https://github.com/PaddlePaddle/Paddle/issues/50630#task25) |                  |          |
| 26   | ⭐️⭐️   | [为 Paddle 新增 paddle.sparse.nn.Softmax 稀疏 API 的 coo 格式计算逻辑](https://github.com/PaddlePaddle/Paddle/issues/50630#task26) |                  |          |
| 27   | ⭐️⭐️   | [为 Paddle 新增 paddle.sparse.concat 稀疏 API](https://github.com/PaddlePaddle/Paddle/issues/50630#task27) |                  |          |
| 28   | ⭐️⭐️   | [为 Paddle 新增 paddle.sparse.index_select 稀疏 API](https://github.com/PaddlePaddle/Paddle/issues/50630#task28) |                  |          |
| 29   | ⭐️⭐️   | [为 Paddle 新增 paddle.sparse.slice 稀疏 API](https://github.com/PaddlePaddle/Paddle/issues/50630#task29) |                  |          |
| 30   | ⭐️⭐️   | [为 Paddle 新增 paddle.sparse.sum 稀疏 API](https://github.com/PaddlePaddle/Paddle/issues/50630#task30) |                  |          |
| 31   | ⭐️    | [部分API发生除0、空指针、堆栈溢出等问题的修复](https://github.com/PaddlePaddle/Paddle/issues/50630#task31) |                  |          |

**算子性能优化**<a name='op'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 32   | ⭐️    | [为 Paddle 优化 expand_as 前向&反向 op 在 GPU 上的计算性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task32) |                  |          |
| 33   | ⭐️    | [为 Paddle 优化 Histogram 在GPU上的性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task33) |                  |          |
| 34   | ⭐️    | [为 Paddle 优化 Lerp OP在GPU上的性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task34) |                  |          |
| 35   | ⭐️    | [为 Paddle 优化 Prelu OP在GPU上的性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task35) |                  |          |
| 36   | ⭐️    | [为 Paddle 优化 Tile OP在GPU上的性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task36) |                  |          |
| 37   | ⭐️⭐️   | [为 Paddle 优化 matrix_rank op 在 GPU 上的计算性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task37) |                  |          |
| 38   | ⭐️⭐️   | [为 Paddle 优化 p_norm op 在 GPU 上的计算性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task38) |                  |          |
| 39   | ⭐️⭐️   | [为 Paddle 优化 p_norm_grad op 在 GPU 上的计算性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task39) |                  |          |
| 40   | ⭐️⭐️   | [为 Paddle 优化 kthvalue op 在 GPU 上的计算性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task40) |                  |          |
| 41   | ⭐️⭐️   | [为 Paddle 优化 cumprod_grad op 在 GPU 上的计算性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task41) |                  |          |
| 42   | ⭐️⭐️   | [为 Paddle 优化 FminGrad OP在GPU上的性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task42) |                  |          |
| 43   | ⭐️⭐️   | [为 Paddle 优化 GeluGrad OP在GPU上的性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task43) |                  |          |
| 44   | ⭐️⭐️   | [为 Paddle 优化 logsumexp OP在GPU上的性能](https://github.com/PaddlePaddle/Paddle/issues/50657#task44) |                  |          |

**数据类型扩展**<a name='data'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 45   | ⭐️    | [为 Paddle logical 算子实现 float16 数据类型支持](https://github.com/PaddlePaddle/Paddle/issues/50658#task45) |                  |          |
| 46   | ⭐️    | [为 Paddle gumbel_softmax 算子实现 float16 数据类型支持](https://github.com/PaddlePaddle/Paddle/issues/50658#task46) |                  |          |
| 47   | ⭐️    | [为 Paddle cross 算子实现 float16 数据类型支持](https://github.com/PaddlePaddle/Paddle/issues/50658#task47) |                  |          |
| 48   | ⭐️    | [为 Paddle assign_value、meshgrid、kthvalue、determinant 算子实现 float16 数据类型支持](https://github.com/PaddlePaddle/Paddle/issues/50658#task48) |                  |          |
| 49   | ⭐️    | [为 Paddle bce_loss 算子实现 float16 数据类型支持](https://github.com/PaddlePaddle/Paddle/issues/50658#task49) |                  |          |
| 50   | ⭐️    | [为 Paddle lerp 算子实现 float16 数据类型支持](https://github.com/PaddlePaddle/Paddle/issues/50658#task50) |                  |          |
| 51   | ⭐️    | [为 Paddle maxout 算子实现 float16 数据类型支持](https://github.com/PaddlePaddle/Paddle/issues/50658#task51) |                  |          |
| 52   | ⭐️    | [为 Paddle dist 算子实现 float16 数据类型支持](https://github.com/PaddlePaddle/Paddle/issues/50658#task52) |                  |          |
| 53   | ⭐️    | [为 Paddle label_smooth 算子实现 float16 数据类型支持](https://github.com/PaddlePaddle/Paddle/issues/50658#task53) |                  |          |
| 54   | ⭐️    | [为 Paddle allclose、isclose 算子实现 float16 数据类型支持](https://github.com/PaddlePaddle/Paddle/issues/50658#task54) |                  |          |
| 55   | ⭐️    | [channel_shuffle 等算子FP16/BF16算子及单测完善](https://github.com/PaddlePaddle/Paddle/issues/50658#task55) |                  |          |
| 56   | ⭐️    | [set_value 等算子 FP16/BF16算子及单测完善](https://github.com/PaddlePaddle/Paddle/issues/50658#task56) |                  |          |
| 57   | ⭐️    | [gaussian 等算子 FP16/BF16算子及单测完善](https://github.com/PaddlePaddle/Paddle/issues/50658#task57) |                  |          |
| 58   | ⭐️    | [linear_interp 等算子 FP16/BF16算子及单测完善](https://github.com/PaddlePaddle/Paddle/issues/50658#task58) |                  |          |
| 59   | ⭐️    | [addmm 等算子 FP16/BF16算子及单测完善](https://github.com/PaddlePaddle/Paddle/issues/50658#task59) |                  |          |
| 60   | ⭐️    | [angle 等算子 FP16/BF16算子及单测完善](https://github.com/PaddlePaddle/Paddle/issues/50658#task60) |                  |          |
| 61   | ⭐️    | [unfold 等算子 FP16/BF16算子及单测完善](https://github.com/PaddlePaddle/Paddle/issues/50658#task61) |                  |          |
| 62   | ⭐️    | [masked_select 等算子 FP16/BF16算子及单测完善](https://github.com/PaddlePaddle/Paddle/issues/50658#task62) |                  |          |
| 63   | ⭐️    | [complex 等算子 FP16/BF16算子及单测完善](https://github.com/PaddlePaddle/Paddle/issues/50658#task63) |                  |          |
| 64   | ⭐️    | [trace 等算子 FP16/BF16算子及单测完善](https://github.com/PaddlePaddle/Paddle/issues/50658#task64) |                  |          |
| 65   | ⭐️    | [为 Paddle matmul_with_flatten/matmul 前向算子实现 int8 数据类型支持](https://github.com/PaddlePaddle/Paddle/issues/50658#task65) |                  |          |
| 66   | ⭐    | [为Paddle FC 前向算子实现量化计算](https://github.com/PaddlePaddle/Paddle/issues/50658#task66) |                  |          |

**PHI算子库独立编译**<a name='phi'></a>

注：该类型任务 3月6日前完成 PR 提交，3月13日前完成 PR 合入

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 67   | ⭐    | [解耦 PHI 算子库对 operator.h 头文件的依赖](https://github.com/PaddlePaddle/Paddle/issues/50659#task67) |                  |          |
| 68   | ⭐️    | [解耦 PHI 算子库对 utils.h 头文件的依赖](https://github.com/PaddlePaddle/Paddle/issues/50659#task68) |                  |          |
| 69   | ⭐️    | [解耦 PHI 算子库对 device_wrapper.h 头文件的依赖](https://github.com/PaddlePaddle/Paddle/issues/50659#task69) |                  |          |
| 70   | ⭐️⭐️   | [解耦 PHI 算子库对 kernels.h 头文件的依赖](https://github.com/PaddlePaddle/Paddle/issues/50659#task70) |                  |          |

**TensorRT开发**<a name='tensor'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 71   | ⭐️    | [为 Paddle-TRT 添加 pad3d 算子](https://github.com/PaddlePaddle/Paddle/issues/50660#task71) |                  |          |
| 72   | ⭐️    | [为 Paddle-TRT 添加 flip 算子](https://github.com/PaddlePaddle/Paddle/issues/50660#task72) |                  |          |
| 73   | ⭐️    | [为 Paddle-TRT 添加 temporal_shift 算子](https://github.com/PaddlePaddle/Paddle/issues/50660#task73) |                  |          |
| 74   | ⭐️    | [为 Paddle-TRT 添加 grid_sampler 算子](https://github.com/PaddlePaddle/Paddle/issues/50660#task74) |                  |          |
| 75   | ⭐️    | [为 Paddle-TRT 添加 expand_as_v2 算子](https://github.com/PaddlePaddle/Paddle/issues/50660#task75) |                  |          |
| 76   | ⭐️    | [为 Paddle-TRT 添加elementwise_mod 算子](https://github.com/PaddlePaddle/Paddle/issues/50660#task76) |                  |          |
| 77   | ⭐️    | [为 Paddle-TRT 添加 inverse 算子](https://github.com/PaddlePaddle/Paddle/issues/50660#task77) |                  |          |
| 78   | ⭐️⭐️   | [为 Paddle-TRT 添加 cumsum 算子](https://github.com/PaddlePaddle/Paddle/issues/50660#task78) |                  |          |
| 79   | ⭐️⭐️   | [为 Paddle-TRT 添加 while 算子](https://github.com/PaddlePaddle/Paddle/issues/50660#task79) |                  |          |
| 80   | ⭐️⭐️   | [为 Paddle-TRT 添加 conditional_block 算子](https://github.com/PaddlePaddle/Paddle/issues/50660#task80) |                  |          |

**CINN开发**<a name='cinn'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 81   | ⭐️    | [为神经网络编译器 CINN 增加 bitcast_convert 算子](https://github.com/PaddlePaddle/Paddle/issues/50661#task81) |                  |          |
| 82   | ⭐️    | [为神经网络编译器 CINN 增加 triangular_solve 算子](https://github.com/PaddlePaddle/Paddle/issues/50661#task82) |                  |          |
| 83   | ⭐️⭐️   | [为神经网络编译器 CINN 增加 resize 算子](https://github.com/PaddlePaddle/Paddle/issues/50661#task83) |                  |          |
| 84   | ⭐️    | [为神经网络编译器 CINN 增加 ReverseComputeInline 原语](https://github.com/PaddlePaddle/Paddle/issues/50661#task84) |                  |          |

**开源社区洞察**<a name='insight'></a>

注：该类型任务不需要提交提案（设计文档），合入 PR 即为完成任务，最终排名由评判委员会打分，分数最高者获胜。

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 85   | ⭐️    | [【开源贡献对比研究】以 CI 流水线为例](https://github.com/PaddlePaddle/Paddle/issues/50662#task85) |                  |          |
| 86   | ⭐️    | [【开源贡献对比研究】以贡献文档为例](https://github.com/PaddlePaddle/Paddle/issues/50662#task86) |                  |          |
| 87   | ⭐️    | [【开源贡献对比研究】以代码组织为例](https://github.com/PaddlePaddle/Paddle/issues/50662#task87) |                  |          |
| 88   | ⭐️⭐️   | [飞桨框架API文档发布的工具链的分析](https://github.com/PaddlePaddle/Paddle/issues/50662#task88) |                  |          |

**其他**<a name='other'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 89   | ⭐️    | [清理动态  import语句，解决circle import 问题](https://github.com/PaddlePaddle/Paddle/issues/50663#task89) |                  |          |
| 90   | ⭐️⭐️   | [JITLayer C++ 端暴露AnaLysisConfig 给用户，提升易用性](https://github.com/PaddlePaddle/Paddle/issues/50663#task90) |                  |          |
| 91   | ⭐️⭐️   | [TensorHook支持动转静](https://github.com/PaddlePaddle/Paddle/issues/50630#task91) |                  |          |
| 92   | ⭐️⭐️   | [ppocr det&rec 全量化模型在 tim-vx（晶晨/瑞芯微） 等设备上的精度提升](https://github.com/PaddlePaddle/Paddle/issues/50663#task92) |                  |          |
| 93   | ⭐️    | [增加 linux 下 cpu tensor file_descriptor 传输方案 ](https://github.com/PaddlePaddle/Paddle/issues/50663#task93) |                  |          |
| 94   | ⭐️⭐️   | [GPU tensor 全局引用计数 ](https://github.com/PaddlePaddle/Paddle/issues/50663#task94) |                  |          |
| 95   | ⭐️⭐️   | [CPU tensor mac/win32 传输 + 适配 DataLoader](https://github.com/PaddlePaddle/Paddle/issues/50663#task95) |                  |          |
| 96   | ⭐️⭐️   | [基于 Paddle 的数据并行DataParallel 添加 join 接口，满足数据流的不均衡输入](https://github.com/PaddlePaddle/Paddle/issues/50663#task96) |                  |          |
| 97   | ⭐️⭐️   | [基于 Paddle 实现异构集群数据并行训练自动负载均衡](https://github.com/PaddlePaddle/Paddle/issues/50663#task97) |                  |          |


**赛道二：模型套件开源贡献**<a name='paddlefamily'></a>

**自然语言处理模型库 PaddleNLP**<a name='nlp'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 98   | ⭐️    | [升级paddlenlp.transformers内的模型结构并且增加基础单测](https://github.com/PaddlePaddle/Paddle/issues/50631#task98) |                  |          |
| 99   | ⭐️    | [升级paddlenlp.transformers内的模型结构并且增加基础单测](https://github.com/PaddlePaddle/Paddle/issues/50631#task99) |                  |          |
| 100  | ⭐️    | [升级paddlenlp.transformers内的模型结构并且增加基础单测](https://github.com/PaddlePaddle/Paddle/issues/50631#task100) |                  |          |
| 101  | ⭐️    | [升级paddlenlp.transformers内的模型结构并且增加基础单测](https://github.com/PaddlePaddle/Paddle/issues/50631#task101) |                  |          |
| 102  | ⭐️    | [给AutoConverter增加新的模型组网的支持](https://github.com/PaddlePaddle/Paddle/issues/50631#task102) |                  |          |
| 103  | ⭐️    | [新增tie_weights能力](https://github.com/PaddlePaddle/Paddle/issues/50631#task103) |                  |          |
| 104  | ⭐️    | [生成式API对齐HF，包括sample和contrastive_search](https://github.com/PaddlePaddle/Paddle/issues/50631#task104) |                  |          |
| 105  | ⭐    | [基于PaddleNLP PPDiffusers 训练 AIGC 趣味模型](https://github.com/PaddlePaddle/Paddle/issues/50631#task105) |                  |          |
| 106  | ⭐️⭐️   | [论文名称：Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion](https://github.com/PaddlePaddle/Paddle/issues/50631#task106) |                  |          |
| 107  | ⭐️⭐️   | [论文名称：Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation](https://github.com/PaddlePaddle/Paddle/issues/50631#task107) |                  |          |
| 108  | ⭐️⭐️   | [论文名称：AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://github.com/PaddlePaddle/Paddle/issues/50631#task108) |                  |          |
| 109  | ⭐️⭐️   | [论文名称：Zero-shot Image-to-Image Translation](https://github.com/PaddlePaddle/Paddle/issues/50631#task109) |                  |          |
| 110  | ⭐️    | [论文名称：Multi-Concept Customization of Text-to-Image Diffusion](https://github.com/PaddlePaddle/Paddle/issues/50631#task110) |                  |          |
| 111  | ⭐️⭐️   | [论文名称：DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://github.com/PaddlePaddle/Paddle/issues/50631#task111) |                  |          |

**文字识别开发套件 PaddleOCR**<a name='ocr'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 112  | ⭐️    | [论文复现：Multi-Granularity Prediction for Scene Text Recognition](https://github.com/PaddlePaddle/Paddle/issues/50631#task112) |                  |          |
| 113  | ⭐️    | [论文复现：PageNet: Towards End-to-End Weakly Supervised Page-Level Handwritten Chinese Text Recognition](https://github.com/PaddlePaddle/Paddle/issues/50631#task113) |                  |          |
| 114  | ⭐️    | [论文复现：GLASS: Global to Local Attention for Scene-Text Spotting](https://github.com/PaddlePaddle/Paddle/issues/50631#task114) |                  |          |
| 115  | ⭐️    | [论文复现：TPSNet: Reverse Thinking of Thin Plate Splines for Arbitrary Shape Scene Text Representation](https://github.com/PaddlePaddle/Paddle/issues/50631#task115) |                  |          |
| 116  | ⭐️⭐️   | [论文复现：ABCNet v2: Adaptive Bezier-Curve Network for Real-time End-to-end Text Spotting](https://github.com/PaddlePaddle/Paddle/issues/50631#task116) |                  |          |
| 117  | ⭐️    | [论文复现：CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition](https://github.com/PaddlePaddle/Paddle/issues/50631#task117) |                  |          |
| 118  | ⭐️    | [论文复现：Syntax-Aware Network for Handwritten Mathematical Expression Recognition](https://github.com/PaddlePaddle/Paddle/issues/50631#task118) |                  |          |
| 119  | ⭐️⭐️   | [论文复现：Learning From Documents in the Wild to Improve Document Unwarping](https://github.com/PaddlePaddle/Paddle/issues/50631#task119) |                  |          |
| 120  | ⭐️⭐️   | [论文复现：C3-STISR: Scene Text Image Super-resolution with Triple Clues](https://github.com/PaddlePaddle/Paddle/issues/50631#task120) |                  |          |
| 121  | ⭐️⭐️   | [PaddleOCR js部署](https://github.com/PaddlePaddle/Paddle/issues/50631#task121) |                  |          |
| 122  | ⭐️    | [《动手学OCR》升级](https://github.com/PaddlePaddle/Paddle/issues/50631#task122) |                  |          |
| 123  | ⭐️    | [模型库中文适配](https://github.com/PaddlePaddle/Paddle/issues/50631#task123) |                  |          |

**图像分类开发套件 PaddleClas** <a name='clas'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 124  | ⭐️    | [论文复现：More ConvNets in the 2020s: Scaling up Kernels Beyond 51x51 using Sparsity](https://github.com/PaddlePaddle/Paddle/issues/50631#task124) |                  |          |
| 125  | ⭐️    | [论文复现：Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs](https://github.com/PaddlePaddle/Paddle/issues/50631#task125) |                  |          |
| 126  | ⭐️    | [论文复现：Revisiting ResNets: Improved Training and Scaling Strategies](https://github.com/PaddlePaddle/Paddle/issues/50631#task126) |                  |          |
| 127  | ⭐️    | [论文复现：Separable Self-attention for Mobile Vision Transformers](https://github.com/PaddlePaddle/Paddle/issues/50631#task127) |                  |          |
| 128  | ⭐️    | [论文复现：MobileViTv3: Mobile-Friendly Vision Transformer with Simple and Effective Fusion of Local, Global and Input Features](https://github.com/PaddlePaddle/Paddle/issues/50631#task128) |                  |          |
| 129  | ⭐️    | [论文复现：Model Rubik’s Cube: Twisting Resolution, Depth andWidth for TinyNets](https://github.com/PaddlePaddle/Paddle/issues/50631#task129) |                  |          |
| 130  | ⭐️    | [论文复现：FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](https://github.com/PaddlePaddle/Paddle/issues/50631#task130) |                  |          |
| 131  | ⭐️    | [论文复现：Hypergraph-Induced Semantic Tuplet Loss for Deep Metric Learning](https://github.com/PaddlePaddle/Paddle/issues/50631#task131) |                  |          |
| 132  | ⭐️    | [论文复现：iBOT: Image BERT Pre-Training with Online Tokenizer](https://github.com/PaddlePaddle/Paddle/issues/50631#task132) |                  |          |
| 133  | ⭐️    | [论文复现：Forward Compatible Training for Large-Scale Embedding Retrieval Systems](https://github.com/PaddlePaddle/Paddle/issues/50631#task133) |                  |          |
| 134  | ⭐️    | [论文复现：ICE: Inter-instance Contrastive Encoding for Unsupervised Person Re-identification](https://github.com/PaddlePaddle/Paddle/issues/50631#task134) |                  |          |
| 135  | ⭐️    | [论文复现：VL-LTR: Learning Class-wise Visual-Linguistic Representation for Long-Tailed Visual Recognition](https://github.com/PaddlePaddle/Paddle/issues/50631#task135) |                  |          |
| 136  | ⭐️    | [论文复现：Recall@k Surrogate Loss with Large Batches and Similarity Mixup](https://github.com/PaddlePaddle/Paddle/issues/50631#task136) |                  |          |
| 137  | ⭐️    | [论文题目：Learning Transferable Visual Models From Natural Language Supervision(CLIP)论文题目：BEIT V2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://github.com/PaddlePaddle/Paddle/issues/50631#task137) |                  |          |
| 138  | ⭐️    | [论文题目：Expanding language-image pretrained models for general video recognition(X-CLIP)论文题目：Context Autoencoder for Self-Supervised Representation Learning(CAE)](https://github.com/PaddlePaddle/Paddle/issues/50631#task138) |                  |          |
| 139  | ⭐️    | [论文题目：BEIT: BERT Pre-Training of Image Transformers论文题目：Masked Autoencoders Are Scalable Vision Learners(MAE)论文题目：Exploring Simple Siamese Representation Learning(SimSam)](https://github.com/PaddlePaddle/Paddle/issues/50631#task139) |                  |          |
| 140  | ⭐️    | [论文题目：Improved baselines with momentum contrastive learning(MoCov2)论文题目：Bootstrap your own latent: A new approach to self-supervised learning（BYOL）论文题目：Unsupervised Learning of Visual Features by Contrasting Cluster Assignments （SwAV）](https://github.com/PaddlePaddle/Paddle/issues/50631#task140) |                  |          |
| 141  | ⭐️⭐️   | [PP-LCNet v3 下游场景验证](https://github.com/PaddlePaddle/Paddle/issues/50631#task141) |                  |          |
| 142  | ⭐️⭐️   | [PP-HGNet v2下游场景验证](https://github.com/PaddlePaddle/Paddle/issues/50631#task142) |                  |          |
| 143  | ⭐️⭐️   | [万类通用识别数据集制作](https://github.com/PaddlePaddle/Paddle/issues/50631#task143) |                  |          |

**图像分割开发套件 PaddleSeg**<a name='seg'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 144  | ⭐️    | [SeaFormer: Squeeze-enhanced Axial Transformer for Mobile Semantic Segmentation](https://github.com/PaddlePaddle/Paddle/issues/50631#task144) |                  |          |
| 145  | ⭐️    | [EfficientFormerV2：Rethinking Vision Transformers for MobileNet Size and Speed](https://github.com/PaddlePaddle/Paddle/issues/50631#task145) |                  |          |
| 146  | ⭐️⭐️   | [Fully Convolutional Networks for Panoptic Segmentation](https://github.com/PaddlePaddle/Paddle/issues/50631#task146) |                  |          |
| 147  | ⭐️⭐️   | [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://github.com/PaddlePaddle/Paddle/issues/50631#task147) |                  |          |
| 148  | ⭐️⭐️   | [K-Net: Towards Unified Image Segmentation](https://github.com/PaddlePaddle/Paddle/issues/50631#task148) |                  |          |
| 149  | ⭐️⭐️   | [Highly Accurate Dichotomous Image Segmentation （ECCV 2022）](https://github.com/PaddlePaddle/Paddle/issues/50631#task149) |                  |          |

**遥感影像智能解译开发套件 PaddleRS**<a name='rs'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 150  | ⭐️⭐️   | [PaddleRS集成PaddleDetection的旋转框检测能力](https://github.com/PaddlePaddle/Paddle/issues/50631#task150) |                  |          |
| 151  | ⭐️    | [PaddleRS运行环境打包，并制作端到端遥感建筑物提取教程](https://github.com/PaddlePaddle/Paddle/issues/50631#task151) |                  |          |
| 152  | ⭐️    | [PaddleRS API 文档完善](https://github.com/PaddlePaddle/Paddle/issues/50631#task152) |                  |          |
| 153  | ⭐️    | [PaddleRS 英文文档](https://github.com/PaddlePaddle/Paddle/issues/50631#task153) |                  |          |

**目标检测开发开发套件 PaddleDetection**<a name='detection'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 154  | ⭐️    | [论文复现：YOLOv6 v3.0: A Full-Scale Reloading](https://github.com/PaddlePaddle/Paddle/issues/50631#task154) |                  |          |
| 155  | ⭐️    | [YOLOv8模型复现](https://github.com/PaddlePaddle/Paddle/issues/50631#task155) |                  |          |
| 156  | ⭐️⭐️   | [论文复现：Open-Vocabulary DETR with Conditional Matching](https://github.com/PaddlePaddle/Paddle/issues/50631#task156) |                  |          |
| 157  | ⭐️⭐️   | [论文复现：PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised Object Detection](https://github.com/PaddlePaddle/Paddle/issues/50631#task157) |                  |          |
| 158  | ⭐️⭐️   | [论文复现：DiffusionDet: Diffusion Model for Object Detection](https://github.com/PaddlePaddle/Paddle/issues/50631#task158) |                  |          |
| 159  | ⭐️⭐️   | [论文复现：PromptDet: Towards Open-vocabulary Detection using Uncurated Images](https://github.com/PaddlePaddle/Paddle/issues/50631#task159) |                  |          |
| 160  | ⭐️⭐️   | [论文复现：Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone](https://github.com/PaddlePaddle/Paddle/issues/50631#task160) |                  |          |
| 161  | ⭐️    | [论文复现：CLRNet: Cross Layer Refinement Network for Lane Detection](https://github.com/PaddlePaddle/Paddle/issues/50631#task161) |                  |          |
| 162  | ⭐️    | [论文复现：Attentional Graph Neural Network for Parking Slot Detection](https://github.com/PaddlePaddle/Paddle/issues/50631#task162) |                  |          |
| 163  | ⭐️    | [基于PaddleDetection PP-TinyPose，新增手势关键点检测模型](https://github.com/PaddlePaddle/Paddle/issues/50631#task163) |                  |          |
| 164  | ⭐️    | [PaddleDetection重点模型接入huggingface](https://github.com/PaddlePaddle/Paddle/issues/50631#task164) |                  |          |

**3D任务开发开发套件 Paddle3D**<a name='3d'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 165  | ⭐️    | [Camera标定LiDAR标定](https://github.com/PaddlePaddle/Paddle/issues/50631#task165) |                  |          |
| 166  | ⭐️    | [Paddle3D目标检测结果可视化](https://github.com/PaddlePaddle/Paddle/issues/50631#task166) |                  |          |
| 167  | ⭐️    | [Paddle3D&ROS联合开发Demo](https://github.com/PaddlePaddle/Paddle/issues/50631#task167) |                  |          |
| 168  | ⭐️⭐️   | [Geometry Uncertainty Projection Network for Monocular 3D Object Detection](https://github.com/PaddlePaddle/Paddle/issues/50631#task168) |                  |          |
| 169  | ⭐️⭐️   | [DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](https://github.com/PaddlePaddle/Paddle/issues/50631#task169) |                  |          |
| 170  | ⭐️⭐️   | [TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers](https://github.com/PaddlePaddle/Paddle/issues/50631#task170) |                  |          |
| 171  | ⭐️⭐️   | [FUTR3D: A Unified Sensor Fusion Framework for 3D Detection](https://github.com/PaddlePaddle/Paddle/issues/50631#task171) |                  |          |
| 172  | ⭐️⭐️   | [Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction](https://github.com/PaddlePaddle/Paddle/issues/50631#task172) |                  |          |
| 173  | ⭐️⭐️   | [Point-based Neural Radiance Fields](https://github.com/PaddlePaddle/Paddle/issues/50631#task173) |                  |          |
| 174  | ⭐️⭐️   | [Scenes as Neural Radiance Fields for View Synthesis](https://github.com/PaddlePaddle/Paddle/issues/50631#task174) |                  |          |
| 175  | ⭐️⭐️   | [TensoRF: Tensorial Radiance Fields](https://github.com/PaddlePaddle/Paddle/issues/50631#task175) |                  |          |
| 176  | ⭐️    | [相机去畸变C++自定义算子开发](https://github.com/PaddlePaddle/Paddle/issues/50631#task176) |                  |          |

**全场景高性能AI部署工具 FastDeploy**<a name='fastdeploy'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 177  | ⭐️    | [将PP-YOLOE-R在**算能BM1684**部署。利用FastDeploy，将PP-YOLOE-R在**算能BM1684X**部署](https://github.com/PaddlePaddle/Paddle/issues/50631#task177) |                  |          |
| 178  | ⭐️    | [集成SOLOv2模型到FastDpeloy，并在Paddle Infenence、ONNX Runtime、TernsorRT后端测试验证](https://github.com/PaddlePaddle/Paddle/issues/50631#task178) |                  |          |
| 179  | ⭐️⭐️   | [将PointPillars集成到FastDeploy，并在**Jetson Orin**硬件上部署验证精度和速度](https://github.com/PaddlePaddle/Paddle/issues/50631#task179) |                  |          |
| 180  | ⭐️⭐️   | [在FastDeploy中集成集成**地平线**推理引擎，在PP-YOLOE完成模型转换测试](https://github.com/PaddlePaddle/Paddle/issues/50631#task180) |                  |          |
| 181  | ⭐️⭐️   | [完成**TVM**接入FastDeploy，并在PP-YOLOE模型上验证正确性](https://github.com/PaddlePaddle/Paddle/issues/50631#task181) |                  |          |
| 182  | ⭐️⭐️   | [完成pp-ocrv3在**RK3588**上的部署，并验证正确性](https://github.com/PaddlePaddle/Paddle/issues/50631#task182) |                  |          |
| 183  | ⭐️⭐️   | [使用FastDeploy完成 ELECTRA 模型GLUE任务模型部署](https://github.com/PaddlePaddle/Paddle/issues/50631#task183) |                  |          |
| 184  | ⭐️    | [在FastDeploy C API的基础上，使用rust完成PaddleDetection部署](https://github.com/PaddlePaddle/Paddle/issues/50631#task185) |                  |          |
| 185  | ⭐️    | [在FastDeploy C++ API的基础上，使用java完成PaddleDetection部署](https://github.com/PaddlePaddle/Paddle/issues/50631#task185) |                  |          |
| 186  | ⭐️    | [在FastDeploy C API的基础上，使用go完成PaddleDetection部署](https://github.com/PaddlePaddle/Paddle/issues/50631#task186) |                  |          |

**语音模型库 PaddleSpeech**<a name='speech'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 187  | ⭐️⭐️   | [模型复现： pruned_transducer_stateless8](https://github.com/PaddlePaddle/Paddle/issues/50631#task187) |                  |          |
| 188  | ⭐️⭐️   | [模型复现：hubert](https://github.com/PaddlePaddle/Paddle/issues/50631#task188) |                  |          |
| 189  | ⭐️⭐️   | [模型复现：wavlm](https://github.com/PaddlePaddle/Paddle/issues/50631#task189) |                  |          |
| 190  | ⭐️⭐️   | [模型复现：iSTFTNet](https://github.com/PaddlePaddle/Paddle/issues/50631#task190) |                  |          |
| 191  | ⭐️⭐️   | [模型复现：JETS](https://github.com/PaddlePaddle/Paddle/issues/50631#task191) |                  |          |
| 192  | ⭐️    | [使用 Gradio 为 PaddleSpeech 语音识别训练过程绘制WebUI工具箱（以conformer模型为例）](https://github.com/PaddlePaddle/Paddle/issues/50631#task192) |                  |          |
| 193  | ⭐️    | [使用 Gradio 为 PaddleSpeech 语音合成声学模型训练过程绘制WebUI工具箱（以fastspeech2模型为例）](https://github.com/PaddlePaddle/Paddle/issues/50631#task193) |                  |          |
| 194  | ⭐️    | [使用 Gradio 为 PaddleSpeech 语音合成声学模型训练过程绘制WebUI工具箱（以conformer模型为例）](https://github.com/PaddlePaddle/Paddle/issues/50631#task194) |                  |          |

**飞桨科学计算 PaddleScience**<a name='science'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 195  | ⭐️⭐️   | [多学科物理场可视化组件开发](https://github.com/PaddlePaddle/Paddle/issues/50631#task195) |                  |          |
| 196  | ⭐️    | [FVM/FEM/LBM等主流CAE结果提取组件开发](https://github.com/PaddlePaddle/Paddle/issues/50631#task196) |                  |          |
| 197  | ⭐️    | [论文复现：Robust Regression with Highly Corrupted Data via Physics Informed Neural Networks](https://github.com/PaddlePaddle/Paddle/issues/50631#task197) |                  |          |
| 198  | ⭐️    | [论文复现：SPNets: Differentiable Fluid Dynamics for Deep Neural Networks](https://github.com/PaddlePaddle/Paddle/issues/50631#task198) |                  |          |
| 199  | ⭐️    | [论文复现：Reduced-order Model for Flows via Neural Ordinary Differential Equations](https://github.com/PaddlePaddle/Paddle/issues/50631#task199) |                  |          |
| 200  | ⭐️    | [论文复现：An AI-based Domain-Decomposition Non-Intrusive Reduced-Order Model for Extended Domains applied to Multiphase Flow in Pipes](https://github.com/PaddlePaddle/Paddle/issues/50631#task200) |                  |          |
| 201  | ⭐️⭐️   | [论文复现：Learning to regularize with a variational autoencoder for hydrologic inverse analysis](https://github.com/PaddlePaddle/Paddle/issues/50631#task201) |                  |          |
| 202  | ⭐️    | [论文复现：Disentangling Generative Factors of Physical Fields Using Variational Autoencoders](https://github.com/PaddlePaddle/Paddle/issues/50631#task202) |                  |          |
| 203  | ⭐️⭐️   | [论文复现：Deep Learning Methods for Reynolds-Averaged Navier-Stokes Simulationsof Airfoil Flows](https://github.com/PaddlePaddle/Paddle/issues/50631#task203) |                  |          |
| 204  | ⭐️⭐️   | [开放赛题- 车辆标准部件受力、变形分析](https://github.com/PaddlePaddle/Paddle/issues/50631#task204) |                  |          |


**赛道三：生态伙伴开源贡献**<a name='paddlefriend'></a>

**OpenVINO项目**<a name='openvino'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 205  | ⭐️⭐️   | [为OpenVINO notebook新增demo示例](https://github.com/PaddlePaddle/Paddle/issues/50632#task205) |                  |          |
| 206  | ⭐️⭐️   | [为 OpenVINO 实现 Paddle 算子 flip 转换](https://github.com/PaddlePaddle/Paddle/issues/50632#task206) |                  |          |
| 207  | ⭐️⭐️   | [为 OpenVINO 实现 Paddle 算子 linspace 转换](https://github.com/PaddlePaddle/Paddle/issues/50632#task207) |                  |          |
| 208  | ⭐️⭐️   | [为 OpenVINO 实现 Paddle 算子 set_value 转换](https://github.com/PaddlePaddle/Paddle/issues/50632#task208) |                  |          |
| 209  | ⭐️    | [为 OpenVINO 实现 Paddle 算子 silu 转换](https://github.com/PaddlePaddle/Paddle/issues/50632#task209) |                  |          |
| 210  | ⭐️    | [为 OpenVINO 实现 Paddle 算子one_hot_v2 转换](https://github.com/PaddlePaddle/Paddle/issues/50632#task210) |                  |          |
| 211  | ⭐️    | [为 OpenVINO 实现 Paddle 算子softshrink 转换](https://github.com/PaddlePaddle/Paddle/issues/50632#task211) |                  |          |
| 212  | ⭐️    | [为 OpenVINO 实现 Paddle 算子 mean 转换](https://github.com/PaddlePaddle/Paddle/issues/50632#task212) |                  |          |
| 213  | ⭐️    | [为 OpenVINO 实现 Paddle 算子index_select转换](https://github.com/PaddlePaddle/Paddle/issues/50632#task213) |                  |          |

**Arm项目**<a name='arm'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 214  | ⭐️    | [Arm 虚拟硬件上完成 PP-OCR 文本检测模型的部署与优化](https://github.com/PaddlePaddle/Paddle/issues/50632#task214) |                  |          |
| 215  | ⭐️    | [Arm 虚拟硬件上完成 PP-OCR 文本方向分类模型的部署与优化](https://github.com/PaddlePaddle/Paddle/issues/50632#task215) |                  |          |
| 216  | ⭐️    | [Arm 虚拟硬件上完成 PaddleClas 模型的部署](https://github.com/PaddlePaddle/Paddle/issues/50632#task216) |                  |          |
| 217  | ⭐️⭐️   | [Arm 虚拟硬件上完成 PaddleClas 模型的部署与优化](https://github.com/PaddlePaddle/Paddle/issues/50632#task217) |                  |          |
| 218  | ⭐️    | [Arm 虚拟硬件上完成 PaddleSeg 模型的部署](https://github.com/PaddlePaddle/Paddle/issues/50632#task218) |                  |          |
| 219  | ⭐️⭐️   | [Arm 虚拟硬件上完成 PaddleSeg 模型的部署与优化](https://github.com/PaddlePaddle/Paddle/issues/50632#task219) |                  |          |
| 220  | ⭐️⭐️   | [Arm 虚拟硬件上完成 PP-TinyPose 模型的部署与优化并在物理开发板上进行验证](https://github.com/PaddlePaddle/Paddle/issues/50632#task220) |                  |          |
| 221  | ⭐️⭐️   | [Arm 虚拟硬件上完成 PaddleSpeech 模型的部署与优化](https://github.com/PaddlePaddle/Paddle/issues/50632#task221) |                  |          |
| 222  | ⭐️    | [为 TVM 增加单个 Paddle 算子 yolo_box 并在 Arm 虚拟硬件上完成 PP-Yolo 模型的部署](https://github.com/PaddlePaddle/Paddle/issues/50632#task222) |                  |          |
| 223  | ⭐️⭐️   | [为 TVM 增加多个 Paddle 算子 stack 和 prior_box 并在 Arm 虚拟硬件上完成 SSD 模型的部署](https://github.com/PaddlePaddle/Paddle/issues/50632#task223) |                  |          |

**Jina AI项目**<a name='jina'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 224  | ⭐️    | [利用 Jina AI 来部署开放域聊天大模型 PLATO-XL](https://github.com/PaddlePaddle/Paddle/issues/50632#task224) |                  |          |
| 225  | ⭐️    | [使用 Jina AI 和 UIE 搭建可视化信息抽取系统](https://github.com/PaddlePaddle/Paddle/issues/50632#task225) |                  |          |

**TVM项目**<a name='tvm'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 226 |   待发布  | |                  |          |
| 227 |   待发布  | |                  |          |
| 228 |   待发布  | |                  |          |
| 229 |   待发布  | |                  |          |
| 230 |   待发布  | |                  |          |
| 231 |   待发布  | |                  |          |
| 232 |   待发布  | |                  |          |
| 233 |   待发布  | |                  |          |
| 234 |   待发布  | |                  |          |
| 235 |   待发布  | |                  |          |


**赛道四：产业合作开源贡献**<a name='paddleindustry'></a>

| 序号 | 难度 | 任务 ISSUE                                                   | 队伍名称/状态/PR | 完成队伍 |
| ---- | ---- | ------------------------------------------------------------ | ---------------- | -------- |
| 236  | ⭐️    | [基于PaddleOCR的工频场强计读数识别](https://github.com/PaddlePaddle/Paddle/issues/50633#task236) |                  |          |
| 237  | ⭐️    | [基于PaddleNLP的跨模态文档信息抽取](https://github.com/PaddlePaddle/Paddle/issues/50633#task237) |                  |          |
| 238  | ⭐️    | [基于PaddleClas的中草药识别](https://github.com/PaddlePaddle/Paddle/issues/50633#task238) |                  |          |
| 239  | ⭐️    | [基于PaddleDetection的无人机航拍图像检测](https://github.com/PaddlePaddle/Paddle/issues/50633#task239) |                  |          |








欢迎大家参与~~
