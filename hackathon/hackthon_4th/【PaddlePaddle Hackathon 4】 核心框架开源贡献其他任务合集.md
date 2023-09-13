# 【PaddlePaddle Hackathon 4】核心框架开源贡献其他任务合集

（此 ISSUE 为 PaddlePaddle Hackathon 第四期活动的任务 ISSUE，更多详见 [【PaddlePaddle Hackathon 第四期】任务总览](https://github.com/PaddlePaddle/Paddle/issues/51281)）

注：报名参与其他任务的同学可以向 paddle-hack@baidu.com 发邮件，我们会邀请你加入对应的社群参与讨论。开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下：

### No.89：清理动态  import语句，解决circle import 问题 <a name='task89'></a>

- 任务难度：基础
- 详细描述：
  - 现状：目前飞桨框架 python 目录下部分模块函数存在动态 import 语句，即 import 语句并未放到文件的最开始地方。这种动态 import 语句存在两个问题：
    - 可能因为存在 circle import 而临时添加的，但不合规范
    - 每次调用函数都会解释执行 import，影响性能
- 任务提交：
  - 设计文档：在 [paddle/community](https://github.com/PaddlePaddle/community/tree/master/rfcs) 仓库下新建 RFC，并提供技术设计思路文档
  - 提交 PR 至Paddle代码仓库。在保持 API 不变的情况下，可适当调整函数或文件位置，以实现「分层管理」

### No.90：JITLayer C++ 端暴露AnaLysisConfig 给用户，提升易用性 <a name='task90'></a>

- 任务难度：进阶
- 详细描述：
  - 现状：paddle/fluid/jit 目录下的Layer是孵化中的项目，旨在提供一个与Python端nn.Layer相同使用方式的后端数据结构，底层封装了预测执行引擎：AnalysisPredictor——推理部署的核心引擎。目前存在如下问题：
    - jit/engine/predictor_engine.h 里的 PredictorEngine 数据结构并未向用户提供灵活 AnaLysisConfig 选项
    - AnaLysisConfig 选项可用于设置 GPU、CPU、MKLDNN、以及自定义优化策略，对提升 Layer 易用性有重要意义。
- 任务提交：
  - 设计文档：在 [paddle/community](https://github.com/PaddlePaddle/community/tree/master/rfcs) 仓库下新建 RFC，并提供技术设计思路文档
  - 提交PR代码至 paddle/fluid/jit 目录下

### No.91：TensorHook支持动转静 <a name='task91'></a>

- 任务难度：进阶
- 详细描述：
  - 现状：动态图下 Tensor 提供了 [register_hook](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#register-hook-hook) 方法，支持用户注册一个hook函数自定义反向grad的数据计算逻辑，此接口仅支持动态图，静态图下缺失对 Tensor 的反向grad同等的逻辑实现，属于动静行为不统一。当用户模型代码包含 register_hook 的用法时，动转静会报错。
- 任务提交：
  - 设计文档：在 [paddle/community](https://github.com/PaddlePaddle/community/tree/master/rfcs) 仓库下新建 RFC，并提供技术设计思路文档
  - 提交PR代码至 python/paddle/jit 动转静目录下

### No.92：ppocr det&rec 全量化模型在 tim-vx（晶晨/瑞芯微） 等设备上的精度提升 <a name='task92'></a>

- 技术标签：深度学习，C++、压缩量化
- 任务难度：进阶
- 详细描述：
  - PP-OCRv3 rec FP32 精度为 76.87，使用PaddleSlim auto-compress全量化后 CPU 侧 eval 精度为75.43，但 GPU、NPU 等目前精度较低。需要通过调整量化工具参数、微调模型等手段，让 PP-OCRv3 rec 的全量化模型在 NPU 上的精度趋近 FP32，目标精度为70.0。
  - 硬件平台为 tim-vx（晶晨/瑞芯微）等任一芯原NPU，如瑞芯微RV1126、1109、晶晨A311D等，系统 OS 建议使用 Linux。
- 提交内容：
  - 完成PP-OCRv3 rec的全量化模型提交，提交至 [Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo) repo下 Paddle-Lite-Demo/[ocr](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/ocr)/assets/ 目录中，会有 RD 同学整理归档。
  - 中文文档，修改 PaddleSlim auto-compress 的文档，描述对该模型在使用auto-compress量化过程中做了哪些修改（量化配置修改、模型修改等）以达到精度在端侧的提升。提交至 [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/ocr) repo下PaddleSlim/[example](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example)/[auto_compression](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression)/ocr/README.md 文档文件。
  - 验收标准：先提交该全量化模型以及模型在 tim-vx（晶晨/瑞芯微） 等任一芯原NPU上的精度结果（开源数据集icdar2015，精度70.0以上），待RD验证通过后，提交 PR 到Paddle-Lite-Demo 和 PaddleSlim 仓库。
- 技术要求：部署
  - 熟练掌握 C++、Python 开发。
  - 了解 AI 模型及全量化。
  - 了解 OCR 算法。
  - 掌握 PaddleSlim auto-compress量化工具、PP-OCRv3 模型的修改、Paddle-Lite + TIM-VX 芯原 NPU 部署。

### No.93：增加 linux 下 cpu tensor file_descriptor 传输方案 <a name='task93'></a>

- 技术标签：深度学习，C++
- 任务难度：基础
- 详细描述：
  - 背景：Multiprocessing 是支持进程间 Tensor 传输的一种方式。#[37302](https://github.com/PaddlePaddle/Paddle/pull/37302)  初步支持了paddle的tensor进程间传输，需要继续完善，可参考 [paddle.multiprocessing 设计文档](https://github.com/PaddlePaddle/Paddle/wiki/paddle进程间tensor传输设计文档-paddle.multiprocessing)。
  - 目前 paddle 支持了 file_system 的 cpu 传输方式，以文档形式存储传输tensor 的中间态。file_descriptor 打开文件句柄之后立即删除，更加安全，不容易发生文件残留。
- 提交内容：
  - 方案设计文档，并提 PR 至 community repo 的 [rfcs](https://github.com/PaddlePaddle/community/tree/master/rfcs) 目录；
  - 完成 file_descriptor 的支持。提交到 Paddle 主 repo
  - file_descriptor的功能对齐竞品，全面且完善支持，切换为默认传输方式。
  - 验收标准：
    - 自测传输10000次，不发生文件残留；
    - 传输速度与竞品差距10%内
- 技术要求：
  - 熟练掌握 C++、Python 开发。

### No.94：GPU tensor 全局引用计数 <a name='task94'></a>

- 技术标签：深度学习，C++
- 任务难度：进阶
- 详细描述：
  - 背景：Multiprocessing 是支持进程间 Tensor 传输的一种方式。#[37302](https://github.com/PaddlePaddle/Paddle/pull/37302)  初步支持了paddle的tensor进程间传输，需要继续完善，可参考 [paddle.multiprocessing 设计文档](https://github.com/PaddlePaddle/Paddle/wiki/paddle进程间tensor传输设计文档-paddle.multiprocessing)。
  - 传输过程是生产者消费者场景，为了维护 tensor 的生命周期，需要将cuda 传输的 tensor 与文件绑定，实现全局引用计数。
  - 目前已有初步实现，需要继续完善：[ZHUI/Paddle/commit/d1ec460](https://github.com/ZHUI/Paddle/commit/d1ec460388c9c8efbbaf0bff3abca492d1b81a12) [ZHUI/Paddle/commits/multiprocessing_gpu_ref_count](https://github.com/ZHUI/Paddle/commits/multiprocessing_gpu_ref_count)
- 提交内容：
  - 方案设计文档，并提 PR 至 community repo 的 [rfcs](https://github.com/PaddlePaddle/community/tree/master/rfcs) 目录；
  - 支持`CudaIPCSentData`，`CudaIPCRefcountedFiles`等功能，将ipc 传输后的Tensor与`CudaIPCSentData`使用`UniqueVoidPtr`绑定。全局引用计数。
  - 验收标准：
    - 自测传输10000次，不发生文件残留
    - 传输速度与竞品差距10%内
- 技术要求：
  - 熟练掌握 C++、Python、CUDA 代码编写。

### No.95：CPU tensor mac/win32 传输 + 适配 DataLoader <a name='task95'></a>

- 技术标签：深度学习，C++
- 任务难度：进阶
- 详细描述：
  - 背景：Multiprocessing 是支持进程间 Tensor 传输的一种方式。#[37302](https://github.com/PaddlePaddle/Paddle/pull/37302)  初步支持了paddle的tensor进程间传输，需要继续完善，可参考 [paddle.multiprocessing 设计文档](https://github.com/PaddlePaddle/Paddle/wiki/paddle进程间tensor传输设计文档-paddle.multiprocessing)。
  - 支持 mac/win32 平台上cpu tensor进程间传输，并打通DataLoader支持。
- 提交内容：
  - 方案设计文档，并提 PR 至 community repo 的 [rfcs](https://github.com/PaddlePaddle/community/tree/master/rfcs) 目录；
  - 支持 mac/win32 平台上cpu tensor 进程间传输，并打通 DataLoader 支持。
  - 验收标准：
    - 自测传输10000次，不发生文件残留
    - 传输速度与竞品差距10%内
- 技术要求：
  - 熟练掌握 C++、Python 开发。
  - 熟悉 mac/win 文件系统

### No.96：基于 Paddle 的数据并行DataParallel 添加 join 接口，满足数据流的不均衡输入 <a name='task96'></a>

- 任务难度：进阶
- 详细描述：构造上下文管理器，结合 paddle.DataParallel，使得参与的进程使用不均匀的输入进行训练。这个上下文管理器，将跟踪相关的 DP 进程，并通过通信操作来“隐藏”前向和反向计算，以便匹配未加入的 DP 进程。 这将确保每个通信调用，都有一个由已加入的进程进行调用，从而防止在跨进程输入不均匀的情况下，训练时发生 hang等错误。 或者，设置某个环境变量throw_on_early_termination，一旦某个 rank 用完输入数据，所有训练进程都会抛出错误，从而允许程序捕获和处理这些错误。
- 提交内容
  - API 的设计文档，并提 PR 至 community repo 的 [rfcs/APIs](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录
  - API 功能，提交至 [python/paddle/fluid/dygraph/parallel.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/dygraph/parallel.py)
  - 提交代码至 Paddle 代码仓库：[python/paddle/fluid/tests/unittests](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/tests/unittests)
- 技术要求
  - 熟悉数据并行的计算原理
  - 熟悉掌握 c++、cuda、python
  - 熟悉掌握集合通信的基本原理，使用集合通信方式

### No.97：基于 Paddle 实现异构集群数据并行训练自动负载均衡 <a name='task97'></a>

- 任务难度：进阶
- 详细描述：异构集群（P40，K40，V100，A100）的设备有不同的显存大小，算力吞吐。在异构集群上进行分布式数据并行，需要考虑不同硬件的显存和算力，来实现在所有硬件显存不溢出的前提下达到最高的整体训练吞吐。参赛者需要通过 Cost Model 对不同异构硬件的显存和算力、任务模型进行建模，并实现一套负载均衡的算法； 将建模信息作为均衡算法输入，计算出每个设备的上 local batch size 等具体训练参数。评价指标是：任务模型使用均衡算法得到的训练参数，在异构集群上数据并行整体吞吐。
- 提交内容
  - 方案设计文档，并提 PR 至 community repo 的 [rfcs](https://github.com/PaddlePaddle/community/tree/master/rfcs) 目录；
  - Cost Model 需要新增的模块提交到 [auto_parallel/cost](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distributed/auto_parallel/cost) 目录（目录下已有较完备的cost model 基础设施可以直接使用）；
  - 负责均衡算法的实现需要新增的模块提交到 [auto_parallel/tuner](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distributed/auto_parallel/tuner) 目录
  - 实际模型负责均衡脚本提交到 [PaddleFleetx/example](https://github.com/PaddlePaddle/PaddleFleetX/tree/develop/projects/gpt) 目录
- 技术要求
  - 熟悉数据并行的计算原理
  - 熟悉掌握 Cost Model 和 负载均衡
  - 熟悉掌握 c++、cuda、python



～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

### 答疑交流

- 如果在开发中对于上述任务有任何问题，欢迎在本 ISSUE 下留言交流。
