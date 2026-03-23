> 模版：进阶任务：为 OpenVINO notebook 新增 demo 示例
> * 技术标签：深度学习框架，Python，OpenVINO
> * 任务难度：进阶
> * 详细描述：作为深度学习工具套件，OpenVINO 可以被广泛应用于不同的应用场景，实现 AI 模型的推理部署，为此我们也想收集更多基于 PaddlePaddle 模型所打造的优秀应用案例，丰富示例仓库。 在这个任务中，你需要在 OpenVINO notebook 仓库新增一个 notebook 示例。本次任务评估将分为两个阶段，在第一阶段中，开发者需要提供一份 RFC，用来描述本次任务的设计方案； 在第二阶段中，我们将从第一阶段提交的结果中，挑选出 2 份比较优秀的方案，并请相对应的开发者根据自己的方案提交 PR。
> * 提交内容：
>    * 第一阶段：RFC 方案提交
>      1. 提交方式：1）以 issue 的形式进行提交到[这儿](https://github.com/openvinotoolkit/openvino_notebooks/issues)，2）标题处打上【PaddlePaddle Hackathon 10】，3）RFC 语言不做强制要求
>      2. 基本要求：1）应用场景与现有 notebook demo 不重复，2）该示例中需要使用最新版本的 openvino 完成所有模型的推理部署
>      3. 筛选依据：1）该示例在真实场景下是否具有实际应用价值，2）该示例的流程逻辑是否清晰，3）运行结果是否符合预期
>         
>    * 第二阶段：PR代码提交
>      1. 提交地址： https://github.com/openvinotoolkit/openvino_notebooks ，标题加上【PaddlePaddle Hackathon 10】字样，并在描述处链接之前的 RFC 地址
>      2. 该 PR 需满足 notebook 贡献规范，开发者需要及时根据 review 的结果进行 PR 修改
>      3. 在比赛过半时设置中期检查会，开发者需汇报项目进度、展示已完成的功能、总结当前遇到的问题与挑战、并介绍后半段比赛的计划安排
> * 参考示例：考虑到通用性，选取的应用场景尽量以英文为主，推荐方案场景有：
>   * PaddleDetection 1）行为识别（打架，抽烟，接打电话 ) 2）车辆追踪 3）高空抛物 4）产线上包装盒内产品计数
>   * PaddleOCR 1）古籍电子化保存
> * 技术要求：熟练掌握OpenVINO python API与其他工具组件的使用方法
> * 参考文档：[OpenVINO notebook仓库](https://github.com/openvinotoolkit/openvino_notebooks)、[OpenVINO notebook仓库代码贡献规范](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/CONTRIBUTING.md)

## 赛题详情（厂商排名不分先后）
### 请 Intel 填写

### 请 AMD 填写

#### 进阶任务：为 Paddle 框架适配 HIP BF16 精度类型
* 技术标签：深度学习框架，C++/HIP，ROCm，MIOpen，PaddlePaddle
* 任务难度：进阶
* 详细描述：
  * **背景**：当前 Paddle 框架在 ROCm 平台上未适配 HIP BF16 精度类型，导致该精度类型下的相关算子不可用。在 AMD GPU 上运行 PaddleOCR-VL 等包含卷积视觉编码器的模型时，由于 BF16 算子缺失，不得不通过 `_keep_in_fp32_modules` 等方式将视觉编码器强制回退到 FP32 精度运行。
    HIP BF16 精度类型不可用对 Paddle 框架在 AMD GPU 上的推理能力有显著制约：
    1. 显存开销倍增：FP32 相比 BF16 显存占用翻倍，制约了可部署的模型规模与批处理容量
    2. 推理性能受损：无法利用 BF16 更高的计算吞吐，推理效率大幅降低
    3. 框架生态受限：不仅影响 PaddleOCR-VL，也制约了 Paddle 框架上其他 LLM 及多模态模型在 AMD GPU 上的推理部署能力
  * **现有 Workaround 参考**：目前 PaddleX 中通过以下方式临时绕过此问题：
    1. 在 `paddlex/inference/utils/misc.py` 的 `is_bfloat16_available()` 中，对 ROCm 平台默认返回 False，强制使用 FP32 精度
    2. 在 `PaddleOCRVLForConditionalGeneration` 模型类中设置 `_keep_in_fp32_modules = ["visual", "mlp_AR"]`，将视觉编码器保持在 FP32
    3. 在静态推理配置中禁用 `conv2d_add_act_fuse_pass` 和 `conv2d_add_fuse_pass`（因依赖 cuDNN，HIP 上不可用）
    4. 详细参考：https://github.com/PaddlePaddle/PaddleX/compare/release/3.3...vivienfanghuagood:PaddleX:dev_rocm70
  * **任务目标**：在 Paddle 框架（https://github.com/PaddlePaddle/Paddle）中适配 HIP BF16 精度类型，使得：
    1. PaddleOCR-VL 等模型在 ROCm 上可以原生使用 BF16 精度进行推理，无需将视觉编码器强制回退到 FP32
    2. Paddle 框架的 ROCm BF16 算子能力得到完善，有利于框架上其他 LLM/多模态模型的 AMD GPU 推理
  * **验收标准**：PaddleOCR-VL-1.5 能在 AMD GPU + ROCm 环境下以 BF16 精度完整运行并输出正确结果。
* 提交内容：
  1. 向 Paddle 主仓库（https://github.com/PaddlePaddle/Paddle）develop 分支提交 Issue 描述问题，并提交 PR 实现 HIP BF16 精度类型适配
  2. 向 PaddleX 仓库（https://github.com/PaddlePaddle/PaddleX）develop 分支提交 Issue 和 PR，移除现有 ROCm BF16 的 workaround 代码
  3. PR 中需包含测试用例和在 AMD GPU 上的验证结果截图
* 提交方式：GitHub Issue + PR，并将验证结果截图发送邮件至 ext_paddle_oss@baidu.com，抄送 Zijun.Wei@amd.com, Huaqiang.Fang@amd.com, bingqing.guo@amd.com
* 技术要求：
  * 熟悉 PaddlePaddle 框架的算子注册与编译机制
  * 了解 ROCm/HIP 编程模型与 MIOpen 库的使用
  * 熟悉 BF16 精度计算的基本原理
  * 具备 C++ 和 Python 开发能力
* 参考文档：
  * [Paddle 主仓库](https://github.com/PaddlePaddle/Paddle)
  * [PaddleX 仓库](https://github.com/PaddlePaddle/PaddleX)
  * [PaddleX ROCm BF16 现有 workaround](https://github.com/PaddlePaddle/PaddleX/compare/release/3.3...vivienfanghuagood:PaddleX:dev_rocm70)
  * [PaddleOCR-VL-1.5 模型](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)
  * [MIOpen 文档](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
  * [PaddleOCR-VL AMD GPU 部署教程](../../pfcc/paddle-hardware/AMD-PaddleOCR-VL-GPU打卡任务.md)

### 请 Arm 填写

### 请 天数智芯 填写

### 请 沐曦 填写

### 请 燧原 填写

### 请 海光 填写

### 请 瀚博 填写

### 请 飞腾 填写

### 请 龙芯 填写

### 请 高通 填写

### 请 联发科技 填写

### 请 紫光展锐 填写

### 请 此芯 填写

### 请 瑞芯微 填写

### 请 地瓜机器人 填写

### 请 麒麟 填写

### 请 统信 填写
