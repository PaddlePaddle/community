此文档展示 **PaddlePaddle Hackathon 第七期活动——开源贡献个人挑战赛科学计算方向任务** 详细介绍

### 开发流程

1. **要求基于 PaddleScience 套件进行开发**，开发文档参考：https://paddlescience-docs.readthedocs.io/zh/latest/zh/development/ 。
2. 复现整体流程和验收标准可以参考：https://paddlescience-docs.readthedocs.io/zh/latest/zh/reproduction/#21，复现完成后需供必要的训练产物，包括训练结束后保存的 `train.log`日志文件、`.pdparams`模型权重参数文件（可用网盘的方式提交）、**撰写的 `.md` 案例文档。**
3. 理解复现流程后，可以参考 PaddleScience 开发文档：https://paddlescience-docs.readthedocs.io/zh/latest/zh/development/ ，了解各个模块如何进行开发、修改，以及参考 API 文档，了解各个现有 API 的功能和作用：https://paddlescience-docs.readthedocs.io/zh/latest/zh/api/arch/ 。
4. 案例文档撰写格式可参考 https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/darcy2d/ ，最终合入后会被渲染并展示在 [PaddleScience 官网文档](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/volterra_ide/)。
5. **如在复现过程中出现需添加的功能无法兼容现有 PaddleScience API 体系（[PaddleScience API 文档](https://paddlescience-docs.readthedocs.io/zh/latest/zh/api/arch/)），则可与论文复现指导人说明情况，并视情况允许直接基于 Paddle API 进行复现。**
6. 若参考代码为 pytorch，则复现过程可以尝试使用 [PaConvert](https://github.com/PaddlePaddle/PaConvert) 辅助完成代码转换工作，然后可以尝试使用 [PaDiff](https://github.com/PaddlePaddle/PaDiff) 工具辅助完成前反向精度对齐，从而提高复现效率。

### 验收标准

参考模型复现指南验收标准部分 https://paddlescience-docs.readthedocs.io/zh/latest/zh/reproduction/#3

## 【开源贡献个人挑战赛-科学计算方向】任务详情

### NO.1 为开源符号回归库进行 paddle 适配

**论文链接：**

https://arxiv.org/pdf/2006.11287v2

**代码复现：**

为该库进行 paddle 适配（包含实现代码和单测代码），并为 example 中的 demo 的 “High-dimensional input: Neural Nets + Symbolic Regression” 部分编写 paddle 版本代码，要求可以在 paddle 下跑通，并对齐精度

**参考代码链接：**

https://github.com/MilesCranmer/PySR/tree/master

---

### NO.2 Transolver 论文复现

**论文链接：**

https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=Transolver%3A+A+Fast+Transformer+Solver+for+PDEs+on+General+Geometries&btnG=#:~:text=%E5%8C%85%E5%90%AB%E5%BC%95%E7%94%A8-,%5BPDF%5D%20arxiv.org,-Transolver%3A%20A%20fast

**代码复现：**

复现 List of experiments:

- Core code: see [./Physics_Attention.py](https://github.com/thuml/Transolver/blob/main/Physics_Attention.py)
- Standard benchmarks: see [./PDE-Solving-StandardBenchmark](https://github.com/thuml/Transolver/tree/main/PDE-Solving-StandardBenchmark)
- Car design task: see [./Car-Design-ShapeNetCar](https://github.com/thuml/Transolver/tree/main/Car-Design-ShapeNetCar)
- Airfoil design task: see [./Airfoil-Design-AirfRANS](https://github.com/thuml/Transolver/tree/main/Airfoil-Design-AirfRANS)

精度与论文中对齐，完成文档，符合代码审核要求

**参考代码链接：**

https://github.com/thuml/Transolver

---

### NO.3 DrivAerNet ++ 论文复现

**论文链接：**

https://github.com/Mohamedelrefaie/DrivAerNet#:~:text=preprint%3A%20DrivAerNet%2B%2B%20paper-,here,-DrivAerNet%20Paper%3A

**代码复现：**

复现 RegDGCNN 和 PointNet，在数据集 DrivAer++上，精度与论文中对齐，完成文档，符合代码审核要求

**参考代码链接：**

https://github.com/Mohamedelrefaie/DrivAerNet

---

### NO.4 DrivAerNet 论文复现

**论文链接：**

https://www.researchgate.net/publication/378937154_DrivAerNet_A_Parametric_Car_Dataset_for_Data-Driven_Aerodynamic_Design_and_Graph-Based_Drag_Prediction

**代码复现：**

复现 RegDGCNN 网络，在数据集 DrivAer 上，精度与论文中对齐，完成文档，符合代码审核要求

**参考代码链接：**

https://github.com/Mohamedelrefaie/DrivAerNet

---

### NO.5 Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations 论文复现

**论文链接：**

- Science 正刊：https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7219083/pdf/nihms-1581226.pdf

- 补充材料：https://www.science.org/doi/10.1126/science.aaw4741

**代码复现：**

仓库包含所有案例，精度与论文和补充材料对齐，完成文档，符合代码审核要求

**参考代码链接：**

https://github.com/maziarraissi/HFM

---

### NO.6 Synthetic Lagrangian turbulence by generative diffusion models 论文复现

**论文链接：**

https://www.nature.com/articles/s42256-024-00810-0.pdf

**代码复现：**

复现 DM-1c 和 DM-3c 模型，精度与论文中对齐

**参考代码链接：**

- he code to train the DM and generate new trajectories can be found at https://github.com/SmartTURB/diffusion-lagr (ref. 82).

- A ready-to-runCode Ocean capsule with the complete environment is available at https://codeocean.com/capsule/0870187/tree/v1 (ref. 83).

---

### NO.7 AI-aided geometric design of anti-infection catheters 论文复现

**论文链接：**

https://www.science.org/doi/pdf/10.1126/sciadv.adj1741 （Nature 子刊）

**代码复现：**

复现 GeoFNO 模型，精度与论文中对齐

**参考代码链接：**

https://github.com/zongyi-li/Geo-FNO-catheter

---

### NO.8 A physics-informed diffusion model for high-fidelity flow field reconstruction 论文复现

**论文链接：**

https://www.sciencedirect.com/science/article/pii/S0021999123000670 （Science 子刊）

**代码复现：**

复现 Diffusion models，精度与论文中对齐

**参考代码链接：**

Data and code for this work are publicly available at https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution

---

### NO.9 DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting 论文复现

**论文链接：**

https://arxiv.org/abs/2312.06734 (JCP)

**代码复现：**

复现 DiffCast，精度与论文中对齐，并合入 PaddleScience

**参考代码链接：**

https://github.com/DeminYu98/DiffCast

---

### NO.10 Neural General Circulation Models for Weather and Climate 论文复现

**论文链接：**

https://arxiv.org/pdf/2311.07222

**代码复现：**

复现 neuralgcm，精度与论文中对齐，并合入 PaddleScience

**参考代码链接：**

https://github.com/google-research/neuralgcm

---

### NO.11 FuXi: A cascade machine learning forecasting system for 15-day global weather forecast 论文复现

**论文链接：**

https://arxiv.org/abs/2306.12873

**代码复现：**

复现 FuXi 推理，精度与论文中对齐，并合入 PaddleScience

**参考代码链接：**

https://github.com/tpys/FuXi

---

### NO.12 Adam、AdamW 优化器支持 amsgrad

**论文链接：**

https://openreview.net/forum?id=ryQu7f-RZ

**代码复现：**

为 Adam 和 AdamW 优化器支持 amsgrad 选项，添加动态图和静态图的实现，在单机和分布式环境下，精度与 pytorch 对齐

---

### NO.13 put_along_axis 反向算子实现静态图一阶拆解

**put_along_axis API 文档链接：**

https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/put_along_axis_cn.html#cn-api-paddle-tensor-put-along-axis

**代码复现：**

通过调用已有基础算子，实现 put_along_axis 反向算子的静态图一阶拆解

**参考代码链接：**

静态图一阶拆解示例：https://github.com/PaddlePaddle/Paddle/pull/64432

---

### NO.14 Crystal Diffusion Variational AutoEncoder 论文复现

**论文链接：**

https://arxiv.org/abs/2110.06197

**代码复现：**

复现 cdvae，精度与论文中对齐,并合入 PaddleScience

**参考代码链接：**

https://github.com/txie-93/cdvae

---

### NO.15 SchNet 论文复现

**论文链接：**

https://arxiv.org/abs/1706.08566

**代码复现：**

复现 schnet，精度与论文中对齐,并合入 PaddleScience

**参考代码链接：**

https://github.com/atomistic-machine-learning/SchNet

---

### NO.16 MACE 论文复现

**论文链接：**

https://arxiv.org/abs/2206.07697

**代码复现：**

复现 MACE，精度与论文中对齐，并合入 PaddleScience

**参考代码链接：**

https://github.com/ACEsuit/mace/tree/main

---

### NO.17 PIKAN 论文复现

**论文链接：**

https://arxiv.org/pdf/2407.17611

**代码复现：**

复现 Heat2d_MS、Steady_ns、PoissonNd 三个案例，精度与论文中对齐,并合入 PaddleScience

**参考代码链接：**

https://github.com/srigas/jaxKAN/tree/main
