# 在 PaddleScience 中复现 WGAN-GP 模型

| 任务名称 | 在 PaddleScience 中复现 WGAN-GP 模型            |
| --- |-------------------------------------------|
| 提交作者 | robinbg、XvLingWYY                         |
| 提交时间 | 2025-04-04                                |
| 版本号 | V1.0                                      |
| 依赖飞桨版本 | develop                                   |
| 文件名 | 20250404_add_wgan_gp_for_paddlescience.md |

# 一、概述

## 1、相关背景
Wasserstein GAN with Gradient Penalty (WGAN-GP) 是对原始 Wasserstein GAN 的改进版本，通过引入梯度惩罚项替代权重裁剪，解决了 WGAN 训练中的稳定性和收敛性问题。该方法由 Gulrajani 等人在 2017 年的论文《Improved Training of Wasserstein GANs》中提出，已成为 GAN 训练的重要基准方法。

WGAN-GP 的主要创新点在于：
* 用梯度惩罚（Gradient Penalty）替代了 WGAN 中的权重裁剪（Weight Clipping）
* 通过惩罚判别器梯度范数，强制执行 1-Lipschitz 约束
* 显著提高了训练稳定性，使得 GAN 能够生成更高质量的样本
* 减少了对网络架构和超参数选择的敏感性

## 2、功能目标
在 PaddleScience 中实现 WGAN-GP 模型，对齐原论文中的实验结果，并提供完整的训练和评估流程。具体目标包括：
* 实现 WGAN-GP 的核心算法，包括梯度惩罚项的计算
* 复现论文中的实验案例，包括玩具数据集、MNIST 和 CIFAR-10 数据集上的结果
* 确保与原论文中报告的性能指标对齐
* 提供易于使用的接口，方便用户在自己的数据集上应用 WGAN-GP

## 3、意义
WGAN-GP 作为一种稳定高效的 GAN 训练方法，在 PaddleScience 中的实现将为用户提供更多生成模型选择，特别是在处理复杂分布和高维数据时。此外，该实现也将作为 PaddleScience 中其他生成模型的基础，促进更多生成模型的研究和应用。

# 二、飞桨现状
PaddleScience 目前尚未包含 WGAN-GP 模型的实现。虽然 PaddlePaddle 生态中已有一些 GAN 模型的实现，但缺乏专注于稳定训练的 WGAN-GP 实现，特别是在科学计算领域的应用。

# 三、业内方案调研
目前，WGAN-GP 已在多个深度学习框架中得到实现：

1. TensorFlow 实现：
   * 原作者提供的官方实现：https://github.com/igul222/improved_wgan_training
   * 支持多种数据集和网络架构

2. PyTorch 实现：
   * 多个开源版本，如 PyTorch-GAN 和 PyTorch-WGAN-GP
   * 提供了灵活的接口和丰富的示例

3. 其他框架：
   * Keras、JAX 等框架也有相应实现
   * 各实现在细节上可能有所不同，但核心算法一致

通过对比分析，我们将主要参考原作者的 TensorFlow 实现，确保算法的正确性和性能的一致性。

# 四、对比分析
与传统 GAN 和 WGAN 相比，WGAN-GP 具有以下优势：

1. 训练稳定性：
   * WGAN-GP 通过梯度惩罚替代权重裁剪，避免了梯度消失和模式崩溃问题
   * 相比原始 GAN，WGAN-GP 的训练过程更加稳定，不易出现训练失败的情况

2. 样本质量：
   * WGAN-GP 生成的样本质量通常优于传统 GAN 和原始 WGAN
   * 在高分辨率图像生成和复杂分布建模方面表现尤为突出

3. 架构灵活性：
   * WGAN-GP 对网络架构的要求较低，可以适应各种不同的网络结构
   * 不需要像 WGAN 那样严格控制判别器的参数范围

4. 收敛性：
   * WGAN-GP 的损失函数提供了更有意义的训练信号，有助于判断训练进度
   * 收敛速度通常快于传统 GAN

# 五、设计思路与实现方案

## 1. 核心算法实现
WGAN-GP 的核心在于其损失函数和梯度惩罚项的计算。以下是主要实现步骤：

### 1.1 损失函数
```python
# CIFAR10实验中生成器损失
class Cifar10GenFuncs:
    """
    Loss function for cifar10 generator
    Args
        discriminator_model: discriminator model
        acgan_scale_g: scale of acgan loss for generator

    """

    def __init__(
        self,
        discriminator_model,
        acgan_scale_g=0.1,
    ):
        self.crossEntropyLoss = paddle.nn.CrossEntropyLoss()
        self.acgan_scale_g = acgan_scale_g
        self.discriminator_model = discriminator_model

    def loss(self, output_dict: Dict, *args):
        fake_image = output_dict["fake_data"]
        labels = output_dict["labels"]
        outputs = self.discriminator_model({"data": fake_image, "labels": labels})
        disc_fake, disc_fake_acgan = outputs["disc_fake"], outputs["disc_acgan"]
        gen_cost = -paddle.mean(disc_fake)
        if disc_fake_acgan is not None:
            gen_acgan_cost = self.crossEntropyLoss(disc_fake_acgan, labels)
            gen_cost += self.acgan_scale_g * gen_acgan_cost
        return {"loss_g": gen_cost}

# CIFAR10实验中判别器损失
class Cifar10DisFuncs:
    """
    Loss function for cifar10 discriminator
    Args
        discriminator_model: discriminator model
        acgan_scale: scale of acgan loss for discriminator

    """

    def __init__(self, discriminator_model, acgan_scale):
        self.crossEntropyLoss = paddle.nn.CrossEntropyLoss()
        self.acgan_scale = acgan_scale
        self.discriminator_model = discriminator_model

    def loss(self, output_dict: Dict, label_dict: Dict, *args):
        fake_image = output_dict["fake_data"]
        real_image = label_dict["real_data"]
        labels = output_dict["labels"]
        disc_fake = self.discriminator_model({"data": fake_image, "labels": labels})[
            "disc_fake"
        ]
        out = self.discriminator_model({"data": real_image, "labels": labels})
        disc_real, disc_real_acgan = out["disc_fake"], out["disc_acgan"]
        gradient_penalty = self.compute_gradient_penalty(real_image, fake_image, labels)
        disc_cost = paddle.mean(disc_fake) - paddle.mean(disc_real)
        disc_wgan = disc_cost + gradient_penalty
        if disc_real_acgan is not None:
            disc_acgan_cost = self.crossEntropyLoss(disc_real_acgan, labels)
            disc_acgan = disc_acgan_cost.sum()
            disc_cost = disc_wgan + (self.acgan_scale * disc_acgan)
        else:
            disc_cost = disc_wgan
        return {"loss_d": disc_cost}

    def compute_gradient_penalty(self, real_data, fake_data, labels):
        differences = fake_data - real_data
        alpha = paddle.rand([fake_data.shape[0], 1])
        interpolates = real_data + (alpha * differences)
        gradients = paddle.grad(
            outputs=self.discriminator_model({"data": interpolates, "labels": labels})[
                "disc_fake"
            ],
            inputs=interpolates,
            create_graph=True,
            retain_graph=False,
        )[0]
        slopes = paddle.sqrt(paddle.sum(paddle.square(gradients), axis=1))
        gradient_penalty = 10 * paddle.mean((slopes - 1.0) ** 2)
        return gradient_penalty
```

### 1.2 梯度惩罚计算
```python
# CIFAR-10 判别器中的梯度惩罚计算
def compute_gradient_penalty(self, real_data, fake_data, labels):
  differences = fake_data - real_data
  alpha = paddle.rand([fake_data.shape[0], 1])
  interpolates = real_data + (alpha * differences)
  gradients = paddle.grad(
      outputs=self.discriminator_model({"data": interpolates, "labels": labels})[
          "disc_fake"
      ],
      inputs=interpolates,
      create_graph=True,
      retain_graph=False,
  )[0]
  slopes = paddle.sqrt(paddle.sum(paddle.square(gradients), axis=1))
  gradient_penalty = 10 * paddle.mean((slopes - 1.0) ** 2)
  return gradient_penalty
```

## 2. 网络架构
我们将实现三种不同的网络架构，对应论文中的不同实验：

### 2.1 玩具数据集架构
简单的多层感知机网络，用于 2D 高斯分布等简单数据集。

### 2.2 MNIST 架构
卷积神经网络架构，适用于 MNIST 等低分辨率图像数据集。

### 2.3 CIFAR-10 架构
深度卷积神经网络，适用于 CIFAR-10 等彩色图像数据集。

## 3. 训练流程
WGAN-GP 的训练流程与标准 GAN 有所不同，主要区别在于：
* 每次生成器更新前，判别器需要更新多次（通常为 5 次）
* 使用 Adam 优化器，且推荐的参数设置为 β₁=0.5, β₂=0.9
* 不需要权重裁剪操作

```python
# 训练循环示例
 for i in range(cfg.TRAIN.epochs):
     logger.message(f"\nEpoch: {i + 1}\n")
     optimizer_discriminator.clear_grad()
     solver_discriminator.train()
     optimizer_generator.clear_grad()
     solver_generator.train()
```

## 4. 评估指标
为了评估 WGAN-GP 的性能，我们将实现以下指标：

### 4.1 Inception Score (IS)
用于评估生成图像的质量和多样性。

### 4.2 生成样本可视化
保存生成的样本，用于直观评估模型性能。

## 5. 与 PaddleScience 集成
我们将设计一个模块化的实现，便于与 PaddleScience 集成：

### 5.1 模型结构
```
PaddleScience/
└── examples/
    └── wgangp/
        ├── conf
        │  ├── wgangp_cifar10.yaml # CIFAR-10 配置文件
        │  ├── wgangp_mnist.yaml # MNIST 配置文件
        │  └── wgangp_toy.yaml # 玩具数据集配置文件
        ├── function.py # 损失函数、评估指标、可视化工具
        ├── wgangp_cifr10.py # CIFAR-10 示例 
        ├── wgangp_cifar10_model.py # CIFAR-10实验模型
        ├── wgangp_mnist.py # MNIST 示例
        ├── wgangp_mnist_model.py # MNIST实验模型
        └── wgangp_toy.py # 玩具数据集示例
        └── wgangp_toy_model.py # 玩具数据集实验模型
```

### 5.2 接口设计
提供简洁统一的接口，方便用户使用：

```python
# 示例用法
import os
import paddle
from functions import Cifar10DisFuncs
from functions import Cifar10GenFuncs
from functions import load_cifar10
from omegaconf import DictConfig
from wgangp_cifar10_model import WganGpCifar10Discriminator
from wgangp_cifar10_model import WganGpCifar10Generator

def train(cfg: DictConfig):
    # set model
    generator_model = WganGpCifar10Generator(**cfg["MODEL"]["gen_net"])
    discriminator_model = WganGpCifar10Discriminator(**cfg["MODEL"]["dis_net"])
    if cfg.TRAIN.pretrained_dis_model_path and os.path.exists(
        cfg.TRAIN.pretrained_dis_model_path
    ):
        discriminator_model.load_dict(paddle.load(cfg.TRAIN.pretrained_dis_model_path))

    # set Loss
    generator_funcs = Cifar10GenFuncs(
        **cfg["LOSS"]["gen"], discriminator_model=discriminator_model
    )
    discriminator_funcs = Cifar10DisFuncs(
        **cfg["LOSS"]["dis"], discriminator_model=discriminator_model
    )

    # set dataloader
    inputs, labels = load_cifar10(**cfg["DATA"])
    dataloader_cfg = {
        "dataset": {
            "name": cfg["EVAL"]["dataset"]["name"],
            "input": inputs,
            "label": labels,
        },
        "sampler": {
            **cfg["TRAIN"]["sampler"],
        },
        "batch_size": cfg["TRAIN"]["batch_size"],
        "use_shared_memory": cfg["TRAIN"]["use_shared_memory"],
        "num_workers": cfg["TRAIN"]["num_workers"],
        "drop_last": cfg["TRAIN"]["drop_last"],
    }

    # set constraint
    constraint_generator = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg=dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(generator_funcs.loss),
        output_expr={"labels": lambda out: out["labels"]},
        name="constraint_generator",
    )
    constraint_generator_dict = {constraint_generator.name: constraint_generator}

    constraint_discriminator = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg=dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(discriminator_funcs.loss),
        output_expr={"labels": lambda out: out["labels"]},
        name="constraint_discriminator",
    )
    constraint_discriminator_dict = {
        constraint_discriminator.name: constraint_discriminator
    }

    # set optimizer
    lr_scheduler_generator = Linear(**cfg["TRAIN"]["lr_scheduler_gen"])()
    lr_scheduler_discriminator = Linear(**cfg["TRAIN"]["lr_scheduler_dis"])()

    optimizer_generator = ppsci.optimizer.Adam(
        learning_rate=lr_scheduler_generator,
        beta1=cfg["TRAIN"]["optimizer"]["beta1"],
        beta2=cfg["TRAIN"]["optimizer"]["beta2"],
    )
    optimizer_discriminator = ppsci.optimizer.Adam(
        learning_rate=lr_scheduler_discriminator,
        beta1=cfg["TRAIN"]["optimizer"]["beta1"],
        beta2=cfg["TRAIN"]["optimizer"]["beta2"],
    )
    optimizer_generator = optimizer_generator(generator_model)
    optimizer_discriminator = optimizer_discriminator(discriminator_model)

    # initialize solver
    solver_generator = ppsci.solver.Solver(
        model=generator_model,
        output_dir=os.path.join(cfg.output_dir, "generator"),
        constraint=constraint_generator_dict,
        optimizer=optimizer_generator,
        epochs=cfg.TRAIN.epochs_gen,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch_gen,
        pretrained_model_path=cfg.TRAIN.pretrained_gen_model_path,
    )
    solver_discriminator = ppsci.solver.Solver(
        model=generator_model,
        output_dir=os.path.join(cfg.output_dir, "discriminator"),
        constraint=constraint_discriminator_dict,
        optimizer=optimizer_discriminator,
        epochs=cfg.TRAIN.epochs_dis,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch_dis,
        pretrained_model_path=cfg.TRAIN.pretrained_gen_model_path,
    )

    # train
    for i in range(cfg.TRAIN.epochs):
        logger.message(f"\nEpoch: {i + 1}\n")
        optimizer_discriminator.clear_grad()
        solver_discriminator.train()
        optimizer_generator.clear_grad()
        solver_generator.train()

    # save model weight
    paddle.save(
        generator_model.state_dict(),
        os.path.join(cfg.output_dir, "model_generator.pdparams"),
    )
    paddle.save(
        discriminator_model.state_dict(),
        os.path.join(cfg.output_dir, "model_discriminator.pdparams"),
    )

```

# 六、测试验收的考量

## 1. 功能测试
- [ ] 核心算法实现正确性验证
- [ ] 不同网络架构的兼容性测试
- [ ] 各种数据集上的训练测试

## 2. 性能对齐
- [ ] 与原论文在玩具数据集上的结果对比
- [ ] 与原论文在 MNIST 上的结果对比
- [ ] 与原论文在 CIFAR-10 上的结果对比

## 3. 集成测试
- [ ] 与 PaddleScience 其他模块的集成测试
- [ ] API 兼容性测试
- [ ] 文档和示例完整性测试

## 4. 性能测试
- [ ] 训练速度测试
- [ ] 内存占用测试
- [ ] 生成样本质量评估

# 七、可行性分析和排期规划
基于对原论文和参考实现的分析，在 PaddleScience 中实现 WGAN-GP 是完全可行的。PaddlePaddle 提供了所有必要的功能，包括自动微分、梯度计算和优化器等。

## 实施计划
总体开发周期预计为 4 周，具体排期如下：

### 第 1 周：核心算法实现
- 实现 WGAN-GP 的基本框架
- 实现梯度惩罚计算
- 实现损失函数和训练循环

### 第 2 周：网络架构和数据集
- 实现玩具数据集实验
- 实现 MNIST 实验
- 实现 CIFAR-10 实验

### 第 3 周：评估和优化
- 实现评估指标
- 进行性能对齐
- 优化训练过程

### 第 4 周：集成和文档
- 与 PaddleScience 集成
- 编写文档和示例
- 进行最终测试和调整

# 八、影响面
本实现将对 PaddleScience 产生以下影响：

1. 功能扩展：
   * 为 PaddleScience 添加生成模型能力
   * 为用户提供稳定的 GAN 训练方法

2. 应用拓展：
   * 支持科学计算中的数据生成和增强
   * 为物理模拟、材料设计等领域提供新工具

3. 生态建设：
   * 丰富 PaddleScience 的模型库
   * 促进生成模型在科学计算领域的应用

# 名词解释
* GAN (Generative Adversarial Network)：生成对抗网络，一种生成模型框架
* WGAN (Wasserstein GAN)：基于 Wasserstein 距离的 GAN 变体
* WGAN-GP：带梯度惩罚的 Wasserstein GAN
* 梯度惩罚 (Gradient Penalty)：对判别器梯度范数的惩罚项，用于强制执行 Lipschitz 约束
* Inception Score (IS)：评估生成图像质量的指标
* Fréchet Inception Distance (FID)：评估生成分布与真实分布相似度的指标

# 附件及参考资料
1. [Improved Training of Wasserstein GANs 论文](https://arxiv.org/abs/1704.00028)
2. [原作者 TensorFlow 实现](https://github.com/igul222/improved_wgan_training)
3. [PaddleScience 文档](https://github.com/PaddlePaddle/PaddleScience)
4. [Wasserstein GAN 原始论文](https://arxiv.org/abs/1701.07875)
