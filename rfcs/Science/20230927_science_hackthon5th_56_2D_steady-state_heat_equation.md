# Science 56 设计文档

| API名称                                                      | 新增API名称                                |
| ---------------------------------------------------------- | -------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">     | moosewoler                                   |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2023-09-27                             |
| 版本号                                                        | V1.0                                   |
| 依赖CINN版本<input type="checkbox" class="rowselector hidden"> | PaddlePaddle 2.5.0                              |
| 文件名                                                        | 20230927_science_hackthon5th_56_2D_steady-state_heat_equation.md<br> |

# 一、概述

## 1、相关背景

**基于物理信息的神经网络**（Physics-Informed Neural Networks，PINN）是一种机器学习方法，它利用神经网络和物理信息来建立数学模型。

这种方法在许多领域都有广泛的应用，包括求解偏微分方程、优化控制问题等。在**2D稳态热传导方程**方面，PINN也可以发挥其优势。

<font color="red">
PINN方法通过神经网络学习系统中的隐式解，从而可以更加高效准确地解决这类问题。
为了训练神经网络，需要构造一个损失函数，它包括两部分：数据项和物理项。
数据项用于描述神经网络输出与训练数据之间的差异，物理项则用于确保神经网络输出满足某些已知的物理约束或先验知识，例如热力学定律。
这两项的平衡使得神经网络在学习过程中既能适应训练数据，又能保持解的物理性质。
</font>

## 2、功能目标

本任务中，作者在 **PaddleScience** 的 `XPINN_2D_PoissonsEqn` 的基础上，完成了以下任务：

<font color="red">
- 探讨了2D 非定常圆柱绕流的物理场预测任务中的监督测点数量和位置分布的影响；
- 将2D 非定常圆柱绕流的物理场预测任务中的200-300个的监督数据降低到了30个点以内；
- 研究了极端情况下，仅以10个以内的圆柱壁面测点作为监督数据时，本模型物理场预测能力。
</font>


## 3、意义



# 二、飞桨现状

<font color="red">
飞浆框架目前支持PINN模块化建模，针对2D非定常圆柱绕流算例，可实现200-300个监督测点的半监督学习。

![参考 PaddleScience](https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/cylinder/2d_unsteady_continuous)
</font>

# 三、业内方案调研

<font color="red">
|    文献                  | 效果                             | 求解全部计算域 | 完全无监督   |
| ----------------------  | ---------------------------------| ------------ | ---------- |
| Raissi M, et al    [6]  | ![结果文件](https://github.com/tianshao1992/PINNs_Cylinder/tree/main/figs_for_md/paper6.png)  | 否           | 是         |
| Raissi M, et al    [7]  | ![结果文件]([figs_for_md](https://github.com/tianshao1992/PINNs_Cylinder/tree/main/figs_for_md)/paper7.png)  | 否           | 是         |
| Raissi M, et al    [8]  | ![结果文件]([figs_for_md](https://github.com/tianshao1992/PINNs_Cylinder/tree/main/figs_for_md)/paper8.png)  | 是           | 否         |
| Cai S, et al    [9]     | ![结果文件]([figs_for_md](https://github.com/tianshao1992/PINNs_Cylinder/tree/main/figs_for_md)/paper9.png)  | 是           | 否         |
| Jin X, et al    [10]    | ![结果文件]([figs_for_md](https://github.com/tianshao1992/PINNs_Cylinder/tree/main/figs_for_md)/paper10.png) | 否           | 是         |
</font>

# 四、对比分析

<font color="red">
在保证精度的同时，可显著降低监督测点的数量；
此外，考虑了仅有10个以内圆柱壁面测点等极端情况下的物理场预测精度；
具体对比结果可参考 ![结果描述](https://github.com/tianshao1992/PINNs_Cylinder)
</font>

# 五、设计思路与实现方案

<font color="red">
1. [PINNs_Cylinder AI studio](https://aistudio.baidu.com/aistudio/projectdetail/4501565)相关运行结果

  - run_train_pdpd.py   为训练主程序
  - run_tvalidate_pdpd.py  为验证主程序
  - basic_model_pdpd.py  为本问题所涉及到的基础全连接网络,
  - visual_data.py  为数据可视化
  - process_data_pdpd.py  为数据预处理

  - **work文件夹**中为模型训练过程及验证可视化
    - \train  训练集数据 & 训练过程的可视化
    - \validation 验证数据的可视化
    - train.log 所有训练过程中的日志数据保存
    - valida.log 所有训练过程中的日志数据保存
    - latest_model.pth 模型文件

  - **data文件夹**中为非定常2D圆柱绕流数据，可从以下链接获取
链接：https://pan.baidu.com/s/1RtBQaEzZQon0cxSzmau7kg 
提取码：0040
</font>

# 六、测试和验收的考量

<font color="red">
1. 提供完整的网络模型建立、训练、验证和可视化方法
2. 不同监督测点下模型的物理场预测能力验证
</font>


# 七、可行性分析和排期规划

- 可行性分析：非常可行
- 排期规划：已完成。

# 八、影响面

对其他模块无影响。

# 附件及参考资料

<font color="red">
![实现文档](https://github.com/tianshao1992/PINNs_Cylinder)
</font>