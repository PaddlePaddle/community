

# paddlescience中文文档设计方案

|                                                              |                                           |
| ------------------------------------------------------------ | ----------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | Asthestarsfalll,                          |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-22                                |
| 版本号                                                       | V1.0                                      |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                   |
| 文件名                                                       | 20220320_docs_design_for_paddlescience.md |

# 一、概述

## 1、相关背景

PaddleScience是用于开发基于PaddlePaddle的AI驱动的科学计算应用的SDK和库。

## 2、功能目标

翻译英文文档，为PaddleScience添加相应的中文文档。

## 3、意义

PaddleScience增加中文文档，对国内用户更加友好。

# 二、飞桨现状

`PaddleScience`目前并不支持中文文档，但是有相应的英文文档，主要分为`get_started`, `api`, `examples`这三个部分，`get_started`包含PaddleScience的安装和使用，`api`包含了PaddleScience中相关API的介绍文档和使用样例，`examples`包含一些从零开始的教学。

# 三、测试和验收的考量

验收的考量如下：

1. 翻译的正确性：有无错字、别字、漏字等；
2. 翻译的准确性：表意明确，无歧义；
3. 翻译的完整性：意思表达完整，不遗漏必要的信息；
4. 翻译的简洁性：简洁明了，无冗余；
5. 翻译的针对性：明确文档面向对象，内容表述应简单易懂；
6. 翻译的统一性：采用统一的专业术语，保证前后一致。

# 四、可行性分析和排期规划

文档已大致翻译完毕，可满足在当前版本内完成开发。

# 五、影响面

新增文档，可能对文档界面有影响。

# 名词解释

翻译过程中的一些专有名词解释如下：

- Partial Differential Equations(PDE)：偏微分方程，指包含未知函数的偏导数(或偏微分)的方程。描述自变量、未知函数及其偏导数之间的关系。符合这个关系的函数是方程的解。
- Laplace Equation：拉普拉斯方程，又称调和方程、位势方程，是一种偏微分方程，因由法国数学家拉普拉斯首先提出而得名。求解拉普拉斯方程是电磁学、天文学、热力学和流体力学等领域经常遇到的一类重要的数学问题，因为这种方程以势函数的形式描写电场、引力场和流场等物理对象（一般统称为“保守场”或“有势场”）的性质。
- Navier Stokes Equations：纳维尔－斯托克斯方程表达了牛顿流体运动时，动量和质量守恒。有时，还连同状态方程列出，说明流体压强、温度、密度之间的关系。
- Dirichlet Boundary Condition：**狄利克雷边界条件**也被称为常微分方程或偏微分方程的“第一类边界条件”，指定微分方程的解在边界处的值。求出这样的方程的解的问题被称为狄利克雷问题。
- Kinematic Viscosity：运动黏度，即流体的动力粘度与同温度下该流体密度ρ之比。
- Density：密度，指一物质单位体积下的质量，在数学上，密度定义为质量除以体积的商，及物体的质量与体积的比值。
- Discrete Geometry：离散几何学是研究离散几何对象的组合性质和构造方法的几何学的分支。离散几何的大多数问题涉及到基本几何对象的有限集合或离散空间，比如点，线，平面，圆，球，多边形和四维空间。
- Darcy Flow：达西渗流（Darcy flow）是1994年公布的石油名词，**达西定律**是描述液体流过孔隙介质的本构方程。
- Lid-driven Cavity Flow：顶盖驱动方腔流，其理想状态为方腔顶部平板以恒定速度驱动规则区域内封闭的不可压流体（例如水）的流动，在方腔流的流动中可以观察到几乎所有可能发生在不可压流体中的流动现象。
- Porous Medium：多孔介质是由多相物质所占据的共同空间，也是多相物质共存的一种组合体，没有固体骨架的那部分空间叫做孔隙，由液体或气体或气液两相共同占有，相对于其中一相来说，其他相都弥散在其中，并以固相为固体骨架，构成空隙空间的某些空洞相互连通。
- Discretization ：离散化，把无限空间中有限的个体映射到有限的空间中去，以此提高算法的时空效率。
- Moment Estimates：矩估计，即矩估计法，也称“矩法估计”，就是利用样本矩来估计总体中相应的参数。首先推导涉及相关参数的总体矩（即所考虑的随机变量的幂的期望值）的方程。然后取出一个样本并从这个样本估计总体矩。接着使用样本矩取代（未知的）总体矩，解出感兴趣的参数。从而得到那些参数的估计。

# 附件及参考资料

参考资料

- [Partial Differential Equations(PDE)](https://zh.wikipedia.org/zh-hans/%E5%81%8F%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B)
- [Laplace Equation](https://zh.wikipedia.org/wiki/%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF%E6%96%B9%E7%A8%8B)
- [Navier Stokes Equations](https://zh.wikipedia.org/wiki/%E7%BA%B3%E7%BB%B4-%E6%96%AF%E6%89%98%E5%85%8B%E6%96%AF%E6%96%B9%E7%A8%8B)
- [Dirichlet Boundary Condition](https://zh.wikipedia.org/wiki/%E7%8B%84%E5%88%A9%E5%85%8B%E9%9B%B7%E8%BE%B9%E7%95%8C%E6%9D%A1%E4%BB%B6)
- [Kinematic Viscosity](https://baike.baidu.com/item/%E8%BF%90%E5%8A%A8%E9%BB%8F%E5%BA%A6/5472926?fr=aladdin)
- [Density](https://zh.wikipedia.org/wiki/%E5%AF%86%E5%BA%A6)
- [Discrete Geometry](https://zh.wikipedia.org/wiki/%E7%A6%BB%E6%95%A3%E5%87%A0%E4%BD%95%E5%AD%A6)
- [Darcy flow](https://zh.wikipedia.org/wiki/%E8%BE%BE%E8%A5%BF%E5%AE%9A%E5%BE%8B)
- [Lid-driven Cavity Flow](https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node14.html)
- [Porous Medium](https://baike.baidu.com/item/%E7%9F%A9%E4%BC%B0%E8%AE%A1/7994290?fr=aladdin)
- [Discretization](https://baike.baidu.com/item/%E7%A6%BB%E6%95%A3%E5%8C%96/10501557?fr=aladdin) 
- [Moment Estimates](https://baike.baidu.com/item/%E5%A4%9A%E5%AD%94%E4%BB%8B%E8%B4%A8/2593234?fromtitle=porous%20medium&fromid=11329461&fr=aladdin)

