# Paddle Frawework Contributor Club 第六次会议纪要

## 会议概况

- 会议时间：2022-06-23 19：00 - 20：00
- 会议地点：线上会议
- 参会人：本次会议共有 28 名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由本次轮值主席虞凯民（[BrilliantYuKaimin](https://github.com/BrilliantYuKaimin)，浙江大学）主持。

## 会议分享与讨论

### 文档工作介绍

#### 文档工作小组成果汇报

[Ligoml](https://github.com/Ligoml) 对近两个月以来文档小组的工作进行了汇报：

- 介绍了近两个月以来进行的近 1200 篇文档的全量评估和修复工作，目标是为了让 API 文档的评分达到 80 以上并形成一套文档写作规范。
- 为了让 PFCC 成员对飞桨 API 文档问题有进一步的认知，以 [paddle.prod](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.2/api/paddle/prod_cn.html) 为例介绍了目录层级混乱和示例代码不一致等问题，以 [paddle.vision.transforms.RandomCrop](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/transforms/RandomCrop_cn.html) 为例介绍了中英文文档不一致和默认参数错误等问题。
- 汇报中期成果：
  - 总结了 17 条常见问题
  - 完成了 500 多篇文档的评估，对 200 多篇文档进行了修复
  - 在第四期评估任务中得到 79.3 分，仍有红线问题存在
- 欣赏了几个工作小组成员的 PR：
  - [Liyulingyue](https://github.com/Liyulingyue) 的 [PaddlePaddle/docs#4824](https://github.com/PaddlePaddle/docs/pull/4824) 和 [PaddlePaddle/Paddle#42916](https://github.com/PaddlePaddle/Paddle/pull/42916)
  - [liyongchao911](https://github.com/liyongchao911) 的 [PaddlePaddle/docs#4638](https://github.com/PaddlePaddle/docs/pull/4638) 和 [PaddlePaddle/Paddle#42058](https://github.com/PaddlePaddle/Paddle/pull/42058)
  - [SigureMo](https://github.com/SigureMo) 的 [PaddlePaddle/docs#4936](https://github.com/PaddlePaddle/docs/pull/4936) 和 [PaddlePaddle/Paddle#43636](https://github.com/PaddlePaddle/Paddle/pull/43636)
  - [Yulv-git](https://github.com/Yulv-git) 的 [PaddlePaddle/docs#4901](https://github.com/PaddlePaddle/docs/pull/4901) 和 [PaddlePaddle/docs#4919](https://github.com/PaddlePaddle/docs/pull/4919)
  - [BrilliantYuKaimin](https://github.com/BrilliantYuKaimin) 的 [PaddlePaddle/docs#4850](https://github.com/PaddlePaddle/docs/pull/4850) 和 [PaddlePaddle/Paddle#42942](https://github.com/PaddlePaddle/Paddle/pull/42942)

#### 文档工作小组成员分享

[Ligoml](https://github.com/Ligoml) 邀请了两位成员分享文档修复工作的心得：

- [Liyulingyue](https://github.com/Liyulingyue) 指出在开发新的 API 的同时要兼顾撰写规范的文档确实存在困难，希望能有自动化的文档规范性检查工具。

- [SigureMo](https://github.com/SigureMo) 介绍了在修复文档时发现的共性问题：
  - 以 [paddle.vision.models.alexnet](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/models/alexnet_cn.html) 为例指出了有些文档的描述和 API 签名过于紧凑，应当增加一个空行；
  - 中英文字符间应有空格；
  - DenseNet、macOS 等专有名词大小写问题。

#### 展望未来

[Ligoml](https://github.com/Ligoml) 回顾了“定义飞桨框架的明天（Shaping Paddle Framework Tomorrow with You）”的口号，呼吁外部开发者通过各种方式参与飞桨的建设，最后说明了飞桨在 API 文档方面的未来的规划：

- 系统性问题：利用各种工具做出全量的检索和修改，并汇聚这些工具起到 pre-commit 的作用；

- 特例性问题：持续开展文档评估，逐例修复文档的所有问题；

- 规范性问题：在外部开发者的帮助下形成一套统一的文档规范。

### 新的 PFCC-Roadmap 介绍

[BrilliantYuKaimin](https://github.com/BrilliantYuKaimin) 说明了文档规范的重要性和飞桨 API 文档的现状，介绍了[【PFCC-Roadmap】完善 API 文档写作标准](https://github.com/PaddlePaddle/Paddle/issues/43656)任务，并呼吁感兴趣 PFCC 成员加入到文档规范的制定工作中来。

1. 以 [paddle.angle](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/angle_cn.html#angle)、[paddle.argmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/argmax_cn.html#argmax)、[paddle.as_complex](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/as_complex_cn.html#as-complex) 和 [paddle.fft.fft](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/fft_cn.html) 为例介绍了飞桨文档中在逗号、顿号和引号的使用等方面存在的问题。
2. 以 [paddle.argmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/argmax_cn.html#argmax)、[paddle.arange](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/arange_cn.html)、[paddle.allclose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/allclose_cn.html)、[paddle.angle](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/angle_cn.html#angle) 和 [paddle.fft.fft](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fft/fft_cn.html) 介绍了 `x`、$x$ 和 x 的区别。
3. 以 [paddle.cos](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cos_cn.html#cos) 和 [paddle.sin](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sin_cn.html#sin) 为例指出了同类 API 的文档写法的不一致。
4. 以 [paddle.cos](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cos_cn.html#cos) 和 [paddle.deg2rad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/deg2rad_cn.html#deg2rad) 为例指出了飞桨文档中数学公式的不规范之处。
5. 以 [paddle.nn.initializer.KaimingNormal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/initializer/KaimingNormal_cn.html#kaimingnormal) 和 [paddle.nn.initializer.XavierNormal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/initializer/XavierNormal_cn.html) 指出了飞桨文档对人名的处理的不妥。
6. 说明了《[飞桨 API 文档书写规范](https://github.com/PaddlePaddle/docs/wiki/飞桨API文档书写规范)》已经长时间没有更新。

### 自由发言和讨论

对飞桨文档的建设展开了积极的交流，新成员 [yang131313](https://github.com/yang131313) 也借此机会进行的简短的自我介绍。

- [Ligoml](https://github.com/Ligoml) 指出文档规范的制定是一个非常好的由外部开发者来定义飞桨框架的机会。

- [luotao1](https://github.com/luotao1) 指出规范的细节众多，难以全部掌握，最好能写出一个 CI 程序来自动检验大部分规范。

- [Liyulingyue](https://github.com/Liyulingyue) 指出应当尽可能地从英文文档中解析出类似函数签名这些可以复用的部分，对于剩下的必须经过翻译的部分再用类似填表的方式写入。

- [SigureMo](https://github.com/SigureMo) 表示很多细节上的问题必须要自己亲自写一遍才会发现，以亲身经历指出了指定文档规范的必要性，也对 [Liyulingyue](https://github.com/Liyulingyue) 的提议表示了赞同。

### 下次会议安排
确定下次会议的时间为两周后的同一个时间段。并确定下次会议的主席为：张楠（[nlp-zn](https://github.com/nlp-zn)），副主席待定。
