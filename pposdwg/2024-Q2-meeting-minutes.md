# 飞桨社区开源发展工作组 2024-Q2 会议纪要

- 本次会议时间：2024/05/10 19:00-20:30
- 参会人：
  - 飞桨社区开源发展工作组： [@luotao1](https://github.com/luotao1)、 [@Aurelius84](https://github.com/Aurelius84)、 [@Liyulingyue](https://github.com/Liyulingyue)、 [@GreatV](https://github.com/GreatV)
  、[@jinyouzhi](https://github.com/jinyouzhi)、[@jzhang533](https://github.com/jzhang533)、 [@Zheng-Bicheng](https://github.com/Zheng-Bicheng)
  - 意向参与 PaddleOCR PMC 的开发者：[@GreatV](https://github.com/GreatV)、 [@Topdu](https://github.com/Topdu)、 [@SWHL](https://github.com/SWHL)、 [@Liyulingyue](https://github.com/Liyulingyue)、
  [@tink2123](https://github.com/tink2123)、  [@sunting78](https://github.com/sunting78)
  - 其他相关人：[@Harryoung](https://github.com/Harryoung)、 [@Gmgge](https://github.com/Gmgge)
- 会议讨论遵循 [Chatham House Rule](https://www.chathamhouse.org/about-us/chatham-house-rule) 。

## 会议简记
### PaddleOCR 组建 PMC 研讨
1. 介绍 PMC 的定义：[Apache 基金会对于 PMC 的定义]( https://www.apache.org/dev/pmc.html) 和 [Paddle2ONNX PMC 的章程](https://github.com/PaddlePaddle/Paddle2ONNX/issues/1185)。
   简单来说，PMC 的职责是照看项目、持续研发、发展社区。
2. 回顾前期的讨论：[How to get PaddleOCR better maintained.](https://github.com/PaddlePaddle/PaddleOCR/issues/11859)
3. 正式成立 PaddleOCR PMC：
   - [@GreatV](https://github.com/GreatV)（汪昕、 PMC Chair）
   - [@tink2123](https://github.com/tink2123)（殷晓婷、 PMC Chair）
   - [@Topdu](https://github.com/Topdu)（杜永坤）
   - [@SWHL](https://github.com/SWHL)（Joshua Wang）
   - [@Liyulingyue](https://github.com/Liyulingyue)（张一乔）
   - [@Sunting78](https://github.com/Sunting78)（孙婷）

4. PaddleOCR 项目的未来工作事项探讨
   - 正在进行中的项目：
     - 迁移基础设施到 github： [Modify the setuptools configuration from SETUP.py into PYPROJECT.toml](https://github.com/PaddlePaddle/PaddleOCR/pull/12013)、[Add CI for PaddleOCR test](https://github.com/PaddlePaddle/PaddleOCR/pull/12062)
     - [解决 PaddleOCR 历史存在的疑难 Issue](https://github.com/PaddlePaddle/PaddleOCR/issues/11906)
     - [开放原子开源大赛 PaddleOCR 算法模型挑战赛](https://pfcclab.github.io/posts/suzhou-kaifangyuanzi) 的一等奖成果正在合入 PaddleOCR 仓库中：[openocr 团队的代码](https://github.com/PaddlePaddle/PaddleOCR/pull/12033)
     和 [ocr 识别队的代码](https://github.com/PaddlePaddle/PaddleOCR/pull/11999)
   - 讨论的议题：
     - 发布一个新的版本，如 2.8.0 (遵守 semantic versioning）
         - 待上述的进行中的项目完成后，可以发布一个新的版本 2.8.0 。
         - 经过近期的对分支的调整，发布新的版本后，让分支管理走向正规。
     - 重新定位 PaddleOCR，需要充分研讨，待 PaddleOCR PMC 后续单独组织讨论。
     - 梳理整个仓库代码，已经有一些工作在做，重要但不紧急。如 PPOCRLabel 是否要合入 [PFCCLab/PaddleLabel](https://github.com/PFCCLab/PaddleLabel)。
     - 需要发布一些新的模型和算法。
     - 需要对整个仓库的文档体系重新构建。

### 一些活动与通知
1. 2024 年 H1 飞桨开源之星评选筹备：见 [评选规则](https://github.com/PaddlePaddle/community/issues/892)，共 10 人，6 月 11 日前提交申请。
2. 机会分享：[2024 中国互联网发展创新与投资大赛（开源）](https://bs.bjos.club/gong-n255-1.html#cons) ，如【开源应用和开放赛道】适合 PaddleOCR 的下游项目 RapidOCR 申请。
