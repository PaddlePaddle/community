## 第十一次 PaddleOCR 开源共建研讨会

* 会议时间：2024-11-20 19:30
* 本次会议接入方式：
  * 腾讯会议：607-2630-0369
  * 会议密码：1111
  * [点击链接入会](https://meeting.tencent.com/dm/egxl0HKTx7Ow)，或添加至会议列表
* 本次拟参会 member list：
  * [dyning](https://github.com/dyning)
  * [jzhang533](https://github.com/jzhang533)
  * [Topdu](https://github.com/)
  * [greatv](https://github.com/greatv)
  * [Gmgge](https://github.com/Gmgge)
  * [SWHL](https://github.com/SWHL)
  * [Liyulingyue](https://github.com/Liyulingyue)
  * [luotao1](https://github.com/luotao1)
  * [mattheliu](https://github.com/mattheliu)
  * [UserWangZz](https://github.com/UserWangZz)
  * [jingsongliujing](https://github.com/jingsongliujing)
  * [E-Pudding](https://github.com/E-Pudding)
  * [JiehangXie](https://github.com/JiehangXie)

## 会议议程

* PaddleOCR 近期进展同步
  * 近期 issue 和 Discussions 回顾
  * 近期 Pull Request 回顾

* PaddleOCR 最新版本中表格识别算法结果与v2.8版本有差异
  - https://github.com/PaddlePaddle/PaddleOCR/issues/14007
  - https://github.com/PaddlePaddle/PaddleOCR/issues/14163

* `eval()` 消除计划
  - https://github.com/PaddlePaddle/PaddleOCR/issues/13848

* 旧 issues 的处理，复现、close 和 lock

  经常有用户在多年前的 issue 下再次回复，目前是进行 close 和 lock

* PaddleOCR 海外开发者交流会（某次 PFCC）[E-Pudding](https://github.com/E-Pudding) [luotao1](https://github.com/luotao1)

  以下是从 2024.1.1 ~ 2024.10.17 给 PaddleOCR 仓库合入过 PR 的海外开发者，且给提供了邮寄地址接受了快乐开源礼物，可以尝试邀请 & 讨论下交流会的主题：

  | **GithubID** | **PR** | **Country** ｜
  | ------  |  ------ | ----------|
  | [1chimaruGin](https://github.com/1chimaruGin)  |  https://github.com/PaddlePaddle/PaddleOCR/pull/12020 | Japan|
  | [AlexPasqua](https://github.com/AlexPasqua) |  https://github.com/PaddlePaddle/PaddleOCR/pull/12042 、https://github.com/PaddlePaddle/PaddleOCR/pull/12542 | ITALY|
  | [johnlockejrr](https://github.com/johnlockejrr)  | https://github.com/PaddlePaddle/PaddleOCR/pull/13797 、 https://github.com/PaddlePaddle/PaddleOCR/pull/13800  |Spain|
  | [Kayzwer](https://github.com/Kayzwer)  | https://github.com/PaddlePaddle/PaddleOCR/pull/13760  | Malaysia|
  | [MatKollar](https://github.com/MatKollar)  | https://github.com/PaddlePaddle/PaddleOCR/pull/11520 | Slovakia |
  | [taeefnajib](https://github.com/taeefnajib)  | https://github.com/PaddlePaddle/PaddleOCR/pull/13373  | Bangladesh |
  | [zovelsanj](https://github.com/zovelsanj)  | https://github.com/PaddlePaddle/PaddleOCR/pull/12108 | Spain|

* Paddle OCR 识别提取字幕的精准度问题探讨 [JiehangXie](https://github.com/JiehangXie)

  - 【背景】：当前使用 Paddle OCR 进行视频画面的字幕提取，用于生成音频-文本对多模态数据，用于训练语音&多模态模型。
  - 【实现方案】：开发批处理脚本，用 OCR 识别每一帧，根据上下文不同生成一份包含时间戳的字幕 srt 文件。
  - 【问题】：当前OCR开源模型识别准确率不足，导致出现
     - 1）同一句台词有不同的识别结果，时间帧被砍成了若干段；
     - 2）基于1）的问题尝试增加过滤规则，但发生导致较短的句子被删除，字幕文件高频漏字漏句；
     - 3）模型精度不足经常出现画面误识别，出现特殊符号，比如#号。
     - 最终方案的结果是高频出现漏字漏句、出现特殊符号等问题，导致对齐数据的字准率不足，音文不齐，无法通过数据质检。

* 自由发言，可以提需求给大家讨论、提问等

* 确定下次的部分议题及召开时间

## 轮值值班安排

- 职责：处理 issue 区和 Discussion 区的新增问题，召集，主持例会并记录。
- 期间：每隔两周轮换。
- 排班：@greatv, @jzhang533, @SWHL, @jingsongliujing, @Liyulingyue, @Topdu @mattheliu

序号|GITHUB ID|值班开始|值班结束
:------:|:------:|:------:|:------:
1|@greatv|2024-11-07|2024-11-20
2|@jzhang533|2024-11-21|2024-12-04
3|@SWHL|2024-12-05|2024-12-18
4|@jingsongliujing|2024-12-19|2025-01-01 (放假日) 2025-01-08
5|@Liyulingyue|2025-01-09|2025-01-22 (今年最后一次)
6|@Topdu|2025-02-05|2025-02-19
7|@mattheliu|2025-02-20|2025-03-05
