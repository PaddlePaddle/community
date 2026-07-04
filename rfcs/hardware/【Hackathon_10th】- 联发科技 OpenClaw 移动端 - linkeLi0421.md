# 【Hackathon_10th】在天玑 9500 上运行 OpenClaw：以"拍照交互"为切入点的移动端 AI 助手

| 字段 | 值 |
|---|---|
| 任务 | 进阶任务 #23：联发科技 — 在天玑 9500 手机上运行 OpenClaw |
| 提交人 | linkeLi0421 |
| 提交日期 | 2026-05-27 |
| 联系 | lilinke99@gmail.com / GitHub: https://github.com/linkeLi0421 |
| RFC 阶段 | Stage 1（方案设计） |

---

## 0. 摘要 / Executive summary

把 [OpenClaw](https://openclaw.ai/) 移植到天玑 9500 Android 手机上运行，
**以"拍照交互"为核心场景**：用户用手机拍下物体或文字，OpenClaw 调用天玑 9500
APU 在端侧跑轻量视觉模型识别内容，再把识别结果交给文心大模型 API
（ERNIE-4.5）生成自然语言回复或后续动作。这条路线兼顾**端云协同**与
**NPU 加分项**两个评估维度。

技术路线（分层交付）：

| 层级 | 内容 | 必交付? |
|---|---|---|
| MVP | OpenClaw Gateway 在 Android（Termux Node.js）上跑通 + 文心 API 接入 + 3 类基础任务（信息查询、日程提醒、文件管理） | ✅ |
| 差异化（NPU） | 拍照识物：调天玑 9500 APU 跑 MobileNetV2 INT8，把识别结果喂给 OpenClaw | ✅ |
| Stretch | 通讯录管理、Widget 快捷入口、悬浮窗、语音唤醒（按时间裁减） | 可选 |


---

## 1. 背景与目标

OpenClaw 是当下最火的开源个人 AI 助手平台，社区俗称"养龙虾"。原生形态运行在
桌面 / 服务器环境，通过 WhatsApp / Telegram / Discord 等聊天应用与用户交互。
联发科技天玑 9500 是新一代旗舰移动平台，搭载强大的 APU。

本任务的核心问题：

> 把"养龙虾"搬到口袋里 —— 让一个真正能做事的 AI 助手随身可用，
> 且充分利用天玑 9500 的端侧算力，减少对云端的依赖。

本 RFC 的方案要回答：

- OpenClaw（Node.js 服务）如何在 Android 上稳定运行？
- 文心大模型 API 怎么作为"大脑"接入？
- 天玑 9500 APU 用来跑什么、怎么跑、提升多少？
- 拿什么样的演示能让人一眼看懂"我手机里有个 AI 助手"？

---

## 2. 总体架构

```
┌──────────────────────────────────────────────────────────────┐
│   天玑 9500 Android Device                                    │
│                                                              │
│  ┌───────────────────┐    ┌─────────────────────────────┐    │
│  │  Android Shell    │    │   Termux / Linux 子环境      │    │
│  │  (Kotlin / Java)  │    │                             │    │
│  │                   │    │  ┌───────────────────────┐  │    │
│  │  - Camera intent  │    │  │  OpenClaw Gateway     │  │    │
│  │  - Photo picker   │    │  │  (Node.js)            │  │    │
│  │  - 通知中心 UI     │◄──►│  │  ┌─────────────────┐  │  │    │
│  │  - WebSocket 桥   │    │  │  │ Sessions /      │  │  │    │
│  │  - NPU 调用入口    │    │  │  │ Channels /      │  │  │    │
│  └─────────┬─────────┘    │  │  │ Tools / Skills  │  │  │   │
│            │              │  │  └─────────────────┘  │  │   │
│            │ JNI          │  │  ┌─────────────────┐  │  │   │
│            ▼              │  │  │ ERNIE Adapter   │──┼──┼───┼──► 文心 API
│  ┌───────────────────┐    │  │  └─────────────────┘  │  │   │
│  │  NeuroPilot       │    │  │  ┌─────────────────┐  │  │   │
│  │  Lite Runtime     │◄───┼──┼──│ NPU Adapter     │  │  │   │
│  │  + MobileNetV2    │    │  │  └─────────────────┘  │  │   │
│  │    PaddleOCR-mob  │    │  └───────────────────────┘  │   │
│  └───────────────────┘    └─────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

三个隔离的部分：

1. **Android Shell**（薄壳，Kotlin/Java）—— 负责系统集成：相机、文件、
   通知、悬浮窗。本身不跑 LLM 逻辑。
2. **OpenClaw Gateway**（Node.js, 跑在 Termux 子环境）—— 复用上游
   OpenClaw 的全部代码，最大程度保留 Skills/Channels/Memory 等已有能力。
3. **NPU Runtime**（NeuroPilot Lite Runtime）—— 通过 JNI 暴露给上层，
   完成端侧视觉模型推理。

通信：Android Shell ↔ Node.js 用本地 WebSocket / HTTP（loopback）；
Node.js ↔ NPU 通过命令行管道或 Native bridge。

---

## 3. OpenClaw 移动端适配方案

### 3.1 Node.js 运行环境：Termux

OpenClaw 是纯 Node.js 服务，我们用 **Termux** 作为运行环境——一个跑在
Android 上的 Linux 终端 + 包管理器，能直接 `pkg install nodejs` 然后
`node openclaw.js`，OpenClaw 几乎零修改即可启动。

预期改动：

- OpenClaw 的 `fs` 操作适配 Android 沙盒路径（`/data/data/com.termux/files`）
- 移除 / 替换需要 root 权限的 Skills（如系统 shell 直接执行）
- 长驻服务用 Termux:Boot 启动，避免被 Android 杀进程
- Android Shell（Java/Kotlin）通过 Termux:API 调用相机、通知、Intent 等
  系统能力，再通过本地 WebSocket 回传给 OpenClaw

选 Termux 的理由：

- 开发迭代最快——命令行直接调 Node，bug fix → 重启 service 一气呵成
- OpenClaw 上游代码改动最小，便于跟主线同步
- Stage 2 比赛周期短，把工程精力留给 NPU 集成和 Skills 开发，而不是
  打包发行

### 3.2 服务架构

OpenClaw 的核心抽象（Gateway / Sessions / Channels / Tools / Skills）
**完全不动**。我们做的事情：

- **新增一个 mobile Channel**：让 Android Shell 通过 WebSocket 接入，
  代替原本的 WhatsApp / Telegram channel
- **新增一个 NPU Tool**：把 NeuroPilot 推理封装成一个 Tool，OpenClaw 可以
  像调用任何其它 Tool 一样调用（保持 OpenClaw 的扩展模型不变）

### 3.3 资源约束应对

| 约束 | 预算 | 策略 |
|---|---|---|
| 内存 | < 2 GB（含 Node + NPU runtime + Android shell） | OpenClaw 默认 ~400 MB；NeuroPilot Lite Runtime + MobileNetV2 INT8 model < 100 MB；预留 1 GB 给 ERNIE response 缓冲 |
| 存储 | ~500 MB | OpenClaw 安装 ~150 MB，端侧 model ~50 MB，本地数据库 ~预留 300 MB |
| 电量 | 不显著高于普通 IM 应用 | NPU 仅 on-demand 调用（用户主动拍照才跑）；后台 Gateway 用心跳调度，无持续轮询 |
| 网络 | ERNIE 调用走 Wi-Fi / 5G | 失败自动降级（仅本地 NPU 识别结果，跳过对话） |

---

## 4. 文心大模型 API 接入

### 4.1 模型选择

| 模型 | 适用场景 | 估算延迟 |
|---|---|---|
| **ERNIE-4.5-Turbo** | 默认对话 / 任务规划 | 200-500 ms / request |
| ERNIE-Speed | 短回复 / 快速反馈 | < 200 ms |
| ERNIE-4.5-VL（备选）| 如果要纯云端图像理解 | 800 ms-2 s |

主路径用 **ERNIE-4.5-Turbo** 作为 Agent reasoning brain；ERNIE-Speed 用于
诸如"通知摘要"这类不需要复杂推理的回复。

### 4.2 认证 & 调用

使用千帆平台的 OpenAI-兼容接口，OpenClaw 的 LLM Adapter 直接复用 OpenAI
SDK 即可：

```typescript
const client = new OpenAI({
  apiKey: process.env.QIANFAN_API_KEY,
  baseURL: 'https://qianfan.baidubce.com/v2',
})

await client.chat.completions.create({
  model: 'ernie-4.5-turbo',
  messages: [...],
})
```

API key 通过 Android Shell 的安全存储（EncryptedSharedPreferences）
传给 Node 子进程，不写入磁盘明文。

---

## 5. NPU 端侧能力：拍照交互

### 5.1 场景：拍照 → NPU 识别 → ERNIE 自然语言回复

具体演示流程（demo 视频里就是这个）：

```
用户在 OpenClaw 对话框里点"拍照" 
   ↓
Android Camera intent 拿到图片
   ↓
NPU Tool 调 NeuroPilot Lite Runtime 跑 MobileNetV2 INT8（端侧物体分类）
   ↓
拿到 top-5 标签（label + score）
   ↓
OpenClaw 把标签 + 用户原 prompt 一起喂给 ERNIE
   ↓
ERNIE 生成自然语言回复（"这是一只英国短毛猫，常见特征..."）
   ↓
回到 OpenClaw 对话框
```

为什么这个场景最值得做：

1. **NPU 价值明确**：图像分类是 NPU 最经典的负载，比 LLM-on-NPU 现实
2. **端云协同**：NPU 出"事实"（标签），ERNIE 出"语言"（解释），分工清楚
3. **演示密度高**：3 分钟视频里能演 3 个不同的拍照 case
4. **延迟可控**：NPU 推理 50-100 ms + ERNIE 300-500 ms ≈ 半秒响应

### 5.2 端侧模型选择

主交付：**MobileNetV2 1.0_224（INT8 量化版）**。

| 项 | 值 |
|---|---|
| 输入 | 224 × 224 × 3 RGB image |
| 输出 | 1000 类 ImageNet 标签 + 概率 |
| 模型大小（INT8） | ~3.4 MB |
| 预期延迟（天玑 9500 APU） | < 5 ms / 张 |
| 来源 | TFLite Model Zoo 官方版本；NeuroPilot 官方 sample 默认带 |

选 MobileNetV2 INT8 的理由：

- **NPU SDK 的"hello world"**：几乎所有移动 NPU 工具链都把 MobileNetV2
  作为参考样例，NeuroPilot 也不例外，最大化降低 SDK 入门风险
- **极小且足够好**：3.4 MB 装机零负担；1000 类 ImageNet 标签对"拍照识物"
  日常场景已足够；天玑 9500 APU 把它当玩具跑，<5 ms 延迟意味着整个
  拍照交互几乎瞬时
- **可后续替换**：MobileNetV2 跑通后，工具链已经熟悉，按需要换更大的
  分类模型或换 YOLOv5n 加边框都很快

---

## 6. 移动端专属 Skills

按优先级排序：

| Skill | 描述 | 优先级 |
|---|---|---|
| **photo-classify** | §5 主场景，必交付 | P0 |
| **notification-summary** | 把通知栏内容打成摘要发给用户 | P1 |
| **schedule-reminder** | 主动心跳提醒待办事项 | P2 |
| **contact-lookup** | 通讯录查询 / 给某人发消息 | P2 |
| **photo-ocr** | 拍照 OCR → ERNIE 总结 / 翻译（OCR 模型另装） | P3 / Stretch |
| **voice-wake** | 语音唤醒（用 NPU 跑端侧 KWS） | P3 / Stretch |

---

## 7. 可行性分析

### 7.1 内存 / 存储

实测前的估算（基于 OpenClaw 同类项目数据 + Android 移动 NPU SDK 典型占用）：

```
Component                    Resident Memory       Persistent Storage
─────────────────────────────────────────────────────────────────────
Termux + Node 18                ~150 MB              ~100 MB
OpenClaw Gateway (warm)         ~400 MB              ~150 MB
NeuroPilot Lite Runtime          ~80 MB               ~50 MB
MobileNetV2 INT8                  ~5 MB               ~4 MB
Android Shell                    ~50 MB               ~30 MB
─────────────────────────────────────────────────────────────────────
Total (idle)                    ~700 MB              ~350 MB
Peak (during NPU + ERNIE)      ~1.2 GB              (no growth)
─────────────────────────────────────────────────────────────────────
任务要求上限                    < 2 GB               n/a
预算余量                        ~800 MB              ample
```

### 7.2 功耗

| 阶段 | 估算功耗 | 备注 |
|---|---|---|
| 后台 idle（仅 Gateway 心跳） | < 100 mW | 类似一个 IM 后台 |
| 用户拍照交互（NPU 跑 + ERNIE 调用） | ~2-3 W 持续 1 秒 | 单次操作可忽略 |
| 30 min 连续运行无 NPU 调用 | < 200 mAh | 估算基于 5000 mAh 电池 ~4% |

### 7.3 网络依赖

ERNIE API 调用是**唯一**对网络的硬依赖（NPU、文件、通知等都在端侧）。
降级策略：网络不可用时，对话回退到"本地缓存的常见回复 + NPU 识别结果"
（degraded mode），不影响 demo 体验。

---

## 8. 里程碑 / 时间规划

假设拿到天玑 9500 工程设备后开始计时：

| Week | 里程碑 | 交付 |
|---|---|---|
| W1 | Termux + OpenClaw + ERNIE adapter 跑通；Android shell project 起步 | OpenClaw 在 Termux 内能响应一条 ERNIE 消息 |
| W2 | NeuroPilot Lite Runtime 学习 + MobileNetV2 sample 跑通；JNI bridge 设计 | NPU 能从命令行跑出分类结果 |
| W3 | 端到端：Android camera → JNI → NPU → OpenClaw → ERNIE → 回 UI | demo 雏形 |
| W4 | notification-summary / schedule-reminder 等次级 Skill；性能 / 内存调优 | 至少 3 类任务可演 |
| W5 | demo 视频录制、文档完善、连续运行稳定性测试 | 提交 PR |
| W6（缓冲）| 修 issue、refine、按 review 反馈调整 | |
