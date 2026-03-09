# 新硬件厂商仓库独立指导文档

## 一、文档目标
本文档用于指导一家新的硬件厂商，参考 `Paddle-iluvatar` 的方式，将自己的硬件后端从多硬件共享仓演进为厂商独立仓，并支持与 Paddle 联合编译，最终产出单独的硬件版 Paddle whl。

本文档重点回答以下问题：

1. 新仓库应该如何组织目录。
2. 厂商如何只通过修改自己的独立仓，接入 Paddle 的联合编译体系。
3. 如何把硬件产物一起打进 Paddle whl。
4. 如何规划 CI、验收和后续升级。

## 二、适用范围
适用于满足以下条件的硬件厂商：

1. 已经有基于 PaddleCustomDevice 的现有后端实现，或至少已有 runtime/kernel 代码。
2. 希望独立维护自己的仓库、CI 和发版节奏。
3. 希望用户只安装一个 whl，而不是 `paddlepaddle` + 硬件插件双包。
4. 希望后续新增功能时，逐步从函数指针模式迁移到“统一接口 + 基类继承”的实现方式。

## 三、目标形态
厂商完成改造后，应达到以下目标：

1. 厂商拥有独立仓库，例如 `Paddle-<Vendor>`。
2. `Paddle` 作为该仓库的子模块或等价方式纳入版本管理。
3. 构建入口由厂商仓中的 `build_paddle.sh` 驱动。
4. Paddle 在打开对应厂商开关后，可以纳入厂商后端子工程。
5. 硬件产物被复制到 `build/python/paddle/paddle_custom_device`。
6. 最终只产出一个可安装 whl。
7. 安装后 `import paddle` 即可自动发现并加载硬件插件。

## 四、推荐仓库结构
建议以 `Paddle-iluvatar` 为参考，采用如下结构：

```text
Paddle-<Vendor>/
├── Paddle/                         # Paddle 子模块
├── backends/
│   └── <device_name>/
│       ├── CMakeLists.txt
│       ├── build_paddle.sh
│       ├── install_paddle.sh
│       ├── cmake/
│       ├── common/
│       ├── runtime/
│       ├── kernels/
│       ├── plugins/
│       ├── include/
│       ├── tests/
│       └── tools/
├── ci/
├── docker/
├── scripts/
└── README.md
```

说明：

1. `Paddle/` 由厂商仓自行维护版本，可通过子模块固定到指定 commit。
2. `backends/<device_name>/` 是厂商后端主目录，建议保持与原 PaddleCustomDevice 后端目录相近的组织方式，降低迁移成本。
3. `common/` 用于承载“统一接口 + 基类继承”模式下的子类声明和实现。
4. `ci/` 和 `docker/` 由厂商自己维护，避免再依赖多硬件共享仓的统一流水线。

## 五、推荐迁移步骤
建议按以下五个阶段推进，而不是一次性重构。

### 前置说明：厂商职责边界
本文档面向硬件厂商，因此默认以下内容已经由 Paddle 框架侧预置或统一提供：

1. Paddle 已具备联合编译主流程。
2. Paddle 已预留厂商子工程接入点和对应编译开关。
3. Paddle 已支持将 `paddle.paddle_custom_device` 打进最终 whl。
4. 如需“统一接口 + 基类继承”扩展模式，Paddle 已提供对应统一声明和调用入口。

因此，厂商的工作重点应当是：

1. 维护自己的独立仓。
2. 在独立仓内准备后端代码、构建脚本和打包逻辑。
3. 对接 Paddle 已经提供好的联合编译和打包能力。

不要求厂商修改 Paddle 源码本身。

### 阶段 1：完成仓库独立
目标是先把代码管理关系拆开。

建议动作：

1. 新建 `Paddle-<Vendor>` 仓库。
2. 将现有后端代码迁移到 `backends/<device_name>/`。
3. 将 `Paddle` 以子模块方式引入到仓库根目录。
4. 在厂商仓维护自己的分支策略、tag、CI 和发布说明。

验收标准：

1. 厂商仓可以独立 checkout、独立触发 CI。
2. Paddle 版本由厂商仓自行固定，不再与其他硬件共享同一升级节奏。

### 阶段 2：完成联合编译接入
目标是基于 Paddle 已提供的联合编译入口，让厂商后端被主工程拉起。

建议动作：

1. 在 `backends/<device_name>/build_paddle.sh` 中直接调用 Paddle 顶层 CMake。
2. 由厂商脚本透传硬件相关编译参数，而不是再单独编译一个插件仓。
3. 对接 Paddle 已提供的厂商开关，例如：
   4. `WITH_CUSTOM_DEVICE`
   5. `WITH_CUSTOM_DEVICE_SUB_BUILD`
   6. `CUSTOM_DEVICE_CMAKE_ARGS`
7. 保证厂商仓目录布局、变量命名和 CMake 入口符合 Paddle 预置接入规范，使主工程可以自动纳入厂商子工程。

验收标准：

1. 只执行一次主构建即可同时编译 Paddle 和厂商后端。

### 阶段 3：完成单包打包
目标是把厂商插件一起打进 Paddle whl。

建议动作：

1. 在厂商子工程编译完成后，将以下产物复制到 `Paddle/build/python/paddle/paddle_custom_device/`：
   1. `libpaddle-<device_name>.so`
   2. 厂商头文件目录
   3. `__init__.py`
2. 按 Paddle 既有打包规范准备目录内容，确保 `paddle.paddle_custom_device` 能被自动纳入 whl。
3. 如需自定义最终包名，使用 Paddle 已支持的 `PADDLE_PYTHON_PACKAGE_NAME` 能力。
4. 保证在没有厂商插件目录时，不影响 CPU-only 打包流程。

验收标准：

1. 构建结束后，whl 中包含 `paddle/paddle_custom_device/`。
2. 用户安装一个 whl 后，即可自动加载厂商设备插件。

### 阶段 4：新增统一接口扩展模式
目标是在不修改 Paddle 源码的前提下，对接 Paddle 已提供的统一扩展入口。

建议动作：

1. 使用 Paddle 已提供的统一声明，例如 `CustomDeviceFuncBase`。
2. 按 Paddle 已提供的工厂或注册入口接入厂商实现。
3. 在厂商仓 `common/` 目录实现子类 `CustomDeviceFunc` 并覆写目标能力。
4. 厂商实现只放在独立仓中，不把具体实现回写到 Paddle 仓。
5. 对存量功能继续保留函数指针 + 动态加载路径，新增功能优先走继承模式。

验收标准：

1. 新能力接入时，不再需要先抽象一层新的函数指针接口。
2. 调用侧代码不感知具体硬件类型。

### 阶段 5：完成 CI 与发版闭环
目标是让厂商仓真正可长期维护。

建议动作：

1. 建立至少三类 CI：
   1. 编译 CI：验证联合编译和打包。
   2. 单测 CI：验证已有 kernel/runtime/plugin 功能。
   3. 模型 CI：验证 PaddleFormers、FastDeploy 或厂商重点模型。
4. 建立固定的 Paddle 升级流程：
   5. 更新 Paddle 子模块 commit。
   6. 执行编译和回归测试。
   7. 再决定是否发布新版本。
8. 版本号建议与 Paddle 主版本保持一致，避免 ABI 混淆。

验收标准：

1. 厂商仓可以独立发版，不阻塞其他硬件。
2. Paddle 升级影响范围可以被厂商仓 CI 独立识别。

## 六、关键实现约束
为了让后续更多厂商都能复用同一模式，建议统一遵守以下约束。

### 1. Paddle 必须是主工程
不要让厂商后端继续作为完全独立的顶层工程单独产出主包。推荐模式必须是：

1. Paddle 顶层负责配置、编译和打包。
2. 厂商后端作为子工程参与构建。

这样做的原因是：

1. 最终 whl 归属清晰。
2. Python 打包逻辑只维护一套。
3. 更容易保证 ABI、头文件和运行时路径一致。

### 2. 运行时仍兼容动态加载
联合编译并不意味着要立刻删除旧有 `LoadCustomDevice` 路径。建议保留动态加载能力，原因如下：

1. 存量插件逻辑可以继续工作。
2. 新老方案可以并行过渡。
3. 问题定位时可以区分“编译接入问题”和“运行时注册问题”。

### 3. 打包目录必须统一
建议所有厂商统一使用：

```text
build/python/paddle/paddle_custom_device/
```

不要每家厂商定义不同的 Python 包内路径，否则后续安装、调试和 `CUSTOM_DEVICE_ROOT` 逻辑都会碎片化。

### 4. 包名允许自定义，但导入路径不变
建议：

1. 通过 `PADDLE_PYTHON_PACKAGE_NAME` 控制最终 whl 名称。
2. Python 内部导入路径仍保持 `paddle`。
3. 插件目录仍保持 `paddle/paddle_custom_device`。

这样做可以兼顾厂商品牌诉求和 Paddle 生态兼容性。

## 七、厂商侧需要完成的代码改动
厂商只需要完成自己独立仓中的改动：

1. `backends/<device_name>/build_paddle.sh`
   1. 驱动 Paddle 主构建。
   2. 透传厂商编译参数。
2. `backends/<device_name>/CMakeLists.txt`
   1. 编译 runtime/kernel/plugin。
   2. 将产物复制到 Paddle 打包目录。
3. `common/`
   1. 在 Paddle 已提供扩展入口的前提下，提供继承基类后的厂商实现。
4. `runtime/`、`kernels/`、`plugins/`
   5. 迁移或复用原有 PaddleCustomDevice 代码。
6. `ci`
   7. 建立厂商独立 CI。

## 十一、验收清单
一家新厂商完成改造后，按以下清单验收：

1. 是否已有独立仓 `Paddle-<Vendor>`。
2. 是否将 Paddle 纳入厂商仓管理。
3. 是否存在厂商自己的 `build_paddle.sh`。
4. 是否实现 Paddle 主工程驱动的联合编译。
5. 是否只产出一个可安装 whl。
6. 安装后是否能识别出目标设备类型。
7. `paddle.utils.run_check()` 是否通过。
8. 是否具备厂商独立 CI。
9.  Paddle 已开放新的统一扩展入口，是否已支持至少一个新能力按“统一接口 + 继承”模式接入。