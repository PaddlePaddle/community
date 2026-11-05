# 支持 Windows 平台 DataLoader 多进程数据加载

|任务名称 | Windows DataLoader 多进程支持 |
|---|---|
|提交作者 | PlumBlossomMaid |
|提交时间 | 2026-07-05 |
|版本号 | V1.0 |
|依赖飞桨版本 | develop |
|文件名 | 20260705_design_windows_dataloader_multiprocess.md |

# 一、概述

## 1、相关背景

飞桨 DataLoader 在 Windows 平台上长期不支持 `num_workers > 0`。当用户在 Windows 上创建 DataLoader 并设置 `num_workers > 0` 时，框架会发出警告并自动将 `num_workers` 降为 0，退化为单进程模式：

```python
# python/paddle/io/reader.py
if num_workers > 0 and sys.platform == 'darwin':
    warnings.warn("...")
    num_workers = 0
```

而 PyTorch 自 1.0 版本起就支持 Windows 上 `num_workers > 0`。在深度学习训练中，多进程数据加载能显著减少数据预处理对 GPU 计算的空闲等待，尤其对于数据增强复杂、IO 密集的场景，性能差距可达数倍。

## 2、功能目标

- 移除 Windows 上 `num_workers` 强制降为 0 的限制
- 实现与 Linux 行为对齐的多进程数据加载
- 支持 `use_shared_memory`、`persistent_workers` 等全部 DataLoader 参数
- 提供对应的 CI 测试保障

## 3、意义

- 填补飞桨在 Windows 平台上的关键功能空白
- 消除 Windows 用户在数据加载环节的性能瓶颈
- 降低飞桨在 Windows 上的开发体验门槛
- 覆盖中国教育、政企市场中大量使用 Windows 的开发场景

# 二、飞桨现状

飞桨 DataLoader 的多进程数据加载基于 Linux 的系统调用实现，核心依赖：

1. **`fork()` 系统调用**：用于创建 worker 子进程，子进程继承父进程内存（写时复制）
2. **`signalfd` / `sigaction`**：用于处理子进程退出信号（SIGCHLD）
3. **`waitid()`**：用于监控子进程退出状态
4. **POSIX 共享内存（`shm_open` / `mmap`）**：用于 worker 与主进程间的数据共享

在 worker 进程间共享数据时，Paddle 使用 `ForkingPickler` 序列化 `DenseTensor`，通过 `_share_filename` 将 tensor 数据写入 POSIX 共享内存，跨进程传递 IPC 名称，接收方通过 `_new_shared_filename` 重建 tensor。

当前限制代码位于：

```python
# python/paddle/io/reader.py:501-507
if num_workers > 0 and (sys.platform == 'darwin' or sys.platform == 'win32'):
    warnings.warn(...)
    num_workers = 0
```

此外，以下模块也包含平台相关的条件编译限制：

- `paddle/phi/core/memory/allocation/mmap_allocator.{h,cc}` — 整个文件被 `#ifndef _WIN32` 包裹
- `paddle/fluid/imperative/data_loader.{h,cc}` — 进程管理代码被 `#ifndef _WIN32` 包裹
- `paddle/fluid/pybind/imperative.cc` — Python 绑定注册被 `#ifndef _WIN32` 包裹
- `python/paddle/io/multiprocess_utils.py` — 清理逻辑被平台守卫限制
- `python/paddle/incubate/multiprocessing/reductions.py` — 存在 `_supported_check()` 禁止非 Linux 平台

# 三、业内方案调研

## 1、PyTorch Windows DataLoader

PyTorch 对 Windows 多进程数据加载的实现方案：

### 进程创建

- 使用 Python `multiprocessing.spawn` 而非 `fork`
- Worker 进程通过 `spawn` 启动，每个 worker 是独立进程，从零加载 Python 解释器
- Worker 启动时间相对 fork 更长（约 2-5 秒 vs <1 毫秒），内存占用更高（每个 worker 约 500-600MB）

### 进程监控

- 使用 `multiprocessing.Process` 的 `is_alive()` 判断 worker 存活
- 使用 `ManagerWatchdog` 基于 Windows API 监控进程状态
- 使用 `WaitForMultipleObjects` 替代 Linux 的 `waitid()`

### 共享内存

- 默认不使用共享内存（`use_shared_memory` 在 Windows 上默认关闭）
- 使用 `file_system` 策略：数据通过 multiprocessing Queue 序列化传输
- 对于大 tensor，使用 `multiprocessing.reduction` 的 `DuplicateHandle` 实现句柄级共享

### 已知限制

- `num_workers` 建议不超过 CPU 核心数（Windows spawn 内存开销大）
- 数据加载类必须定义在模块顶层（pickle 序列化要求）
- `IterableDataset` 的 worker 间数据分割依赖用户自行实现 `get_worker_info()`
- 偶发 worker 初始化竞争，在高 worker 数时可能出现 CUDA 驱动加载冲突

## 2、TensorFlow Windows tf.data

TensorFlow 使用 C++ 实现的数据管道 `tf.data`，其多线程实现不依赖进程 fork，因此跨平台一致性较好。但 TensorFlow 的 C++ 线程模型与 Paddle/PyTorch 的 Python 多进程模型不同，不具直接可比性。

# 四、对比分析

| 维度 | PyTorch Windows | Paddle Linux（原实现） | 本方案 |
|------|-----------------|----------------------|--------|
| 进程创建 | spawn | fork | spawn |
| 进程监控 | WaitForMultipleObjects | waitid() | OpenProcess + GetExitCodeProcess |
| 共享内存 | file_system（Queue） | POSIX shm / mmap | CreateFileMapping / MapViewOfFile |
| Worker 数量建议 | ≤ CPU 核心数 | 无特殊限制 | ≤ CPU 核心数，≥7 时偶发竞争 |
| 内存开销 | 高（每个 worker ~500MB） | 低（fork 共享） | 高（同 PyTorch） |

**核心决策**：保留 Paddle 现有的共享内存传输方案（`_share_filename` / `_new_shared_filename`），而非退化为纯 Queue 传输。原因：
1. 保持与 Linux 一致的数据传输路径，减少条件分支
2. 共享内存在大 tensor 传输时性能优于 Queue 序列化
3. 代码复用度高，仅需替换操作系统 API 调用层

# 五、设计思路与实现方案

## 1、主体设计思路与折衷

### 整体全貌

本方案在飞桨架构中处于 DataLoader 数据流的下层：

```
用户代码 (Python)
    ↓
DataLoader (reader.py)
    ↓
_DataLoaderIterMultiProcess (dataloader_iter.py)
    ↓ 创建 worker 进程
Worker 进程 (_worker_loop in worker.py)
    ↓ 读取数据 → DenseTensor → 序列化
ForkingPickler (_reduce_lodtensor in reductions.py)
    ↓ 共享内存传输
_share_filename / _new_shared_filename (C++ pybind)
    ↓
mmap_allocator.{h,cc}  ← 本方案主要修改点
```

### 主体设计具体描述

方案的核心设计是：**保留 Paddle 现有的 _share_filename 共享内存架构，将底层的操作系统 API 替换为 Windows 等效实现**。

涉及修改三个层面：

**C++ 层（mmap_allocator.cc）：**
- `AllocateMemoryMap`：替换 `shm_open`+`mmap` 为 `CreateFileMappingA`+`MapViewOfFile`
- 关键修复：Windows 上 `UnmapViewOfFile` 会销毁 named file mapping section，必须在所有 reader 完成读取后才释放

**Python 进程管理层（worker.py / dataloader_iter.py）：**
- 用 `OpenProcess`+`GetExitCodeProcess` 替代 `os.getppid()` 的 parent watchdog
- 移除 `if sys.platform != 'win32'` 的平台守卫
- 添加 ForkingPickler 注册确保 DenseTensor 可序列化

**Python 绑定层（core.py / imperative.cc / tensor.cc）：**
- 注册 `_set_process_pids`、`_throw_error_if_process_failed` 等函数为无条件导出
- 修复 pybind11 在 Windows 上的类方法注册问题（`_share_filename` 等）

### 主体设计选型考量

**为什么不退化为纯 Queue 传输（类似 PyTorch file_system 模式）？**

- Paddle 现有架构深度依赖共享内存，纯 Queue 传输需要重写整个数据管道
- 共享内存在大 tensor（如图像、音频特征）传输时避免了一次额外的内存拷贝
- 保持与 Linux 一致的代码路径，减少条件分支的维护成本

**为什么不使用命名管道或 socket 替代共享内存？**

- 命名管道的性能与共享内存接近，但需要额外的序列化/反序列化开销
- 共享内存在多进程间传输零拷贝的 tensor 元数据方面更高效
- 与现有代码架构的耦合度最低

## 2、关键技术点/子模块设计与实现方案

### 技术点一：Windows 共享内存生命周期管理（核心修复）

**问题：** Windows 上 `CreateFileMapping` 创建的 named file mapping section 在最后一个引用（HANDLE 或 view）关闭时自动销毁。当 worker 进程的 tensor 被 GC 后，`RefcountedMemoryMapAllocation::close()` 调用 `UnmapViewOfFile` 销毁了 section，此时 reader（主进程）可能还没通过 Queue 拿到 IPC 名称。

Linux 上无此问题，因为 `munmap` 不会销毁 `/dev/shm` 中的共享内存文件（只有 `shm_unlink` 才会删除名字，且数据保留至所有映射解除）。

**修复方案：** 在 Windows 上调整 close() 函数的释放时机：

1. `AllocateMemoryMap` 不关闭 `CreateFileMapping` 返回的 HANDLE，始终通过 `*shared_fd` 返回给 caller
2. `RefcountedMemoryMapAllocation::close()` 只在 `info->refcount == 0` 时才调用 `UnmapViewOfFile`。当 refcount > 0 时保持 view 映射，让 section 存活
3. HANDLE 在 `close()` 中无条件关闭（通过 `CloseHandle`）

**生命周期示意：**

```
CreateFileMapping → HANDLE=H1, View=V1, refcount=1
  ↓ _shared_incref → refcount=2
  ↓ worker 的 tensor GC → close(): refcount=1, CloseHandle(H1)
  ↓   refcount!=0 → 保持 V1 映射 → section 存活
  ↓ reader → OpenFileMapping → View=V2 → refcount=2
  ↓ reader → _shared_decref → refcount=1
  ↓ reader 的 tensor GC → close(): refcount=0, UnmapViewOfFile(V2)
  ↓   V1 仍由 worker 保持 → section 存活
  ↓ worker 退出 → OS 自动清理 V1 → section 销毁
```

**涉及文件：**
- `paddle/phi/core/memory/allocation/mmap_allocator.{h,cc}`

### 技术点二：Windows worker 进程监控

**问题：** 原代码使用 Linux 特有的 `waitid(P_PID, ..., WEXITED | WNOHANG | WNOWAIT)` 监控 worker 退出状态。Windows 上无等效系统调用。

**修复方案：** 使用 Windows API `OpenProcess` + `WaitForSingleObject` + `GetExitCodeProcess` 替代：

```cpp
// Windows 实现
HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE,
                              FALSE, process_pid);
if (WaitForSingleObject(hProcess, 0) == WAIT_OBJECT_0) {
    DWORD exit_code;
    GetExitCodeProcess(hProcess, &exit_code);
    // exit_code != 0 && exit_code != STILL_ACTIVE → 异常退出
}
```

此外，worker 进程中的 `ParentWatchDog` 原使用 `os.getppid()` 判断父进程死活。在 Windows `spawn` 模式下，`os.getppid()` 可能返回引导进程的 PID 而非真正的父进程 PID，导致 watchdog 误判。修复为使用 `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` + `GetExitCodeProcess` 直接查询父进程状态。

**涉及文件：**
- `paddle/fluid/imperative/data_loader.cc`
- `paddle/fluid/imperative/data_loader.h`
- `python/paddle/io/dataloader/worker.py`

### 技术点三：Windows 进程信号处理

**问题：** Linux 使用 `sigaction` 注册 SIGCHLD、SIGSEGV、SIGBUS 等信号处理器。Windows 使用结构化异常处理（SEH）。

**修复方案：**
- 使用 `SetUnhandledExceptionFilter` 注册最后机会异常处理器，记录崩溃信息（PID、异常码、地址）
- 不注册向量化异常处理（VEH），因为 `MemoryMapFdSet::Clear()` 在持有 mutex 时可能递归崩溃
- 异常码 `0xC0000409`（STATUS_STACK_BUFFER_OVERRUN）由 `/GS` 安全检查触发，通过 `__fastfail` 直接终止进程，SEH 无法捕获

**涉及文件：**
- `paddle/fluid/imperative/data_loader.cc`

### 技术点四：Windows 共享内存的写入/读取路径

**问题：** 原 `_share_filename` 在 worker 端创建共享内存并写入 tensor 数据，`_new_shared_filename` 在 reader 端打开共享内存并读取。Windows 上需要替换底层 API。

**修复方案：**
- 写入端（worker）：`CreateFileMappingA` + `MapViewOfFile` + 拷贝数据，关闭 handle（view 保持 section 存活）
- 读取端（reader）：`OpenFileMappingA` + `MapViewOfFile`，读取数据，关闭 handle

此外，pybind11 在 Windows 上存在类方法注册问题，`DenseTensor.def("_share_filename", ...)` 注册的类方法在 Windows 上不可调用。解决方法是在模块级注册包装函数：

```python
# reductions.py 中，替换：
#   lodtensor._share_filename(...)   →  core._share_filename(lodtensor, ...)
#   cls._new_shared_filename(...)    →  core._new_shared_filename(...)
```

**涉及文件：**
- `paddle/phi/core/memory/allocation/mmap_allocator.{h,cc}`
- `paddle/fluid/pybind/tensor.cc`
- `python/paddle/incubate/multiprocessing/reductions.py`

### 技术点五：Worker 启动竞争缓解

**问题：** Windows spawn 模式下，≥7 个 worker 同时启动时，每个 worker 都独立加载 Paddle 和 CUDA 运行时。NVIDIA 驱动在处理多进程同时 CUDA 初始化时存在竞争条件，偶发 `0xC0000409` 栈缓冲区溢出。

**缓解方案：** 在 worker 启动循环中添加 0.05s 的间隔（stagger）：

```python
for i in range(num_workers):
    worker.start()
    if sys.platform == 'win32' and num_workers > 4:
        time.sleep(0.05)  # 错开 CUDA 初始化窗口
```

**备选方案讨论（留待 reviewer 决策）：**
- **方案 A（当前实现）：** 固定 0.05s stagger — 简单有效，12 workers 总延迟 0.55s
- **方案 B（自适应）：** 检测到上一个 worker 意外退出才加延迟 — 只在有竞争时减速
- **方案 C（无 stagger）：** 不加延迟，仅文档说明 — 代码干净但偶发 worker 死亡可能让用户困惑

**涉及文件：**
- `python/paddle/io/dataloader/dataloader_iter.py`

## 3、主要影响的模块接口变化

### 核心接口变化

- 无新增 Python 公开 API
- 无修改现有 API 签名
- 内部 C++ 接口变化仅在 Windows 平台生效，Linux 行为完全不变

### 对各模块影响排查

| 模块 | 影响 |
|------|------|
| 网络定义 | 无影响 |
| 底层数据结构 | mmap_allocator 新增 Windows 实现路径 |
| OP | 无影响 |
| 数据 IO | DataLoader 现支持 Windows 多进程 |
| 执行 | 无影响 |
| 分布式 | 无影响 |
| 模型保存 | 无影响 |
| 预测部署 | 无影响 |

# 六、测试和验收的考量

## 单元测试

新增 CI 测试文件：
- `test/legacy_test/test_dataloader_multiprocess.py`

包含 5 个测试用例：
1. `test_multiprocess_singleprocess_loss_close` — nw=2 vs nw=0 损失值一致性
2. `test_multiprocess_with_shared_memory` — use_shared_memory=True
3. `test_multiprocess_more_workers` — nw=4 压力测试
4. `test_multiprocess_persistent_workers` — persistent_workers=True
5. `test_multiprocess_iterable_dataset` — IterableDataset

## 验收标准

- CI 测试 5/5 通过
- `num_workers>0` 不再降级为 0
- `use_shared_memory=True` 模式下数据正确性验证（与单进程结果一致）
- 大数据集（5000 样本 × 128 维）+ nw=8 稳定运行

## 手动验证

使用 `test_windows_dataloader.py`（不提交至 CI）进行本地验证，涵盖 nw=1~12 的全部扫测。

# 七、影响面

## 对用户的影响

- 正面：Windows 用户现在可以使用 `num_workers>0` 加速数据加载
- 负面：无（功能新增，不破坏现有行为）

## 对二次开发用户的影响

- 无新增 API
- Linux 上行为完全不变
- Windows 上原先被强制降为 0 的行为现在正常工作

## 对框架架构的影响

- Linux 代码路径完全不变
- Windows 代码路径新增实现，遵循「只补不添」原则——仅补充缺失的平台实现，不新增函数、不扩展接口

## 对性能的影响

- Linux：无变化
- Windows：多进程数据加载相比单进程有显著性能提升（实测数据依赖具体硬件和数据集）
- 注意：Windows spawn 模式的内存开销高于 Linux fork（每个 worker ~580MB），建议 `num_workers` 不超过 CPU 核心数

## 对比业内深度学习框架的差距与优势

- 补齐了与 PyTorch 在 Windows 多进程数据加载方面的功能差距
- 相比 PyTorch 的 file_system 回退方案，Paddle 保留了共享内存传输路径，在大 tensor 场景下可能更高效
- 与 TensorFlow 的 C++ 线程方案无可比性

## 其他风险

- **0xC0000409 已知问题**：Paddle C++ 层在高并发数据处理时的隐藏 bug（NVIDIA 驱动多进程初始化竞争）。0.05s stagger 已大幅降低触发概率。该问题同样存在于原生 Linux DataLoader 的极低概率下（仅 Windows spawn 加剧了触发频率）。建议 Paddle 官方后续排查。

# 八、排期规划

| Milestone | 时间 | 内容 |
|-----------|------|------|
| RFC 评审 | 2026-07 | 提交 RFC 设计文档，收集 reviewer 反馈 |
| 代码整理 | RFC 通过后 | 分拆 PR（mmap 修复、Python 改进、测试），整理 commit 历史 |
| PR 提交 | 代码整理后 | 提交流程，CI 验证，合并 |
| 回滚预案 | 合并前 | 若发现严重问题，可通过还原移除 Windows 守卫的 commit 快速回滚 |

# 名词解释

- **spawn**：Python multiprocessing 的进程启动方式，子进程从零加载 Python 解释器，常用于 Windows
- **fork**：Unix 系统调用，子进程是父进程的副本（写时复制），Linux 默认的 multiprocessing 启动方式
- **named file mapping**：Windows 内核对象，通过名称在不同进程间共享内存
- **__fastfail**：Windows 快速失败机制，直接终止进程，不经过任何异常处理器

# 附件及参考资料

- [RFC 设计文档模板](./design_template.md)
- [PyTorch DataLoader Windows 实现](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py)
- [Windows CreateFileMapping 文档](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-createfilemappinga)
