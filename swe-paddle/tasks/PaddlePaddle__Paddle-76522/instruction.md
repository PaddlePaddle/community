# 完善 Torch Proxy 对 `paddle.compat` 接口的支持

## 详细描述

Paddle 提供了 Torch Proxy 功能。调用 `paddle.compat.enable_torch_proxy()` 后，用户可以继续使用 `import torch` 和 PyTorch 风格的模块路径，但实际调用的是 Paddle 提供的对应接口。例如，访问 `torch.nn` 时会使用 Paddle 中对应的功能。

`paddle.compat` 中还提供了一些专门用于兼容 PyTorch 行为的接口。当前启用 Torch Proxy 后，这些兼容接口不会自动出现在对应的 `torch` 路径下，导致用户通过 PyTorch 风格路径访问时，得到的仍然是普通 Paddle 接口，而不是已经准备好的兼容实现。

需要完善这一行为：启用 Torch Proxy 后，如果 `paddle.compat` 已经提供了某个公开兼容接口，那么通过对应的 `torch` 路径访问时应使用该兼容接口。该规则也要适用于 `torch.nn`、`torch.nn.functional` 等子模块，并保证属性访问与 `import` 导入得到相同结果。

对于 `paddle.compat` 中没有提供兼容实现的接口，继续沿用现有的 Torch Proxy 行为。

## 验收说明

* 调用 `paddle.compat.enable_torch_proxy()` 后，已有的公开兼容接口可以通过对应的 `torch` 路径使用。
* `torch.nn`、`torch.nn.functional` 等子模块中的兼容接口可以通过属性访问和 `import` 正常取得，且结果一致。
* 没有兼容实现的接口继续使用原有代理逻辑，已有的接口覆盖行为不能受到影响。
* 未启用 Torch Proxy 时，Paddle 和 `torch` 模块原有行为保持不变。

## 技术要求

- 熟悉 Python
- 了解 Python 模块导入机制
- 了解 Paddle Torch Proxy 和 `paddle.compat`
