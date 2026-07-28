# 完善 Torch Proxy 的 Compat Override 行为

## 详细描述

Paddle 的 torch proxy 在启用后需要自动暴露 `paddle.compat` 中声明为公开接口的兼容实现，使调用方可以通过对应的 `torch` 命名空间访问这些实现，而不需要额外手工注册 override。

当兼容接口位于嵌套模块中时，从父模块访问子模块也应继续应用相应的 override，而不是回退到原始 Paddle 子模块中的同名对象。已有的 proxy fallback 行为需要保持兼容。

## 验收说明

- 启用 torch proxy 后，`paddle.compat` 子模块公开导出的兼容接口应自动出现在对应的 `torch` 命名空间中，私有接口不应被注册。
- 位于嵌套子模块中的兼容接口应能通过父级 proxy 模块正确访问，并返回兼容实现而不是原始 Paddle 对象。
- 对没有 override 的属性，已有的 proxy fallback 行为应保持不变。

## 技术要求

- 熟悉 Python import system、module proxy 和动态属性访问机制。
- 理解 Paddle torch compatibility layer 的模块映射与生命周期。
- 测试应验证运行期可观察行为，不依赖源码字符串、局部变量名或具体内部容器实现。

## 参考资料

- https://github.com/PaddlePaddle/Paddle/pull/76522

## Acceptance Criteria

- The behavior described above should be implemented correctly.
- Existing valid proxy behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or broadly bypassing proxy validation.
