# Paddle Frawework Contributor Club 第十次会议纪要

## 会议概况

- 会议时间：2022-09-08 19:00 - 20:00
- 会议地点：线上会议
- 参会人：本次会议共有 33 名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由轮值主席徐晓健（[Nyakku Shigure](https://github.com/SigureMo)）主持。

## 会议分享与讨论

### 新成员的自我介绍

PFCC 新成员 [六个骨头](https://github.com/zrr1999) 进行了自我介绍，欢迎加入 PFCC！

### 开发者体验提升计划——Python 工具链

[Nyakku Shigure](https://github.com/SigureMo) 分享《开发者体验提升计划——Python 工具链》主题内容，以 Python 工具链为例，介绍了开发者体验提升的一些关键工具，以及如何制作和添加这些工具。主要包括以下几部分内容：

- 常见开发工作流及工具介绍：介绍了从 Editor / IDE 到 git hooks 再到 CI 这一常见的工作流阶段，以及在这些工作流中可以插入的工具，包含 Linter 和 Formatter
- 工具的实现方法：介绍了这些工具的主要实现方案，包含利用纯文本（含正则）到词法分析，再进一步到语法分析得到的 AST，在这些不同的结构上进行操作的一些方式，着重介绍和演示了 AST 的操作模式
- 新工具的引入所需要解决的问题：提到了引入新工具要面临的两大问题，一是如何修复存量代码中的问题，二是如何将新的工具引入到 CI 中
- 一些将来可能/可以做的事情：列举了一些 Paddle 在开发者体验提升部分可以进一步做的一些事情
  - [flake8](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/code_style_flake8.md)
  - [文档规范性检查工具](https://github.com/ShigureLab/dochooks/tree/main/dochooks/api_doc_checker)
  - 各种 Warning 的消除
    - [Sphinx](https://github.com/PaddlePaddle/docs/issues/5177)
    - [C++ 编译](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/code_style_compiler_warning.md)
  - [静态类型提示](https://github.com/cattidea/paddlepaddle-stubs)

QA 交流环节：

- [gglin001](https://github.com/gglin001)：Paddle 目前是否有一些文档来介绍一些工具上或者编辑器上的推荐配置吗？
- [Nyakku Shigure](https://github.com/SigureMo)：目前应该还是没有的，因此在这里我也主要是推荐将来在文档中添加相关的内容。

### 使用工具提升 Paddle 开发效率

[gglin001](https://github.com/gglin001) 分享《使用工具提升 Paddle 开发效率》主题内容，主要包含以下几部分内容：

- 在 Linux 上支持 Ninja 编译，大幅提高重编译的效率，详情见 [PR #44210](https://github.com/PaddlePaddle/Paddle/pull/44210)
- 介绍在修改少量 Python 端代码时提升开发效率的一种方式：开发调试过程中直接修改 Python 源码进行调试（而不是 build 目录下的 Python 代码），需要将 build 目录下的构建产物 copy 到源代码目录，另外重编译时可仅构建 `paddle_pybind` 这一 target，这样可以节省 copy 源码及重新 packing 一个 wheel 包的时间，提升开发效率
- 通过简单的方式支持 Clang 编译并通过 demo 展示支持 Clang 编译后通过 VS Code 中 CodeLLDB 扩展带来的单步调试体验的提升（比 GDB 调试快很多），这个与上一个 demo 中的配置文件可见 [gglin001/Paddle-fork](https://github.com/gglin001/Paddle-fork/tree/pfcc_demo_20220908)
- 对 Paddle 未来工具使用上的一些期望
  - Linux Clang 编译的支持
  - 支持更多比较现代化的基础设施（[mold](https://github.com/rui314/mold) linker [#45761](https://github.com/PaddlePaddle/Paddle/issues/45761)、最新版本 gcc 等）
  - 更加优雅的 CMake 配置（目前静态库太多，在仅仅编译单一单测时会缺失某一些库）
  - 为 C++ 代码暴露的 API 提供 `.pyi` 文件，以提供更好的 Python 端智能提示和代码补全
  - 官方发布一些开发时的最佳实践文档，为开发者提供参考

QA 交流环节：

- [Nyakku Shigure](https://github.com/SigureMo)：对于为 C++ 代码提供 `.pyi` 文件的话，我刚刚提到的[类型提示项目 paddlepaddle-stubs](https://github.com/cattidea/paddlepaddle-stubs) 是以一个第三方包的形式提供了一个 [stub-only 的包](https://peps.python.org/pep-0561/#stub-only-packages)，可以为框架的使用者提供更好的类型提示，但对于开发 Paddle 的开发者可能并没有太大帮助。当然我非常希望 Paddle 可以内部直接集成类型提示，不过这可能需要花费很多的时间去做。

---

- [luotao1](https://github.com/luotao1)：Ninja 进行重编译与直接使用 make 编译的速度有进行对比过吗？
- [gglin001](https://github.com/gglin001)：由于时间的原因没有来得及进行测试，不过这个加速效果在体验上是非常明显的。
- [luotao1](https://github.com/luotao1)：最后提到的《More possible improvements》内容非常有帮助，比如 gcc 版本和静态库太多的问题都是需要解决的，一些其他的更多的问题也可以直接提到 issue 里。

### 开源社区信息同步

[Ligoml](https://github.com/Ligoml) 进行了一些信息的同步：

- [飞桨框架代码仓库的角色及权限介绍](https://github.com/PaddlePaddle/community/blob/master/contributors/community-membership.md)。
- AI Studio 新上线的「框架开发任务」功能介绍和展示。

QA 交流环节：

- [luotao1](https://github.com/luotao1)：AI Studio 上新的「框架开发任务」功能中提供的 VS Code 环境如果有预装某些扩展的需求也是可以提出来的。

### GTC 2022 Watch Party 活动预告

[luotao1](https://github.com/luotao1) 介绍了参与 GTC 2022 Watch Party 的方式。

### 下次会议安排

确定下次会议的时间为两周后的同一个时间段。并确定下次会议的主席为：林旭（[isLinXu](https://github.com/isLinXu)），副主席待定。
