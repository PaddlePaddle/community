# Python 端引入 ruff 作为代码风格检查/自动修复工具

| 任务名称     | Python 端引入 ruff 作为代码风格检查/自动修复工具 |
| ------------ | ---------------------------------------------- |
| 提交作者     | @SigureMo                                      |
| 提交时间     | 2023-03-05                                     |
| 版本号       | v0.2                                           |
| 依赖飞桨版本 | develop                                        |
| 文件名       | 20230305_introducing_ruff.md                  |

## 一、概述

### 1、相关背景

不久前 Paddle 刚刚完成了 [Flake8 代码风格检查工具的引入](../../pfcc/call-for-contributions/code_style/code_style_flake8.md)（Tracking issue: [PaddlePaddle/Paddle#46039](https://github.com/PaddlePaddle/Paddle/issues/46039)）、[Python 2.7 相关代码退场](../../pfcc/call-for-contributions/code_style/legacy_python2.md)、[Python 3.5/3.6 相关代码退场](../../pfcc/call-for-contributions/code_style/legacy_python36minus.md)（Tracking issue: [PaddlePaddle/Paddle#46837](https://github.com/PaddlePaddle/Paddle/issues/46837)）三项 Call for Contribution 任务，Python 端代码的风格、整洁性已经得到了极大的提高。但是在我们实践的过程中也遇到了很多问题，部分问题现在也没有解决。本 RFC 旨在引入一个新的 Linter [Ruff](https://github.com/charliermarsh/ruff)，以解决我们前几个 Call for Contribution 任务中的遗留问题，并进一步利用 Ruff 中内置的 rules 来优化代码风格。

Ruff 是一个利用 Rust 编写的 Linter，拥有极佳的运行速度（10-100 倍于现有的 Linter）。它重新实现了 Flake8 内置的绝大多数 rules 和若干受欢迎的 Flake8 插件。特别的是，它为大多数 rules 提供了**自动修复**功能。

在引入 Flake8 的过程中，由于 Flake8 没有提供自动修复功能，所以在存量修复时往往需要借助一些其他工具来完成，比如我们在 F401 错误码的修复过程（存量修复）中主要使用了 [autoflake](https://github.com/PyCQA/autoflake)，并在之后 [[Tools] Add autoflake pre-commit hook to remove unused-imports](https://github.com/PaddlePaddle/Paddle/pull/47455) 引入了 autoflake 以自动移除未使用的 import（增量自动修复）。因此如果引入 Ruff，则可以利用一个工具直接完成存量修复和增量拦截和增量自动修复的功能。

在 Python 旧版本退场系列任务的开发前期，Ruff 还是一个非常早期的项目，只实现了较少的 [pyupgrade rules](https://beta.ruff.rs/docs/rules/#pyupgrade-up)，而 [pyupgrade](https://github.com/asottile/pyupgrade) 本身则因为不支持禁用掉某一条或几条 rule 而难以使用，因此当时大多使用的是手动修复 + 写代码转换脚本来完成的。随着 Ruff 连续几个月的开发，目前 pyupgrade 相应的 rules 已经完全实现（相关 tracking issue：[Implement pyupgrade](https://github.com/charliermarsh/ruff/issues/827)），因此引入 Ruff 一方面可以大大减轻后续的 Python 旧版本退场的清理工作，另一方面可以对增量进行控制。

在[升级飞桨代码中使用 NumPy 1.20 数据类型的用法](https://github.com/PaddlePaddle/Paddle/issues/49949)中，我们对 NumPy 的弃用用法已经进行了替换，但是由于没有控制增量，因此目前代码库中已经出现了少许增量。而 Ruff 近期也新增了 NumPy 相关的一些 rules，比如 [NPY001](https://beta.ruff.rs/docs/rules/#numpy-specific-rules-npy) 对应的就是对 NumPy 1.20 弃用的数据类型的检查，引入 Ruff 即可直接对增量进行控制。

以上即是我们在相关社区任务中遇到的一些问题，而引入 Ruff 即可同时解决以上多个问题。

### 2、功能目标

引入 Ruff 工具，利用 Ruff 完成以下效果：

- 引入 Ruff 的 [pyupgrade rules](https://beta.ruff.rs/docs/rules/#pyupgrade-up)：可自动对旧版本遗留代码进行升级，一方面可以完成旧版本清理任务中未做的「增量控制」，另一方面可以降低未来旧版本清理时的成本（比如 4 个月后[即将 EOL 的 Python 3.7](https://endoflife.date/python)）；
- 引入 Ruff 实现的 Flake8 其余插件：如比较受欢迎的 [flake8-bugbear](https://github.com/PyCQA/flake8-bugbear)、[flake8-comprehensions](https://github.com/adamchainz/flake8-comprehensions)；
- 引入 Ruff 实现的专有 rule：比如 NumPy 弃用类型别名的检测 [NPY001](https://beta.ruff.rs/docs/rules/#numpy-specific-rules-npy)，引入该 rule 即可避免[升级飞桨代码中使用 NumPy 1.20 数据类型的用法](https://github.com/PaddlePaddle/Paddle/issues/49949)出现增量；
- 利用 Ruff 的自动修复功能替换掉 autoflake：目前 autoflake 是仅仅为了自动 Flake8 F401 rule 而引入的，因此可替换掉以精简工具数量；
- 引入 Ruff 实现的 Pylint rule 来替代原有的 iScan Python 流水线中的 Pylint 功能（[PFCC Call for contribution - IDEA：iScan 流水线退场](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/README.md#ideaiscan-%E6%B5%81%E6%B0%B4%E7%BA%BF%E9%80%80%E5%9C%BA)）。

### 3、意义

- 进一步规范 Python 端代码风格；
- 通过自动化的方式来将部分代码转换为更加高效的代码（如 [flake8-comprehensions](https://github.com/adamchainz/flake8-comprehensions) 相关 rules）；
- 精简 Python 端代码风格检查工具数量；
- 部分 rule 可启用自动修复功能，避免手动修复增量，降低开发者解决 Linter Error 的成本；
- Ruff 速度很快，可以使开发者有着更好的开发体验。

## 二、飞桨现状

Paddle 目前引入了 Black、isort、Flake8、autoflake 以及一个[仅用于检查 Docstring 但是已经失效的 pylint](https://github.com/PaddlePaddle/Paddle/issues/47821) 共五个工具用于 Python 端的代码风格监控。Black 和 isort 属于 Formatter，Flake8 则属于 Linter，autoflake 目前仅仅用于 Flake8 F401 rule 的自动修复。

## 三、业内方案调研

### 代码风格检查工具对比

Ruff 的目标是成为 Python 语言的全能型 Linter，因此 Ruff 实现了大多数现有主流 Linter 的功能，并实现了部分 Formatter 的功能，整体对比如下：

| | Flake8 | PyLint | Black | isort | Pyupgrade | Ruff |
| - | - | - | - | - | - | - |
| 速度 | 慢 | 非常慢 | 快 | 一般 | 一般 | 非常快 |
| 是否支持插件 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 支持的 rule 数量 | 默认包含 132 条 rules，可通过安装插件进一步扩展 | 默认包含 395 条 rules，可通过自定义插件进一步扩展 | 仅格式化 | 仅排序 | 包含 43 条 rules | 默认包含 500+ 条 rules，且在持续增长中 |
| 支持自动修复 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |

Ruff 虽然目前不支持插件，但 Ruff 现有的内置 rules 基本可以覆盖所有常见的需求。

此外，Ruff 对于用户来说拥有统一的配置和 CLI 选项，不必同时学习多个工具，使用一个工具即可完成多个工具的工作。

### 社区使用情况调研

由于 Ruff 自开始开发以来时间还不长，因此使用 Ruff 的项目还不多，但目前正在高速增长着，在 GitHub 上已经拥有 9.5k Star，在 Python 社区非常受欢迎。目前 Ruff 已经被 pandas、Transformers (Hugging Face)、Diffusers (Hugging Face)、SciPy、Jupyter、Pylint 等知名项目所使用。

PyTorch 等深度学习框架尚未引入 Ruff，但是 PyTorch 已经引入了 Flake8，除了启用了默认的 pycodestyle、pyflakes、mccabe、还引入了 flake8-bugbear、flake8-comprehensions、flake8-executable、flake8-logging-format、flake8-coding、flake8-pyi（见 [pytorch/pytorch - requirements-flake8.txt](https://github.com/pytorch/pytorch/blob/master/requirements-flake8.txt)），其中 flake8-bugbear、flake8-comprehensions、flake8-executable、flake8-logging-format 均已被 Ruff 实现，flake8-pyi 正在[开发中](https://github.com/charliermarsh/ruff/issues/848)，flake8-coding 尚未被实现。此外，PyTorch 目前也在考虑引入 Ruff，见 [[BE]: Add ruff to lintrunner - use for additional plugins like pyupgrade etc](https://github.com/pytorch/pytorch/issues/94737)。

TensorFlow 的代码风格管控较为宽松，目前并未使用 Flake8 这类 Linter，但使用了 pylint 来规范代码风格，见 [tensorflow/tensorflow - tensorflow/tools/ci_build/pylintrc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/ci_build/pylintrc)。目前 Ruff 也已经实现了很多 Pylint 的 rules，可通过引入 Ruff 实现的 Pylint rules 来达到同样的效果。

## 四、设计思路与实现方案

### 1、主体设计思路与折衷

#### 基本配置

同 Flake8，Ruff 也需要 ignore 部分文件，初始化的配置如下：

`pyproject.toml`：

```toml
[tool.ruff]
exclude = [
    "./build",
    "./python/paddle/fluid/[!t]**",
    "./python/paddle/fluid/tra**",
    "./python/paddle/utils/gast/**",
    "./python/paddle/fluid/tests/unittests/npu/**",
    "./python/paddle/fluid/tests/unittests/mlu/**",
]
target-version = "py37"
select = []
```

并添加相应的 pre-commit hook：

`.pre-commit-config.yaml`：

```yaml
-   repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.254
    hooks:
    -   id: ruff
        args: [--fix, --exit-non-zero-on-fix, --no-cache]
```

测试 PR 见：[[CodeStyle] initialize ruff config](https://github.com/PaddlePaddle/Paddle/pull/51201)

#### 确定需要引入的 rules

根据「相关背景」和「功能目标」中的调研，我们可以引入 Pyupgrade（UP）、NumPy-specific rules（NPY001）、flake8-bugbear（B）、flake8-comprehensions（C4）、Pylint（PL）、Pyflakes（F401），其余 Rules 在后续调研后认为合适即可引入。

引入 Rules 所需要考察的要点如下：

- 调研存量修复是否方便，是否 Ruff 是否已经提供了自动修复功能，如果没有，存量是否较少可手动修复，否则并不适合引入；
- 调研该 rule 是否会引起性能的倒退，比如 UP038（isinstance-with-tuple）在 Python 3.10 上会[引起性能下降](https://github.com/charliermarsh/ruff/issues/2923)，因此并不适合在 Python 3.10 上引入，仅仅适合在 Python 3.11 上引入；另外由于 UP031（printf-string-formatting）	会稍微引起性能下降，而 UP032	（f-string）则会引起性能提升，因此建议两个 rule 一起引入（即在合并成一个 PR）；
- 调研该 rule 是否会引起代码风格的倒退，比如 UP015（redundant-open-modes）并不满足「Explicit is better than implicit」的原则（参考 [removing "r" in open(..., "r") is not an upgrade](https://github.com/asottile/pyupgrade/issues/714)、[PEP 20 – The Zen of Python](https://peps.python.org/pep-0020/)），因此不适合引入。

此外应该注意 Paddle 的动转静单测部分，部分测试 case 会依赖于某一特定语法结构，相关代码应该注意避免 lint 和 autofix，遇到相关问题应该在配置文件的 `per-file-ignores` 中配置以跳过。

#### 确定该 rule 是否使用 Ruff 提供的自动修复功能

虽然 Ruff 为大多数可以自动修复的 rule 都提供了自动修复的功能，但并不是所有的自动修复效果都是合理的，因此是否要使用该 rule 的自动修复功能也需要逐 rule 进行判断，对于不合理的情况，应该仅仅拦截错误，由开发者手动修复。

主要要注意以下几点：

- 该 rule 的自动修复是否可能导致代码语义的变化；
- 该 rule 的自动修复是否会引起代码风格的倒退，如 NPY001（numpy-deprecated-type-alias），该 rule 的自动修复会将 `np.int` 替换为 `int` 以确保代码的语义不变，但是对于这种情况，使用合适的 `np.int32` 或者 `np.int64` 是更推荐的修复方式，因此该 rule 并不推荐使用自动修复功能；


#### 可移除的旧有拦截 / 修复工具

- 在引入 F401 自动修复功能后可以从 `.pre-commit-config.yaml` 移除 autoflake，引入 PR 见 [[Tools]Add autoflake pre-commit hook to remove unused-imports/var](https://github.com/PaddlePaddle/Paddle/pull/47455)；
- 在引入 UP010 rule 后可以从 `tools/check_file_diff_approvals.sh` 移除相关检查项，引入 PR 见 [[CodeStyle] add CI script to prevent future import](https://github.com/PaddlePaddle/Paddle/pull/46466)

#### 存量统计

pyupgrade（UP）- 16 条、2358 处：

```
$ ruff --select UP . --statistics                
  14    UP004   [*] Class `Event` inherits from `object`
   6    UP005   [*] `assertEquals` is deprecated, use `assertEqual`
   1    UP006   [*] Use `list` instead of `List` for type annotations
  80    UP008   [*] Use `super()` instead of `super(__class__, self)`
  16    UP009   [*] UTF-8 encoding declaration is unnecessary
   3    UP010   [*] Unnecessary `__future__` import `print_function` for target Python version
   2    UP012   [*] Unnecessary call to `encode` as UTF-8
 151    UP015   [*] Unnecessary open mode parameters
  51    UP018   [*] Unnecessary call to `str`
   9    UP024   [*] Replace aliased errors with `OSError`
  10    UP027   [*] Replace unpacked list comprehension with a generator expression
  14    UP028   [*] Replace `yield` over `for` loop with `yield from`
 279    UP030   [*] Use implicit references for positional format fields
 162    UP031   [*] Use format specifiers instead of percent format
1335    UP032   [*] Use f-string instead of `format` call
 225    UP034   [*] Avoid extraneous parentheses
```

Pylint（PL）- 17 条、7684 处：

```
$ ruff --select PL . --statistics
 215    PLR5501 [ ] Consider using `elif` instead of `else` then `if` to remove one indentation level
  13    PLC0414 [*] Import alias does not rename original package
   6    PLC3002 [ ] Lambda expression called directly. Execute the expression inline instead.
   2    PLR0206 [ ] Cannot have defined parameters for properties
2113    PLR0402 [*] Use `from paddle import nn` in lieu of alias
   5    PLR0133 [ ] Two constants compared in a comparison, consider replacing `10 > 5`
  80    PLR1701 [ ] Merge these isinstance calls: `isinstance(norm, (float, int))`
  41    PLR1722 [*] Use `sys.exit()` instead of `exit`
2182    PLR2004 [ ] Magic value used in comparison, consider replacing 3 with a constant variable
 177    PLW0603 [ ] Using the global statement to update `_g_amp_state_` is discouraged
  83    PLW0602 [ ] Using global for `_g_amp_state_` but no assignment is done
  46    PLR0911 [ ] Too many return statements (8/6)
1446    PLR0913 [ ] Too many arguments to function call (8/5)
 469    PLR0912 [ ] Too many branches (20/12)
 469    PLR0915 [ ] Too many statements (51/50)
 336    PLW2901 [ ] Outer for loop variable `pair` overwritten by inner assignment target
   1    PLE1205 [ ] Too many arguments for `logging` format string
```

NumPy-specific rules（NPY001）- 1 条、2 处：

```
$ ruff --select NPY001 . --statistics
2       NPY001  [*] Type alias `np.bool` is deprecated, replace with builtin type
```

pyflakes（F401）：无存量

flake8-comprehensions（C4）- 14 条、750 处：

```
$ ruff --select C4 . --statistics
 19     C400    [*] Unnecessary generator (rewrite as a `list` comprehension)
  5     C401    [*] Unnecessary generator (rewrite as a `set` comprehension)
  4     C402    [*] Unnecessary generator (rewrite as a `dict` comprehension)
 22     C403    [*] Unnecessary `list` comprehension (rewrite as a `set` comprehension)
  2     C404    [*] Unnecessary `list` comprehension (rewrite as a `dict` comprehension)
172     C405    [*] Unnecessary `list` literal (rewrite as a `set` literal)
355     C408    [*] Unnecessary `tuple` call (rewrite as a literal)
  3     C409    [*] Unnecessary `list` literal passed to `tuple()` (rewrite as a `tuple` literal)
  8     C410    [*] Unnecessary `list` literal passed to `list()` (remove the outer call to `list()`)
  6     C411    [*] Unnecessary `list` call (remove the outer call to `list()`)
  1     C413    [*] Unnecessary `reversed` call around `sorted()`
 17     C414    [*] Unnecessary `list` call within `sorted()`
104     C416    [*] Unnecessary `list` comprehension (rewrite using `list()`)
 32     C417    [*] Unnecessary `map` usage (rewrite using a `list` comprehension)
```

flake8-bugbear（B）- 17 条、1373 处：

```
$ ruff --select B . --statistics
  1     B004    [ ] Using `hasattr(x, '__call__')` to test if x is callable is unreliable. Use `callable(x)` for consistent results.
  9     B005    [ ] Using `.strip()` with multi-character strings is misleading the reader
196     B006    [ ] Do not use mutable data structures for argument defaults
801     B007    [*] Loop control variable `atype` not used within loop body
 24     B008    [ ] Do not perform function call `fluid.global_scope` in argument defaults
 59     B009    [*] Do not call `getattr` with a constant attribute value. It is not any safer than normal property access.
 29     B010    [*] Do not call `setattr` with a constant attribute value. It is not any safer than normal property access.
 34     B011    [*] Do not `assert False` (`python -O` removes these calls), raise `AssertionError()`
  7     B015    [ ] Pointless comparison. This comparison does nothing but waste CPU instructions. Either prepend `assert` or remove it.
  1     B016    [ ] Cannot raise a literal. Did you intend to return it or raise an Exception?
 14     B017    [ ] `assertRaises(Exception)` should be considered evil
  7     B020    [ ] Loop control variable `data` overrides iterable it iterates
113     B023    [ ] Function definition does not bind loop variable `opt_step`
  1     B024    [ ] `FLClientBase` is an abstract base class, but it has no abstract methods
  9     B026    [ ] Star-arg unpacking after a keyword argument is strongly discouraged
  3     B027    [ ] `AlgorithmBase.collect_model_info` is an empty method in an abstract base class, but has no abstract decorator
 65     B904    [ ] Within an except clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
```

其中标记有 `[*]` 的表示 Ruff 提供自动修复功能

### 2、关键技术点/子模块设计与实现方案

#### 可行性验证

[[CodeStyle][pyupgrade] automatically rewrite code with ruff](https://github.com/PaddlePaddle/Paddle/pull/50477) 已经尝试了引入 Ruff 的全部 pyupgrade rules（UP），可以通过全量单测。

#### 推进方式

第一个 PR 会修改配置，引入部分没有存量的 rule，之后由外部开发者提交 PR 来逐步引入有存量的 rule。具体实施步骤如下：

首先可按照「确定需要引入的 rules」和「确定该 rule 是否使用 Ruff 提供的自动修复功能」两小节确定是否引入该 rule 以及是否使用该 rule 的自动修复功能，如果确定引入该 rule 则在 `pyproject.toml` 的 Ruff 配置部分添加该 rule 对应的 violation，如添加 `UP010`：

```diff
  [tool.ruff]
  # ...
- select = []
+ select = ["UP010"]
```

如果 Ruff 为该错误码提供自动修复方案且**自动修复方案合适**，之后在 Paddle 项目根目录运行 `ruff . --fix` 自动修复即可，否则需要 `ruff .` 后手动修复。

对于引入但不引入自动修复功能的 rule，需要在配置项 `unfixable` 中添加该 rule 的 violation，如 `NPY001`：

```diff
  [tool.ruff]
  # ...
  select = ["NPY001"]
- unfixable = []
+ unfixable = ["NPY001"]
```

对于多个存量较少的 rule 可以合并为一个 PR 提交，但尽可能不要超过 20 个文件的修改量。

建议使用单独的 PR 来修改配置，以避免 PR 频繁冲突。但可以将多个 rule 的配置 PR 合并成为一个。

### 3、主要影响的模块接口变化

不会对模块接口产生影响。

# 五、测试和验收的考量

确保不会引起性能倒退，确保不会引起代码风格倒退，通过 CI 各条流水线。

## 六、影响面

### 对用户的影响

用户对于框架内部代码风格的变动不会有任何感知，不会有任何影响。

### 对二次开发用户的影响

可以提高 Paddle 代码风格，极大提高开发体验。

### 对框架架构的影响

在 pre-commit 工作流中引入 Ruff，因此在该 hook 引入后开发者首次 commit 需要稍微等一段时间用于初始化 Ruff 环境，后续提交代码不受影响。

### 对性能的影响

在确保 Rule 是安全的情况下，对性能不会产生任何影响，部分 Rule 可能会提高性能。

### 对比业内深度学习框架的差距与优势的影响

引入 Ruff 的 flake8-bugbear、flake8-comprehensions rules 可以对齐 PyTorch 的代码风格，引入 Ruff 的 Pylint rules 可以对齐 TensorFlow 的代码风格。在完成本 RFC 中所述的全部 rules 后，Paddle 的代码风格管控将会超越 TensorFlow 和 PyTorch。

### 其他风险

Ruff 本身还处于早期阶段，因此部分选项和 rule 的作用可能会在未来变动，但在锁版本的情况下不存在该问题（pre-commit 配置中强制锁版本）。

关于用于更新 Ruff 版本的成本，只需要偶尔更新即可（比如一个月），并在更新版本的时候仔细阅读 Release Note 来确定升级方案，不需要频繁更新，维护成本并不高。

> 相关链接：Ruff 作者关于 stable 版本的回复 [Open Version 1 road map, or whatever you decide to call “stable”… (comment)](https://github.com/charliermarsh/ruff/issues/1992#issuecomment-1405497643)

## 七、排期规划

| 任务 | 存量 | 预计完成时间 | 备注 |
| - | - | - | - |
| Ruff 配置初始化 | - | 1 人 1 天 | [@SigureMo](https://github.com/SigureMo) [PR #51201](https://github.com/PaddlePaddle/Paddle/pull/51201) |
| NumPy-specific rules（NPY001）| 存量 1 条、2 处 | 1 人 1 天 | |
| pyupgrade（UP）| 存量 16 条、2358 处 | 1 人 1 周 |（测试 PR 见 [PR #50477](https://github.com/PaddlePaddle/Paddle/pull/50477)，实际合入拆分成多个 PR 以便 review）|
| pyflakes（F401）| 存量 14 条、750 处 | 1 人 1 天 | |
| flake8-comprehensions（C4）| 存量 14 条、750 处 | 1 人 1 周 | |
| flake8-bugbear（B）| 存量 17 条、1373 处 | 1 人 2 周 | |
| Pylint（PL）| 存量 17 条、7684 处 | 2 周 | |
| 6 月更新 Ruff，清理存量 | - | 1 人 2 天 | |
| 9 月更新 Ruff，清理存量 | - | 1 人 2 天 | |
| 12 月更新 Ruff，清理存量 | - | 1 人 2 天 | |

待 Ruff 稳定后（0.1），可按照版本号来更新 Ruff，比如 0.1、0.2、0.3 等。

上表以优先级排序，不代表实际完成顺序，可并行进行。具体执行将会由 SigureMo 和外部开发者一起完成。

## 八、替代方案

### 直接引入 pyupgrade

我们已经对引入 pyupgrade 进行了测试（见 [[CodeStyle][pyupgrade] automatically rewrite code with pyupgrade](https://github.com/PaddlePaddle/Paddle/pull/48140)），但 pyupgrade 不提供选项来禁用某一个或多个 rule，这意味着只能全盘接受或者不使用，而部分 rule 对于代码风格并不是提升，因此不会选择，另外 PyTorch 社区因为同样的原因没有选择引入 pyupgrade（见 [Option to disable `Unpacking list comprehensions`](https://github.com/asottile/pyupgrade/issues/794)），因此也在考虑利用 Ruff 来引入 pyupgrade 的 rules。

### 直接引入 flake8-bugbear、flake8-comprehensions 等 Flake8 插件

Flake8 不提供自动修复功能，而 Ruff 可以尽可能地提供自动修复功能，可以同时减少引入时存量修复的工作量和之后开发者引入增量时修复的工作量。

### 利用 Ruff 取代现有 Flake8 中内置的 pycodestyle、pyflakes、mccabe 插件

目前 Ruff 实现的功能尚不能完全完成这三个插件的功能，部分社区在直接使用 Ruff 替换掉原有的 Flake8 后引起了代码风格回归的问题，见 [ENH: Added analytical formula for truncnorm entropy - discussion](https://github.com/scipy/scipy/pull/17874#discussion_r1103885021)，因此在 Ruff 完全实现 Flake8 内置插件全部 rules 且功能稳定之前，不会考虑直接使用 Ruff 替代这三个内置插件。

### 利用 Ruff 取代现有的 isort 和 black

Ruff 目前同样实现了 isort 和 black 的功能，但这些功能实现尚处于早期，甚至还在持续开发中，因此现阶段不会考虑引入，日后如果 Ruff 的这两项功能稳定，且成为 Python 社区的主流解决方案之后，将会考虑使用 Ruff 直接替代这两项功能。

## 名词解释

- rule：规则，即对应于某一类错误的检查项，如 `UP010` 检查不必要的 future import。

## 附件及参考资料

1. [关于引入 Ruff 的前期调研](https://github.com/PaddlePaddle/Paddle/pull/50458#issuecomment-1431280278)
2. [Ruff documentation](https://beta.ruff.rs/docs/)
3. [引入 Ruff 初始化配置 PR](https://github.com/PaddlePaddle/Paddle/pull/51201)
