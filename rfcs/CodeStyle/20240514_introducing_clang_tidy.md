# clang-tidy代码风格检查工具的引入

|任务名称|clang-tidy代码风格检查工具的引入|
|------|------|
|提交作者|@ApricityXX|
|提交时间|2024-05-14|
|版本号|v0.2|
|依赖飞桨版本|develop|
|文件名| 20240514_introducing_clang_tidy.md |

## 一、概述

### 1.1 背景

当前 `Paddle` 已引入 `precommit`、`pylint`、`remove-ctrlf`、`cpplint`、`clang-format`、`yapf`、`cmakelint`、`cmake-format` 和 `flake8` 等多种代码风格检查工具。在v0.1中，Paddle引入了 `clang-tidy` 静态分析工具。

但是目前`Paddle`开启的拦截数量仍然较少，本次技术文档主要对标`pytorch`进行规则的引入，在此基础上，希望可以针对各个错误进行存量的修复，并且能够在CI中开启相应规则的拦截，以实现增量拦截

### 1.2 意义

- 进一步规范 C++ 代码风格；
- 进一步提升 C++ 代码稳健性，提升工程质量和 C++ 代码的可维护性，减少潜在的 bug；
- 自动修复部分典型的编程错误，降低开发者解决 Linter Error 的成本。
- 开启CI拦截，后续自动化进行增量的拦截，避免后续无止境的存量修复

### 1.3 前置工作

PaddlePaddle于2023年已经进行过一轮clang-tidy的技术修复方案，具体见：

1. [20230501_introducing_clang_tidy.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/CodeStyle/20230501_introducing_clang_tidy.md)
2. [赛题四：在飞桨框架中引入 clang-tidy Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/54073)

工作修复了clang-tidy40多项存量，并且在pre-commit中加入了clang-tidy的检查，这些项目的优化使得PaddlePaddle的代码风格有了显著改善。

具体工作见下表

> | 编号            | 错误类型                                                     | 错误数量                                | 认领人                                                       | PR链接                                                       |
> | --------------- | ------------------------------------------------------------ | --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
> | 1               | cppcoreguidelines-init-variables                             | 11002                                   |                                                              |                                                              |
> | 2               | modernize-redundant-void-arg                                 | 1294                                    |                                                              |                                                              |
> | 3               | bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions | 327                                     | [@enkilee](https://github.com/enkilee)                       | [#61759](https://github.com/PaddlePaddle/Paddle/pull/61759) ✅ [#62109](https://github.com/PaddlePaddle/Paddle/pull/62109) ✅ |
> | 4               | cppcoreguidelines-pro-type-member-init                       | 216                                     |                                                              |                                                              |
> | 5✅ (2024/3/1)   | cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays    | 151                                     | [@enkilee](https://github.com/enkilee)                       | [#61751](https://github.com/PaddlePaddle/Paddle/pull/61751)  |
> | 6               | modernize-unary-static-assert                                | 132                                     |                                                              |                                                              |
> | 7✅ (2024/3/1)   | bugprone-branch-clone                                        | 132                                     | [@enkilee](https://github.com/enkilee)                       | [#61735](https://github.com/PaddlePaddle/Paddle/pull/61735)  |
> | 8               | performance-unnecessary-copy-initialization                  | 127                                     |                                                              |                                                              |
> | 9               | cppcoreguidelines-avoid-goto,hicpp-avoid-goto                | 109                                     |                                                              |                                                              |
> | 10              | cppcoreguidelines-pro-type-const-cast                        | 90                                      |                                                              |                                                              |
> | 11              | modernize-pass-by-value                                      | 65                                      |                                                              |                                                              |
> | 12✅ (2024/3/7)  | modernize-loop-convert                                       | 46                                      | [@enkilee](https://github.com/enkilee)                       | [#61725](https://github.com/PaddlePaddle/Paddle/pull/61725)  |
> | 13✅ (2024/2/21) | modernize-deprecated-headers                                 | 44                                      | [@enkilee](https://github.com/enkilee)                       | [#61721](https://github.com/PaddlePaddle/Paddle/pull/61721)  |
> | 14✅ (2024/2/20) | misc-unused-alias-decls                                      | 44                                      | [@enkilee](https://github.com/enkilee)                       | [#61716](https://github.com/PaddlePaddle/Paddle/pull/61716)  |
> | 15✅ (2024/2/20) | performance-inefficient-vector-operation                     | 30                                      | [@enkilee](https://github.com/enkilee)                       | [#61715](https://github.com/PaddlePaddle/Paddle/pull/61715)  |
> | 16              | clang-analyzer-optin.cplusplus.VirtualCall                   | 26                                      |                                                              |                                                              |
> | 17✅ (2024/3/1)  | cppcoreguidelines-explicit-virtual-functions,modernize-use-override | 20                                      | [@enkilee](https://github.com/enkilee)                       | [#61714](https://github.com/PaddlePaddle/Paddle/pull/61714)  |
> | 18              | clang-analyzer-core.NullDereference                          | 18                                      |                                                              |                                                              |
> | 19✅ (2024/2/26) | readability-container-size-empty                             | 16                                      | [@enkilee](https://github.com/enkilee)                       | [#61713](https://github.com/PaddlePaddle/Paddle/pull/61713)  |
> | 20✅ (2023/12/5) | modernize-use-nullptr                                        | 15                                      | [@ccsuzzh](https://github.com/ccsuzzh)                       | [#59626](https://github.com/PaddlePaddle/Paddle/pull/59626)  |
> | 21✅ (2024/2/21) | performance-for-range-copy                                   | 14                                      | [@enkilee](https://github.com/enkilee)                       | [#61712](https://github.com/PaddlePaddle/Paddle/pull/61712)  |
> | 22              | cppcoreguidelines-no-malloc                                  | 13                                      |                                                              |                                                              |
> | 23              | modernize-use-emplace                                        | 11                                      |                                                              |                                                              |
> | 24✅ (2024/3/1)  | hicpp-exception-baseclass                                    | 11                                      | [@enkilee](https://github.com/enkilee)                       | [#61691](https://github.com/PaddlePaddle/Paddle/pull/61691)  |
> | 25✅ (2024/2/26) | modernize-use-transparent-functors                           | 9                                       | [@enkilee](https://github.com/enkilee)                       | [#61689](https://github.com/PaddlePaddle/Paddle/pull/61689)  |
> | 26✅ (2024/2/20) | misc-unused-using-decls                                      | 9                                       | [@enkilee](https://github.com/enkilee)                       | [#61616](https://github.com/PaddlePaddle/Paddle/pull/61616)  |
> | 27✅ (2024/2/21) | performance-move-const-arg                                   | 7                                       | [@enkilee](https://github.com/enkilee)                       | [#61615](https://github.com/PaddlePaddle/Paddle/pull/61615)  |
> | 28✅ (2024/2/21) | modernize-use-equals-default                                 | 7                                       | [@enkilee](https://github.com/enkilee)                       | [#61614](https://github.com/PaddlePaddle/Paddle/pull/61614)  |
> | 29✅ (2024/2/20) | bugprone-exception-escape                                    | 7                                       | [@enkilee](https://github.com/enkilee)                       | [#61609](https://github.com/PaddlePaddle/Paddle/pull/61609)  |
> | 30              | performance-inefficient-string-concatenation                 | 5                                       |                                                              |                                                              |
> | 31✅ (2024/2/29) | clang-analyzer-cplusplus.NewDeleteLeaks                      | 5                                       | [@enkilee](https://github.com/enkilee)                       | [#62129](https://github.com/PaddlePaddle/Paddle/pull/62129)  |
> | 32✅ (2024/2/29) | bugprone-unused-raii                                         | 5                                       | [@enkilee](https://github.com/enkilee)                       | [#62129](https://github.com/PaddlePaddle/Paddle/pull/62129)  |
> | 33✅ (2024/2/20) | bugprone-inaccurate-erase                                    | 5                                       | [@enkilee](https://github.com/enkilee)                       | [#61589](https://github.com/PaddlePaddle/Paddle/pull/61589)  |
> | 34✅ (2024/2/29) | bugprone-copy-constructor-init                               | 5                                       | [@enkilee](https://github.com/enkilee)                       | [#62129](https://github.com/PaddlePaddle/Paddle/pull/62129)  |
> | 35✅ (2024/2/20) | modernize-use-bool-literals                                  | 3                                       | [@enkilee](https://github.com/enkilee)                       | [#61580](https://github.com/PaddlePaddle/Paddle/pull/61580)  |
> | 36✅ (2024/2/20) | clang-analyzer-core.DivideZero                               | 3                                       | [@enkilee](https://github.com/enkilee)                       | [#61580](https://github.com/PaddlePaddle/Paddle/pull/61580)  |
> | 37✅ (2024/2/20) | bugprone-integer-division                                    | 3                                       | [@enkilee](https://github.com/enkilee)                       | [#61580](https://github.com/PaddlePaddle/Paddle/pull/61580)  |
> | 38✅ (2024/2/26) | performance-trivially-destructible                           | 2                                       | [@enkilee](https://github.com/enkilee)                       | [#61556](https://github.com/PaddlePaddle/Paddle/pull/61556)  |
> | 39✅ (2024/2/26) | modernize-make-unique                                        | 2                                       | [@enkilee](https://github.com/enkilee)                       | [#61556](https://github.com/PaddlePaddle/Paddle/pull/61556)  |
> | 40✅ (2024/2/26) | modernize-avoid-bind                                         | 2                                       | [@enkilee](https://github.com/enkilee)                       | [#61556](https://github.com/PaddlePaddle/Paddle/pull/61556)  |
> | 41✅ (2024/2/29) | cppcoreguidelines-slicing                                    | 2                                       | [@enkilee](https://github.com/enkilee)                       | [#62129](https://github.com/PaddlePaddle/Paddle/pull/62129)  |
> | 42✅ (2024/2/20) | performance-noexcept-move-constructor                        | 1                                       | [@enkilee](https://github.com/enkilee)                       | [#61555](https://github.com/PaddlePaddle/Paddle/pull/61555)  |
> | 43✅ (2024/2/20) | clang-diagnostic-unused-but-set-variable                     | 1                                       | [@enkilee](https://github.com/enkilee)                       | [#61555](https://github.com/PaddlePaddle/Paddle/pull/61555)  |
> | 44✅ (2024/2/20) | clang-analyzer-security.FloatLoopCounter                     | 1                                       | [@enkilee](https://github.com/enkilee)                       | [#61555](https://github.com/PaddlePaddle/Paddle/pull/61555)  |
> | 45✅ (2024/2/29) | clang-analyzer-cplusplus.NewDelete                           | 1                                       | [@enkilee](https://github.com/enkilee)                       | [#62129](https://github.com/PaddlePaddle/Paddle/pull/62129)  |
> | 46✅ (2024/2/21) | clang-analyzer-core.NonNullParamChecker                      | 1                                       | [@enkilee](https://github.com/enkilee)                       | [#61494](https://github.com/PaddlePaddle/Paddle/pull/61494)  |
> | 47✅ (2024/2/21) | bugprone-unhandled-self-assignment                           | 1                                       | [@enkilee](https://github.com/enkilee)                       | [#61494](https://github.com/PaddlePaddle/Paddle/pull/61494)  |
> | 48✅ (2024/2/20) | bugprone-string-integer-assignment                           | 1                                       | [@enkilee](https://github.com/enkilee)                       | [#61492](https://github.com/PaddlePaddle/Paddle/pull/61492)  |
> | 49✅ (2024/2/20) | bugprone-misplaced-widening-cast                             | 1                                       | [@enkilee](https://github.com/enkilee)                       | [#61492](https://github.com/PaddlePaddle/Paddle/pull/61492)  |
> | 50✅ (2024/2/20) | bugprone-infinite-loop                                       | 1                                       | [@enkilee](https://github.com/enkilee)                       | [#61492](https://github.com/PaddlePaddle/Paddle/pull/61492)  |
> | 67✅ (2023/7/11) | 在 `.pre-commit-config.yaml` 在添加 `clang-tidy` 检查项      | [@GreatV](https://github.com/GreatV)    | [#55279](https://github.com/PaddlePaddle/Paddle/pull/55279) [#55894](https://github.com/PaddlePaddle/Paddle/pull/55894) |                                                              |
> | 68✅(2023/7/11)  | 使用单独的脚本运行 `clang-tidy`，开发者便于手动执行检查      | 运行`tools/codestyle/clang-tidy.py`即可 | [#55279](https://github.com/PaddlePaddle/Paddle/pull/55279)  |                                                              |

但是美中不足的是工作并没有开启CI的拦截，导致pr的不断增多的同时，存量也在不断增加，这是一个急需改善并解决的问题，否则PaddlePaddle只会陷入无止境的存量修复之中，因此这也是本次工作的重点内容。

## 二、飞桨现状

Paddle目前引入了`clang-format`、`cpplint`等工具用于C++端的代码风格监控。

其中，`clang-format`用于格式化代码，它可以根据用户提供的配置文件（`.clang-format`或`_clang-format`）将代码自动格式化成指定的风格，如[LLVM](https://llvm.org/docs/CodingStandards.html)风格、[Google](https://google.github.io/styleguide/cppguide.html)风格等。

`cpplint`是由Google开发和维护的一个代码风格检查工具，用于检查C/C++文件的风格问题，遵循[Google的C++风格指南](http://google.github.io/styleguide/cppguide.html)。`cpplint`可以检查代码中的格式错误、命名规范、注释、头文件和其他编程约定等问题。

`clang-tidy`是一个静态代码分析工具，可用于检查代码中的潜在问题，提升代码稳健性。`clang-tidy`可以检查代码中的内存管理错误、类型不匹配错误、未定义行为等问题。`clang-tidy`可以使用不同的检查器配置，每个配置可以启用或禁用一组相关的检查器。

总的来说，`clang-format`与`cpplint`用于提高代码可读性和风格的工具，而`clang-tidy`则是用于发现和修复潜在问题和错误的工具。在飞桨中引入`clang-tidy`，并与`clang-format`和`cpplint`结合使用，可进一步提高代码质量与可维护性。

而引入`clang-tidy`之后，任务的重中之重是能够开启增量拦截，使得后续提交的代码不会出现`clang-tidy`规则内的错误，然后逐渐修复`paddle`内部目前所存的代码错误以及风格修复。

## 三、`clang-tidy` 相关调研

### 3.1`pytorch`调研

- 版本： **15.0.6**

- 针对`pytorch`的规则，修改`.clang-tidy`如下：

  ---
  ```yaml
  ---
  Checks: '
  bugprone-*,
  -bugprone-easily-swappable-parameters,
  -bugprone-forward-declaration-namespace,
  -bugprone-macro-parentheses,
  -bugprone-lambda-function-name,
  -bugprone-reserved-identifier,
  -bugprone-swapped-arguments,
  -bugprone-unchecked-optional-access,
  clang-diagnostic-missing-prototypes,
  cppcoreguidelines-*,
  -cppcoreguidelines-avoid-do-while,
  -cppcoreguidelines-avoid-magic-numbers,
  -cppcoreguidelines-avoid-non-const-global-variables,
  -cppcoreguidelines-interfaces-global-init,
  -cppcoreguidelines-macro-usage,
  -cppcoreguidelines-owning-memory,
  -cppcoreguidelines-pro-bounds-array-to-pointer-decay,
  -cppcoreguidelines-pro-bounds-constant-array-index,
  -cppcoreguidelines-pro-bounds-pointer-arithmetic,
  -cppcoreguidelines-pro-type-cstyle-cast,
  -cppcoreguidelines-pro-type-reinterpret-cast,
  -cppcoreguidelines-pro-type-static-cast-downcast,
  -cppcoreguidelines-pro-type-union-access,
  -cppcoreguidelines-pro-type-vararg,
  -cppcoreguidelines-special-member-functions,
  -cppcoreguidelines-non-private-member-variables-in-classes,
  -facebook-hte-RelativeInclude,
  hicpp-exception-baseclass,
  hicpp-avoid-goto,
  misc-*,
  -misc-const-correctness,
  -misc-include-cleaner,
  -misc-use-anonymous-namespace,
  -misc-unused-parameters,
  -misc-no-recursion,
  -misc-non-private-member-variables-in-classes,
  -misc-confusable-identifiers,
  modernize-*,
  -modernize-macro-to-enum,
  -modernize-return-braced-init-list,
  -modernize-use-auto,
  -modernize-use-default-member-init,
  -modernize-use-using,
  -modernize-use-trailing-return-type,
  -modernize-use-nodiscard,
  performance-*,
  readability-container-size-empty,
  readability-delete-null-pointer,
  readability-duplicate-include
  readability-misplaced-array-index,
  readability-redundant-function-ptr-dereference,
  readability-redundant-smartptr-get,
  readability-simplify-subscript-expr,
  readability-string-compare,
  '
  HeaderFilterRegex: '^(paddle/(?!cinn)).*$'
  AnalyzeTemporaryDtors: false
  WarningsAsErrors: '*'
  ...
  ```

  而展开来看的话，pytorch支持**246**种规则的检查，规则数量在使用`clang-tidy`并行检查工具之后可以得到，并且要及时更新。

  

### 3.2`clang-tidy`进行存量扫描

注意：以下规则检测基于2024.5.13的`paddlepaddle`的`develop`分支以及`pytorch`的检查规则

首先，我们将`paddlepaddle`编译`cmake`之后,我们使用`clang-tidy`的并行检查工具进行扫描，使用的规则便是`pytorch`设置的规则

扫描命令：

```shell
python ./tools/codestyle/clang-tidy.py -p=build -j$(nproc) \
> -clang-tidy-binary=clang-tidy \
> -extra-arg=-Wno-unknown-warning-option \
> -extra-arg=-Wno-pessimizing-move \
> -extra-arg=-Wno-braced-scalar-init \
> -extra-arg=-Wno-deprecated-copy \
> -extra-arg=-Wno-dangling-gsl \
> -extra-arg=-Wno-final-dtor-non-final-class \
> -extra-arg=-Wno-implicit-int-float-conversion \
> -extra-arg=-Wno-inconsistent-missing-override \
> -extra-arg=-Wno-infinite-recursion \
> -extra-arg=-Wno-mismatched-tags  \
> -extra-arg=-Wno-self-assign \
> -extra-arg=-Wno-sign-compare \
> -extra-arg=-Wno-sometimes-uninitialized \
> -extra-arg=-Wno-tautological-overlap-compare \
> -extra-arg=-Wno-unused-const-variable \
> -extra-arg=-Wno-unused-lambda-capture \
> -extra-arg=-Wno-unused-private-field \
> -extra-arg=-Wno-unused-value \
> -extra-arg=-Wno-unused-variable  \
> -extra-arg=-Wno-overloaded-virtual  \
> -extra-arg=-Wno-defaulted-function-deleted  \
> -extra-arg=-Wno-delete-non-abstract-non-virtual-dtor  \
> -extra-arg=-Wno-return-type-c-linkage  > clang-tidy.log  2>&1
```

检索`clang-tidy.log`文件：

```shell
grep "error:" clang-tidy.log | wc -l
```

得到总 error错误数量为：**9148**

整理一下各模块的错误数量，如下所示：

| 错误类型                                                     | 数量 |
| ------------------------------------------------------------ | ---- |
| bugprone-argument-comment                                    | 0    |
| bugprone-assert-side-effect                                  | 0    |
| bugprone-assignment-in-if-condition                          | 5    |
| bugprone-bad-signal-to-kill-thread                           | 0    |
| bugprone-bool-pointer-implicit-conversion                    | 0    |
| bugprone-branch-clone                                        | 79   |
| bugprone-copy-constructor-init                               | 2    |
| bugprone-dangling-handle                                     | 0    |
| bugprone-dynamic-static-initializers                         | 0    |
| bugprone-exception-escape                                    | 5    |
| bugprone-fold-init-type                                      | 1    |
| bugprone-forwarding-reference-overload                       | 0    |
| bugprone-implicit-widening-of-multiplication-result          | 149  |
| bugprone-inaccurate-erase                                    | 0    |
| bugprone-incorrect-roundings                                 | 0    |
| bugprone-infinite-loop                                       | 0    |
| bugprone-integer-division                                    | 0    |
| bugprone-macro-repeated-side-effects                         | 0    |
| bugprone-misplaced-operator-in-strlen-in-alloc               | 0    |
| bugprone-misplaced-pointer-arithmetic-in-alloc               | 0    |
| bugprone-misplaced-widening-cast                             | 0    |
| bugprone-move-forwarding-reference                           | 0    |
| bugprone-multiple-statement-macro                            | 0    |
| bugprone-narrowing-conversions                               | 472  |
| bugprone-no-escape                                           | 0    |
| bugprone-not-null-terminated-result                          | 0    |
| bugprone-parent-virtual-call                                 | 0    |
| bugprone-posix-return                                        | 0    |
| bugprone-redundant-branch-condition                          | 0    |
| bugprone-shared-ptr-array-mismatch                           | 0    |
| bugprone-signal-handler                                      | 0    |
| bugprone-signed-char-misuse                                  | 3    |
| bugprone-sizeof-container                                    | 0    |
| bugprone-sizeof-expression                                   | 0    |
| bugprone-spuriously-wake-up-functions                        | 0    |
| bugprone-string-constructor                                  | 0    |
| bugprone-string-integer-assignment                           | 0    |
| bugprone-string-literal-with-embedded-nul                    | 0    |
| bugprone-stringview-nullptr                                  | 0    |
| bugprone-suspicious-enum-usage                               | 0    |
| bugprone-suspicious-include                                  | 0    |
| bugprone-suspicious-memory-comparison                        | 0    |
| bugprone-suspicious-memset-usage                             | 0    |
| bugprone-suspicious-missing-comma                            | 0    |
| bugprone-suspicious-semicolon                                | 0    |
| bugprone-suspicious-string-compare                           | 0    |
| bugprone-terminating-continue                                | 0    |
| bugprone-throw-keyword-missing                               | 0    |
| bugprone-too-small-loop-variable                             | 0    |
| bugprone-undefined-memory-manipulation                       | 0    |
| bugprone-undelegated-constructor                             | 0    |
| bugprone-unhandled-exception-at-new                          | 0    |
| bugprone-unhandled-self-assignment                           | 0    |
| bugprone-unused-raii                                         | 0    |
| bugprone-unused-return-value                                 | 0    |
| bugprone-use-after-move                                      | 1    |
| bugprone-virtual-near-miss                                   | 0    |
| clang-analyzer-apiModeling.Errno                             | 0    |
| clang-analyzer-apiModeling.StdCLibraryFunctions              | 0    |
| clang-analyzer-apiModeling.TrustNonnull                      | 0    |
| clang-analyzer-apiModeling.TrustReturnsNonnull               | 0    |
| clang-analyzer-apiModeling.google.GTest                      | 0    |
| clang-analyzer-apiModeling.llvm.CastValue                    | 0    |
| clang-analyzer-apiModeling.llvm.ReturnValue                  | 0    |
| clang-analyzer-core.CallAndMessage                           | 3    |
| clang-analyzer-core.CallAndMessageModeling                   | 0    |
| clang-analyzer-core.DivideZero                               | 0    |
| clang-analyzer-core.DynamicTypePropagation                   | 0    |
| clang-analyzer-core.NonNullParamChecker                      | 0    |
| clang-analyzer-core.NonnilStringConstants                    | 0    |
| clang-analyzer-core.NullDereference                          | 21   |
| clang-analyzer-core.StackAddrEscapeBase                      | 0    |
| clang-analyzer-core.StackAddressEscape                       | 0    |
| clang-analyzer-core.UndefinedBinaryOperatorResult            | 0    |
| clang-analyzer-core.VLASize                                  | 0    |
| clang-analyzer-core.builtin.BuiltinFunctions                 | 0    |
| clang-analyzer-core.builtin.NoReturnFunctions                | 0    |
| clang-analyzer-core.uninitialized.ArraySubscript             | 0    |
| clang-analyzer-core.uninitialized.Assign                     | 0    |
| clang-analyzer-core.uninitialized.Branch                     | 0    |
| clang-analyzer-core.uninitialized.CapturedBlockVariable      | 0    |
| clang-analyzer-core.uninitialized.UndefReturn                | 0    |
| clang-analyzer-cplusplus.InnerPointer                        | 0    |
| clang-analyzer-cplusplus.Move                                | 0    |
| clang-analyzer-cplusplus.NewDelete                           | 0    |
| clang-analyzer-cplusplus.NewDeleteLeaks                      | 2    |
| clang-analyzer-cplusplus.PlacementNew                        | 0    |
| clang-analyzer-cplusplus.PureVirtualCall                     | 0    |
| clang-analyzer-cplusplus.SelfAssignment                      | 0    |
| clang-analyzer-cplusplus.SmartPtrModeling                    | 0    |
| clang-analyzer-cplusplus.StringChecker                       | 0    |
| clang-analyzer-cplusplus.VirtualCallModeling                 | 0    |
| clang-analyzer-deadcode.DeadStores                           | 1    |
| clang-analyzer-fuchsia.HandleChecker                         | 0    |
| clang-analyzer-nullability.NullPassedToNonnull               | 0    |
| clang-analyzer-nullability.NullReturnedFromNonnull           | 0    |
| clang-analyzer-nullability.NullabilityBase                   | 0    |
| clang-analyzer-nullability.NullableDereferenced              | 0    |
| clang-analyzer-nullability.NullablePassedToNonnull           | 0    |
| clang-analyzer-nullability.NullableReturnedFromNonnull       | 0    |
| clang-analyzer-optin.cplusplus.UninitializedObject           | 0    |
| clang-analyzer-optin.cplusplus.VirtualCall                   | 26   |
| clang-analyzer-optin.mpi.MPI-Checker                         | 0    |
| clang-analyzer-optin.osx.OSObjectCStyleCast                  | 0    |
| clang-analyzer-optin.osx.cocoa.localizability.EmptyLocalizationContextChecker | 0    |
| clang-analyzer-optin.osx.cocoa.localizability.NonLocalizedStringChecker | 0    |
| clang-analyzer-optin.performance.GCDAntipattern              | 0    |
| clang-analyzer-optin.performance.Padding                     | 0    |
| clang-analyzer-optin.portability.UnixAPI                     | 0    |
| clang-analyzer-osx.API                                       | 0    |
| clang-analyzer-osx.MIG                                       | 0    |
| clang-analyzer-osx.NSOrCFErrorDerefChecker                   | 0    |
| clang-analyzer-osx.NumberObjectConversion                    | 0    |
| clang-analyzer-osx.OSObjectRetainCount                       | 0    |
| clang-analyzer-osx.ObjCProperty                              | 0    |
| clang-analyzer-osx.SecKeychainAPI                            | 0    |
| clang-analyzer-osx.cocoa.AtSync                              | 0    |
| clang-analyzer-osx.cocoa.AutoreleaseWrite                    | 0    |
| clang-analyzer-osx.cocoa.ClassRelease                        | 0    |
| clang-analyzer-osx.cocoa.Dealloc                             | 0    |
| clang-analyzer-osx.cocoa.IncompatibleMethodTypes             | 0    |
| clang-analyzer-osx.cocoa.Loops                               | 0    |
| clang-analyzer-osx.cocoa.MissingSuperCall                    | 0    |
| clang-analyzer-osx.cocoa.NSAutoreleasePool                   | 0    |
| clang-analyzer-osx.cocoa.NSError                             | 0    |
| clang-analyzer-osx.cocoa.NilArg                              | 0    |
| clang-analyzer-osx.cocoa.NonNilReturnValue                   | 0    |
| clang-analyzer-osx.cocoa.ObjCGenerics                        | 0    |
| clang-analyzer-osx.cocoa.RetainCount                         | 0    |
| clang-analyzer-osx.cocoa.RetainCountBase                     | 0    |
| clang-analyzer-osx.cocoa.RunLoopAutoreleaseLeak              | 0    |
| clang-analyzer-osx.cocoa.SelfInit                            | 0    |
| clang-analyzer-osx.cocoa.SuperDealloc                        | 0    |
| clang-analyzer-osx.cocoa.UnusedIvars                         | 0    |
| clang-analyzer-osx.cocoa.VariadicMethodTypes                 | 0    |
| clang-analyzer-osx.coreFoundation.CFError                    | 0    |
| clang-analyzer-osx.coreFoundation.CFNumber                   | 0    |
| clang-analyzer-osx.coreFoundation.CFRetainRelease            | 0    |
| clang-analyzer-osx.coreFoundation.containers.OutOfBounds     | 0    |
| clang-analyzer-osx.coreFoundation.containers.PointerSizedValues | 0    |
| clang-analyzer-security.FloatLoopCounter                     | 0    |
| clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling | 0    |
| clang-analyzer-security.insecureAPI.SecuritySyntaxChecker    | 0    |
| clang-analyzer-security.insecureAPI.UncheckedReturn          | 0    |
| clang-analyzer-security.insecureAPI.bcmp                     | 0    |
| clang-analyzer-security.insecureAPI.bcopy                    | 0    |
| clang-analyzer-security.insecureAPI.bzero                    | 0    |
| clang-analyzer-security.insecureAPI.decodeValueOfObjCType    | 0    |
| clang-analyzer-security.insecureAPI.getpw                    | 0    |
| clang-analyzer-security.insecureAPI.gets                     | 0    |
| clang-analyzer-security.insecureAPI.mkstemp                  | 0    |
| clang-analyzer-security.insecureAPI.mktemp                   | 0    |
| clang-analyzer-security.insecureAPI.rand                     | 0    |
| clang-analyzer-security.insecureAPI.strcpy                   | 0    |
| clang-analyzer-security.insecureAPI.vfork                    | 0    |
| clang-analyzer-unix.API                                      | 0    |
| clang-analyzer-unix.DynamicMemoryModeling                    | 0    |
| clang-analyzer-unix.Malloc                                   | 0    |
| clang-analyzer-unix.MallocSizeof                             | 0    |
| clang-analyzer-unix.MismatchedDeallocator                    | 0    |
| clang-analyzer-unix.Vfork                                    | 0    |
| clang-analyzer-unix.cstring.BadSizeArg                       | 0    |
| clang-analyzer-unix.cstring.CStringModeling                  | 0    |
| clang-analyzer-unix.cstring.NullArg                          | 0    |
| clang-analyzer-valist.CopyToSelf                             | 0    |
| clang-analyzer-valist.Uninitialized                          | 0    |
| clang-analyzer-valist.ValistBase                             | 0    |
| clang-analyzer-webkit.NoUncountedMemberChecker               | 0    |
| clang-analyzer-webkit.RefCntblBaseVirtualDtor                | 0    |
| clang-analyzer-webkit.UncountedLambdaCapturesChecker         | 0    |
| cppcoreguidelines-avoid-c-arrays                             | 76   |
| cppcoreguidelines-avoid-goto                                 | 0    |
| cppcoreguidelines-c-copy-assignment-signature                | 0    |
| cppcoreguidelines-explicit-virtual-functions                 | 2    |
| cppcoreguidelines-init-variables                             | 4654 |
| cppcoreguidelines-narrowing-conversions                      | 472  |
| cppcoreguidelines-no-malloc                                  | 20   |
| cppcoreguidelines-prefer-member-initializer                  | 299  |
| cppcoreguidelines-pro-type-const-cast                        | 168  |
| cppcoreguidelines-pro-type-member-init                       | 229  |
| cppcoreguidelines-slicing                                    | 9    |
| cppcoreguidelines-virtual-class-destructor                   | 15   |
| hicpp-avoid-goto                                             | 0    |
| hicpp-exception-baseclass                                    | 0    |
| misc-definitions-in-headers                                  | 0    |
| misc-misleading-bidirectional                                | 0    |
| misc-misleading-identifier                                   | 0    |
| misc-misplaced-const                                         | 0    |
| misc-new-delete-overloads                                    | 0    |
| misc-non-copyable-objects                                    | 0    |
| misc-redundant-expression                                    | 20   |
| misc-static-assert                                           | 0    |
| misc-throw-by-value-catch-by-reference                       | 0    |
| misc-unconventional-assign-operator                          | 0    |
| misc-uniqueptr-reset-release                                 | 0    |
| misc-unused-alias-decls                                      | 3    |
| misc-unused-using-decls                                      | 6    |
| modernize-avoid-bind                                         | 0    |
| modernize-avoid-c-arrays                                     | 76   |
| modernize-concat-nested-namespaces                           | 1314 |
| modernize-deprecated-headers                                 | 3    |
| modernize-deprecated-ios-base-aliases                        | 0    |
| modernize-loop-convert                                       | 32   |
| modernize-make-shared                                        | 0    |
| modernize-make-unique                                        | 0    |
| modernize-pass-by-value                                      | 136  |
| modernize-raw-string-literal                                 | 0    |
| modernize-redundant-void-arg                                 | 13   |
| modernize-replace-auto-ptr                                   | 0    |
| modernize-replace-disallow-copy-and-assign-macro             | 0    |
| modernize-replace-random-shuffle                             | 0    |
| modernize-shrink-to-fit                                      | 0    |
| modernize-unary-static-assert                                | 0    |
| modernize-use-bool-literals                                  | 6    |
| modernize-use-emplace                                        | 1    |
| modernize-use-equals-default                                 | 5    |
| modernize-use-equals-delete                                  | 0    |
| modernize-use-noexcept                                       | 0    |
| modernize-use-nullptr                                        | 41   |
| modernize-use-override                                       | 2    |
| modernize-use-transparent-functors                           | 4    |
| modernize-use-uncaught-exceptions                            | 0    |
| performance-faster-string-find                               | 2    |
| performance-for-range-copy                                   | 5    |
| performance-implicit-conversion-in-loop                      | 0    |
| performance-inefficient-algorithm                            | 0    |
| performance-inefficient-string-concatenation                 | 5    |
| performance-inefficient-vector-operation                     | 10   |
| performance-move-const-arg                                   | 9    |
| performance-move-constructor-init                            | 0    |
| performance-no-automatic-move                                | 0    |
| performance-no-int-to-ptr                                    | 19   |
| performance-noexcept-move-constructor                        | 0    |
| performance-trivially-destructible                           | 1    |
| performance-type-promotion-in-math-fn                        | 0    |
| performance-unnecessary-copy-initialization                  | 255  |
| performance-unnecessary-value-param                          | 353  |
| readability-container-size-empty                             | 26   |
| readability-delete-null-pointer                              | 10   |
| readability-duplicate-include                                | 0    |
| readability-misplaced-array-index                            | 19   |
| readability-redundant-function-ptr-dereference               | 0    |
| readability-redundant-smartptr-get                           | 32   |
| readability-simplify-subscript-expr                          | 0    |
| readability-string-compare                                   | 10   |

下面去除0错误的检查项，也即目前需要存量修复的检查项

共得到：可待优化项**53**项，总错误**9148**项

| 错误类型                                            | 数量 |
| --------------------------------------------------- | ---- |
| bugprone-assignment-in-if-condition                 | 5    |
| bugprone-branch-clone                               | 79   |
| bugprone-copy-constructor-init                      | 2    |
| bugprone-exception-escape                           | 5    |
| bugprone-fold-init-type                             | 1    |
| bugprone-implicit-widening-of-multiplication-result | 149  |
| bugprone-narrowing-conversions                      | 472  |
| bugprone-signed-char-misuse                         | 3    |
| bugprone-use-after-move                             | 1    |
| clang-analyzer-core.CallAndMessage                  | 3    |
| clang-analyzer-core.NullDereference                 | 21   |
| clang-analyzer-cplusplus.NewDeleteLeaks             | 2    |
| clang-analyzer-deadcode.DeadStores                  | 1    |
| clang-analyzer-optin.cplusplus.VirtualCall          | 26   |
| cppcoreguidelines-avoid-c-arrays                    | 76   |
| cppcoreguidelines-explicit-virtual-functions        | 2    |
| cppcoreguidelines-init-variables                    | 4654 |
| cppcoreguidelines-narrowing-conversions             | 472  |
| cppcoreguidelines-no-malloc                         | 20   |
| cppcoreguidelines-prefer-member-initializer         | 299  |
| cppcoreguidelines-pro-type-const-cast               | 168  |
| cppcoreguidelines-pro-type-member-init              | 229  |
| cppcoreguidelines-slicing                           | 9    |
| cppcoreguidelines-virtual-class-destructor          | 15   |
| misc-redundant-expression                           | 20   |
| misc-unused-alias-decls                             | 3    |
| misc-unused-using-decls                             | 6    |
| modernize-avoid-c-arrays                            | 76   |
| modernize-concat-nested-namespaces                  | 1314 |
| modernize-deprecated-headers                        | 3    |
| modernize-loop-convert                              | 32   |
| modernize-pass-by-value                             | 136  |
| modernize-redundant-void-arg                        | 13   |
| modernize-use-bool-literals                         | 6    |
| modernize-use-emplace                               | 17   |
| modernize-use-equals-default                        | 5    |
| modernize-use-nullptr                               | 41   |
| modernize-use-override                              | 2    |
| modernize-use-transparent-functors                  | 4    |
| performance-faster-string-find                      | 2    |
| performance-for-range-copy                          | 5    |
| performance-inefficient-string-concatenation        | 5    |
| performance-inefficient-vector-operation            | 10   |
| performance-move-const-arg                          | 9    |
| performance-no-int-to-ptr                           | 19   |
| performance-trivially-destructible                  | 1    |
| performance-unnecessary-copy-initialization         | 255  |
| performance-unnecessary-value-param                 | 353  |
| readability-container-size-empty                    | 26   |
| readability-delete-null-pointer                     | 10   |
| readability-misplaced-array-index                   | 19   |
| readability-redundant-smartptr-get                  | 32   |
| readability-string-compare                          | 10   |

因此，上述表格所涉及到的错误项便可进行进行存量修复，后续可引入CI实现增量拦截。

## 四、可行性分析与排期计划

根据Paddle目前的状况以及需求，我们认为可以通过最新规则扫描的最新检查项以及最新的错误数量进行分类排查，具体来讲：

- 首先将0错误项和非0错误项分类
- 0错误项可以近期较快地引入到CI规则，从而实现增量拦截
- 非0错误项进行存量修复，存量修复基本完成后紧跟CI引入实现增量拦截

### 4.1 存量修复

下面是引入`pytorch`规则后的`.clang-tidy`文件的修改

```yaml
Checks: '
bugprone-*,
-bugprone-easily-swappable-parameters,
-bugprone-forward-declaration-namespace,
-bugprone-macro-parentheses,
-bugprone-lambda-function-name,
-bugprone-reserved-identifier,
-bugprone-swapped-arguments,
-bugprone-unchecked-optional-access,
clang-diagnostic-missing-prototypes,
cppcoreguidelines-*,
-cppcoreguidelines-avoid-do-while,
-cppcoreguidelines-avoid-magic-numbers,
-cppcoreguidelines-avoid-non-const-global-variables,
-cppcoreguidelines-interfaces-global-init,
-cppcoreguidelines-macro-usage,
-cppcoreguidelines-owning-memory,
-cppcoreguidelines-pro-bounds-array-to-pointer-decay,
-cppcoreguidelines-pro-bounds-constant-array-index,
-cppcoreguidelines-pro-bounds-pointer-arithmetic,
-cppcoreguidelines-pro-type-cstyle-cast,
-cppcoreguidelines-pro-type-reinterpret-cast,
-cppcoreguidelines-pro-type-static-cast-downcast,
-cppcoreguidelines-pro-type-union-access,
-cppcoreguidelines-pro-type-vararg,
-cppcoreguidelines-special-member-functions,
-cppcoreguidelines-non-private-member-variables-in-classes,
-facebook-hte-RelativeInclude,
hicpp-exception-baseclass,
hicpp-avoid-goto,
misc-*,
-misc-const-correctness,
-misc-include-cleaner,
-misc-use-anonymous-namespace,
-misc-unused-parameters,
-misc-no-recursion,
-misc-non-private-member-variables-in-classes,
-misc-confusable-identifiers,
modernize-*,
-modernize-macro-to-enum,
-modernize-return-braced-init-list,
-modernize-use-auto,
-modernize-use-default-member-init,
-modernize-use-using,
-modernize-use-trailing-return-type,
-modernize-use-nodiscard,
performance-*,
readability-container-size-empty,
readability-delete-null-pointer,
readability-duplicate-include
readability-misplaced-array-index,
readability-redundant-function-ptr-dereference,
readability-redundant-smartptr-get,
readability-simplify-subscript-expr,
readability-string-compare,
'
```

简单介绍存量修复的流程：

- 可通过修改`.clang-tidy`文件，使用`“-”`来屏蔽相应的检查项，为了方便，也可以单项检查，只需要在上述内容中的展开项摘取即可。

- 检测出的错误项，可通过log进行手动修复，并提交PR待流水线检测。

- 某些代码错误项也可以根据clang-tidy提供的自动修复工具进行自动修复

  ```
  clang-tidy -p=./build/ --fix-errors -extra-arg=-Wno-unknown-warning-option <FILE_NEED_FIXED>
  ```

- 如果要检测单个文件，可以使用以下命令：

  ```
  clang-tidy -p /build <target_file>   -checks=-*,<test_name>
  ```

- 对于某些情况下，需要跳过 `clang-tidy` 检查的，可以使用 `NOLINT`, `NOLINTNEXTLINE`, `NOLINTBEGIN ... NOLINTEND` 来抑制检查诊断。



### 4.2 引入CI

针对现实状况，可以首先打开目前错误为0的检查项，列出如下：

```txt
bugprone-argument-comment: 0
bugprone-assert-side-effect: 0
bugprone-bad-signal-to-kill-thread: 0
bugprone-bool-pointer-implicit-conversion: 0
bugprone-dangling-handle: 0
bugprone-dynamic-static-initializers: 0
bugprone-forwarding-reference-overload: 0
bugprone-inaccurate-erase: 0
bugprone-incorrect-roundings: 0
bugprone-infinite-loop: 0
bugprone-integer-division: 0
bugprone-macro-repeated-side-effects: 0
bugprone-misplaced-operator-in-strlen-in-alloc: 0
bugprone-misplaced-pointer-arithmetic-in-alloc: 0
bugprone-misplaced-widening-cast: 0
bugprone-move-forwarding-reference: 0
bugprone-multiple-statement-macro: 0
bugprone-no-escape: 0
bugprone-not-null-terminated-result: 0
bugprone-parent-virtual-call: 0
bugprone-posix-return: 0
bugprone-redundant-branch-condition: 0
bugprone-shared-ptr-array-mismatch: 0
bugprone-signal-handler: 0
bugprone-sizeof-container: 0
bugprone-sizeof-expression: 0
bugprone-spuriously-wake-up-functions: 0
bugprone-string-constructor: 0
bugprone-string-integer-assignment: 0
bugprone-string-literal-with-embedded-nul: 0
bugprone-stringview-nullptr: 0
bugprone-suspicious-enum-usage: 0
bugprone-suspicious-include: 0
bugprone-suspicious-memory-comparison: 0
bugprone-suspicious-memset-usage: 0
bugprone-suspicious-missing-comma: 0
bugprone-suspicious-semicolon: 0
bugprone-suspicious-string-compare: 0
bugprone-terminating-continue: 0
bugprone-throw-keyword-missing: 0
bugprone-too-small-loop-variable: 0
bugprone-undefined-memory-manipulation: 0
bugprone-undelegated-constructor: 0
bugprone-unhandled-exception-at-new: 0
bugprone-unhandled-self-assignment: 0
bugprone-unused-raii: 0
bugprone-unused-return-value: 0
bugprone-virtual-near-miss: 0
clang-analyzer-apiModeling.Errno: 0
clang-analyzer-apiModeling.StdCLibraryFunctions: 0
clang-analyzer-apiModeling.TrustNonnull: 0
clang-analyzer-apiModeling.TrustReturnsNonnull: 0
clang-analyzer-apiModeling.google.GTest: 0
clang-analyzer-apiModeling.llvm.CastValue: 0
clang-analyzer-apiModeling.llvm.ReturnValue: 0
clang-analyzer-core.CallAndMessageModeling: 0
clang-analyzer-core.DivideZero: 0
clang-analyzer-core.DynamicTypePropagation: 0
clang-analyzer-core.NonNullParamChecker: 0
clang-analyzer-core.NonnilStringConstants: 0
clang-analyzer-core.StackAddrEscapeBase: 0
clang-analyzer-core.StackAddressEscape: 0
clang-analyzer-core.UndefinedBinaryOperatorResult: 0
clang-analyzer-core.VLASize: 0
clang-analyzer-core.builtin.BuiltinFunctions: 0
clang-analyzer-core.builtin.NoReturnFunctions: 0
clang-analyzer-core.uninitialized.ArraySubscript: 0
clang-analyzer-core.uninitialized.Assign: 0
clang-analyzer-core.uninitialized.Branch: 0
clang-analyzer-core.uninitialized.CapturedBlockVariable: 0
clang-analyzer-core.uninitialized.UndefReturn: 0
clang-analyzer-cplusplus.InnerPointer: 0
clang-analyzer-cplusplus.Move: 0
clang-analyzer-cplusplus.NewDelete: 0
clang-analyzer-cplusplus.PlacementNew: 0
clang-analyzer-cplusplus.PureVirtualCall: 0
clang-analyzer-cplusplus.SelfAssignment: 0
clang-analyzer-cplusplus.SmartPtrModeling: 0
clang-analyzer-cplusplus.StringChecker: 0
clang-analyzer-cplusplus.VirtualCallModeling: 0
clang-analyzer-fuchsia.HandleChecker: 0
clang-analyzer-nullability.NullPassedToNonnull: 0
clang-analyzer-nullability.NullReturnedFromNonnull: 0
clang-analyzer-nullability.NullabilityBase: 0
clang-analyzer-nullability.NullableDereferenced: 0
clang-analyzer-nullability.NullablePassedToNonnull: 0
clang-analyzer-nullability.NullableReturnedFromNonnull: 0
clang-analyzer-optin.cplusplus.UninitializedObject: 0
clang-analyzer-optin.mpi.MPI-Checker: 0
clang-analyzer-optin.osx.OSObjectCStyleCast: 0
clang-analyzer-optin.osx.cocoa.localizability.EmptyLocalizationContextChecker: 0
clang-analyzer-optin.osx.cocoa.localizability.NonLocalizedStringChecker: 0
clang-analyzer-optin.performance.GCDAntipattern: 0
clang-analyzer-optin.performance.Padding: 0
clang-analyzer-optin.portability.UnixAPI: 0
clang-analyzer-osx.API: 0
clang-analyzer-osx.MIG: 0
clang-analyzer-osx.NSOrCFErrorDerefChecker: 0
clang-analyzer-osx.NumberObjectConversion: 0
clang-analyzer-osx.OSObjectRetainCount: 0
clang-analyzer-osx.ObjCProperty: 0
clang-analyzer-osx.SecKeychainAPI: 0
clang-analyzer-osx.cocoa.AtSync: 0
clang-analyzer-osx.cocoa.AutoreleaseWrite: 0
clang-analyzer-osx.cocoa.ClassRelease: 0
clang-analyzer-osx.cocoa.Dealloc: 0
clang-analyzer-osx.cocoa.IncompatibleMethodTypes: 0
clang-analyzer-osx.cocoa.Loops: 0
clang-analyzer-osx.cocoa.MissingSuperCall: 0
clang-analyzer-osx.cocoa.NSAutoreleasePool: 0
clang-analyzer-osx.cocoa.NSError: 0
clang-analyzer-osx.cocoa.NilArg: 0
clang-analyzer-osx.cocoa.NonNilReturnValue: 0
clang-analyzer-osx.cocoa.ObjCGenerics: 0
clang-analyzer-osx.cocoa.RetainCount: 0
clang-analyzer-osx.cocoa.RetainCountBase: 0
clang-analyzer-osx.cocoa.RunLoopAutoreleaseLeak: 0
clang-analyzer-osx.cocoa.SelfInit: 0
clang-analyzer-osx.cocoa.SuperDealloc: 0
clang-analyzer-osx.cocoa.UnusedIvars: 0
clang-analyzer-osx.cocoa.VariadicMethodTypes: 0
clang-analyzer-osx.coreFoundation.CFError: 0
clang-analyzer-osx.coreFoundation.CFNumber: 0
clang-analyzer-osx.coreFoundation.CFRetainRelease: 0
clang-analyzer-osx.coreFoundation.containers.OutOfBounds: 0
clang-analyzer-osx.coreFoundation.containers.PointerSizedValues: 0
clang-analyzer-security.FloatLoopCounter: 0
clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling: 0
clang-analyzer-security.insecureAPI.SecuritySyntaxChecker: 0
clang-analyzer-security.insecureAPI.UncheckedReturn: 0
clang-analyzer-security.insecureAPI.bcmp: 0
clang-analyzer-security.insecureAPI.bcopy: 0
clang-analyzer-security.insecureAPI.bzero: 0
clang-analyzer-security.insecureAPI.decodeValueOfObjCType: 0
clang-analyzer-security.insecureAPI.getpw: 0
clang-analyzer-security.insecureAPI.gets: 0
clang-analyzer-security.insecureAPI.mkstemp: 0
clang-analyzer-security.insecureAPI.mktemp: 0
clang-analyzer-security.insecureAPI.rand: 0
clang-analyzer-security.insecureAPI.strcpy: 0
clang-analyzer-security.insecureAPI.vfork: 0
clang-analyzer-unix.API: 0
clang-analyzer-unix.DynamicMemoryModeling: 0
clang-analyzer-unix.Malloc: 0
clang-analyzer-unix.MallocSizeof: 0
clang-analyzer-unix.MismatchedDeallocator: 0
clang-analyzer-unix.Vfork: 0
clang-analyzer-unix.cstring.BadSizeArg: 0
clang-analyzer-unix.cstring.CStringModeling: 0
clang-analyzer-unix.cstring.NullArg: 0
clang-analyzer-valist.CopyToSelf: 0
clang-analyzer-valist.Uninitialized: 0
clang-analyzer-valist.Unterminated: 0
clang-analyzer-valist.ValistBase: 0
clang-analyzer-webkit.NoUncountedMemberChecker: 0
clang-analyzer-webkit.RefCntblBaseVirtualDtor: 0
clang-analyzer-webkit.UncountedLambdaCapturesChecker: 0
cppcoreguidelines-avoid-goto: 0
cppcoreguidelines-c-copy-assignment-signature: 0
hicpp-avoid-goto: 0
hicpp-exception-baseclass: 0
misc-definitions-in-headers: 0
misc-misleading-bidirectional: 0
misc-misleading-identifier: 0
misc-misplaced-const: 0
misc-new-delete-overloads: 0
misc-non-copyable-objects: 0
misc-static-assert: 0
misc-throw-by-value-catch-by-reference: 0
misc-unconventional-assign-operator: 0
misc-uniqueptr-reset-release: 0
modernize-avoid-bind: 0
modernize-deprecated-ios-base-aliases: 0
modernize-make-shared: 0
modernize-make-unique: 0
modernize-raw-string-literal: 0
modernize-replace-auto-ptr: 0
modernize-replace-disallow-copy-and-assign-macro: 0
modernize-replace-random-shuffle: 0
modernize-shrink-to-fit: 0
modernize-unary-static-assert: 0
modernize-use-equals-delete: 0
modernize-use-noexcept: 0
modernize-use-uncaught-exceptions: 0
performance-implicit-conversion-in-loop: 0
performance-inefficient-algorithm: 0
performance-move-constructor-init: 0
performance-no-automatic-move: 0
performance-noexcept-move-constructor: 0
performance-type-promotion-in-math-fn: 0
readability-duplicate-include: 0
readability-redundant-function-ptr-dereference: 0
readability-simplify-subscript-expr: 0
```

针对后续非0错误项的存量修复解决之后，可以进行CI的引入

### 4.3 后续改进的可行方案

上述修改主要是针对`Pytorch`的规则，而当`pytorch`规则在`PaddlePaddle`上完善之后，我们希望可以增加更对的规则，来使得`PaddlePaddle`具有更好的`code style`

为了使得上述问题可行并且有质量保证，我调研了`tensorflow`的检查项，可以对额外检查项进行引进，`tensorflow`检查项如下：

```yaml
	bugprone-argument-comment
    bugprone-assert-side-effect
    bugprone-branch-clone
    bugprone-copy-constructor-init
    bugprone-dangling-handle
    bugprone-dynamic-static-initializers
    bugprone-macro-parentheses
    bugprone-macro-repeated-side-effects
    bugprone-misplaced-widening-cast
    bugprone-move-forwarding-reference
    bugprone-multiple-statement-macro
    bugprone-suspicious-semicolon
    bugprone-swapped-arguments
    bugprone-terminating-continue
    bugprone-unused-raii
    bugprone-unused-return-value
    llvm-else-after-return
    llvm-header-guard
    llvm-include-order
    llvm-namespace-comment
    llvm-prefer-isa-or-dyn-cast-in-conditionals
    llvm-prefer-register-over-unsigned
    llvm-qualified-auto
    llvm-twine-local
    misc-confusable-identifiers
    misc-const-correctness
    misc-definitions-in-headers
    misc-misleading-bidirectional
    misc-misleading-identifier
    misc-misplaced-const
    misc-new-delete-overloads
    misc-non-copyable-objects
    misc-redundant-expression
    misc-static-assert
    misc-throw-by-value-catch-by-reference
    misc-unconventional-assign-operator
    misc-uniqueptr-reset-release
    misc-unused-alias-decls
    misc-unused-parameters
    misc-unused-using-decls
    modernize-loop-convert
    modernize-make-unique
    modernize-raw-string-literal
    modernize-use-bool-literals
    modernize-use-default-member-init
    modernize-use-emplace
    modernize-use-equals-default
    modernize-use-nullptr
    modernize-use-override
    modernize-use-using
    performance-for-range-copy
    performance-implicit-conversion-in-loop
    performance-inefficient-algorithm
    performance-inefficient-vector-operation
    performance-move-const-arg
    performance-no-automatic-move
    performance-trivially-destructible
    performance-unnecessary-copy-initialization
    performance-unnecessary-value-param
    readability-avoid-const-params-in-decls
    readability-const-return-type
    readability-container-size-empty
    readability-identifier-naming
    readability-inconsistent-declaration-parameter-name
    readability-misleading-indentation
    readability-redundant-control-flow
    readability-simplify-boolean-expr
    readability-simplify-subscript-expr
    readability-use-anyofallof
```

找到比pytorch多出的检查规则：

```
	misc-confusable-identifiers
    readability-avoid-const-params-in-decls
    readability-simplify-boolean-expr
    llvm-header-guard
    llvm-qualified-auto
    bugprone-swapped-arguments
    llvm-include-order
    llvm-prefer-register-over-unsigned
    readability-const-return-type
    llvm-namespace-comment
    readability-inconsistent-declaration-parameter-name
    misc-const-correctness
    readability-misleading-indentation
    readability-use-anyofallof
    readability-redundant-control-flow
    llvm-twine-local
    llvm-else-after-return
    readability-identifier-naming
    misc-unused-parameters
    modernize-use-default-member-init
    llvm-prefer-isa-or-dyn-cast-in-conditionals
    bugprone-macro-parentheses
    modernize-use-using
```

这样就可以继续完善PaddlePaddle的检查项和CI拦截，当然，这里只是拿tensorflow举个例子，还可以寻找一些其他的知名项目进行类似的操作。

## 五、测试和验收的考量

- 确保不会引起性能倒退
- 确保不会引起代码风格倒退
- 通过 CI 各条流水线

## 六、影响面

- 对用户的影响

  用户对于框架内部代码风格的变动不会有任何感知，不会有任何影响。

- 对 Paddle 框架开发者的影响

  代码风格更加统一，代码更加稳健，项目可以自动化进行增量拦截

## 参考资料

1. [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)
2. [clang-tidy代码风格检查工具的引入](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/code_style/code_style_clang_tidy.md)
3. [20230501_introducing_clang_tidy.md](https://github.com/PaddlePaddle/community/blob/master/rfcs/CodeStyle/20230501_introducing_clang_tidy.md)
4. [赛题四：在飞桨框架中引入 clang-tidy Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/54073)
