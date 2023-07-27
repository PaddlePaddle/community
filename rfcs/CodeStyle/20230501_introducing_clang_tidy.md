# clang-tidy代码风格检查工具的引入

|任务名称|clang-tidy代码风格检查工具的引入|
|------|------|
|提交作者|@GreatV (汪昕)|
|提交时间|2023-05-01|
|版本号|v0.1|
|依赖飞桨版本|develop|
|文件名| 20230501_introducing_clang_tidy.md|

## 一、概述

### 1、相关背景

当前 `Paddle` 已引入 `precommit`、`pylint`、`remove-ctrlf`、`cpplint`、`clang-format`、`yapf`、`cmakelint`、`cmake-format` 和 `flake8` 等多种代码风格检查工具。为了进一步增强 `C++` 代码的稳健性，还可引入 `clang-tidy` 静态分析工具。

### 2、`clang-tidy` 简介

[clang-tidy](https://clang.llvm.org/extra/clang-tidy/) 是一个基于 `clang` 的 `C` linter 工具。它的目的是提供一个可扩展的框架，用于诊断和修复典型的编程错误，如风格违规、接口误用，或通过静态分析推断出的错误。`clang-tidy` 是模块化的，为编写新的检查提供了一个方便的接口。

### 3、意义

- 进一步规范 C++ 代码风格；
- 进一步提升 C++ 代码稳健性，提升工程质量和 C++ 代码的可维护性，减少潜在的 bug；
- 自动修复部分典型的编程错误，避免手动修复，降低开发者解决 Linter Error 的成本。

## 二、飞桨现状

Paddle 目前引入了 `clang-format`，`cpplint` 等工具用于 C++ 端的代码风格监控。

其中，`clang-format` 用于格式化代码，它可以根据用户提供的配置文件（`.clang-format` 或 `_clang-format`）将代码自动格式化成指定的风格，如 [LLVM](https://llvm.org/docs/CodingStandards.html) 风格、[Google](https://google.github.io/styleguide/cppguide.html) 风格等。

`cpplint` 是由 Google 开发和维护的一个代码风格检查工具，用于检查 C/C++ 文件的风格问题，遵循 [Google 的 C++ 风格指南](http://google.github.io/styleguide/cppguide.html)。`cpplint` 可以检查代码中的格式错误、命名规范、注释、头文件和其他编程约定等问题。

`clang-tidy` 是一个静态代码分析工具，可用于检查代码中的潜在问题，提升代码稳健性。`clang-tidy` 可以检查代码中的内存管理错误、类型不匹配错误、未定义行为等问题。`clang-tidy` 可以使用不同的检查器配置，每个配置可以启用或禁用一组相关的检查器。

总的来说，`clang-format` 与 `cpplint` 用于提高代码可读性和风格的工具，而 `clang-tidy` 则是用于发现和修复潜在问题和错误的工具。在飞桨中引入 `clang-tidy`，并与 `clang-format` 和 `cpplint` 结合使用，可进一步提高代码质量与可维护性。

## 三、`clang-tidy` 相关调研

### 部分知名项目使用`clang-tidy`版本调研

- pytorch 使用 15.0.6
- tensorflow 使用 6.0 (ubuntu 16.04)
- pybind11 使用 15

### 存量代码`clang-tidy`扫描结果调研

首先使用 `clang-tidy` (LLVM version 10.0.0) 对代码仓库 (commit 0c2ab714befeaac35a3df5d92cfe7cb1631ce716) 进行扫描，其中，编译选项为 `cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=ON -DWITH_TENSORRT=ON -DWITH_DISTRIBUTE=ON -DWITH_MKL=ON -DWITH_AVX=ON -DCUDA_ARCH_NAME=Volta -DWITH_PYTHON=ON  -DWITH_TESTING=ON -DWITH_COVERAGE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DWITH_CONTRIB=ON -DWITH_INFERENCE_API_TEST=ON -DPY_VERSION=3.8 -DWITH_PSCORE=ON -DWITH_GLOO=ON -DWITH_LITE=ON -DWITH_STRIP=ON -DWITH_UNITY_BUILD=ON`。扫描用的脚本如下所示：

```shell
for file in $(find paddle/ -name *.cc); do
    clang-tidy -p=./build/ -extra-arg=-Wno-unknown-warning-option $file >> clang-tidy.log 2>&1
done

for file in $(find paddle/ -name *.cu); do
    clang-tidy -p=./build/ -extra-arg=-Wno-unknown-warning-option $file >> clang-tidy.log 2>&1
done

for file in $(find paddle/ -name *.h); do
    clang-tidy -p=./build/ -extra-arg=-Wno-unknown-warning-option $file >> clang-tidy.log 2>&1
done

for file in $(find paddle/ -name *.cpp); do
    clang-tidy -p=./build/ -extra-arg=-Wno-unknown-warning-option $file >> clang-tidy.log 2>&1
done
```

使用的配置文件该文件来自[pytorch](https://github.com/pytorch/pytorch/blob/main/.clang-tidy)，进行了少量修改）如下所示：

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
misc-unused-alias-decls,
misc-unused-using-decls,
modernize-*,
-modernize-concat-nested-namespaces,
-modernize-macro-to-enum,
-modernize-return-braced-init-list,
-modernize-use-auto,
-modernize-use-default-member-init,
-modernize-use-using,
-modernize-use-trailing-return-type,
-modernize-use-nodiscard,
performance-*,
-performance-noexcept-move-constructor,
-performance-unnecessary-value-param,
readability-container-size-empty,
'
# display errors from all non-system headers
# HeaderFilterRegex: '.*'
AnalyzeTemporaryDtors: false
...

```

得到总 `warning` 数为 **18822**，`error` 数为 **4134**。

```shell
grep warning: clang-tidy.log  | grep "\[.*\]" | wc -l
# 18822
grep error: clang-tidy.log  | grep "\[.*\]" | wc -l
# 4134
```

需要修改的文件数为 **1434**。

```shell
grep warning: clang-tidy.log | grep "\[.*\]" | grep -o "/paddle/.*\.cc"| awk '{print $1}'|sort|uniq -c|sort -nr | wc -l
# 1391
grep warning: clang-tidy.log | grep "\[.*\]" | grep -o "/paddle/.*\.cu"| awk '{print $1}'|sort|uniq -c|sort -nr | wc -l
# 27
grep warning: clang-tidy.log | grep "\[.*\]" | grep -o "/paddle/.*\.h"| awk '{print $1}'|sort|uniq -c|sort -nr | wc -l
# 6
grep warning: clang-tidy.log | grep "\[.*\]" | grep -o "/paddle/.*\.cpp"| awk '{print $1}'|sort|uniq -c|sort -nr | wc -l
# 1
grep error: clang-tidy.log | grep "\[.*\]" | grep -o "/paddle/.*\.cc"| awk '{print $1}'|sort|uniq -c|sort -nr | wc -l
# 4
grep error: clang-tidy.log | grep "\[.*\]" | grep -o "/paddle/.*\.cu"| awk '{print $1}'|sort|uniq -c|sort -nr | wc -l
# 1
grep error: clang-tidy.log | grep "\[.*\]" | grep -o "/paddle/.*\.h"| awk '{print $1}'|sort|uniq -c|sort -nr | wc -l
# 4
grep error: clang-tidy.log | grep "\[.*\]" | grep -o "/paddle/.*\.cpp"| awk '{print $1}'|sort|uniq -c|sort -nr | wc -l
# 0
```

去重后需要修改的检查项，如下所示：

错误类型|错误数量
:------:|:------
[clang-diagnostic-error]|4134
[cppcoreguidelines-pro-bounds-pointer-arithmetic]|4415
[cppcoreguidelines-pro-type-cstyle-cast]|4337
[cppcoreguidelines-pro-type-reinterpret-cast]|2279
[cppcoreguidelines-pro-type-vararg]|1882
[modernize-redundant-void-arg]|1280
[cppcoreguidelines-pro-bounds-array-to-pointer-decay]|1054
[readability-container-size-empty]|766
[modernize-loop-convert]|543
[modernize-use-nullptr]|436
[cppcoreguidelines-pro-bounds-constant-array-index]|363
[cppcoreguidelines-pro-type-static-cast-downcast]|315
[cppcoreguidelines-pro-type-const-cast]|249
[modernize-use-auto]|171
[modernize-pass-by-value]|132
[modernize-use-override]|125
[cppcoreguidelines-pro-type-union-access]|120
[clang-diagnostic-pessimizing-move]|70
[misc-unused-alias-decls]|52
[clang-analyzer-core.StackAddressEscape]|35
[modernize-use-default]|33
[clang-analyzer-core.CallAndMessage]|32
[clang-diagnostic-missing-braces]|22
[clang-analyzer-core.NullDereference]|19
[performance-unnecessary-copy-initialization]|16
[modernize-make-unique]|15
[clang-diagnostic-unused-value]|9
[clang-diagnostic-sign-compare]|8
[clang-analyzer-core.UndefinedBinaryOperatorResult]|7
[clang-analyzer-deadcode.DeadStores]|6
[cppcoreguidelines-c-copy-assignment-signature]|4
[clang-analyzer-unix.Malloc]|3
[clang-diagnostic-overloaded-virtual]|2
[clang-diagnostic-mismatched-tags]|2
[clang-diagnostic-inconsistent-missing-override]|2
[clang-diagnostic-compare-distinct-pointer-types]|2
[clang-diagnostic-braced-scalar-init]|2
[clang-analyzer-cplusplus.NewDelete]|2
[clang-analyzer-core.DivideZero]|2
[clang-diagnostic-unused-variable]|1
[clang-diagnostic-unused-private-field]|1
[clang-diagnostic-unused-const-variable]|1
[clang-diagnostic-self-assign]|1
[clang-diagnostic-return-type-c-linkage]|1
[clang-analyzer-unix.Vfork]|1
[clang-analyzer-security.insecureAPI.vfork]|1
[clang-analyzer-cplusplus.NewDeleteLeaks]|1
[clang-analyzer-core.uninitialized.Assign]|1
[clang-analyzer-core.NonNullParamChecker]|1


## 四、可行性分析与排期计划

为 Paddle 引入 `clang-tidy` 可分为如下几步，分别进行：

### 1.1 在 `.pre-commit-config.yaml` 在添加 `clang-tidy` 检查项

```yaml
repos:
    - repo: https://github.com/pocc/pre-commit-hooks
      rev: v1.3.5
      hooks:
          - id: clang-tidy
            args: [-p=build, -extra-arg=-Wno-unknown-warning-option]
```

并添加 `.clang-tidy` 配置文件，内容如本文档第三节配置文件所示。并关闭所有检查项。

### 1.2 使用单独的脚本运行 `clang-tidy`，开发者手动执行

考虑到引入 `clang-tidy` 到 `pre-commit`，可能会导致 `pre-commit` 运行缓慢，因此，也可提供单独的脚本运行 `clang-tidy`，开发者在修改文件后，手动执行。如：

```shell
#!/bin/bash

# for ubuntu
# check clang-tidy exists
if ! command -v clang-tidy &> /dev/null
then
    echo "clang-tidy could not be found"
    sudo apt update
    sudo apt install clang-tidy
    exit
fi

clang-tidy -p=./build/ -extra-arg=-Wno-unknown-warning-option $1
```

### 2. 存量修复

- 根据 `clang-tidy` 检查项创建 `tracking issue`，邀请小伙伴一起欢乐开源；
- 将 `clang-tidy` 检查项逐一打开，并逐步修复；
- 部分代码可借助 `clang-tidy` 的自动修复功能修复，如 `clang-tidy -p=./build/ --fix-errors -extra-arg=-Wno-unknown-warning-option <FILE_NEED_FIXED>`；
- 对于某些情况下，需要跳过 `clang-tidy` 检查的，可以使用 `NOLINT`, `NOLINTNEXTLINE`, `NOLINTBEGIN ... NOLINTEND` 来抑制检查诊断。

### 3. 将 `clang-tidy` 同步到release分支

此处可参考 [cherry pick code format check upgrade to release/2.3](https://github.com/PaddlePaddle/Paddle/pull/43732)。

### 4. 将 `clang-tidy` 同步到CI镜像中，避免重复安装下载

此出可参考 [upgrade pre-commit tools in docker](https://github.com/PaddlePaddle/Paddle/pull/43534)。

## 五、测试和验收的考量

确保不会引起性能倒退，确保不会引起代码风格倒退，通过 CI 各条流水线。

## 六、影响面

- 对用户的影响

  用户对于框架内部代码风格的变动不会有任何感知，不会有任何影响。

- 对 Paddle 框架开发者的影响

  代码风格更加统一，代码更加稳健，副作用是可能造成 `pre-commit` 运行缓慢。

## 参考资料

1. [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)
2. [clang-tidy代码风格检查工具的引入](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/code_style/code_style_clang_tidy.md)
