# 编译warning的消除

> This project will be mentored by [@zhiqiu](https://github.com/zhiqiu) and [@luotao1](https://github.com/luotao1)
> 
> Tracking issue: [PaddlePaddle/Paddle#47143](https://github.com/PaddlePaddle/Paddle/issues/47143)
## 目的

作为一个大型的基础设施类开源项目，减少飞桨框架在编译时的warning，可以提升工程质量和整个代码仓库的可维护性，以及减少潜在的bug。

## 现状
### linux 明面warning
以8.16日某成功PR的流水线 [PR-CI-Build](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/builds/21068?module=github%2FPaddlePaddle%2FPaddle&pipeline=PR-CI-Build&branch=branches) 为例，
编译选项：`cmake .. -DPY_VERSION=3.8 -DWITH_DISTRIBUTE=ON -DWITH_GPU=ON -DWITH_LITE=ON`。
可下载 [日志](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/6367838/job/17573866)，
统计编译warning如下（linux明面上的warning相对比较少，因为修过很多轮了）：
```shell
grep warning: a.txt |grep "\[\-W" | wc -l 
# 305
grep warning: a.txt |grep "\[\-W" | grep party |wc -l
# 265，这里很多是第三方库lite repo引入的，可以先不做处理
grep warning: a.txt |grep "\[\-W" |grep -v party | awk '{print $NF}'|sort|uniq -c|sort -nr
# 
  25 [-Wsign-compare]
   6 [-Wterminate]
   4 [-Wunused-variable]
   2 [-Wunknown-pragmas]
   2 [-Wmaybe-uninitialized]
   1 [-Wunused-local-typedefs]
```
### mac 明面warning
以8.16日某成功PR的流水线 [PR-CI-Mac-Python3](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/builds/10496?module=github%2FPaddlePaddle%2FPaddle&pipeline=PR-CI-Mac-Python3&branch=branches) 为例，
编译选项：`cmake .. -DPY_VERSION=3.7 -DWITH_DISTRIBUTE=ON -DWITH_TESTING=ON`
可下载[日志](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/6373552/job/17594587)，
统计编译warning如下（mac的warning相对就多多了）
```shell
grep warning: b.txt |grep "\[\-W" | wc -l 
# 7909
grep warning: b.txt |grep "\[\-W" | grep party |wc -l
# 7166，很多是eigen引入的，看如何屏蔽eigen带来的warning
grep warning: b.txt |grep "\[\-W" |grep -v party | awk '{print $NF}'|sort|uniq -c|sort -nr
#
 685 [-Winconsistent-missing-override]
  31 [-Wformat]
  13 [-Wbraced-scalar-init]
   4 [-Wliteral-conversion]
   4 [-Wc++17-extensions]
   2 [-Wexceptions]
   1 [-Wuninitialized]
   1 [-Wtautological-constant-out-of-range-compare]
   1 [-Wreturn-type-c-linkage]
   1 [-Wpragma-pack]
```
### Windows 明面warning（可忽略）
以8.16日某成功PR的效率云流水线 [PR-CI-Windows](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/builds/10090?module=github%2FPaddlePaddle%2FPaddle&pipeline=PR-CI-Windows&branch=branches) 为例，
可下载[日志](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/6368225/job/17575121)，
统计编译warning如下（windows有专项，所以剩余warning非常少，可以忽略）
```shell
grep warning c.txt |wc -l 
# 8
```
### 第三方库引入的warning情况
历史上为了兼容第三方库的编译，在cmake中加了一些warning选项。当第三方库发生变化（移除或更新）时，这些warning选项也需要同步更新。

#### 由Boost库引入的warning
Boost库作为C++标准库的“预备军”，一直以来在飞桨框架开发中充当C++扩展库的角色，便于开发者使用一些C++标准库未实现或只在较高版本C++中有实现的功能。Boost库虽使用方便，但其中存在大量宏展开以及声明和实现未做分离的数据类型，整个库体积庞大，在飞桨代码中导入Boost相关的文件对源码编译速度和build目录体积都会产生较大的负面影响。经过近期针对Boost库使用的集中整治，飞桨代码中已有的所有Boost相关代码均已被替换和移除，其中一些C++ 14标准已支持的功能替换成了标准库中的对应实现，另一些标准库中未支持的功能在paddle/utils目录下实现了功能相同的轻量化版本。

目前飞桨代码已不依赖Boost库，因此`flags.cmake`中为了兼容boost库引入的warning选项都可以去掉，可避免出现 [PR#47254](https://github.com/PaddlePaddle/Paddle/pull/47254) 中的问题。如全局[flags.cmake](https://github.com/PaddlePaddle/Paddle/blob/2d0bb2c3961c7bb06746051732b460829e2450dd/cmake/flags.cmake#L170)中就有针对boost库的warning选项：
```CMake
if(NOT APPLE)
    if((${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 8.0) OR (WITH_ROCM))
      set(COMMON_FLAGS
          ${COMMON_FLAGS}
          -Wno-format-truncation # Warning in boost gcc 8.2
          -Wno-error=parentheses # Warning in boost gcc 8.2
          -Wno-error=catch-value # Warning in boost gcc 8.2
          -Wno-error=nonnull-compare # Warning in boost gcc 8.2
          -Wno-error=address # Warning in boost gcc 8.2
          -Wno-ignored-qualifiers # Warning in boost gcc 8.2
          -Wno-ignored-attributes # Warning in Eigen gcc 8.3
          -Wno-parentheses # Warning in Eigen gcc 8.3
      )
    endif()
  endif()
```

### 未打开的warning情况（非常多，几十万+）
在cmake中还有些手动修改了warning选项的，因此不会统计出来。目前GCC选项：指定了一些-Wno-error，可以在修复一类后，把这些都去掉即可。
```
CXX_FLAGS =  -Wno-error=deprecated-declarations -Wno-deprecated-declarations -std=c++14 -m64 -fPIC 
-fno-omit-frame-pointer -Werror -Wall -Wextra -Wno-non-virtual-dtor -Wdelete-non-virtual-dtor 
-Wno-unused-parameter -Wno-unused-function -Wno-error=literal-suffix -Wno-error=unused-local-typedefs 
-Wno-error=ignored-attributes -Wno-error=terminate -Wno-error=int-in-bool-context -Wimplicit-fallthrough=0 
-Wno-error=maybe-uninitialized -Wno-format-truncation -Wno-error=cast-function-type -Wno-error=parentheses 
-Wno-error=catch-value -Wno-error=nonnull-compare -Wno-error=address -Wno-ignored-qualifiers -Wno-ignored-attributes 
-Wno-parentheses -fopenmp -mavx -O3 -DNDEBUG
```
手动修改了warning选项：
- 全局[flags.cmake](https://github.com/PaddlePaddle/Paddle/blob/2d0bb2c3961c7bb06746051732b460829e2450dd/cmake/flags.cmake#L140)中：
```CMake
if(NOT WIN32)
  set(COMMON_FLAGS
      -fPIC
      -fno-omit-frame-pointer
      -Werror
      -Wall
      -Wextra
      -Wnon-virtual-dtor
      -Wdelete-non-virtual-dtor
      -Wno-unused-parameter
      -Wno-unused-function
      -Wno-error=literal-suffix
      -Wno-error=unused-local-typedefs
      -Wno-error=ignored-attributes # Warnings in Eigen, gcc 6.3
      -Wno-error=terminate # Warning in PADDLE_ENFORCE
      -Wno-error=int-in-bool-context # Warning in Eigen gcc 7.2
      -Wimplicit-fallthrough=0 # Warning in tinyformat.h
      -Wno-error=maybe-uninitialized # Warning in boost gcc 7.2
      ${fsanitize})
```
- 某些编译选项手动加入，可搜索关键字[-Wno-error](https://github.com/PaddlePaddle/Paddle/search?l=CMake&q=-Wno-error)， 
  如[分布式方向](https://github.com/PaddlePaddle/Paddle/blob/2d0bb2c3961c7bb06746051732b460829e2450dd/paddle/fluid/framework/CMakeLists.txt#L742)
```CMake
set(DISTRIBUTE_COMPILE_FLAGS
        "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=parentheses"
    )
```
- 打开 `unused-parameter`、`unused-function`和`implicit-fallthrough`的结果（历史数据可做参考）
```shell
grep "unused-parameter" build.txt|wc -l
#141684
grep "unused-function" build.txt|wc -l  
#106
grep "implicit-fallthrough" build.txt|wc -l
#6894，大部分是tinyformat.h（可忽略）
```
- 参考：[gcc 和warning相关的编译选项](https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html)
## 解决原则和方法
### 原则
* 存量：大部分编译warning与代码逻辑依赖较少，可以分类解决。可参考 
[remove Wno-error=parentheses-equality](https://github.com/PaddlePaddle/Paddle/pull/42993), 
[fix gcc warning of [-Wint-in-bool-context]](https://github.com/PaddlePaddle/Paddle/pull/42268), 
[fix gcc warning of cast-function-type](https://github.com/PaddlePaddle/Paddle/pull/42235), 
[fix unused-variable warning](https://github.com/PaddlePaddle/Paddle/pull/43791), 
[fix sign-compare warning](https://github.com/PaddlePaddle/Paddle/pull/43625), 
[fix Wtype-limits](https://github.com/PaddlePaddle/Paddle/pull/42676), 
[remove maybe-uninitialized](https://github.com/PaddlePaddle/Paddle/pull/45204) 等
* 增量：
  * 解决一类问题后，将这一类 warning 改为error。例如，`-Werror=sign-compare`
  * 针对某些CMakeLists.txt手动加入-Wno-error的情况，用CI关键字进行监控【已完成 [PR42875](https://github.com/PaddlePaddle/Paddle/pull/42875)】
* 最终态：
  * 相对实际：消除大部分明面和隐藏的warning，提升代码质量和开发者体验
  * 理想态：至少开启 `-Wall -Wextra -Werror`，即开启所有的warning，并将所有的warning视为error

### 推进方法
1. **修复明面warning**：按目前的cmake flags设置，在linux GPU和Mac编译下，把现有的warning修复。
2. **修复隐藏warning**：针对代码中设置的cmake flag，分析判断合理性，选择将部分或全部设置 -Wno- 删除，并修复代码中的warning。
    - 分布式（`WITH_DISTRIBUTE=ON`）手动改了编译选项`DISTRIBUTE_COMPILE_FLAGS`，优先级可以往后放
3. **在Linux GPU/Mac编译选项下**，重复上述1、2（暂不考虑其他设备，比如Windows/XPU/NPU/IPU/MLU等）

### 一些问题
Wno-maybe-uninitialized的已知问题
* 检查存在false negative。具体来说，将 Wno-maybe-uninitialized删除后，理论上来说，因为有-Wall -Wextra的存在，
  代码中的这类warning应该都报出error编译失败。[PR#42902](https://github.com/PaddlePaddle/Paddle/pull/42902) 确实修改了一部分这类warning以通过CI，但是代码中还存在相似写法却不会报error的。
* 在其他编译环境中爆出问题 [Issues#43156](https://github.com/PaddlePaddle/Paddle/issues/43156)，最后revert了。
  见[stackoverflow上的解释](https://stackoverflow.com/questions/14132898/gcc-wuninitialized-wmaybe-uninitialized-issues/14132910#14132910) 
  和 [gcc bug list中的相关问题](https://gcc.gnu.org/bugzilla/buglist.cgi?quicksearch=may%20be%20uninitialized)
