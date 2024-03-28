# clang-tidy代码风格检查工具的引入

> This project will be mentored by [@Galaxy1458](http://github.com/Galaxy1458) and [@zhangbo9674](http://github.com/zhangbo9674)

## 背景
Paddle目前使用的代码风格检查工具及hooks版本较低，导致开发者在提交代码时会碰到一些问题，加强代码规范检查，有利于提高Paddle代码质量，增强代码的可读性。

内部已经完成了precommit, pylint, remove-ctrlf,cpplint,cmake-format,yapf,cmakelint,cmake-format 8大检查工具的升级，还剩下两大检查工具clang-tidy和flake8还未引入，期待社区开发者主导完成。

## clang-tidy调研情况
【新增】clang-tidy：13.0.0 （总错误数：40288，需修改文件数：3055，见[详细列表](https://shimo.im/sheets/RKAWM7b2BwsQGJq8/vLv2M)）
* 原因：和clang-format保持一致。
* 收益：使得C++代码更加规范、更加现代化
* 风险：
  * 根据历史全量检测统计结果 clang-tidy表，基本按照pytorch的检查规则。需要修改的文件和错误数量很多，且涉及到很多C++语法知识，需要深入理解代码含义才能修复。
  * 若需要加上该检查项，则本地和static-check流水线在cmake时需要加上选项`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`。
    用和static-check流水线相同的cmake命令执行一次，并把所需的第三方库编译好。（这些操作可以集成在本地hook脚本中）
  * 检查时间较长：全量扫描耗时2天。取数量前三的错误列举如下表：

错误类型  | 数量  | 含义
 ----          | ----- | ------  
cppcoreguidelines-init-variables  | 12471 | 变量未初始化
 bugprone-narrowing-conversions   | 6569 | 收缩转换  
[readability/mixedcase]           | 5320 | 可转换类型的参数直接相互跟随，导致调用时即使参数顺序错了也不会报错

* 检查步骤：
   * 下载clang+llvm预编译包（13.0.0版本）：https://releases.llvm.org/ 。
      clang-tidy位于 clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang-tidy 。
   * 在build目录下执行cmake命令并加上`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`选项，如
     `cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=ON -DWITH_PYTHON=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` ，
      将在build目录下生成compile_commands.json文件，包含每个文件的编译命令。
     （注意：clang-tidy的检查需要对文件进行真正的编译，所以cmake命令中的编译选项，会影响文件的检查结果，
      如某个文件cpu和gpu走不同分支，当WITH_GPU=OFF时，就检查不了GPU分支的代码）。
   * 使用clang-tidy检查abs_op.cc文件：`/clang-tidy-path/clang-tidy -p=build -extra-arg=-Wno-unknown-warning-option /workspace/Paddle/paddle/fluid/operators/abs_op.cc`。
     下图的检查结果显示：需要在虚函数后面加上 override 关键字。检查结果取决于配置文件中的设置。
![image](https://user-images.githubusercontent.com/6836917/185282945-35cc927b-33c6-418f-8b04-5e65a1125b48.png)

* clang-tidy 对于部分检查到的错误，提供了自动的修复功能，可以尝试使用。（但是，因为这种修复只在其对应的编译选项下进行，可能会导致在其他的编译选项下会出错，还需要认真检查其修复后的代码并做适当调整）


## 可行性分析和规划排期
1. 因为存量较大，需要进行存量修复的可行性分析（本任务最有难度的地方），并制定如何修复的步骤
   * 将检查工具的检测项逐一打开，并逐步修复对应的存量问题。
   * 部分存量较大的问题，需要有自动化工具进行修复。
2. 格式检查工具升级和存量修复，可以参考[code format check upgrade](https://github.com/PaddlePaddle/Paddle/search?q=code%20format%20check%20upgrade&type=commits) 已有工作
3. 仅格式检查工具同步到release分支，参考 [PR43732](http://agroup.baidu.com/paddle-ci/md/article/2https://github.com/PaddlePaddle/Paddle/pull/43732)
4. 将检查工具升级同步到CI镜像中，避免重复安装下载，参考 [PR43534](https://github.com/PaddlePaddle/Paddle/pull/43534)
5. 代码风格检查指南文档，[docs#4933](https://github.com/PaddlePaddle/docs/pull/4933)
