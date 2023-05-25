# paddle代码自动生成机制讲解

一、代码自动生成体系
==========

1.1 CodeGen的概念
-------------

### 1.1.1 基本概念

代码自动生成（Code generation）是指根据指定规范或高级描述，自动生成源代码或其他软件部件的过程。代码自动生成的目标是提高生产力，减少错误，提高软件的可维护性。   代码自动生成有不同的技术，包括基于模板的代码生成、基于模型的代码生成等。代码自动生成在现代软件开发中被广泛应用，例如在神经网络、编译器、数据库等领域中都有应用。它可以大大提高软件开发的效率，同时减少人工出错的可能性。

### 1.1.2 引入CodeGen的意义

Paddle作为国内第一的深度学习框架，功能已经十分丰富，代码量十分庞大。为了实现更多的硬件接入Paddle体系，也开放给第三方的硬件平台修改Paddle主框架的权限，与此同时也增加了框架的维护成本。鉴于此，引入CodeGen这套设计模式来解决这些问题。CodeGen能够给paddle带来以下收益：

1.  减少框架代码量：相关代码通过简单的yaml文件和Python脚本在编译阶段(或构建阶段)自动生成，大大减少了Paddle的可见代码量；
    
2.  降低开发成本：如果要进一步往Paddle中增加新的算子，只需要配置yaml文件、添加对应的Kernel文件和Python端接口代码，即可完成添加；
    
3.  实现Op定义统一，降低维护成本：通过固定的机制来生成相关代码，规范了代码添加规则，降低框架维护升级成本。
    

厂内的CINN中也大量使用CodeGen的技术生成目标代码。

1.2 Paddle内的CodeGen体系
--------------------

 Paddle核心训练框架中，目前主要存在的是三套CodeGen设计体系，即动态图、静态图、旧动态图，但旧动态图在完全退场后，旧动态图的CodeGen体系也预计会被完全清除。

![代码自动生成总体框架](./images/auto_gen_ops/%E4%BB%A3%E7%A0%81%E8%87%AA%E5%8A%A8%E7%94%9F%E6%88%90%E6%80%BB%E4%BD%93%E6%A1%86%E6%9E%B6.png)

<center>图1-1 代码自动生成总体框</center>

目前动态图和静态图均是依据"**yaml配置文件+Python脚本**"的设计自动生成相关代码，区别点在于静态图同时使用了[Jinja](https://jinja.palletsprojects.com/en/3.1.x/)作为生成模板。旧动态图API依据静态图注册的OpInfoMap，使用C++程序进行生成，所以旧动态图的生成的代码和静态图OP耦合度极高，这也是因为旧动态图的调用机制复用了静态图Op。这些模块内部还隐含了**组合算子CodeGen**、**运算符重载CodeGen**等模块。   首先介绍CodeGen相关的yaml配置文件，最主要的yaml文件位于paddle/phi/api/yaml/，(tensor_operators.yaml是运算符重载yaml文件)，此外还有组合算子的一个单独yaml文件位于paddle/fluid/prim/api/api.yaml，如图1-2所示。


![](./images/auto_gen_ops/%E4%BB%A3%E7%A0%81%E8%87%AA%E5%8A%A8%E7%94%9F%E6%88%90%E7%9B%B8%E5%85%B3yaml%E6%96%87%E4%BB%B61.png)
![](./images/auto_gen_ops/%E4%BB%A3%E7%A0%81%E8%87%AA%E5%8A%A8%E7%94%9F%E6%88%90%E7%9B%B8%E5%85%B3yaml%E6%96%87%E4%BB%B62.png)

<center>图1-2 代码自动生成相关yaml文件</center>

相关文件yaml文件的功能如下所示：

![](./images/auto_gen_ops/%E4%BB%A3%E7%A0%81%E8%87%AA%E5%8A%A8%E7%94%9F%E6%88%90%E7%9B%B8%E5%85%B3yaml%E6%96%87%E4%BB%B6%E5%8A%9F%E8%83%BD%E8%AF%B4%E6%98%8E.png)

<center>图1-3 代码自动生成相关yaml文件功能说明</center>

二、动态图生成体系
=========

2.1 新动态图生成体系
------------

### 2.1.1 整体设计

新动态图的codegen体系分为三层：**c++api层**、**dygraph_function网络构建层**、**python-c映射层**。C++API层是纯粹的调用kernel进行计算，dygraph_function层会根据需求建立起backward网络，将forward和backward联系起来，Python-C映射层则是方便Python端可以直接调用动态图算子接口，三者的详细介绍可以参见["动态图调用过程详解"](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/Dygraph)。

![](./images/auto_gen_ops/%E6%96%B0%E5%8A%A8%E6%80%81%E5%9B%BE%E4%BB%A3%E7%A0%81%E8%87%AA%E5%8A%A8%E7%94%9F%E6%88%90%E6%A1%86%E6%9E%B6.png)

<center>图2-1 新动态图代码自动生成框架</center>

从图2-1可以看出，动态图三个层次的代码生成均是依据原始的yaml配置文件，只有组合算子网络构建层的API是依据parse_op.py处理后的yaml文件。

### 2.1.2 API生成设计

API的生成脚本文件(paddle/phi/api/yaml/generator/)如图2-2所示：
![](./images/auto_gen_ops/%E5%8A%A8%E6%80%81%E5%9B%BEAPI%E7%94%9F%E6%88%90%E8%84%9A%E6%9C%AC.png)

<center>图2-2 动态图API生成脚本 </center>

每个文件的功能如下：

![](./images/auto_gen_ops/API%E8%84%9A%E6%9C%AC%E5%8A%9F%E8%83%BD.png)

<center>图2-3 API脚本功能</center>

以api_gen.py为例，生成代码的流程如下所示，相应的CMakeLists文件为`paddle/phi/api/lib/CMakeLists.txt`：

![](./images/auto_gen_ops/c%2B%2BAPI%E7%94%9F%E6%88%90%E6%B5%81%E7%A8%8B.png)

<center>图2-4 c++API生成流程</center>  

简单的生成流程如图2-4，输入yaml文件并解析后，构造相应的生成类，调用相应的接口即可生成相应的文本，最后将文本写入文件即可。细节可以阅读相关源码。图2-4仅仅列出了普通ops也fused_ops的生成过程，其他的sparse_ops,infermediate_ops,string_ops的调用流程类似。注意Python脚本直接生成的是xxx.cc.tmp文件，最后需要通过add_custom_command添加编译target(output)进行拷贝生成最终的xxx.cc文件。

### 2.1.3 网络构建层生成设计

动态图网络构建层分为两个部分，一部分是基础网络构建层，一部分是组合算子网络构建层。

#### 2.1.3.1 普通算子网络构建层

普通算子网络构建层生成脚本文件(paddle/fluid/eager/auto_code_generator/generator/)如下：
![](./images/auto_gen_ops/%E5%8A%A8%E6%80%81%E5%9B%BE%E7%BD%91%E7%BB%9C%E6%9E%84%E5%BB%BA%E5%B1%82%E5%92%8CPython-C%E4%BA%A4%E4%BA%92%E5%B1%82%E7%94%9F%E6%88%90%E8%84%9A%E6%9C%AC.png)

<center>图2-5 动态图网络构建层和Python-C交互层生成脚本</center> 

由于网络构建层和Python-C交互层的脚本较为简单，二者放置在同一路径下。

![](./images/auto_gen_ops/%E7%BD%91%E7%BB%9C%E6%9E%84%E5%BB%BA%E5%B1%82%E5%92%8Cpython-C%E4%BA%A4%E4%BA%92%E5%B1%82%E8%84%9A%E6%9C%AC%E5%8A%9F%E8%83%BD.png)

<center>图2-6 网络构建层和python-C交互层脚本功能</center>

执行流程如下，相关CMakeLIsts文件(paddle/fluid/eager/auto_code_generator/generator/CMakeLists.txt)：

![](./images/auto_gen_ops/%E5%8A%A8%E6%80%81%E5%9B%BE%E7%BD%91%E7%BB%9C%E6%9E%84%E5%BB%BA%E5%B1%82%E7%94%9F%E6%88%90%E6%B5%81%E7%A8%8B.png)

<center>图2-7 动态图网络构建层生成流程  </center>

这里生成的文件位于paddle/fluid/eager/api/generated/eager_generated/路径下。  注意Python脚本直接生成的是xxx.cc.tmp文件，最后需要通过add_custom_command添加编译target(output)进行拷贝生成最终的xxx.cc文件。

#### 2.1.3.2 组合算子网络构建层

组合算子网络构建层生成脚本文件(paddle/fluid/prim/api/auto_code_generated/)如下：
![](./images/auto_gen_ops/%E5%8A%A8%E9%9D%99%E6%80%81%E5%9B%BE%E7%BB%84%E5%90%88%E7%AE%97%E5%AD%90%E7%BD%91%E7%BB%9C%E6%9E%84%E5%BB%BA%E5%B1%82.png)

<center>图2-8 动静态图组合算子网络构建层</center>

组合算子网络构建层分为动态图和静态图，这里静态图代码功能类似于Python端调用append_op这个函数。每个文件的功能如下：

![](./images/auto_gen_ops/%E7%BB%84%E5%90%88%E7%AE%97%E5%AD%90%E7%BD%91%E7%BB%9C%E6%9E%84%E5%BB%BA%E5%B1%82%E8%84%9A%E6%9C%AC.png)

<center>图2-9 组合算子网络构建层脚本</center>

调用流程可以参考paddle/fluid/prim/api/auto_code_generated/CMakeLists.txt。注意Python脚本直接生成的是xxx.cc.tmp文件，最后需要通过execute_process添加构建(cmake)期脚本命令进行拷贝生成最终的xxx.cc文件。

### 2.1.4 Python-C交互层生成设计

Python-C交互层相关文件和功能参见图2-5和图2-6中的python_c_gen.py文件.从CMakeLists文件paddle/fluid/eager/auto_code_generator/generator/CMakeLists.txt中看出，通过add_custom_target添加target调用python_c_gen.py脚本直接生成最终文件。

```
add_custom_target(  eager_python_c_codegen
  COMMAND    "${PYTHON_EXECUTABLE}"    "${PADDLE_SOURCE_DIR}/paddle/fluid/eager/auto_code_generator/generator/python_c_gen.py"    "--api_yaml_path=${api_yaml_path},${fwd_api_yaml_path}"    "--output_path=${tmp_python_c_output_path}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_python_c_output_path}          ${python_c_output_path}  VERBATIM)
```

 这里的target--eager_python_c_codegen会被paddle/fluid/pybind/CMakeLists.txt下的add_custom_command依赖，保证add_custom_target会被执行：

```
    if(NOT ((NOT WITH_PYTHON) AND ON_INFER))      add_custom_command(        OUTPUT ${eager_impl_file}
        COMMAND          ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:."          "${CMAKE_CURRENT_BINARY_DIR}/eager_legacy_op_function_generator"          "${tmp_eager_impl_file}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_eager_impl_file}                ${eager_impl_file}
        COMMENT "copy_if_different ${tmp_eager_impl_file} to ${eager_impl_file}"        DEPENDS ${EAGER_OP_IMPL_DEPS}        VERBATIM)    endif()
```

2.2 旧动态图生成体系
------------

### 2.2.1 整体设计

由于旧动态图复用了静态图的Op(Trace模式)，所以旧动态图的代码生成的只有两部分组成，也就是网络构建层和Python-C交互层。

![](./images/auto_gen_ops/%E6%97%A7%E5%8A%A8%E6%80%81%E5%9B%BE%E4%BB%A3%E7%A0%81%E8%87%AA%E5%8A%A8%E7%94%9F%E6%88%90%E6%A1%86%E6%9E%B6.png)

<center>图2-10 旧动态图代码自动生成框架</center>

### 2.2.2 网络构建层生成设计

旧动态图网络构建层使用的C++程序，之所以使用C++程序是方便从OpInfoMap::Instance()中获取Op的信息。文件生成器构成如下：
![](./images/auto_gen_ops/%E6%97%A7%E5%8A%A8%E6%80%81%E5%9B%BE%E7%AE%97%E5%AD%90%E7%BD%91%E7%BB%9C%E6%9E%84%E5%BB%BA%E5%B1%82%E7%94%9F%E6%88%90%E5%99%A8%E6%96%87%E4%BB%B6.png)

<center>图2-11 旧动态图算子网络构建层生成器文件</center>

相关文件功能如下：

![](./images/auto_gen_ops/%E6%97%A7%E5%8A%A8%E6%80%81%E5%9B%BE%E7%BD%91%E7%BB%9C%E6%9E%84%E5%BB%BA%E5%B1%82%E7%94%9F%E6%88%90%E5%99%A8.png)

<center>图2-12 旧动态图网络构建层生成器</center>

旧动态图的相关代码生成过程如下：

![](./images/auto_gen_ops/%E6%97%A7%E5%8A%A8%E6%80%81%E5%9B%BE%E7%BD%91%E7%BB%9C%E6%9E%84%E5%BB%BA%E5%B1%82%E7%94%9F%E6%88%90%E6%B5%81%E7%A8%8B.png)

<center>图2-13 旧动态图网络构建层生成流程</center>

通过观察生成的CMakLists.txt文件，如paddle/fluid/eager/api/generated/fluid_generated/forwards/CMakeLists.txt：

![](./images/auto_gen_ops/%E7%9B%B8%E5%85%B3CMakeLists%E6%96%87%E4%BB%B6.png)

<center>图2-14 相关CMakeLists.txt文件</center>  

生成lib库时只用到了带有数字后缀的相关文件，dygraph_forward_functions.cc并没有用到。这里分割文件的意义是防止单个文件过大，导致编译时间急剧增加。

### 2.2.3 Python-C交互层生成设计

旧动态图的Python-C交互层生成器组成较为简单仅有一个文件paddle/fluid/pybind/eager_legacy_op_function_generator.cc，执行流程如下：

![](./images/auto_gen_ops/%E6%97%A7%E5%8A%A8%E6%80%81%E5%9B%BEPython-C%E7%94%9F%E6%88%90%E6%B5%81%E7%A8%8B.png)

<center>图2-15 旧动态图Python-C生成流程</center>

三、静态图op生成体系
===========

在Python层不会通过函数形式直接调用静态图op，并且静态图组网是通过Program、Block等模块实现的，因此静态图只需要自动生成Op相关的代码。

![](./images/auto_gen_ops/%E9%9D%99%E6%80%81%E5%9B%BEOp%E4%BB%A3%E7%A0%81%E8%87%AA%E5%8A%A8%E7%94%9F%E6%88%90%E6%A1%86%E6%9E%B6.png)

<center>图3-1静态图Op代码自动生成框架</center>

如图3-1所示，yaml文件经过parse_op.py解析后输出对应的parsed文件，静态图生成的Python脚本借助Jinja模板生成对应的op文件和sig文件。相关文件如下：

![](./images/auto_gen_ops/%E9%9D%99%E6%80%81%E5%9B%BEOp%E4%BB%A3%E7%A0%81%E7%94%9F%E6%88%90%E5%99%A8%E7%9B%B8%E5%85%B3%E6%96%87%E4%BB%B6.png)

<center>图3-2 静态图Op代码生成器相关文件</center>

相关的文件功能如下：

![](./images/auto_gen_ops/%E9%9D%99%E6%80%81%E5%9B%BEOp%E4%BB%A3%E7%A0%81%E7%94%9F%E6%88%90%E5%99%A8%E7%9B%B8%E5%85%B3%E6%96%87%E4%BB%B6%E5%8A%9F%E8%83%BD.png)

<center>图3-3 静态图Op代码生成器相关文件功能</center>

从图中可以看出，静态图Op的生成器主要包括三部分：jinja模板，Python脚本和C++文件。静态图通过这三者之间的配合生成相关的代码文件，执行流程如下：

![](./images/auto_gen_ops/%E9%9D%99%E6%80%81%E5%9B%BEOp%E4%BB%A3%E7%A0%81%E7%94%9F%E6%88%90%E6%B5%81%E7%A8%8B.png)

<center>图3-4 静态图Op代码生成流程 </center> 

从图3-4可以看到，yaml文件经过parse_op.py解析后生成静态图规范格式的yaml文件，这一步骤的主要原始是原始的yaml配置会缺省很多配置，格式不够规整，不利于后续的jinja模板读取。为保证yaml配置的正确性，这里通过cross_validate.py进行交叉检验，主要是通过同一个op的forward和backward的配置进行检验。经过检验正确的yaml文件会输入到generate_op.py中，并结合op_compat.yaml、op_version.yaml中的信息，利用Jinja模板生成兼容手写Op文件的代码。这里的op_compat.yaml的主要功能是参数名字映射和增加一些原始ops.yaml中没有的信息，确保生成的Op和原始手写的文件一致，新增的算子一般不需要在op_compat.yaml中增加配置。   由于sparse相关的算子是新增到Phi下的，没有历史兼容性问题，所以不需要输入op_compat.yaml、op_version.yaml。ops的extra attrs的信息全部配置在op_compat.yaml中，可以直接生成。

附录
==

1.  "快乐开源"--[框架静态图算子自动生成一期](https://github.com/PaddlePaddle/Paddle/issues/51842)
    
2.  "快乐开源"--[框架静态图算子自动生成二期](https://github.com/PaddlePaddle/Paddle/issues/53267)
    
3.  CMake关键命令([官方文档](https://cmake.org/cmake/help/latest/index.html))：
    
    a.  execute_process：直接算构建(cmake)期间执行脚本
    
    b.  add_custom_command：在编译(make)期间执行脚本，相当于构建一个target。通常有两个中用法(OUTPUT和绑定TARGET)
    
    c.  add_custom_target：在编译(make)期间执行脚本，相当于构建一个target，但是默认是不会构建该target的，当显示指明该target(cmake .. --build --target my_target)或这有其他target依赖该target时才会生成。
    
    d.  copy_if_different:如果是多个文件，确保文件路径已经存在。