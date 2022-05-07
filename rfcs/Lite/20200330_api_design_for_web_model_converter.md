# Web 版 Paddle Lite 模型转换工具设计文档

| API名称                                                      | -                                                  |
| ------------------------------------------------------------ | -------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | 中国功夫窝们喜欢                                   |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-30                                         |
| 版本号                                                       | V1.0                                               |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                            |
| 文件名                                                       | 20200330_api_design_for_web_model_converter.md<br> |


# 一、概述

## 1、相关背景

模型转换是模型部署的必备一环，传统方法需要下载安装特定的模型转换工具，很可能遇到缺少依赖库、版本不匹配等问题。Web 模型转换不需要用户下载安装模型转换工具，在浏览器里就可以一键得到转换后的模型，大大增加了用户的便利性。

## 2、功能目标

1. 实现 Paddle 模型到 Paddle-Lite 模型的 Web 模型转换
2. 将 opt 工具的 WebAssembly 编译加入到 Paddle Lite 的 CI 里

## 3、意义

Paddle Lite 用户可非常方便的完成 Paddle 到 Paddle Lite 的模型转换，同时还可以提供对应版本的 Paddle Lite 库下载的服务。

# 二、飞桨现状

Paddle Lite 目前不支持 Web 模型转换。

# 三、业内方案调研

https://convertmodel.com/ 支持了训练框架到 NCNN、MNN 和 Tengine 模型的 Web 转换，以及 ONNX Optimizer 等模型优化器。

# 四、对比分析

无

# 五、设计思路与实现方案

使用 emsdk 将 opt 工具编译为 WebAssembly，集成进 https://convertmodel.com 已有的前端页面里，或挂载在 Paddle 提供的专用域名下（如有）。

# 六、测试和验收的考量

使用常见的 Paddle 模型来测试，产生的 Paddle Lite 模型的 md5 和使用 native 版 opt 工具得到的 Paddle Lite 模型的 md5 值相同。

注意：Paddle 模型的体积不能太大（比如 300MB 以上），否则可能触发某些浏览器本身的问题。

# 七、可行性分析和排期规划

前两周：实现相关功能，上线到 https://www.convertmodel.com/ 以及 Paddle Lite 官方模型转换网站（如有）；

第三周：测试和迭代

# 八、影响面

除了对 opt 工具本身可能需要做源码修改之外，Paddle Lite 的 CI 中也需要加入将 opt 工具编译到 WebAssembly 的测试。

# 名词解释

# 附件及参考资料
