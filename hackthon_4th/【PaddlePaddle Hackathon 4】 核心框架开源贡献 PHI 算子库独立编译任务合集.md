# 【PaddlePaddle Hackathon 4】核心框架开源贡献 PHI 算子库独立编译任务合集

（此 ISSUE 为 PaddlePaddle Hackathon 第四期活动的任务 ISSUE，更多详见 [【PaddlePaddle Hackathon 第四期】任务总览](https://github.com/PaddlePaddle/Paddle/issues/51281)）

注：为飞桨框架 PHI 算子库解耦头文件依赖，**3月6日前完成 PR 提交，3月13日前完成 PR 合入**，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下，其他说明事项在任务列表后：

### No.67：解耦 PHI 算子库对 operator.h 头文件的依赖 <a name='task67'></a>

- 任务难度：基础
- 详细描述：
  - 现状：目前 PHI 算子库依赖头文件 "paddle/fluid/framework/operator.h"，为了实现 PHI 算子库独立编译的目标，需要清理该头文件
  - 目标：移除掉 PHI 下所有文件对 "paddle/fluid/framework/operator.h" 头文件的使用，并且对依赖该头文件的数据结构替换成 phi 命名空间的数据结构；如果当前 phi 命名空间下没有该数据结构，需要将该头文件中的组件迁移到 PHI 算子库下，并删除迁移前原目录下的代码。

### No.68：解耦 PHI 算子库对 utils.h 头文件的依赖 <a name='task68'></a>

- 任务难度：基础
- 详细描述：
  - 现状：目前 PHI 算子库依赖头文件 "paddle/fluid/operators/utils.h"，为了实现 PHI 算子库独立编译的目标，需要清理该头文件
  - 目标：移除掉 PHI 下所有文件对 "paddle/fluid/operators/utils.h" 头文件的使用，并且对依赖该头文件的数据结构替换成 phi 命名空间的数据结构；如果当前 phi 命名空间下没有该数据结构，需要将该头文件中的组件迁移到 PHI 算子库下，并删除迁移前原目录下的代码。

### No.69：解耦 PHI 算子库对 device_wrapper.h 头文件的依赖 <a name='task69'></a>

- 任务难度：基础
- 详细描述：
  - 现状：目前 PHI 算子库依赖头文件 "paddle/fluid/platform/device/device_wrapper.h"，为了实现 PHI 算子库独立编译的目标，需要清理该头文件
  - 目标：移除掉 PHI 下所有文件对 "paddle/fluid/platform/device/device_wrapper.h" 头文件的使用，并且对依赖该头文件的数据结构替换成 phi 命名空间的数据结构；如果当前 phi 命名空间下没有该数据结构，需要将该头文件中的组件迁移到 PHI 算子库下，并删除迁移前原目录下的代码。

### No.70：解耦 PHI 算子库对 kernels.h 头文件的依赖 <a name='task70'></a>

- 任务难度：进阶
- 详细描述：
  - 现状：目前 PHI 算子库依赖头文件 "paddle/fluid/operators/jit/kernels.h"，为了实现 PHI 算子库独立编译的目标，需要清理该头文件
  - 目标：移除掉 PHI 下所有文件对 "paddle/fluid/operators/jit/kernels.h" 头文件的使用，并且对依赖该头文件的数据结构替换成 phi 命名空间的数据结构；如果当前 phi 命名空间下没有该数据结构，需要将该头文件中的组件迁移到 PHI 算子库下，并删除迁移前原目录下的代码。



～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

### 合入标准

- 解耦头文件代码 PR，提交目录至：[paddle/phi](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/phi) 和 [paddle/fluid](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid)
- 可以引入 PHI 下已经存在的 Fluid 目录头文件，但不能引入新的 Fluid 目录头文件
- 通过所有 CI，代码符合 PHI 算子库下命名/设计规范
- 可参考该 [Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/47615) ,这个里边提供了 PHI 算子库独立编译相关文档及头文件解耦依赖 PR 示例。

### 技术要求
- 了解 Paddle PHI 算子库
- 熟悉使用 C++ 开发及 cmake 编译

### 答疑交流

- 如果在开发中对于上述任务有任何问题，欢迎在本 ISSUE 下留言交流。
- 对于开发中的共性问题，在活动过程中，会定期组织答疑，请关注官网&微信群的通知，及时参与。
