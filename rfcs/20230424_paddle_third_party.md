# 对常用库编译进行优化

|任务名称 | 对常用库编译进行优化 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | gouzil | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-04-24 | 
|版本号 | v1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20230424_paddle_third_party.md<br> | 

# 一、概述
## 1、相关背景
有多个编译目录时，重新克隆第三方库，导致编译时间过长。
## 2、功能目标
- 把通用的基础库移动到`third_party`目录下，避免重复克隆。

## 3、意义
- 减少编译时因频繁克隆第三方库而导致的时间消耗和编译失败问题。

# 二、飞桨现状
目前如果有多个编译目录，当第一次执行`make -j$(nproc)`, 都会重新克隆第三方库.


# 三、业内方案调研

## pytorch: 

* 方案1: 在`third_party`目录下使用`Bazel`和子模块的形式管理第三方库. 
* 方案2: 在`cmake/Modules`目录下配置`FindXXX`文件, 使用预编译文件库.

## tensorflow:

直接将第三方库放在`third_party`下,使用`Bazel`进行构建. (这里的第三方库不是全量的,只是`Bazel`的配置文件)

# 四、对比分析

| 框架 | 构建方式 |
| ---- | ---- |
| paddlepaddle | cmake + git clone(http download) |
| pytorch | cmake + 子模块 + bazel + FindXXX |
| tensorflow | bazel |

# 五、设计思路与实现方案

## 1、主体设计思路与折衷
### 整体全貌

为了避免发布 releases 版本时过大, 所以仅添加部分通用的第三方库, 其他的第三方库仍然使用`git clone`和`http download`的方式进行下载.

### 主体设计具体描述

* 将`ExternalProject_Add`的搜索方式从`git clone`改为索引到`third_party`目录下的子模块.

### 主体设计选型考量

* 引入`Vpkg`(目前不是主流, 暂时不考虑)
* 引入`Bazel`(目前支持不是很好, 没有支持的ide, 暂时不考虑)
* 引入`FindXXX`通过查找或指定预编译 lib 来跳过编译一些第三方库编译. 示例:[Findpybind11.cmake](https://github.com/pytorch/pytorch/blob/main/cmake/Modules/Findpybind11.cmake)

如果引入了`FindXXX`需要在[贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)或[安装指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/index_cn.html)中专门说明如何配置, 不建议在附录中追加.

## 2、关键技术点/子模块设计与实现方案

### 需要迁移的第三方库

| 第三方库 | 说明 |
| ---- | ---- |
| glog | [branch](https://github.com/gouzil/Paddle/tree/add_third_party)
| cryptopp |
| dlpack |
| eigen3 |
| gflags |
| lapack |
| pocketfft |
| protobuf |
| pybind |
| threadpool |
| utf8proc |
| warpctc |
| warprnnt |
| xxhash |
| zlib |



### 迁移前工作:

* 在`setup.py`添检查子模块是否加载.[在这次提交](https://github.com/gouzil/Paddle/commit/e34eb0584e48bbc5492e7a4eb48547e61131350a)
```bash
def check_submodules():
    for sub_dir in os.listdir(TOP_DIR + "/third_party"):
        if not os.path.exists(TOP_DIR + "/third_party/" + sub_dir +
                              "/.git"):
            raise RuntimeError(
                "Please run 'git submodule update --init --recursive' to make sure you have all the submodules installed."
            )
```

* 在文档中补充说明

### 正式迁移

以`glog`为例, 参考[pr](https://github.com/gouzil/Paddle/tree/add_third_party)这个分支的最近几次提交

在`third_party`目录下使用终端执行:
```bash
git submodule add -b v4.0.0 https://github.com/google/glog.git
```

```cmake
# https://github.com/google/glog/tree/v0.4.0 # 添加
set(GLOG_REPOSITORY ${GIT_URL}/google/glog.git) # 删除
set(GLOG_TAG v0.4.0) # 删除
```
```cmake
ExternalProject_Add(
    extern_glog
    ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
    GIT_REPOSITORY ${GLOG_REPOSITORY} # 删除
    GIT_TAG ${GLOG_TAG} # 删除
    URL ${PROJECT_SOURCE_DIR}/third_party/glog #添加
    DEPENDS gflags
    PREFIX ${GLOG_PREFIX_DIR}
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_FLAGS=${GLOG_CMAKE_CXX_FLAGS}
               -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
               -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
               -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
               -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR}
               -DCMAKE_INSTALL_LIBDIR=${GLOG_INSTALL_DIR}/lib
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
               -DWITH_GFLAGS=OFF
               -DBUILD_TESTING=OFF
               -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
               ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${GLOG_INSTALL_DIR}
      -DCMAKE_INSTALL_LIBDIR:PATH=${GLOG_INSTALL_DIR}/lib
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    BUILD_BYPRODUCTS ${GLOG_LIBRARIES})
```

# 六、测试和验收的考量

ci通过即可

# 七、影响面

* 在发布 releases 版本时, 不会将子模块打包进去, 会导致编译失败. 需要单独添加一个带子模块的压缩包.
* 推荐使用一个pr合入, 已克隆的开发者需要执行`git submodule update --init --recursive`加载子模块.参考[releases](https://github.com/gouzil/Paddle/releases/tag/add_third_party_test)
* 此次引入主要是为了解决开发者在新创建编译时, 需要下载大量的第三方库, 从而导致编译时间过长的问题. 


## 对用户的影响

无

## 对二次开发用户的影响

向下兼容问题: 无

影响面: 如果此时用户再创建编译目录, 需使用`git submodule update --init --recursive`初始化子模块, 对原有编译目录无影响.

对下载 releases 中的 Source code 编译用户有影响. (在发布 releases 版本时并不会打包子模块, 需手动添加一个压缩包版本)

## 其他风险

需要添加一个带子模块的压缩包, 用于发布 releases 版本.

# 八、排期规划

1-2周即可

# 附件及参考资料
 * [git submodule](https://git-scm.com/book/zh/v2/Git-%E5%B7%A5%E5%85%B7-%E5%AD%90%E6%A8%A1%E5%9D%97)
 * [Vpkg](https://vcpkg.io/en/)
 * [Bazel](https://bazel.build/)