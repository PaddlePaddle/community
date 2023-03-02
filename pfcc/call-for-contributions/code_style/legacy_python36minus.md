# python 3.5/3.6 相关代码退场
> This project will be mentored by [@luotao1](http://github.com/luotao1)

# 一、概要
## 1、相关背景

- Python 3.5 在2020年9月30日终止支持，Python 3.6 在2021年12月23日终止支持，https://devguide.python.org/versions/。
- Paddle 从 2.1 版本（2021年4月）开始不再发 Python 3.5 的包，计划从2.5版本（2023年）开始不再维护Python 3.6。
- Develop分支已于2022年9月27日限制只能编译Python 3.7及以上的包，见 [PR#46477](https://github.com/PaddlePaddle/Paddle/pull/46477)。
```
if sys.version_info < (3,7):
    raise RuntimeError("Paddle only support Python version>=3.7 now")
```
对 Paddle 中 Python 3.5/3.6 相关代码进行退场，可以提高源码整洁性，提升开发者阅读的便利性。

## 2、功能目标
对 Paddle 中 Python 3.5/3.6 相关代码进行退场，提升开发者阅读和开发源码的便利性。

## 3、方案要点

Python 3.5/3.6 相关代码的退场比 [Python 2.7 退场](http://agroup.baidu.com/paddlepaddle/md/article/5041740) 更为简单，
主要涉及到针对 module（模块） 与 requirement（运行环境依赖）等几个方面的处理：

* 删除非必要的环境依赖
* 清理 Python 3.x 相关逻辑分支
* 清理文档中涉及到 Python 3.x 的内容

# 二、意义
Paddle 从 2.1 版本（2021年4月）开始不再维护 Python 3.5，从2.5版本（2023年）开始不再维护Python 3.6，develop分支从2022年9月已无法编译 Python 3.6的包，
因此，同步对 Python 3.5/3.6 相关代码进行退场，提升开发者阅读和开发源码的便利性。

举例：

1. Tensor类型注释，对于提升编程体验至关重要，但此功能需要python 3.7+ 版本支持。
见 [Type Hint for Tensor 联合开发项目](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/type_hint.md) 
和 [issue#45979](https://github.com/PaddlePaddle/Paddle/issues/45979)。
![](https://user-images.githubusercontent.com/10242208/183284234-a608cf6f-16a0-4e5b-bc76-4e88aa988630.gif)
2. 可以直接使用 Python 3.5+ 版本支持的 `os.scandir`，性能更快，不需要写两个分支。
如 [python/paddle/vision/datasets/folder.py#L249-L257](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/datasets/folder.py#L249-L257)
```
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [
                d
                for d in os.listdir(dir)
                if os.path.isdir(os.path.join(dir, d))
            ]
```
# 三、业内调研

Pytorch在1.11版本（2022年3月11日）开始不再维护Python 3.6，见 [Deprecating Python 3.6 support](https://github.com/pytorch/pytorch/issues/66462):

* [Move all CI workflows off of Python 3.6](https://github.com/pytorch/pytorch/issues/66462)：下线 Python 3.6 版本的 CI 流水线
* Remove Python 3.6 from our binary builds workflows：不发 Python 3.6 版本的包
* 没有看到相关代码退场的 PR

TensorFlow在2.7.0版本（2021年11月5日）开始不再维护Python 3.6，见 [tensorflow_tested_build_configurations](https://www.tensorflow.org/install/source#tested_build_configurations)，
没有看到相关代码退场的 PR。

# 四、设计思路与实现方案
##1、主体设计思路与折衷
Paddle Python 3.5/3.6 相关代码退场涉及如下内容：

- 删除非必要的环境依赖
- 清理 Python 3.5/3.6 相关逻辑分支
- 清理文档中涉及到 Python 3.5/3.6 的内容

由于 Paddle 不再维护 Python 3.5/3.6，因此，代码退场的 PR 只要 CI 能通过，就可以合入，无其他风险。

## 2、 关键技术点/子模块设计与实现方案

### 删除非必要的环境依赖
Paddle 开发镜像 [ubuntu16_dev.sh](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/dockerfile/ubuntu16_dev.sh) 和 [ubuntu18_dev.sh](https://github.com/PaddlePaddle/Paddle/blob/develop/tools/dockerfile/ubuntu18_dev.sh)  安装了 Python 3.5/3.6，可以进行删除来减少镜像体积大小。此项工作完成后，会统计一下包体积变化。

Paddle CI脚本 [paddle/scripts/paddle_build.sh](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/paddle_build.sh) 安装了 Python 3.5/3.6，可以进行删除来简化脚本。

### 清理 Python 3.5/3.6 相关逻辑分支

部分代码中使用 `sys.version_info` （共47处）来区分不同 Python 版本，并对不同版本做不同处理， Python 3.6以下的逻辑分支可以删除。

### 清理文档中涉及到 Python 3.5/3.6 的内容
在 [docs](https://github.com/PaddlePaddle/docs) 仓库下用`grep -irn 'python3.5\|python3.6' | wc -l`， 可以看到有30条结果。如 [Linux 下的 PIP 安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/pip/linux-pip.html)：
<img width="402" alt="image" src="https://user-images.githubusercontent.com/6836917/202664493-65b1820d-3405-4735-9d62-2e8c20604ae7.png">


# 五、影响和风险总结
## 影响
对开发者的影响：

- 提高源码整洁性，提升开发者阅读的便利性；

对用户使用模型影响：

* Python 3.5 相关代码退场没有风险，因为：
   - Paddle 从 2.1 版本（2021年4月）开始不再发 Python 3.5 的包。
* Python 3.6 相关代码退场没有风险，因为：
   - Paddle 从 2.5 版本（2023年）开始计划不再发 Python 3.6 的包。
   - 各套件 Repo 中 develop 分支的 CE 任务和测试环境，均已下线了 Python 3.6 的监控。
   - 主框架的 develop 分支已于 2022年9月27日 限制只能编译 Python 3.7及以上的包。

## 风险

由于 Paddle 不再维护 Python 3.5/3.6，因此，代码退场的 PR 只要 CI 能通过，就可以合入，无其他风险。
