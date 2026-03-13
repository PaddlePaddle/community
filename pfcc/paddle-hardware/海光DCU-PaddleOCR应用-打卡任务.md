# 打卡任务名称：海光DCU平台PaddleOCR应用开发与实践
海光的DCU是一款国产的GPGPU，在国内超算与智算中心被广泛采用，也全面接入了PaddlePaddle生态。

小伙伴们可以通过此次打卡活动熟悉飞桨生态与DCU平台，然后进入DCU平台上的飞桨生态的高级功能开发。

## 任务目标
通过本次打卡，你将掌握：
* 在DCU平台上搭建Paddle架构的方法
* 基于PaddleOCR的应用开发
* 通过vllm等工具加速PaddleOCR的方法

## 提交方式
参与热身打卡活动并按照邮件模板格式将截图，输出结果和代码发送至 ext_paddle_oss@baidu.com + liuyun@hygon.cn

## 算力/环境支持
本次热身打卡活动需要使用海光第二代DCU K100-AI，我们将在国家超算互联网平台（www.scnet.cn）提供打卡所需的环境。
我们将会为小伙伴们提供的手机号配置账号并分配活动所用卡时。

## 搭建环境指导
### 创建DCU实例
登录www.scnet.cn，控制台页面中，进入【人工智能】页面，然后点击【创建Notebook】，选择一台【异构加速卡AI】的实例，
然后镜像选择【paddle -> 3.0.0 -> py3.10-ubuntu22.04】，创建后，等待启动实例，可以通过ssh登录到实例上来。

### 升级paddlepaddle-dcu到最新版本
```
python -m pip install paddlepaddle-dcu -U -i https://www.paddlepaddle.org.cn/packages/stable/dcu/
```

### 安装paddlex和paddleocr等上层应用框架
```
python -m pip install paddleocr[doc-parser]
```

### 验证paddlepaddle-dcu可以正常运行
```
python3 -c "import paddle; paddle.utils.run_check()"
```

### 运行示例
```
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png --device dcu
```
注意：
- 建议设置环境变量：PADDLE_PDX_DISABLE_DEV_MODEL_WL=true   PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true
- 默认PaddleOCR模型会下载到/root/目录下，学员需要把模型权重等文件放到/root/private_data/下，不然超出image容量限制，无法保存

## 打卡任务
### 编写PaddleOCR应用
- 任务：当前目录dcu_tasks/下有10张图片，编写应用脚本，通过paddleocr-vl 1.5的模型，把他扫描成文本。
- 打卡提交物：编写的代码（python程序，后缀名改为'py_'），扫描出来的文本（txt文档）。

### 使用vllm后端加速OCR
- 扫描整个数据集：https://opendatalab.com/OpenDataLab/MSRA-TD500
- 打卡提交物：代码（python程序，后缀名改为'py_'），运行结果（txt文件），性能数据（屏幕截图）

## 邮件格式
* 标题： [飞桨黑客松第十期DCU平台任务打卡]
* 内容：
   * 飞桨团队你好，
   * 【GitHub ID】：XXX
   * 【打卡内容】：编写PaddleOCR应用 / 使用vllm后端加速OCR
   * 【附件与截图】：
