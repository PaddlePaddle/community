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
本次热身打卡活动需要使用海光第二代DCU K100-AI，我们将在 [国家超算互联网平台](www.scnet.cn) 提供打卡所需的环境。
我们将会为小伙伴们提供的手机号配置账号并分配活动所用卡时。

## 搭建环境指导
### 创建DCU实例
- 登录 [国家超算互联网平台](www.scnet.cn) 中的控制台页面；
- 进入【人工智能】页面，点击【创建Notebook】，选择一台【异构加速卡AI】的实例；
- 镜像选择【PyTorch -> 2.5.1 -> py3.10-ubuntu22.04 -> DTK25.04.2】；
- 镜像创建后，等待启动实例，可以通过 ssh 登录到实例上来。

### 升级paddlepaddle-dcu到最新版本
```
python -m pip install paddlepaddle-dcu -U -i https://www.paddlepaddle.org.cn/packages/stable/dcu/
```

### 安装paddlex和paddleocr等上层应用框架
```
python -m pip install paddleocr[doc-parser]
pip install PyYAML==6.0.3 
```

### 验证paddlepaddle-dcu可以正常运行
```
python3 -c "import paddle; paddle.utils.run_check()"
```

### 设置环境变量
设置如下环境变量，最好写入.bashrc中，每次登陆后自动执行：
```
export PADDLE_PDX_DISABLE_DEV_MODEL_WL=true
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true
unset CUDA_HOME
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "cuda" | tr '\n' ':' | sed 's/:$//')
export HOME=/root/private_data/
```
退出后，重新ssh登录。以避免后续把paddleocr的模型文件下载到/root/下。（应下载到/root/private_data/下）

### 运行示例
```
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png --device dcu
```
此时会自动下载PaddleOCR的模型权重。

## 打卡任务
### 第一步：编写PaddleOCR应用
- 下载 [数据集](https://www.scnet.cn/ui/aihub/dataset/acsepoahfb/paddleocr_dcu_task/)，
编写 python 代码，调用 PaddleOCR-vl 1.5的模型，把它扫描成文本。
- 打卡提交物：编写的代码（python程序，后缀名改为'py_'，以防邮件被过滤），扫描出来的文本（txt文档）。

### 第二步：使用vllm后端加速OCR
- 下载 [数据集](https://www.scnet.cn/ui/aihub/dataset/acsepoahfb/MSRA-Text_Detection_500_Database/)，
对test下的200张图片进行 OCR 扫描。要求通过启动 vllm 后端的方式，通过打高 batch 的方式尽可能得到一个较高的性能。
- 启动vllm参考命令： 
```
paddleocr genai_server \
  --model_name PaddleOCR-VL-1.5-0.9B \
  --host 0.0.0.0 \
  --port 8118 \
  --backend vllm
``` 
- 验证vllm后端可以正常运行：
```
paddleocr doc_parser --input paddleocr_vl_demo.png --device dcu \
  --vl_rec_backend vllm-server \
  --vl_rec_server_url http://localhost:8118/v1
```
- 代码要求：输出时把扫描的文本内容保存进txt文件，并打印出处理200张图片的总耗时，以及batch_size等信息。
- 参考性能：batch=1时，扫描200张图片需要约411秒。
- 打卡提交物：代码（python程序，后缀名改为'py_'），运行结果（txt文件），性能数据（屏幕截图）

## 邮件格式
* 标题： [飞桨黑客松第十期DCU平台任务打卡]
* 内容：
   * 飞桨团队你好，
   * 【GitHub ID】：XXX
   * 【打卡内容】：编写PaddleOCR应用 / 使用vllm后端加速OCR
   * 【附件与截图】：
