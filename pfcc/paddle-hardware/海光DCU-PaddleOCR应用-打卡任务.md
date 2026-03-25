# 🎯 海光打卡任务：免费领DCU算力资源

🎁 **报名海光打卡任务，可领 200小时 DCU卡时！** 名额限 **40名**！ <br/>
 <br/>
🚀 **完成进阶任务，额外再领第三代DCU BW系列显卡 200小时卡时！** 名额限 **20名**！ <br/>

---

## 📋 任务概览

| 项目 | 内容 |
|:---|:---|
| **任务名称** | 海光DCU平台PaddleOCR应用开发与实践 |
| **平台** | 海光DCU（国产GPGPU，全面接入PaddlePaddle生态） |
| **目标** | 熟悉飞桨生态与DCU平台，掌握高级功能开发 |

---

## 🎓 你将学到什么

- ✅ 在DCU平台上搭建Paddle架构
- ✅ 基于PaddleOCR的应用开发
- ✅ 通过vLLM等工具加速PaddleOCR

---

## 📤 提交方式

将以下内容按邮件模板格式发送至：
- `ext_paddle_oss@baidu.com`
- `liuyun@hygon.cn`

📎 需包含：截图、输出结果、代码

---

## 💻 算力与环境支持

| 配置项 | 详情 |
|:---|:---|
| **硬件** | 海光第二代DCU K100-AI |
| **平台** | [国家超算互联网平台](https://www.scnet.cn) |
| **账号** | 提供手机号即可配置，自动分配活动卡时 |

---

## 🛠️ 环境搭建指南

### 1️⃣ 创建DCU实例

1. 登录 [国家超算互联网平台](https://www.scnet.cn) 控制台
2. 进入 **【人工智能】** → 点击 **【创建Notebook】**
3. 选择 **【异构加速卡AI】** 实例
4. **镜像选择**：`PyTorch 2.5.1` → `py3.10-ubuntu22.04` → `DTK25.04.2`
5. 启动实例后，通过 SSH 登录

### 2️⃣ 升级 PaddlePaddle-DCU

```bash
python -m pip install paddlepaddle-dcu -U -i https://www.paddlepaddle.org.cn/packages/stable/dcu/
```

### 3️⃣ 升级 PaddlePaddle-DCU
```bash
python -m pip install paddleocr[doc-parser]
pip install PyYAML==6.0.3 
```

### 4️⃣ 验证安装
```bash
python3 -c "import paddle; paddle.utils.run_check()"
```

### 5️⃣ 设置环境变量
⚠️ 建议写入 ~/.bashrc，每次登录自动生效
```bash
export PADDLE_PDX_DISABLE_DEV_MODEL_WL=true
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true
unset CUDA_HOME
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "cuda" | tr '\n' ':' | sed 's/:$//')
export HOME=/root/private_data/
```
🔁 退出后重新 SSH 登录，确保模型下载到 /root/private_data/ 而非 /root/

### 6️⃣ 运行示例
```bash
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png --device dcu
```

## 📝 打卡任务
### 任务一：编写PaddleOCR应用
| 项目      | 要求                                                                           |
| :------ | :--------------------------------------------------------------------------- |
| **数据集** | [下载地址](https://www.scnet.cn/ui/aihub/dataset/acsepoahfb/paddleocr_dcu_task/) |
| **任务**  | 调用 PaddleOCR-vl 1.5 模型，将图片扫描成文本                                              |
| **提交物** | Python代码（后缀改为`.py_`防过滤）、扫描结果（TXT）                                            |

### 任务二：使用vLLM后端加速OCR
| 项目       | 要求                                                                                                                                    |
| :------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **数据集**  | [MSRA-Text Detection 500 Database](https://www.scnet.cn/ui/aihub/dataset/acsepoahfb/MSRA-Text_Detection_500_Database/)（test目录下200张图片） |
| **目标**   | 启动vLLM后端，通过高batch提升性能                                                                                                                 |
| **参考性能** | batch=1时，200张图片约需411秒                                                                                                                 |
| **提交物**  | 代码（`.py_`）、运行结果（TXT）、性能数据截图                                                                                                           |

### 参考命令：
启动vLLM服务
```bash 
paddleocr genai_server \
  --model_name PaddleOCR-VL-1.5-0.9B \
  --host 0.0.0.0 \
  --port 8118 \
  --backend vllm
```

验证vLLM后端
```bash
paddleocr doc_parser --input paddleocr_vl_demo.png --device dcu \
  --vl_rec_backend vllm-server \
  --vl_rec_server_url http://localhost:8118/v1
```

代码要求
 * 输出扫描文本保存为TXT文件
 * 打印总耗时
 * 显示batch_size等信息

## 📧 邮件提交模板
```
标题：[飞桨黑客松第十期DCU平台任务打卡]

飞桨团队你好，

【GitHub ID】：XXX
【打卡内容】：编写PaddleOCR应用 / 使用vllm后端加速OCR
【附件与截图】：（请在此处添加）
```

# 🎁 领取算力资源
<p>报名通过后，加入 【飞桨黑客松-海光】 微信群，辅导老师将为你充值账号。</p>
<img src="images/hygon_group_260401.jpg" width="300">

<p>💡 提示：名额有限，建议尽快报名并完成任务！</p>