## 统信：在 deepin 25 操作系统上调通文心大模型

为帮助开发者快速上手 deepin 操作系统与文心大模型的联合开发环境，特设置此打卡任务。通过简单的几步操作，开发者就能在 deepin 系统上调用文心大模型进行对话，为后续更复杂的应用开发打下基础。

### 1.1 任务目标

通过本次打卡，你将掌握：

- **deepin 25 系统的基本操作**与终端环境使用。
  
- **erniebot SDK** 在 Linux 环境下的安装与配置。
  
- **云端文心大模型 API** 的基本调用流程。

### 1.2 提交方式

参与热身打卡活动并按照“邮件格式”要求，将全屏截图发送至：`luzhen@uniontech.com`并抄送 `ext_paddle_oss@baidu.com`。

### 1.3 算力/环境支持

- **系统镜像**：[deepin 25 最新版镜像](https://www.deepin.org/zh/download/ "null")，根据网页中的指导文档完成安装。

- **硬件要求**：支持 deepin 25 安装的物理机（推荐）或虚拟机环境，参与者需自行准备实体或虚拟硬件设备完成打卡任务。

- **API 资源**：开发者需提前获取 [飞桨 AI Studio 平台](https://aistudio.baidu.com/ "null") Access Token。

  欢迎扫码加入“deepin赛道百度黑客松打卡任务群”:

  <img src="./images/deepin赛道.png" width="200">

### 1.4 任务指导

以下步骤均在 deepin 25 系统上进行。
#### 1.4.1 安装环境依赖

在 **deepin-terminal** 中执行以下命令，安装 Python 环境：

```bash
sudo apt update
sudo apt install -y python3-pip
```

#### 1.4.2 安装飞桨 SDK

使用 pip 安装 erniebot 库：

```bash
pip install erniebot --break-system-packages
```

### 1.5 详细打卡流程

**Step 1: 编写交互脚本**

在终端输入 `nano task.py`，粘贴以下内容并替换你的 Token：

```python
import erniebot

# 配置 API Token
erniebot.api_type = 'aistudio'
erniebot.access_token = '在这里粘贴你的Access Token'

# 创建一次对话
response = erniebot.ChatCompletion.create(
    model='ernie-lite',
    messages=[{'role': 'user', 'content': '你好，我是 deepin 25 用户，请问 deepin 系统的主要特点是什么？'}]
)

print(f"\n[文心大模型回复]: \n{response.get_result()}")
```

**Step 2: 运行并验证**

在终端执行：`python3 task.py`。

### 1.6 邮件格式

**标题**：【飞桨黑客松打卡】deepin 25 专项任务 - [你的 GitHub ID 或姓名]

**内容**：

deepin 团队你好，

【GitHub ID】：XXX

【打卡内容】：在 deepin 25 操作系统上调通文心大模型。

【打卡截图】：需包含 OS 版本显示（终端输入 `cat /etc/os-release` ）、task.py 执行过程及模型返回结果、完整的 deepin 桌面环境。

![](https://github.com/deepin-mozart/Hackathon-deepin/raw/master/capture.png)
