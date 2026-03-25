# 打卡任务名称：基于LoongArch架构完成OpenClaw搭建及文心模型API调用（实现本地问答）

为帮助开发者快速掌握龙芯LoongArch架构下的工具搭建与AI API集成核心能力，夯实本次黑客松任务开发基础，飞桨社区推出本次热身打卡任务。开发者需在龙芯LoongArch架构中完成OpenClaw工具的完整搭建与环境适配，同时集成文心模型API，开发实现针对OpenClaw的本地问答工具，实现OpenClaw相关问题的智能交互解答。

## 1.1 任务目标

通过本次打卡，开发者将熟练掌握：

1. LoongArch架构下OpenClaw工具的搭建流程、依赖适配及环境验证技巧

2. 文心模型API的鉴权机制、参数配置

3. 基于OpenClaw+文心API构建本地智能问答工具的开发逻辑

4. 飞桨社区指定仓库的PR提交规范与技术文档编写要求

## 1.2 提交方式

参与热身打卡活动并按照"邮件格式"要求，将全屏截图发送至：huangshuang@loongson.cn

## 1.3 算力/环境支持

本阶段不提供算力/资源支持，需选手自备相应算力。

## 1.4 任务指导

本次打卡所有操作均需在龙芯 LoongArch 架构硬件设备中完成，所有关键步骤需记录执行命令及操作截图，整理完成后发送到制定邮箱。

### Step 1：进行龙芯 LoongArch架构基础环境验证

1. 登录系统，验证LoongArch架构与基础开发工具，确认环境符合要求：

```bash
uname -m
```

2. 更新系统基础依赖，避免后续安装出现权限或依赖缺失问题：

```bash
sudo yum update -y

sudo yum install -y make cmake libffi-devel openssl-devel git  gcc 
```

【操作截图：环境验证命令运行结果】

### Step 2：安装适配 LoongArch 的 Node.js/npm（ OpenClaw 官方要求≥v22.x ）

确认nodejs 版本，是否满足要求

如不满足要求，请https://unofficial-builds.nodejs.org/下载loongarch新版nodejs

### Step3： LoongArch 下 OpenClaw 一键安装与验证

1. 全局安装 OpenClaw（官方标准方式）：

```bash
npm install -g openclaw@latest
```

2. 验证 OpenClaw 安装成功：

```bash
# 查看OpenClaw版本（输出版本号即为成功） 
openclaw -v 

# 查看OpenClaw帮助（输出命令列表，确认核心功能） 
openclaw --help
```

3. 文心API配置：

【openclaw更改内容及配置文件所在路径，均需截图上传】

### Step 4：集成文心模型API，实现OpenClaw本地问答核心功能

浏览器打开openclaw UI，进行5轮以上问答。

【操作截图：本地问答工具启动及自定义问题交互结果】

## 1.5 邮件格式

标题：【飞桨黑客松打卡】基于LoongArch架构完成OpenClaw搭建及文心模型API调用 - [你的 GitHub ID 或姓名]

内容：

龙芯中科 团队你好，

【GitHub ID】：XXX

【打卡内容】：基于LoongArch架构完成OpenClaw搭建及文心模型API调用（实现本地问答）

【打卡截图】：环境搭建全流程（关键命令 + 操作截图）、至少 3 个 OpenClaw 相关测试用例（问题 + 实际问答结果 + 截图）。
