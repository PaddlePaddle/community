龙芯中科：基于 LoongArch 架构完成 OpenClaw 搭建及文心模型 API 调用（实现本地问答）
1.1 任务目标
通过本次打卡，开发者将熟练掌握：
龙芯 LoongArch 架构系统基础操作与系统环境配置方法
LoongArch 架构下 OpenClaw 工具的搭建流程、依赖适配及环境验证技巧
文心模型 API 的鉴权机制、参数配置
基于 OpenClaw + 文心 API 构建本地智能问答工具的开发逻辑
1.2 提交方式
完成打卡任务所有操作后，需按要求参加热身打卡活动并按照邮件模板格式将截图发送至指定邮箱。
1.3 算力 / 环境支持
本次打卡任务龙芯中科硬件设备需要参赛人员自行准备，主办方提供文心一言 API 作为后端调用算力。
1.4 任务指导
本次打卡所有操作均需在龙芯 LoongArch 架构硬件设备中完成，所有关键步骤需记录执行命令、运行日志及操作截图，最终完整整理至提交的 md 文档中。所有代码需保证可直接运行，无语法错误与环境依赖问题。
Step 1：登录龙芯 LoongArch 基础环境验证
登录并验证 LoongArch 架构与基础开发工具，确认环境符合要求：
bash
运行
# 验证架构（输出需为loongarch64）
uname -m
更新系统基础依赖，避免后续安装出现权限或依赖缺失问题：
bash
运行
sudo yum update -y
sudo yum install -y make cmake libffi-devel openssl-devel git gcc
【操作截图：环境验证命令运行结果】
Step 2：安装适配 LoongArch 的 Node.js/npm（OpenClaw 官方要求≥v22.x）
确认 nodejs 版本，是否满足要求
如不满足要求，通过https://unofficial-builds.nodejs.org/下载 loongarch 新版 nodejs
Step 3：LoongArch 下 OpenClaw 一键安装与验证
全局安装 OpenClaw（官方标准方式）：
bash
运行
npm install -g openclaw@latest
验证 OpenClaw 安装成功：
bash
运行
# 查看OpenClaw版本（输出版本号即为成功）
openclaw -v
# 查看OpenClaw帮助（输出命令列表，确认核心功能）
openclaw --help
文心 API 配置：
【openclaw 更改内容及配置文件所在路径，均需截图上传】
Step 4：集成文心模型 API，实现 OpenClaw 本地问答核心功能
浏览器打开 openclaw UI，进行 5 轮以上问答。
【操作截图：本地问答工具启动及自定义问题交互结果】
1.5 邮件格式
邮件标题
[龙芯中科 × 百度飞桨热身打卡]
邮件正文格式
飞桨团队你好，
【GitHub ID】：XXX（例如 paddle-hack）
【打卡内容】：
文档与 PR 规范
新建文件命名规则：【xxx 厂商 - LoongArch 搭建 OpenClaw 及文心 API 本地问答.md】（xxx 厂商替换为实际参与厂商名称）
PR 标题标注规范：【飞桨黑客松第十期 - 硬件生态 - xxx 厂商 - 打卡任务】
PR 描述简要说明打卡任务完成情况，无需额外附件。
md 文档必须包含核心内容
环境搭建全流程（关键命令 + 操作截图）
文心 API 调用测试的完整代码及运行结果日志
OpenClaw 本地问答工具的实现代码（可直接运行）
至少 3 个 OpenClaw 相关测试用例（问题 + 实际问答结果 + 截图）
开发过程中遇到的问题、排查思路及最终解决方法
PR 提交后，由飞桨硬件生态团队与龙芯技术团队联合进行 review，审核通过即完成本次打卡。