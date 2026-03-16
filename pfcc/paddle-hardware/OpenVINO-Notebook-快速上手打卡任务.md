# OpenVINO Notebook 快速上手（PaddleOCR-VL 文档理解）打卡任务
为了帮助参赛者快速进入 OpenVINO + PaddleOCR-VL 的开发与推理部署流程，我们设置本次热身打卡任务。完成后你将具备参与正式赛题的基础能力：能在本地/云端跑通 Notebook，并掌握基本的运行、调试与结果验证方法。

## 任务目标
通过本次打卡，你将掌握：
* 如何成功运行一个 OpenVINO Notebook 示例（推荐 [PaddleOCR-VL Notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddleocr_vl) ）。
* 理解端到端流程：模型准备/推理调用/结果展示（含关键输出截图）。
* 提交一份针对进阶任务的“初步方案说明”（场景+思路+工具框架+预期效果）。


## 提交方式
参与热身打卡活动并按照邮件模板格式将截图发送至 ext_paddle_oss@baidu.com + 厂商邮件组 zhuo.wu@intel.com 及 ethan.yang@intel.com 

## 算力/环境支持
本次热身打卡活动需要使用 Intel 平台的CPU或GPU，赶快行动起来吧~

## 任务指导
### 获取并运行 OpenVINO Notebooks
```
git clone https://github.com/openvinotoolkit/openvino_notebooks.git
```
### 安装OpenVINO_notebooks环境
根据自己使用的操作系统或常用的安装方式，在 openvino_notebooks 的 Readme 文件中找到 [对应的环境安装方式链接](https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-installation-guide) ，如使用的是Windows操作系统，安装指南链接请参考 [这儿](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) 。

### 加载notebooks
环境安装完成后，打开一个终端窗口，运行命令jupyter lab notebooks，加载所有的 notebook 代码示例，并选择 paddleocr-vl notebook 打开。

### 执行
按顺序执行全部 Cell，确保最终可得到可视化结果与推理输出。

### 编译打卡流程
产出打卡截图/录屏
* 截图至少包含：依赖安装成功、推理运行成功日志/输出、可视化结果或关键结构化输出。例如：
  ![Result1](https://github.com/zhuo-yoyowz/dino_bone_finding_OpenVINO/blob/main/paddleocr-vl-ov%20result1.png)
  
  ![Result2](https://github.com/zhuo-yoyowz/dino_bone_finding_OpenVINO/blob/main/paddleocr-vl-ov%20result2.png)
* 可选：录制 30~60 秒短视频，展示从输入样例到输出结果的完整过程。

## 邮件格式
* 标题： [飞桨黑客松第十期OpenVINO任务打卡]
* 内容：
   * 飞桨团队你好，
   * 【GitHub ID】：参赛选手本人 GitHub 打卡任务仓库地址
   * 【运行 Notebook】：PaddleOCR-VL Notebook（链接/路径）
   * 【环境信息】：OS / CPU / GPU / OpenVINO 版本
   * 【打卡截图】：（粘贴截图或提供链接）
