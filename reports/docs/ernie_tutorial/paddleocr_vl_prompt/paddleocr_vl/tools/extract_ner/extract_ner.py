#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态图像识别脚本
通过调用OpenAI接口识别图片信息并返回JSON格式数据
支持本地图片和多模态大模型处理
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
from openai import OpenAI
from io import BytesIO

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# 导入配置
from config import Config

# 配置日志
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class MultimodalImageRecognizer:
    """多模态图像识别器"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        初始化多模态图像识别器

        Args:
            api_key: OpenAI API密钥，如果不提供则从配置文件获取
            base_url: API基础URL，如果不提供则从配置文件获取
            model: 模型名称，如果不提供则从配置文件获取
        """
        # 使用配置文件中的默认值
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.base_url = base_url or Config.OPENAI_BASE_URL
        self.model = model or Config.OPENAI_MODEL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logger.info(f"初始化多模态图像识别器，模型: {self.model}")

    def convert_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        将图片文件转换为base64编码，统一转换为JPEG格式

        Args:
            image_path: 图片文件路径

        Returns:
            base64编码字符串，失败时返回None
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"图片文件不存在: {image_path}")
                return None

            # 如果安装了PIL，使用PIL转换图片格式
            if HAS_PIL:
                try:
                    # 打开图片并转换为RGB（处理RGBA或其他格式）
                    img = Image.open(image_path)

                    # 如果图片有RGBA通道，转换为RGB
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # 创建白色背景
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        # 粘贴原图片到背景上
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = rgb_img
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')

                    # 将图片转换为JPEG字节
                    img_buffer = BytesIO()
                    img.save(img_buffer, format='JPEG', quality=95)
                    image_data = img_buffer.getvalue()

                    logger.info(f"图片已转换为JPEG格式，长度: {len(image_data)}")
                except Exception as e:
                    logger.warning(f"使用PIL转换图片失败: {e}，尝试直接读取")
                    # 如果PIL转换失败，直接读取原始数据
                    with open(image_path, 'rb') as image_file:
                        image_data = image_file.read()
            else:
                # 如果没有PIL，直接读取原始文件
                logger.warning("未安装PIL库，将直接读取图片文件，建议安装Pillow库以支持格式转换")
                with open(image_path, 'rb') as image_file:
                    image_data = image_file.read()

            # 转换为base64
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
            logger.info(f"图片base64转换成功，长度: {len(base64_encoded)}")
            return base64_encoded

        except Exception as e:
            logger.error(f"转换图片为base64时发生错误: {e}")
            return None

    def _get_mime_type(self) -> str:
        """
        获取MIME类型，因为所有图片都转换为JPEG格式

        Returns:
            MIME类型字符串
        """
        return 'image/jpeg'

    def create_multimodal_message(self, text: str, image_input: Union[str, bytes]) -> list:
        """
        创建多模态消息

        Args:
            text: 文本内容
            image_input: 图片路径或base64编码

        Returns:
            多模态消息列表
        """
        content = [
            {"type": "text", "text": text}
        ]

        # 判断输入类型
        if isinstance(image_input, bytes):
            # 如果是bytes，转换为base64
            base64_image = base64.b64encode(image_input).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{base64_image}"
        elif os.path.exists(image_input):
            # 如果是文件路径，转换为base64
            base64_image = self.convert_image_to_base64(image_input)
            if base64_image:
                mime_type = self._get_mime_type()
                image_url = f"data:{mime_type};base64,{base64_image}"
            else:
                logger.error("图片转换失败")
                return [{"type": "text", "text": text}]
        else:
            # 如果是base64或URL，直接使用
            if image_input.startswith('data:'):
                image_url = image_input
            else:
                logger.error("不支持的图片输入格式")
                return [{"type": "text", "text": text}]

        content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

        return content

    def recognize_image(
        self,
        image_input: Union[str, bytes],
        prompt: str,
        system_prompt: str,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        识别图片信息

        Args:
            image_input: 图片路径、URL或base64编码
            prompt: 用户提示词
            system_prompt: 系统提示词
            max_tokens: 最大令牌数

        Returns:
            识别结果的JSON格式数据
        """
        try:
            # 创建多模态消息
            content = self.create_multimodal_message(prompt, image_input)

            # 构建消息列表
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]

            logger.info(f"开始调用API识别图片，模型: {self.model}")

            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2
            )

            # 获取响应内容
            response_content = response.choices[0].message.content
            logger.info(f"API响应: {response_content}")

            # 尝试解析JSON响应
            try:
                # 预处理响应内容，移除代码块标记
                cleaned_content = self._preprocess_response(response_content)

                # 尝试直接解析JSON
                if cleaned_content.strip().startswith('{'):
                    result = json.loads(cleaned_content)
                    return {
                        "success": True,
                        "data": result,
                        "raw_response": response_content
                    }
                else:
                    # 尝试从文本中提取JSON
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', cleaned_content)
                    if json_match:
                        json_str = json_match.group()
                        # 尝试修复常见的JSON格式问题
                        json_str = self._fix_json_format(json_str)
                        result = json.loads(json_str)
                        return {
                            "success": True,
                            "data": result,
                            "raw_response": response_content
                        }
                    else:
                        # 如果无法解析JSON，返回原始文本
                        return {
                            "success": True,
                            "data": {"content": response_content},
                            "raw_response": response_content
                        }
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}")
                logger.error(f"原始响应内容: {response_content}")
                return {
                    "success": False,
                    "error": f"JSON解析失败: {e}",
                    "raw_response": response_content
                }

        except Exception as e:
            logger.error(f"调用API时发生错误: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _preprocess_response(self, response_content: str) -> str:
        """
        预处理API响应内容，移除代码块标记等

        Args:
            response_content: 原始响应内容

        Returns:
            预处理后的内容
        """
        try:
            import re

            # 移除代码块标记
            content = re.sub(r'```(?:json)?\s*', '', response_content)
            content = re.sub(r'\s*```', '', content)

            # 移除多余的空白行
            content = re.sub(r'\n\s*\n', '\n', content)

            return content.strip()
        except Exception as e:
            logger.warning(f"响应预处理失败: {e}")
            return response_content

    def _fix_json_format(self, json_str: str) -> str:
        """
        尝试修复常见的JSON格式问题

        Args:
            json_str: 可能格式有问题的JSON字符串

        Returns:
            修复后的JSON字符串
        """
        try:
            import re

            # 移除可能的注释
            json_str = re.sub(r'//.*?\n', '', json_str)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

            # 修复控制字符问题（在字符串值中）- 改进版本
            def escape_control_chars(match):
                # 获取引号内的内容
                content = match.group(1)
                # 先将已有的反斜杠进行转义，然后处理其他控制字符
                content = content.replace('\\', '\\\\')  # 转义反斜杠
                content = content.replace('"', '\\"')    # 转义引号
                content = content.replace('\n', '\\n')   # 转义换行符
                content = content.replace('\r', '\\r')   # 转义回车符
                content = content.replace('\t', '\\t')   # 转义制表符
                return f'"{content}"'

            # 匹配所有包含反斜杠或控制字符的字符串
            json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_control_chars, json_str)

            # 修复以0开头的数字（在JSON中必须用字符串表示）
            json_str = re.sub(r':\s*(0\d+)', r': "\1"', json_str)

            # 尝试修复缺少引号的问题（简单情况）
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)

            # 尝试修复单引号问题
            json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)

            # 尝试修复尾部逗号问题
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            return json_str
        except Exception as e:
            logger.warning(f"JSON修复失败: {e}")
            return json_str

    def analyze_image(
        self,
        image_input: Union[str, bytes],
        analysis_type: str = "document"
    ) -> Dict[str, Any]:
        """
        分析图片内容（简化版本）

        Args:
            image_input: 图片路径、URL或base64编码
            analysis_type: 分析类型，固定为 "document"

        Returns:
            分析结果的JSON格式数据
        """
        # 统一使用文档分析提示词
        prompt = "请分析这张文档图片中的所有信息，并返回完整的JSON格式数据。如果有的字段没有值，那么保留此字段，值为空。注意：所有的值都以string的形式返回，不要使用数字类型等。"
        system_prompt = '''
你是一个专业的文档分析助手，能够准确分析文档内容并返回结构化的JSON数据。

注意：数据的语言与文档的语言保持一致。
注意：需要保留完整的字段层级关系，不要把所有字段都放到一级字段中。
注意：JSON数据中不要包含注释，也不需要任何解释或说明。
注意：对于特殊字符需要进行转义。

注意：对于选项字段，只保留所选择的字段值，如果没有选择，则置为空。
比如，`业务类型` 包括 `账户开户、账户登记` 等选项，文档中`账户登记`是选中状态，则，返回 `{"业务类型"："账户登记"}`，不返回`账户开户`等其他选项。
再比如，`业务类型` 包括 `账户开户、账户登记` 等选项，文档中没有标记选中的选项，则，返回 `{"业务类型"：""}`，也就是说，只保留键，不需要有值。
选中的样式包括但不限于打勾等情况。
举例，如果你识别到的是：
```
{
    "志愿捐献": {
        "器官": "☑", # 这里可以是 `√` `X` 等情况
        "眼角膜": "☐",
        "其他组织": "☐",
    }
}
```
正确的返回应该是：
```
{"志愿捐献":"器官"}
```
而不是
```
{
    "志愿捐献": {
        "器官": "☑"
    }
}
```
'''

        return self.recognize_image(
            image_input=image_input,
            prompt=prompt,
            system_prompt=system_prompt
        )


def main():
    """主函数，演示用法"""
    # 验证配置
    if not Config.validate_config():
        print("配置验证失败，请检查配置文件")
        return

    # 创建识别器实例
    recognizer = MultimodalImageRecognizer()

    # 示例：识别本地图片
    image_path = "/media/shun/bigdata/Dataset/机动车发票/train/3.jpg"  # 替换为实际图片路径

    if os.path.exists(image_path):
        print(f"正在识别图片: {image_path}")

        # 文档分析
        result = recognizer.analyze_image(image_path, "document")
        print("文档分析结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        print(f"图片文件不存在: {image_path}")
        print("请将图片文件放置在当前目录下，或修改image_path变量")
        print("\n配置说明:")
        print("1. 复制 .env.example 为 .env")
        print("2. 在 .env 文件中设置您的配置")


if __name__ == "__main__":
    main()
