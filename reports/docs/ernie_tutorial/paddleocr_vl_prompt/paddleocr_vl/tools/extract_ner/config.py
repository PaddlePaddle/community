#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件
管理API密钥和相关配置
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """配置类"""

    # OpenAI API配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "292a9a1fd77e793cc795e91c02a18dd52b93ab5b")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "ernie-5.0-thinking-preview")

    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate_config(cls):
        """验证配置"""
        # OpenAI配置验证
        if not cls.OPENAI_API_KEY:
            print("缺少 OPENAI_API_KEY 配置")
            return False

        if not cls.OPENAI_BASE_URL:
            print("缺少 OPENAI_BASE_URL 配置")
            return False

        if not cls.OPENAI_MODEL:
            print("缺少 OPENAI_MODEL 配置")
            return False

        return True

    @classmethod
    def get_openai_config(cls):
        """获取OpenAI配置"""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "base_url": cls.OPENAI_BASE_URL,
            "model": cls.OPENAI_MODEL
        }

# 使用示例
if __name__ == "__main__":
    if Config.validate_config():
        print("配置验证通过")
        print("OpenAI配置:", Config.get_openai_config())
    else:
        print("配置验证失败")
