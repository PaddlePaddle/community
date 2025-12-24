#!/usr/bin/env python3
"""
找到 paddlex 中的 pipeline.py 文件，将 assert prompt_label 改为 warning 提示。
"""

import re
import sys
from pathlib import Path

try:
    import paddlex
except ImportError:
    print("错误：未能导入 paddlex，请确保已安装")
    sys.exit(1)


def main():
    # 根据 paddlex 模块的位置确定目标文件
    paddlex_file = Path(paddlex.__file__)
    paddlex_root = paddlex_file.parent
    target_file = paddlex_root / "inference/pipelines/paddleocr_vl/pipeline.py"

    # 检查文件是否存在
    if not target_file.exists():
        print(f"错误：文件不存在: {target_file}")
        sys.exit(1)

    # 读取文件内容
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 定义要替换的模式
    old_pattern = r'''            assert prompt_label\.lower\(\) in \[
                "ocr",
                "formula",
                "table",
                "chart",
            \], f"Layout detection is disabled \(use_layout_detection=False\)\. 'prompt_label' must be one of \['ocr', 'formula', 'table', 'chart'\], but got '\{prompt_label\}'\."'''

    # 新的代码（使用 warning 代替 assert）
    new_code = '''            if prompt_label.lower() not in [
                "ocr",
                "formula",
                "table",
                "chart",
            ]:
                import warnings
                warnings.warn(
                    f"Layout detection is disabled (use_layout_detection=False). "
                    f"'prompt_label' must be one of ['ocr', 'formula', 'table', 'chart'], "
                    f"but got '{prompt_label}'. Program will continue anyway.",
                    UserWarning
                )'''

    # 执行第一个替换：assert 改为 warning
    new_content = re.sub(old_pattern, new_code, content)

    # 检查是否进行了第一个替换
    if new_content == content:
        print("警告：未找到匹配的 assert 代码模式，请检查文件内容")
    else:
        print("成功：已将 assert 改为 warning")

    # 执行第二个替换：text_prompt 改为条件表达式
    old_text_prompt = 'text_prompt = "OCR:"'
    new_text_prompt = 'text_prompt = "OCR:" if block_label.lower() == "ocr" else block_label; block_label = block_label.lower()'

    if old_text_prompt in new_content:
        new_content = new_content.replace(old_text_prompt, new_text_prompt)
        print("成功：已将 text_prompt 改为条件表达式")
    else:
        print("警告：未找到 text_prompt 的代码")

    # 执行第三个替换：label 移除 .lower()
    old_label = '"label": prompt_label.lower(),'
    new_label = '"label": prompt_label,'

    if old_label in new_content:
        new_content = new_content.replace(old_label, new_label)
        print("成功：已将 label 改为 prompt_label")
    else:
        print("警告：未找到 label 的代码")

    # 写回文件
    try:
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"成功：已完成所有 patch，文件位置: {target_file}")
    except Exception as e:
        print(f"错误：写入文件时出现异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
