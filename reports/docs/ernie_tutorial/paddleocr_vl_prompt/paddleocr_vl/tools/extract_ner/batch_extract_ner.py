#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量图片NER提取脚本
遍历指定目录中的图片文件，使用extract_ner.py提取JSON格式数据并保存
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse

from extract_ner import MultimodalImageRecognizer


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    获取目录中的所有图片文件

    Args:
        directory: 要搜索的目录路径
        extensions: 支持的图片扩展名列表，默认为常见格式

    Returns:
        图片文件路径列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

    image_files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"错误：目录不存在 - {directory}")
        return image_files

    # 递归搜索所有图片文件
    for ext in extensions:
        pattern = f"**/*{ext}"
        image_files.extend(directory_path.glob(pattern))
        # 同时搜索大写扩展名
        pattern_upper = f"**/*{ext.upper()}"
        image_files.extend(directory_path.glob(pattern_upper))

    # 转换为字符串路径并去重
    image_paths = list(set(str(path) for path in image_files))
    image_paths.sort()

    return image_paths


def process_single_image(recognizer: MultimodalImageRecognizer, image_path: str, output_dir: str) -> bool:
    """
    处理单个图片文件

    Args:
        recognizer: 图片识别器实例
        image_path: 图片文件路径
        output_dir: 输出目录

    Returns:
        处理成功返回True，失败返回False
    """
    try:
        # 生成输出文件名
        image_name = Path(image_path).stem
        output_file = os.path.join(output_dir, f"{image_name}_ner.json")

        # 检查输出文件是否已存在，如果存在则跳过处理
        if os.path.exists(output_file):
            print(f"跳过已处理的图片: {image_path} (输出文件已存在: {output_file})")
            return True

        print(f"正在处理: {image_path}")

        # 调用extract_ner的功能进行图片识别
        result = recognizer.analyze_image(image_path, "document")

        # 检查提取是否成功
        if not result.get("success", False):
            print(f"图片 {image_path} 提取失败，跳过生成文件")
            return False

        # 构建输出数据结构，只保留data字段
        output_data = {
            "image": image_path,
            "data": result.get("data", {})
        }

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"保存结果到: {output_file}")
        return True

    except Exception as e:
        print(f"处理图片 {image_path} 时发生错误: {e}")
        return False


def batch_process_images(input_dir: str, output_dir: str = None) -> None:
    """
    批量处理图片

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为None则在输入目录下创建output子目录
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(input_dir, "output")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图片文件
    print(f"正在扫描目录: {input_dir}")
    image_files = get_image_files(input_dir)

    if not image_files:
        print("未找到任何图片文件")
        return

    print(f"找到 {len(image_files)} 个图片文件")

    # 初始化识别器
    try:
        recognizer = MultimodalImageRecognizer()
    except Exception as e:
        print(f"初始化识别器失败: {e}")
        return

    # 处理每个图片文件
    success_count = 0
    total_count = len(image_files)
    failed_images = []  # 记录失败的图片路径

    for image_path in image_files:
        if process_single_image(recognizer, image_path, output_dir):
            success_count += 1
        else:
            failed_images.append(image_path)

    # 输出处理结果统计
    print(f"\n处理完成！")
    print(f"总文件数: {total_count}")
    print(f"成功处理: {success_count}")
    print(f"失败数量: {total_count - success_count}")
    print(f"结果保存在: {output_dir}")

    # 打印所有失败的图片路径
    if failed_images:
        print(f"\n失败的图片路径列表：")
        for failed_path in failed_images:
            print(f"  - {failed_path}")
    else:
        print("\n没有失败的图片")


def find_deepest_folders(root_dir: str) -> List[str]:
    """
    递归查找最底层的文件夹（不包含任何子文件夹的文件夹）

    Args:
        root_dir: 根目录路径

    Returns:
        最底层文件夹路径列表
    """
    deepest_folders = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 如果当前目录没有子文件夹，则认为是最底层文件夹
        if not dirnames:
            # 排除以 _ner 结尾的文件夹
            if not dirpath.endswith('_ner'):
                deepest_folders.append(dirpath)

    return deepest_folders


def main():
    """主函数"""
    # parser = argparse.ArgumentParser(description='批量图片NER提取工具')
    # parser.add_argument('input_dir', help='输入图片目录路径')
    # parser.add_argument('-o', '--output', help='输出目录路径（可选）')

    # args = parser.parse_args()

    # input_dir = args.input_dir
    # output_dir = args.output
    # input_dir = '/media/shun/bigdata/Dataset/机动车发票/train'
    # output_dir = '/media/shun/bigdata/Dataset/机动车发票/train_ner'

    # input_dir = '/media/shun/bigdata/Dataset/增值税普通发票/zzsptfp'
    # output_dir = '/media/shun/bigdata/Dataset/增值税普通发票/zzsptfp_ner'

    # input_dir = '/media/shun/bigdata/Dataset/ocr/基于OCR的表单识别数据集/XFUND_ori/zh.train'
    # output_dir = '/media/shun/bigdata/Dataset/ocr/基于OCR的表单识别数据集/XFUND_ori/zh.train_ner'

    # input_dir = '/media/shun/bigdata/Dataset/ocr/基于OCR的表单识别数据集/XFUND_ori/zh.val'
    # output_dir = '/media/shun/bigdata/Dataset/ocr/基于OCR的表单识别数据集/XFUND_ori/zh.val_ner'

    # 设置根目录
    root_dir = '/media/shun/bigdata/Dataset/ocr/DkbRrByl/wildreceipt/'

    # 验证根目录
    if not os.path.exists(root_dir):
        print(f"错误：根目录不存在 - {root_dir}")
        sys.exit(1)

    # 查找所有最底层文件夹
    print(f"正在扫描目录结构，查找最底层文件夹...")
    deepest_folders = find_deepest_folders(root_dir)

    if not deepest_folders:
        print("未找到任何最底层文件夹")
        sys.exit(1)

    print(f"找到 {len(deepest_folders)} 个最底层文件夹")

    # 处理每个最底层文件夹
    for i, input_dir in enumerate(deepest_folders, 1):
        print(f"\n[{i}/{len(deepest_folders)}] 处理文件夹: {input_dir}")

        # 生成对应的输出文件夹路径
        output_dir = input_dir + "_ner"

        # 验证输入目录
        if not os.path.exists(input_dir):
            print(f"警告：输入目录不存在 - {input_dir}，跳过处理")
            continue

        # 执行批量处理
        batch_process_images(input_dir, output_dir)

    print(f"\n所有文件夹处理完成！共处理了 {len(deepest_folders)} 个最底层文件夹")


if __name__ == "__main__":
    main()
