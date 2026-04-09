#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to process NER dataset and convert it to JSONL format.

This script traverses a parent directory to find subdirectories ending with '_ner',
then processes JSON files ending with '_ner.json' in those subdirectories.
Each input JSON file is converted to a line in the output JSONL file.
"""

import os
import json
import argparse
import random
from pathlib import Path


def _flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary.

    Args:
        d (dict): Dictionary to flatten
        parent_key (str): Parent key for nested dictionaries
        sep (str): Separator for nested keys

    Returns:
        dict: Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _randomly_select_fields(data_field):
    """
    Randomly select fields from the data.

    Args:
        data_field (dict): Data field to select from

    Returns:
        list: List of selected top-level keys
    """
    # Get all top-level keys
    all_keys = list(data_field.keys())

    # Randomly select number of fields (at least 1, at most all fields)
    num_fields = random.randint(1, len(all_keys))

    # Randomly select fields
    selected_keys = random.sample(all_keys, num_fields)

    return selected_keys


def _generate_mask_text(selected_keys, data_field=None):
    """
    Generate mask text from selected keys in JSON format.

    Uses different placeholders based on the value type:
    - Single values (str, int, float, bool, etc.) → ""
    - list → []
    - dict (nested JSON) → {}

    Args:
        selected_keys (list): List of selected top-level keys
        data_field (dict, optional): Original data field to inspect value types

    Returns:
        str: Mask text in JSON format
    """
    # Build a dict with selected keys and appropriate placeholders based on value types
    mask_dict = {}

    for key in selected_keys:
        if data_field and key in data_field:
            value = data_field[key]
            # Determine placeholder based on value type
            if isinstance(value, dict):
                mask_dict[key] = {}
            elif isinstance(value, list):
                mask_dict[key] = []
            else:
                # For single values (str, int, float, bool, None, etc.)
                mask_dict[key] = ""
        else:
            # Default to empty string if no data_field provided
            mask_dict[key] = ""

    # Convert to JSON string with proper formatting
    json_str = json.dumps(mask_dict, ensure_ascii=False)

    return f"OCR:{json_str}"


def _generate_no_mask_text(selected_keys, data_field):
    """
    Generate no-mask text from selected keys and data in JSON format.

    Args:
        selected_keys (list): List of selected top-level keys
        data_field (dict): Original data field

    Returns:
        str: No-mask text (JSON string with actual values)
    """
    # Build a dict with selected keys and their actual values
    result_dict = {}
    for key in selected_keys:
        if key in data_field:
            result_dict[key] = data_field[key]

    # Return as formatted JSON string
    return json.dumps(result_dict, ensure_ascii=False)


def process_ner_dataset(parent_dir, output_file, image_root=None, n_entries=1, common_prefix=None, url_root=None):
    """
    Process NER dataset and generate JSONL output.

    Args:
        parent_dir (str): Parent directory to traverse
        output_file (str): Output JSONL file path
        image_root (str, optional): Root directory for image URLs. If provided,
                                     it will be prepended to the relative path.
                                     If not provided, relative path will be used.
        n_entries (int, optional): Number of output entries to generate per input file.
                                   Default is 1 (original behavior).
        common_prefix (str, optional): Common prefix to strip from image paths.
                                      Default is ""
        url_root (str, optional): Root directory to prepend to image_url.
                                  This is applied after processing common_prefix.
        """
    # List to store all output data
    output_data = []

    # Set default common_prefix if not provided
    if common_prefix is None:
        common_prefix = ""

    # Traverse parent directory
    parent_path = Path(parent_dir)

    # Find all subdirectories ending with '_ner' recursively
    for subdir in parent_path.rglob('*_ner'):
        if subdir.is_dir():
            print(f"Processing directory: {subdir.relative_to(parent_path)}")

            # Find JSON files ending with '_ner.json'
            for json_file in subdir.glob('*_ner.json'):
                print(f"Processing file: {json_file.name}")

                # Read input JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    try:
                        input_data = json.load(f)

                        # Extract image URL
                        image_path = input_data.get('image', '')
                        # Process image path according to image_root parameter
                        if image_root:
                            # If image_root is provided, use it as the root directory
                            # Remove the common prefix from image_path and prepend image_root
                            if image_path.startswith(common_prefix):
                                relative_path = image_path[len(common_prefix):]
                                image_url = os.path.join(image_root, relative_path).replace(os.sep, '/')
                            else:
                                # If the path doesn't match the expected prefix, use it as is
                                image_url = image_path
                        else:
                            # If no image_root is provided, use relative path
                            if image_path.startswith(common_prefix):
                                image_url = image_path[len(common_prefix):].lstrip('/')
                            else:
                                # Fallback to original relative path processing
                                if image_path.startswith('/'):
                                    # Find the position of the second '/' to remove the first directory
                                    first_slash = image_path.find('/', 1)
                                    if first_slash != -1:
                                        image_url = image_path[first_slash:].lstrip('/')  # Convert to relative path
                                    else:
                                        image_url = image_path  # Fallback to original if path format is unexpected
                                else:
                                    image_url = image_path

                        # Prepend url_root or './' based on whether url_root is provided
                        if url_root:
                            image_url = os.path.join(url_root, image_url).replace(os.sep, '/')
                        else:
                            # If no url_root, prepend './' to make it a relative path
                            image_url = os.path.join('.', image_url).replace(os.sep, '/')

                        # Extract data field
                        data_field = input_data.get('data', {})

                        # Generate N output entries
                        for i in range(n_entries):
                            if i == 0:
                                # First entry: Keep original behavior (all data fields)
                                output_item = {
                                    "image_info": [
                                        {
                                            "matched_text_index": 0,
                                            "image_url": image_url
                                        }
                                    ],
                                    "text_info": [
                                        {
                                            "text": "OCR:{}",
                                            "tag": "mask"
                                        },
                                        {
                                            "text": json.dumps(data_field, ensure_ascii=False),
                                            "tag": "no_mask"
                                        }
                                    ]
                                }
                            else:
                                # Subsequent entries: Randomly select fields
                                selected_fields = _randomly_select_fields(data_field)
                                mask_text = _generate_mask_text(selected_fields, data_field)
                                no_mask_text = _generate_no_mask_text(selected_fields, data_field)

                                output_item = {
                                    "image_info": [
                                        {
                                            "matched_text_index": 0,
                                            "image_url": image_url
                                        }
                                    ],
                                    "text_info": [
                                        {
                                            "text": mask_text,
                                            "tag": "mask"
                                        },
                                        {
                                            "text": no_mask_text,
                                            "tag": "no_mask"
                                        }
                                    ]
                                }

                            # Add to output data list
                            output_data.append(output_item)

                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {json_file}: {e}")
                        continue

    # Write all output data to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Processed {len(output_data)} files. Output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Process NER dataset and convert to JSONL format')
    parser.add_argument('parent_dir', help='Parent directory to traverse')
    parser.add_argument('-o', '--output', default='output.jsonl', help='Output JSONL file path')
    parser.add_argument('-r', '--image-root', help='Root directory for image URLs')
    parser.add_argument('-n', '--n-entries', type=int, default=1, help='Number of output entries to generate per input file')
    parser.add_argument('-p', '--prefix', help='Common prefix to strip from image paths')
    parser.add_argument('-u', '--url-root', help='Root directory to prepend to image URLs')

    args = parser.parse_args()

    if not os.path.exists(args.parent_dir):
        print(f"Error: Parent directory '{args.parent_dir}' does not exist.")
        return

    process_ner_dataset(args.parent_dir, args.output, args.image_root, args.n_entries, args.prefix, args.url_root)


if __name__ == '__main__':
    main()
