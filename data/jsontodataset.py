#!/usr/bin/env python3
"""
Script to convert a JSON file of strings into a Hugging Face dataset.
The script takes an input JSON file and creates a dataset with a single column named 'article'.
"""

import json
import argparse
from datasets import Dataset


def convert_json_to_dataset(input_file, output_name):
    """
    Convert a JSON file containing strings to a Hugging Face dataset
    
    Args:
        input_file (str): Path to the input JSON file
        output_name (str): Name for the output dataset
    """
    # Load the JSON data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure the data is in the correct format
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of strings")
    
    # Create the dataset dictionary with one column named 'article'
    dataset_dict = {"article": data}
    
    # Create a Hugging Face Dataset object
    dataset = Dataset.from_dict(dataset_dict)
    
    # Save the dataset
    print(f"Saving dataset as {output_name}...")
    dataset.save_to_disk(output_name)
    
    # Print some statistics
    print(f"Dataset created successfully!")
    print(f"Number of examples: {len(dataset)}")
    print(f"Sample data: {dataset[0] if len(dataset) > 0 else 'No data'}")
    
    return dataset


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert JSON file to Hugging Face dataset')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('output_name', help='Name for the output dataset')
    args = parser.parse_args()
    
    # Convert the JSON file to a dataset
    convert_json_to_dataset(args.input_file, args.output_name)


if __name__ == "__main__":
    main()