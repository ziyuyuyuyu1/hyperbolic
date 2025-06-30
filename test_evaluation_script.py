#!/usr/bin/env python3
"""
Test script to verify the evaluation script works correctly.
This script tests the data loading and processing without loading a full model.
"""

import json
import sys
import os

def load_test_data(data_file):
    """Load test data from file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        if data_file.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    return data

def test_data_loading():
    """Test that the test data can be loaded correctly."""
    print("Testing data loading...")
    
    # Test loading the sample data
    test_files = [
        "data/test_sample.json",
        "data/balanced_test_sample.json"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nLoading {test_file}...")
            try:
                data = load_test_data(test_file)
                print(f"✓ Successfully loaded {len(data)} samples")
                
                # Check format
                if len(data) > 0:
                    sample = data[0]
                    required_fields = ["input", "expected_answer", "expression"]
                    missing_fields = [field for field in required_fields if field not in sample]
                    
                    if missing_fields:
                        print(f"✗ Missing required fields: {missing_fields}")
                    else:
                        print(f"✓ Data format is correct")
                        print(f"  Sample input: {sample['input']}")
                        print(f"  Sample expression: {sample['expression']}")
                        print(f"  Sample expected answer: {sample['expected_answer']}")
                        
                        if 'operator' in sample:
                            print(f"  Sample operator: {sample['operator']}")
                
            except Exception as e:
                print(f"✗ Failed to load {test_file}: {e}")
        else:
            print(f"✗ Test file {test_file} not found")

def test_data_format():
    """Test that the data format matches the expected format from test_model.py."""
    print("\nTesting data format compatibility...")
    
    # Load the test data
    if os.path.exists("data/test_sample.json"):
        data = load_test_data("data/test_sample.json")
        
        # Check if the format matches test_model.py expectations
        expected_format = "<bos> <unk> [expression] <|eom_id|> <unk>"
        
        for i, sample in enumerate(data[:3]):  # Check first 3 samples
            input_text = sample['input']
            expression = sample['expression']
            expected_answer = sample['expected_answer']
            
            # Check if input contains the expected tokens
            has_bos = '<bos>' in input_text
            has_unk = '<unk>' in input_text
            has_eom = '<|eom_id|>' in input_text
            has_expression = expression in input_text
            
            print(f"\nSample {i+1}:")
            print(f"  Input: {input_text}")
            print(f"  Expression: {expression}")
            print(f"  Expected Answer: {expected_answer}")
            print(f"  Format check:")
            print(f"    ✓ Contains <bos>: {has_bos}")
            print(f"    ✓ Contains <unk>: {has_unk}")
            print(f"    ✓ Contains <|eom_id|>: {has_eom}")
            print(f"    ✓ Contains expression: {has_expression}")
            
            if all([has_bos, has_unk, has_eom, has_expression]):
                print(f"    ✓ Format is compatible with test_model.py")
            else:
                print(f"    ✗ Format is NOT compatible with test_model.py")

def test_balanced_dataset():
    """Test that the balanced dataset has equal distribution."""
    print("\nTesting balanced dataset distribution...")
    
    if os.path.exists("data/balanced_test_sample.json"):
        data = load_test_data("data/balanced_test_sample.json")
        
        # Count operators
        operator_counts = {}
        for sample in data:
            if 'operator' in sample:
                op = sample['operator']
                operator_counts[op] = operator_counts.get(op, 0) + 1
        
        print(f"Operator distribution:")
        for op, count in operator_counts.items():
            print(f"  {op}: {count} samples")
        
        # Check if balanced
        if len(set(operator_counts.values())) == 1:
            print("✓ Dataset is perfectly balanced")
        else:
            print("✗ Dataset is not balanced")
            print(f"  Expected equal counts, got: {operator_counts}")

def main():
    print("="*60)
    print("EVALUATION SCRIPT TESTING")
    print("="*60)
    
    # Test data loading
    test_data_loading()
    
    # Test data format
    test_data_format()
    
    # Test balanced dataset
    test_balanced_dataset()
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)
    print("\nTo run full evaluation, use:")
    print("python evaluate_model.py --model_path saves/your_model --test_data data/test_data.json")
    print("\nOr use the combined script:")
    print("python run_evaluation.py --model_path saves/your_model")

if __name__ == "__main__":
    main() 