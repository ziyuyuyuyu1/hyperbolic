#!/usr/bin/env python3
"""
Script to generate test data for model evaluation in the format used by test_model.py.
Format: <bos> <unk> [expression] <|eom_id|> <unk> with expected numerical answer.
"""

import json
import random
import argparse
from typing import List, Dict, Any, Tuple
import sys
import os


class TestDataGenerator:
    """Generate test data for model evaluation in the specific format used by test_model.py."""
    
    def __init__(self, max_num: int = 999):
        self.max_num = max_num
        self.operators = ["+", "-", "*", "/"]
        
    def generate_expression(self) -> Tuple[str, str, int]:
        """Generate a mathematical expression and its result. Returns (expression, result_str, result_int)."""
        op = random.choice(self.operators)
        
        if op == "+":
            # For addition, ensure result doesn't exceed max_num
            a = random.randint(0, self.max_num)
            b = random.randint(0, self.max_num - a)
            result = a + b
        elif op == "-":
            # For subtraction, ensure result is non-negative
            a = random.randint(0, self.max_num)
            b = random.randint(0, a)
            result = a - b
        elif op == "*":
            # For multiplication, ensure result doesn't exceed max_num
            a = random.randint(0, min(self.max_num, int(self.max_num ** 0.5)))
            b = random.randint(0, min(self.max_num // max(a, 1), self.max_num))
            result = a * b
        elif op == "/":
            # For division, ensure clean integer division and result doesn't exceed max_num
            b = random.randint(1, min(50, self.max_num))  # Smaller divisor for cleaner results
            result = random.randint(0, min(self.max_num // b, self.max_num))
            a = result * b  # Ensure clean division
        
        expression = f"{a} {op} {b}"
        return expression, str(result), result
    
    def generate_test_sample(self) -> Dict[str, str]:
        """Generate a single test sample in the format used by test_model.py."""
        expression, result, _ = self.generate_expression()
        
        # Format: <bos> <unk> [expression] <|eom_id|> <unk>
        input_text = f"<bos> <unk> {expression} <|eom_id|> <unk>"
        
        return {
            "input": input_text,
            "expected_answer": result,
            "expression": expression
        }
    
    def generate_dataset(self, num_samples: int) -> List[Dict[str, str]]:
        """Generate a dataset of test samples."""
        dataset = []
        
        for _ in range(num_samples):
            sample = self.generate_test_sample()
            dataset.append(sample)
        
        return dataset
    
    def generate_balanced_dataset(self, samples_per_operator: int = 250) -> List[Dict[str, str]]:
        """Generate a balanced dataset with equal samples per operator."""
        dataset = []
        
        for op in self.operators:
            for _ in range(samples_per_operator):
                # Force specific operator
                if op == "+":
                    a = random.randint(0, self.max_num)
                    b = random.randint(0, self.max_num - a)
                    result = a + b
                elif op == "-":
                    a = random.randint(0, self.max_num)
                    b = random.randint(0, a)
                    result = a - b
                elif op == "*":
                    a = random.randint(0, min(self.max_num, int(self.max_num ** 0.5)))
                    b = random.randint(0, min(self.max_num // max(a, 1), self.max_num))
                    result = a * b
                elif op == "/":
                    b = random.randint(1, min(50, self.max_num))
                    result = random.randint(0, min(self.max_num // b, self.max_num))
                    a = result * b
                
                expression = f"{a} {op} {b}"
                input_text = f"<bos> <unk> {expression} <|eom_id|> <unk>"
                
                sample = {
                    "input": input_text,
                    "expected_answer": str(result),
                    "expression": expression,
                    "operator": op
                }
                dataset.append(sample)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, str]], output_file: str, format_type: str = "json"):
        """Save the dataset to a file."""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        if format_type.lower() == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
        elif format_type.lower() == "jsonl":
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in dataset:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        print(f"Generated {len(dataset)} test samples and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate test data for model evaluation")
    parser.add_argument("--num_samples", "-n", type=int, default=1000, 
                       help="Number of samples to generate (default: 1000)")
    parser.add_argument("--output", "-o", type=str, default="data/test_data.json",
                       help="Output file path (default: data/test_data.json)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (if specified, output file will be placed in this directory)")
    parser.add_argument("--format", "-f", type=str, choices=["json", "jsonl"], default="json",
                       help="Output format (default: json)")
    parser.add_argument("--max_num", "-m", type=int, default=999,
                       help="Maximum number in calculations (default: 999)")
    parser.add_argument("--balanced", "-b", action="store_true",
                       help="Generate balanced dataset with equal samples per operator")
    parser.add_argument("--samples_per_operator", type=int, default=250,
                       help="Number of samples per operator for balanced dataset (default: 250)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None)")
    
    args = parser.parse_args()
    
    # Handle output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # Extract filename from output path
        filename = os.path.basename(args.output)
        output_path = os.path.join(args.output_dir, filename)
    else:
        output_path = args.output
    
    if args.seed is not None:
        random.seed(args.seed)
    
    generator = TestDataGenerator(max_num=args.max_num)
    
    if args.balanced:
        print(f"Generating balanced dataset with {args.samples_per_operator} samples per operator...")
        dataset = generator.generate_balanced_dataset(samples_per_operator=args.samples_per_operator)
    else:
        print(f"Generating {args.num_samples} test samples...")
        dataset = generator.generate_dataset(num_samples=args.num_samples)
    
    generator.save_dataset(dataset, output_path, args.format)
    
    print("\nExample samples:")
    for i, entry in enumerate(dataset[:5]):
        print(f"\n{i+1}. Input: {entry['input']}")
        print(f"   Expression: {entry['expression']}")
        print(f"   Expected Answer: {entry['expected_answer']}")
        if 'operator' in entry:
            print(f"   Operator: {entry['operator']}")
    
    # Print statistics
    if 'operator' in dataset[0]:
        operator_counts = {}
        for entry in dataset:
            op = entry['operator']
            operator_counts[op] = operator_counts.get(op, 0) + 1
        
        print(f"\nOperator distribution:")
        for op, count in operator_counts.items():
            print(f"  {op}: {count} samples")


if __name__ == "__main__":
    main() 