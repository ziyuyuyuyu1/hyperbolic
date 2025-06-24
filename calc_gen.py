#!/usr/bin/env python3
"""
Script to generate calculation data for LLaMA Factory pretraining.
Generates simple mathematical expressions with numbers 0-999 and operators +, -, *, /, =.
Format: a op b = c
"""

import json
import random
import argparse
from typing import List, Dict, Any
import sys
import os


class CalcDataGenerator:
    """Generate simple calculation data for pretraining or SFT."""
    
    def __init__(self, max_num: int = 999):
        self.max_num = max_num
        self.operators = ["+", "-", "*", "/"]
        # self.operators = ["+", "-"]
        
    def generate_simple_expression(self) -> (str, str, str):
        """Generate a simple two-number expression: a op b = c. Returns (expression, result_str, full_eqn)."""
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
        
        expr = f"{a} {op} {b}"
        full_eqn = f"{expr} = {result}"
        return expr, str(result), full_eqn, result
    
    def generate_dataset(self, num_samples: int, format_type: str = "json") -> List[Dict[str, Any]]:
        """Generate a dataset of simple calculation examples for pretraining."""
        dataset = []
        
        for i in range(num_samples):
            expression = self.generate_simple_expression()
            
            # Create data entry for pretraining - single text field
            entry = {
                "text": expression[0]
            }
            
            dataset.append(entry)
        
        return dataset
    
    def generate_sft_dataset(self, num_samples: int) -> list:
        """Generate SFT-style dataset for math: instruction, input, output."""
        dataset = []
        for _ in range(num_samples):
            expr, _, _, result = self.generate_simple_expression()
            entry = {
                "instruction": expr,
                "input": "",
                "output": str(result)
            }
            dataset.append(entry)
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_file: str, format_type: str = "json"):
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
        
        print(f"Generated {len(dataset)} samples and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate simple calculation or SFT identity data for LLaMA Factory")
    parser.add_argument("--num_samples", "-n", type=int, default=1000, 
                       help="Number of samples to generate or select (default: 1000)")
    parser.add_argument("--output", "-o", type=str, default="data/calc_pretrain.json",
                       help="Output file path (default: data/calc_pretrain.json)")
    parser.add_argument("--format", "-f", type=str, choices=["json", "jsonl"], default="json",
                       help="Output format (default: json)")
    parser.add_argument("--max_num", "-m", type=int, default=999,
                       help="Maximum number in calculations (default: 999)")
    parser.add_argument("--mode", type=str, choices=["calc", "sft-identity", "sft"], default="calc",
                       help="Generation mode: 'calc' for calculation data, 'sft' for SFT math data, 'sft-identity' for SFT identity data (default: calc)")
    parser.add_argument("--identity_json", type=str, default="data/identity.json",
                       help="Path to identity.json for SFT identity mode (default: data/identity.json)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling (default: None)")
    
    args = parser.parse_args()
    
    if args.mode == "sft-identity":
        # SFT identity mode: load identity.json, shuffle/sample, and save
        with open(args.identity_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if args.seed is not None:
            random.seed(args.seed)
        random.shuffle(data)
        if args.num_samples < len(data):
            data = data[:args.num_samples]
        # Save in requested format
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        if args.format.lower() == "json":
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif args.format.lower() == "jsonl":
            with open(args.output, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"Unsupported format: {args.format}")
        print(f"Generated {len(data)} SFT identity samples and saved to {args.output}")
        print("\nExample samples:")
        for i, entry in enumerate(data[:5]):
            print(f"\n{i+1}. Instruction: {entry['instruction']}\n   Input: {entry['input']}\n   Output: {entry['output']}")
        return
    
    if args.mode == "sft":
        generator = CalcDataGenerator(max_num=args.max_num)
        print(f"Generating {args.num_samples} SFT math samples...")
        dataset = generator.generate_sft_dataset(num_samples=args.num_samples)
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        if args.format.lower() == "json":
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
        elif args.format.lower() == "jsonl":
            with open(args.output, 'w', encoding='utf-8') as f:
                for entry in dataset:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"Unsupported format: {args.format}")
        print(f"Generated {len(dataset)} SFT math samples and saved to {args.output}")
        print("\nExample samples:")
        for i, entry in enumerate(dataset[:5]):
            print(f"\n{i+1}. Instruction: {entry['instruction']}\n   Input: {entry['input']}\n   Output: {entry['output']}")
        return
    
    # Default: calculation data generation
    generator = CalcDataGenerator(max_num=args.max_num)
    print(f"Generating {args.num_samples} simple calculation samples for pretraining...")
    dataset = generator.generate_dataset(
        num_samples=args.num_samples,
        format_type=args.format
    )
    generator.save_dataset(dataset, args.output, args.format)
    print("\nExample samples:")
    for i, entry in enumerate(dataset[:5]):
        print(f"\n{i+1}. Text: {entry['text']}")


if __name__ == "__main__":
    main()
