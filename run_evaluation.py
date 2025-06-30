#!/usr/bin/env python3
"""
Demo script showing how to use generate_test_data.py and evaluate_model.py together.
This script generates test data and then evaluates a model on it.
"""

import subprocess
import sys
import os
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Command completed successfully")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate test data and evaluate a model")
    parser.add_argument("--model_path", "-m", type=str, required=True,
                       help="Path to the model to evaluate")
    parser.add_argument("--num_samples", "-n", type=int, default=1000,
                       help="Number of test samples to generate (default: 1000)")
    parser.add_argument("--balanced", "-b", action="store_true",
                       help="Generate balanced dataset with equal samples per operator")
    parser.add_argument("--samples_per_operator", type=int, default=250,
                       help="Number of samples per operator for balanced dataset (default: 250)")
    parser.add_argument("--test_data_output", type=str, default="data/test_data.json",
                       help="Output file for test data (default: data/test_data.json)")
    parser.add_argument("--evaluation_output", type=str, default="evaluation_results.json",
                       help="Output file for evaluation results (default: evaluation_results.json)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for all files (if specified, both test data and results will be placed in this directory)")
    parser.add_argument("--test_data_dir", type=str, default=None,
                       help="Output directory for test data (overrides --output_dir for test data)")
    parser.add_argument("--evaluation_dir", type=str, default=None,
                       help="Output directory for evaluation results (overrides --output_dir for evaluation results)")
    parser.add_argument("--max_new_tokens", type=int, default=5,
                       help="Maximum new tokens to generate (default: 5)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for evaluation (default: auto)")
    parser.add_argument("--show_examples", type=int, default=5,
                       help="Number of example results to show (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for data generation (default: 42)")
    
    args = parser.parse_args()
    
    # Handle output directories
    test_data_dir = args.test_data_dir or args.output_dir
    evaluation_dir = args.evaluation_dir or args.output_dir
    
    # Prepare output paths
    if test_data_dir:
        os.makedirs(test_data_dir, exist_ok=True)
        test_data_filename = os.path.basename(args.test_data_output)
        test_data_output_path = os.path.join(test_data_dir, test_data_filename)
    else:
        test_data_output_path = args.test_data_output
    
    if evaluation_dir:
        os.makedirs(evaluation_dir, exist_ok=True)
        evaluation_filename = os.path.basename(args.evaluation_output)
        evaluation_output_path = os.path.join(evaluation_dir, evaluation_filename)
    else:
        evaluation_output_path = args.evaluation_output
    
    print("="*60)
    print("MATHEMATICAL CALCULATION MODEL EVALUATION")
    print("="*60)
    
    # Step 1: Generate test data
    generate_cmd = [
        sys.executable, "generate_test_data.py",
        "--num_samples", str(args.num_samples),
        "--output", test_data_output_path,
        "--seed", str(args.seed)
    ]
    
    if test_data_dir:
        generate_cmd.extend(["--output_dir", test_data_dir])
    
    if args.balanced:
        generate_cmd.extend(["--balanced", "--samples_per_operator", str(args.samples_per_operator)])
    
    if not run_command(generate_cmd, "Step 1: Generating test data"):
        print("Failed to generate test data. Exiting.")
        return 1
    
    # Step 2: Evaluate model
    evaluate_cmd = [
        sys.executable, "evaluate_model.py",
        "--model_path", args.model_path,
        "--test_data", test_data_output_path,
        "--output", evaluation_output_path,
        "--device", args.device,
        "--max_new_tokens", str(args.max_new_tokens),
        "--show_examples", str(args.show_examples)
    ]
    
    if evaluation_dir:
        evaluate_cmd.extend(["--output_dir", evaluation_dir])
    
    if not run_command(evaluate_cmd, "Step 2: Evaluating model"):
        print("Failed to evaluate model. Exiting.")
        return 1
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Test data saved to: {test_data_output_path}")
    print(f"Evaluation results saved to: {evaluation_output_path}")
    
    # Step 3: Show summary
    print(f"\nSummary:")
    print(f"  Model: {args.model_path}")
    print(f"  Test samples: {args.num_samples}")
    if args.balanced:
        print(f"  Balanced dataset: {args.samples_per_operator} samples per operator")
    print(f"  Device: {args.device}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 