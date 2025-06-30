# Mathematical Calculation Model Evaluation

This directory contains scripts for generating test data and evaluating models on mathematical calculation tasks. The scripts are designed to work with the specific format used in `test_model.py`.

## Files Overview

- `generate_test_data.py` - Generates test data in the format used by `test_model.py`
- `evaluate_model.py` - Evaluates a model on the generated test data
- `run_evaluation.py` - Combined script that generates data and evaluates a model
- `README_evaluation.md` - This documentation file

## Data Format

The test data follows the format used in `test_model.py`:
```
Input: <bos> <unk> [expression] <|eom_id|> <unk>
Expected Answer: [numerical_result]
```

Example:
```
Input: <bos> <unk> 2 + 3 <|eom_id|> <unk>
Expected Answer: 5
```

## Usage

### 1. Generate Test Data

Generate test data for evaluation:

```bash
# Generate 1000 random samples
python generate_test_data.py --num_samples 1000 --output data/test_data.json

# Generate balanced dataset with equal samples per operator
python generate_test_data.py --balanced --samples_per_operator 250 --output data/balanced_test_data.json

# Generate with specific seed for reproducibility
python generate_test_data.py --num_samples 500 --seed 42 --output data/test_data.json

# Generate with custom output directory
python generate_test_data.py --num_samples 1000 --output_dir results/experiment1 --output test_data.json
```

**Options:**
- `--num_samples, -n`: Number of samples to generate (default: 1000)
- `--output, -o`: Output file path (default: data/test_data.json)
- `--output_dir`: Output directory (if specified, output file will be placed in this directory)
- `--format, -f`: Output format: json or jsonl (default: json)
- `--max_num, -m`: Maximum number in calculations (default: 999)
- `--balanced, -b`: Generate balanced dataset with equal samples per operator
- `--samples_per_operator`: Number of samples per operator for balanced dataset (default: 250)
- `--seed`: Random seed for reproducibility

### 2. Evaluate Model

Evaluate a model on the generated test data:

```bash
# Basic evaluation
python evaluate_model.py --model_path saves/your_model --test_data data/test_data.json

# Evaluation with custom parameters
python evaluate_model.py \
    --model_path saves/your_model \
    --test_data data/test_data.json \
    --output results.json \
    --device cuda \
    --max_new_tokens 5 \
    --show_examples 10

# Evaluation with custom output directory
python evaluate_model.py \
    --model_path saves/your_model \
    --test_data data/test_data.json \
    --output_dir results/experiment1 \
    --output evaluation_results.json
```

**Options:**
- `--model_path, -m`: Path to the model to evaluate (required)
- `--test_data, -t`: Path to test data file (required)
- `--output, -o`: Output file for results (default: evaluation_results.json)
- `--output_dir`: Output directory (if specified, output file will be placed in this directory)
- `--device, -d`: Device to use: auto, cuda, cpu (default: auto)
- `--max_new_tokens`: Maximum new tokens to generate (default: 5)
- `--do_sample`: Use sampling instead of greedy decoding
- `--show_examples`: Number of example results to show (default: 5)
- `--save_detailed`: Save detailed results for each sample
- `--save_operator_results`: Save operator-specific results to separate files

### 3. Combined Generation and Evaluation

Use the combined script to generate data and evaluate in one step:

```bash
# Basic usage
python run_evaluation.py --model_path saves/your_model

# Advanced usage
python run_evaluation.py \
    --model_path saves/your_model \
    --num_samples 2000 \
    --balanced \
    --samples_per_operator 500 \
    --device cuda \
    --max_new_tokens 5 \
    --show_examples 10

# Usage with custom output directories
python run_evaluation.py \
    --model_path saves/your_model \
    --output_dir results/experiment1 \
    --test_data_output test_data.json \
    --evaluation_output results.json

# Separate directories for test data and results
python run_evaluation.py \
    --model_path saves/your_model \
    --test_data_dir data/experiment1 \
    --evaluation_dir results/experiment1
```

**Additional Options for Combined Script:**
- `--output_dir`: Output directory for all files (if specified, both test data and results will be placed in this directory)
- `--test_data_dir`: Output directory for test data (overrides --output_dir for test data)
- `--evaluation_dir`: Output directory for evaluation results (overrides --output_dir for evaluation results)

## Output Format

### Test Data Format (JSON)
```json
[
  {
    "input": "<bos> <unk> 2 + 3 <|eom_id|> <unk>",
    "expected_answer": "5",
    "expression": "2 + 3",
    "operator": "+"
  }
]
```

### Evaluation Results Format (JSON)
```json
{
  "overall": {
    "accuracy": 0.85,
    "total_correct": 850,
    "total_samples": 1000,
    "avg_loss": 0.234,
    "avg_perplexity": 1.264
  },
  "operator_metrics": {
    "+": {
      "accuracy": 0.92,
      "correct": 230,
      "total": 250,
      "avg_loss": 0.156,
      "avg_perplexity": 1.169
    }
  }
}
```

## Metrics Explained

- **Accuracy**: Percentage of correct predictions
- **Loss**: Cross-entropy loss for the expected answer
- **Perplexity**: Exponential of the loss, measures how "surprised" the model is
- **Operator-specific metrics**: Performance breakdown by mathematical operator

## Operator-Specific Analysis

The evaluation provides detailed analysis for each mathematical operator (`+`, `-`, `*`, `/`):

### Console Output
- **Operator ranking** by accuracy
- **Detailed breakdown** for each operator including:
  - Accuracy and success rate
  - Average loss and perplexity
  - Correct/total sample counts
- **Incorrect examples** grouped by operator
- **Tabular format** for easy comparison

### Saved Results
When using `--save_operator_results`, the following files are created:
- `operator_summary.json`: Overall summary with operator ranking
- `+_operations.json`: Detailed results for addition operations
- `-_operations.json`: Detailed results for subtraction operations  
- `*_operations.json`: Detailed results for multiplication operations
- `/_operations.json`: Detailed results for division operations
- `incorrect_examples_by_operator.json`: All incorrect examples grouped by operator

### Example Usage
```bash
# Evaluate with operator-specific analysis
python evaluate_model.py \
    --model_path saves/your_model \
    --test_data data/balanced_test.json \
    --output_dir results/experiment1 \
    --save_operator_results \
    --show_examples 10
```

## Example Workflow

1. **Generate balanced test data:**
   ```bash
   python generate_test_data.py --balanced --samples_per_operator 250 --output data/balanced_test.json
   ```

2. **Evaluate your model:**
   ```bash
   python evaluate_model.py --model_path saves/your_model --test_data data/balanced_test.json
   ```

3. **Or use the combined script:**
   ```bash
   python run_evaluation.py --model_path saves/your_model --balanced --samples_per_operator 250
   ```

4. **Organized experiment with custom directories:**
   ```bash
   python run_evaluation.py \
       --model_path saves/your_model \
       --output_dir experiments/math_eval_001 \
       --num_samples 1000 \
       --balanced \
       --samples_per_operator 250
   ```

## Tips

- Use `--balanced` flag to ensure equal representation of all operators
- Set `--seed` for reproducible results
- Use `--save_detailed` to get detailed results for each sample (useful for analysis)
- Adjust `--max_new_tokens` based on your expected answer length
- Use `--device cuda` for faster evaluation on GPU
- Use `--output_dir` to organize your experiments in separate directories
- Use separate `--test_data_dir` and `--evaluation_dir` for more granular control

## Troubleshooting

- **Model loading errors**: Ensure the model path is correct and the model is compatible
- **Memory issues**: Reduce `--num_samples` or use CPU evaluation
- **Token errors**: Make sure the model has the `<|eom_id|>` token added
- **Format errors**: Check that the test data file is valid JSON/JSONL
- **Directory errors**: Ensure you have write permissions for the output directories 