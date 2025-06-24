#!/usr/bin/env python3
"""
Debug script to simulate the pretraining processor logic.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.calc_tokenizer import CalcTokenizer


def simulate_pretrain_processor():
    """Simulate the exact pretraining processor logic."""
    print("="*80)
    print("SIMULATING PRETRAIN PROCESSOR LOGIC")
    print("="*80)
    
    # Initialize tokenizer (same as in training)
    tokenizer = CalcTokenizer()
    
    # Simulate the data structure that comes from the dataset
    # Based on the log output, this is what the data looks like
    examples = {
        "_prompt": [
            [{"content": "20 * 7 = <unk> 126 / 18 = <unk> 697 - 107 = <unk> 214 - 12 = <unk> 24 * 39 = <unk> 399 / 21 = <unk> 23 * 6 = <unk> 36 / 36 = <unk> 611 - 217 = <unk> 26 * 3 = <unk> 757 - 73 = <unk> 512 / 32 = <unk> 9 * 41 ="}],
            [{"content": "15 + 25 = 40 100 - 30 = 70 8 * 9 = 72 200 / 10 = 20"}],
            [{"content": "5 + 3 = 8 12 - 4 = 8 6 * 7 = 42 50 / 5 = 10"}]
        ]
    }
    
    print(f"Number of examples: {len(examples['_prompt'])}")
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Tokenizer vocab size: {getattr(tokenizer, 'vocab_size', 'N/A')}")
    
    # Check tokenizer vocabulary for key tokens
    if hasattr(tokenizer, 'vocab'):
        print(f"Sample vocabulary entries:")
        vocab = tokenizer.vocab
        for token in ['<unk>', '0', '1', '2', '20', '140', '+', '-', '*', '/', '=']:
            if token in vocab:
                print(f"  '{token}': ID {vocab[token]}")
            else:
                print(f"  '{token}': NOT FOUND!")
    
    print("\nOriginal examples:")
    for i, messages in enumerate(examples["_prompt"]):
        print(f"  {i}: '{messages[0]['content']}'")
    
    # Step 1: Extract content and add EOS token (line 25 in pretrain.py)
    eos_token = "<eos>"  # Assuming not using llama3 template
    text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]
    
    print(f"\nAfter adding EOS token:")
    for i, text in enumerate(text_examples):
        print(f"  {i}: '{text}'")
    
    # Step 2: Add BOS token if needed (lines 33-36 in pretrain.py)
    if getattr(tokenizer, "add_bos_token", False):
        text_examples = [tokenizer.bos_token + example for example in text_examples]
        print(f"\nAfter adding BOS token:")
        for i, text in enumerate(text_examples):
            print(f"  {i}: '{text}'")
    
    # Step 3: Tokenize with add_special_tokens=False (lines 38-41 in pretrain.py)
    print(f"\nTokenizing with add_special_tokens=False...")
    
    # DEBUG: Test tokenization step by step for each example
    for i, text in enumerate(text_examples):
        print(f"\n--- EXAMPLE {i} ---")
        print(f"Text: '{text}'")
        
        # Manual tokenization
        if hasattr(tokenizer, '_tokenize'):
            manual_tokens = tokenizer._tokenize(text)
            print(f"Manual tokens: {manual_tokens}")
            
            # Check each token
            print(f"Token check:")
            for j, token in enumerate(manual_tokens):
                if hasattr(tokenizer, 'vocab') and token in tokenizer.vocab:
                    token_id = tokenizer.vocab[token]
                    print(f"  [{j}] '{token}': ID {token_id}")
                else:
                    print(f"  [{j}] '{token}': NOT IN VOCABULARY!")
    
    # Full tokenization
    result = tokenizer(
        text_examples, 
        add_special_tokens=False, 
        truncation=True, 
        max_length=512  # Example cutoff length
    )
    
    # DEBUG: Check for <unk> tokens in result
    print(f"\n" + "="*60)
    print("FINAL TOKENIZATION RESULTS")
    print("="*60)
    
    for i, (text, input_ids) in enumerate(zip(text_examples, result["input_ids"])):
        print(f"\nExample {i}:")
        print(f"  Original: '{text}'")
        print(f"  Token IDs: {input_ids}")
        
        # Convert back to tokens to check for <unk>
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        unk_count = tokens.count("<unk>")
        if unk_count > 0:
            print(f"  ⚠️  Found {unk_count} <unk> tokens!")
            unk_positions = [j for j, token in enumerate(tokens) if token == "<unk>"]
            print(f"  <unk> positions: {unk_positions}")
            # Show tokens around <unk> positions
            for pos in unk_positions:
                start = max(0, pos-2)
                end = min(len(tokens), pos+3)
                context = tokens[start:end]
                print(f"  Context around position {pos}: {context}")
        else:
            print(f"  ✓ No <unk> tokens")
        
        # Decode to see final result
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"  Decoded: '{decoded}'")
    
    print("="*80)
    return result


def analyze_unk_pattern():
    """Analyze the pattern of <unk> tokens to understand the issue."""
    print(f"\n" + "="*80)
    print("ANALYZING <UNK> PATTERN")
    print("="*80)
    
    # The problematic input from the log
    problematic_input = "20 * 7 = <unk> 126 / 18 = <unk> 697 - 107 = <unk> 214 - 12 = <unk> 24 * 39 = <unk> 399 / 21 = <unk> 23 * 6 = <unk> 36 / 36 = <unk> 611 - 217 = <unk> 26 * 3 = <unk> 757 - 73 = <unk> 512 / 32 = <unk> 9 * 41 ="
    
    print(f"Problematic input: '{problematic_input}'")
    
    # Split into individual expressions
    expressions = problematic_input.split()
    
    print(f"\nBreaking down into expressions:")
    current_expr = []
    expressions_list = []
    
    for token in expressions:
        current_expr.append(token)
        if token == "=":
            # Next token should be the result
            expressions_list.append(" ".join(current_expr))
            current_expr = []
    
    if current_expr:
        expressions_list.append(" ".join(current_expr))
    
    print(f"Found {len(expressions_list)} expressions:")
    for i, expr in enumerate(expressions_list):
        print(f"  {i+1}: '{expr}'")
    
    # Analyze the pattern
    print(f"\nPattern analysis:")
    print(f"  - All expressions end with '='")
    print(f"  - The result after '=' is always '<unk>'")
    print(f"  - This suggests the data generation process is broken")
    print(f"  - The results are being replaced with '<unk>' instead of actual numbers")


def test_correct_data():
    """Test with correctly formatted data to verify tokenizer works."""
    print(f"\n" + "="*80)
    print("TESTING WITH CORRECT DATA")
    print("="*80)
    
    tokenizer = CalcTokenizer()
    
    # Test with correctly formatted expressions
    correct_expressions = [
        "20 * 7 = 140 126 / 18 = 7 697 - 107 = 590",
        "214 - 12 = 202 24 * 39 = 936 399 / 21 = 19",
        "23 * 6 = 138 36 / 36 = 1 611 - 217 = 394"
    ]
    
    for i, expr in enumerate(correct_expressions):
        print(f"\nTesting correct expression {i+1}: '{expr}'")
        
        # Add EOS token
        expr_with_eos = expr + "<eos>"
        
        # Tokenize
        tokens = tokenizer._tokenize(expr_with_eos)
        token_ids = tokenizer.encode(expr_with_eos, add_special_tokens=False)
        
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        
        # Check for <unk>
        unk_count = tokens.count("<unk>")
        if unk_count > 0:
            print(f"  ⚠️  Found {unk_count} <unk> tokens!")
        else:
            print(f"  ✓ No <unk> tokens")
        
        # Decode
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"  Decoded: '{decoded}'")


if __name__ == "__main__":
    simulate_pretrain_processor()
    analyze_unk_pattern()
    test_correct_data() 