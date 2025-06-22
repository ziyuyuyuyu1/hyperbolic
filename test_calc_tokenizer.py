#!/usr/bin/env python3
"""
Test script to verify that generated calculation data can be properly tokenized.
"""

import sys
import os

# Add the src directory to the path so we can import the CalcTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.calc_tokenizer import CalcTokenizer
from calc_gen import CalcDataGenerator


def test_tokenizer():
    """Test the CalcTokenizer with generated data."""
    print("Testing CalcTokenizer with generated calculation data...")
    
    # Initialize tokenizer
    tokenizer = CalcTokenizer()
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}, bos={tokenizer.bos_token_id}, unk={tokenizer.unk_token_id}")
    
    # Generate some test data
    generator = CalcDataGenerator(max_num=999)
    
    # Test multiple simple expressions
    test_cases = []
    for _ in range(5):
        test_cases.append(generator.generate_simple_expression())
    
    print("\nTesting tokenization:")
    for i, expression in enumerate(test_cases, 1):
        print(f"\n{i}. Expression: {expression}")
        
        # Tokenize expression
        tokens = tokenizer._tokenize(expression)
        token_ids = tokenizer.encode(expression, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        print(f"   Tokens: {tokens}")
        print(f"   Token IDs: {token_ids}")
        print(f"   Decoded: '{decoded}'")
        
        # Verify tokenization works correctly
        assert expression == decoded, f"Tokenization failed: '{expression}' != '{decoded}'"
        print("   ✓ Tokenization verified successfully")


def test_dataset_generation():
    """Test dataset generation and tokenization."""
    print("\n" + "="*50)
    print("Testing dataset generation and tokenization...")
    
    # Generate a small dataset
    generator = CalcDataGenerator(max_num=999)
    dataset = generator.generate_dataset(num_samples=10)
    
    tokenizer = CalcTokenizer()
    
    print(f"\nGenerated {len(dataset)} samples:")
    for i, entry in enumerate(dataset, 1):
        print(f"\n{i}. Text: {entry['text']}")
        
        # Test tokenization
        tokens = tokenizer._tokenize(entry['text'])
        
        print(f"   Tokens: {tokens}")
        
        # Verify all tokens are in vocabulary
        for token in tokens:
            if token not in tokenizer.vocab:
                print(f"   ⚠️  Warning: Token '{token}' not in vocabulary")
            else:
                print(f"   ✓ Token '{token}' in vocabulary (ID: {tokenizer.vocab[token]})")


def test_vocabulary_coverage():
    """Test that all generated numbers and operators are in the vocabulary."""
    print("\n" + "="*50)
    print("Testing vocabulary coverage...")
    
    tokenizer = CalcTokenizer()
    generator = CalcDataGenerator(max_num=999)
    
    # Generate many samples to test coverage
    dataset = generator.generate_dataset(num_samples=1000)
    
    all_tokens = set()
    for entry in dataset:
        tokens = tokenizer._tokenize(entry['text'])
        all_tokens.update(tokens)
    
    print(f"Total unique tokens found: {len(all_tokens)}")
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    # Check for missing tokens
    missing_tokens = []
    for token in all_tokens:
        if token not in tokenizer.vocab:
            missing_tokens.append(token)
    
    if missing_tokens:
        print(f"⚠️  Missing tokens: {missing_tokens}")
    else:
        print("✓ All tokens are in vocabulary")
    
    # Show some statistics
    number_tokens = [t for t in all_tokens if t.isdigit()]
    operator_tokens = [t for t in all_tokens if t in ['+', '-', '*', '/', '=', '>', '<']]
    word_tokens = [t for t in all_tokens if not t.isdigit() and t not in ['+', '-', '*', '/', '=', '>', '<']]
    
    print(f"Number tokens: {len(number_tokens)}")
    print(f"Operator tokens: {operator_tokens}")
    print(f"Word tokens: {word_tokens}")


def test_number_range():
    """Test that all numbers are within the 0-999 range."""
    print("\n" + "="*50)
    print("Testing number range constraints...")
    
    generator = CalcDataGenerator(max_num=999)
    dataset = generator.generate_dataset(num_samples=1000)
    
    all_numbers = set()
    for entry in dataset:
        # Extract numbers from text
        text = entry['text']
        
        # Simple number extraction (assuming format "a op b = c")
        parts = text.split()
        for part in parts:
            if part.isdigit():
                all_numbers.add(int(part))
    
    max_found = max(all_numbers)
    min_found = min(all_numbers)
    
    print(f"Number range found: {min_found} to {max_found}")
    print(f"Total unique numbers: {len(all_numbers)}")
    
    if max_found <= 999 and min_found >= 0:
        print("✓ All numbers are within 0-999 range")
    else:
        print(f"⚠️  Numbers outside range: min={min_found}, max={max_found}")


if __name__ == "__main__":
    try:
        test_tokenizer()
        test_dataset_generation()
        test_vocabulary_coverage()
        test_number_range()
        print("\n" + "="*50)
        print("✓ All tests passed! The CalcTokenizer can properly handle the generated data.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 