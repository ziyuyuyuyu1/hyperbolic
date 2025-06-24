#!/usr/bin/env python3
"""
Debug script to understand why training data shows <unk> tokens.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.calc_tokenizer import CalcTokenizer


def debug_training_input():
    """Debug the specific training input that's showing <unk> tokens."""
    print("Debugging the training input with <unk> tokens...")
    
    # The problematic input from the log
    problematic_input = "20 * 7 = <unk> 126 / 18 = <unk> 697 - 107 = <unk> 214 - 12 = <unk> 24 * 39 = <unk> 399 / 21 = <unk> 23 * 6 = <unk> 36 / 36 = <unk> 611 - 217 = <unk> 26 * 3 = <unk> 757 - 73 = <unk> 512 / 32 = <unk> 9 * 41 ="
    
    print(f"Problematic input: '{problematic_input}'")
    
    # Initialize tokenizer
    tokenizer = CalcTokenizer()
    
    # Tokenize the input
    tokens = tokenizer._tokenize(problematic_input)
    token_ids = tokenizer.encode(problematic_input, add_special_tokens=False)
    
    print(f"\nTokenization result:")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    # Check which tokens are <unk>
    unk_count = 0
    for i, token in enumerate(tokens):
        if token == "<unk>":
            unk_count += 1
            print(f"  <unk> found at position {i}")
        elif token not in tokenizer.vocab:
            print(f"  Token '{token}' not in vocabulary!")
    
    print(f"\nTotal <unk> tokens: {unk_count}")
    
    # Check vocabulary size and some key tokens
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    
    # Check if key numbers are in vocabulary
    test_numbers = ["20", "7", "126", "18", "697", "107", "214", "12", "24", "39", "399", "21", "23", "6", "36", "611", "217", "26", "3", "757", "73", "512", "32", "9", "41"]
    
    missing_numbers = []
    for num in test_numbers:
        if num not in tokenizer.vocab:
            missing_numbers.append(num)
        else:
            print(f"  '{num}': ID {tokenizer.vocab[num]}")
    
    if missing_numbers:
        print(f"\nMissing numbers: {missing_numbers}")
    else:
        print(f"\nAll test numbers are in vocabulary ✓")


def check_vocabulary_construction():
    """Check how the vocabulary is being constructed."""
    print("\n" + "="*60)
    print("CHECKING VOCABULARY CONSTRUCTION")
    print("="*60)
    
    tokenizer = CalcTokenizer()
    
    # Check the order of vocabulary construction
    print("Vocabulary construction order:")
    
    # Find the first few entries
    first_entries = []
    for i in range(20):
        if i in tokenizer.ids_to_tokens:
            first_entries.append((i, tokenizer.ids_to_tokens[i]))
    
    print("First 20 vocabulary entries:")
    for token_id, token in first_entries:
        print(f"  ID {token_id}: '{token}'")
    
    # Check if numbers are properly included
    print(f"\nChecking number range 0-999:")
    missing_numbers = []
    for i in range(1000):
        num_str = str(i)
        if num_str not in tokenizer.vocab:
            missing_numbers.append(num_str)
            if len(missing_numbers) <= 10:  # Only show first 10 missing
                print(f"  Missing: '{num_str}'")
    
    if len(missing_numbers) > 10:
        print(f"  ... and {len(missing_numbers) - 10} more missing numbers")
    
    if not missing_numbers:
        print("  ✓ All numbers 0-999 are in vocabulary")
    else:
        print(f"  ✗ {len(missing_numbers)} numbers are missing from vocabulary")


def test_specific_numbers():
    """Test specific numbers that should be in the vocabulary."""
    print("\n" + "="*60)
    print("TESTING SPECIFIC NUMBERS")
    print("="*60)
    
    tokenizer = CalcTokenizer()
    
    # Test numbers from the problematic input
    test_cases = [
        "20 * 7 = 140",
        "126 / 18 = 7", 
        "697 - 107 = 590",
        "214 - 12 = 202",
        "24 * 39 = 936",
        "399 / 21 = 19",
        "23 * 6 = 138",
        "36 / 36 = 1",
        "611 - 217 = 394",
        "26 * 3 = 78",
        "757 - 73 = 684",
        "512 / 32 = 16",
        "9 * 41 = 369"
    ]
    
    for expr in test_cases:
        print(f"\nTesting: '{expr}'")
        
        tokens = tokenizer._tokenize(expr)
        token_ids = tokenizer.encode(expr, add_special_tokens=False)
        
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        
        # Check for any <unk> tokens
        if "<unk>" in tokens:
            print(f"  ✗ <unk> found in tokens!")
        else:
            print(f"  ✓ No <unk> tokens")


def check_tokenizer_registration():
    """Check if the tokenizer is properly registered."""
    print("\n" + "="*60)
    print("CHECKING TOKENIZER REGISTRATION")
    print("="*60)
    
    try:
        from transformers import AutoTokenizer
        
        # Try to load the tokenizer by name
        tokenizer = AutoTokenizer.from_pretrained("calc")
        print("✓ Tokenizer can be loaded via AutoTokenizer")
        
        # Test basic functionality
        test_expr = "20 * 7 = 140"
        tokens = tokenizer._tokenize(test_expr)
        print(f"Test expression: '{test_expr}'")
        print(f"Tokens: {tokens}")
        
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")


if __name__ == "__main__":
    debug_training_input()
    check_vocabulary_construction()
    test_specific_numbers()
    check_tokenizer_registration() 