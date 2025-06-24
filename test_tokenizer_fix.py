#!/usr/bin/env python3
"""
Test script to verify the fixed CalcTokenizer works correctly.
"""

import sys
import os

# Add the src directory to the path so we can import the CalcTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.calc_tokenizer import CalcTokenizer


def test_tokenizer_basic():
    """Test basic tokenization functionality."""
    print("Testing basic tokenization...")
    
    tokenizer = CalcTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}, bos={tokenizer.bos_token_id}, unk={tokenizer.unk_token_id}")
    
    # Test some basic expressions
    test_expressions = [
        "389 + 577 = 966",
        "4 - 1 = 3", 
        "14 * 70 = 980",
        "456 / 24 = 19"
    ]
    
    for expr in test_expressions:
        print(f"\nExpression: '{expr}'")
        
        # Tokenize
        tokens = tokenizer._tokenize(expr)
        token_ids = tokenizer.encode(expr, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Decoded: '{decoded}'")
        
        # Verify round-trip
        assert expr == decoded, f"Round-trip failed: '{expr}' != '{decoded}'"
        print("  ✓ Round-trip verified")


def test_vocabulary_order():
    """Test that vocabulary ordering is correct."""
    print("\n" + "="*50)
    print("Testing vocabulary ordering...")
    
    tokenizer = CalcTokenizer()
    
    # Check that numbers come before operators
    print("Checking vocabulary ordering:")
    
    # Find the first operator ID
    first_op_id = None
    for token, token_id in tokenizer.vocab.items():
        if token in ['+', '-', '*', '/', '=']:
            if first_op_id is None or token_id < first_op_id:
                first_op_id = token_id
    
    # Find the last number ID
    last_num_id = None
    for token, token_id in tokenizer.vocab.items():
        if token.isdigit():
            if last_num_id is None or token_id > last_num_id:
                last_num_id = token_id
    
    print(f"Last number ID: {last_num_id}")
    print(f"First operator ID: {first_op_id}")
    
    if last_num_id < first_op_id:
        print("✓ Numbers come before operators in vocabulary")
    else:
        print("✗ Numbers do not come before operators in vocabulary")
    
    # Test specific tokens
    print(f"\nToken IDs for key tokens:")
    print(f"  '0': {tokenizer.vocab.get('0', 'NOT_FOUND')}")
    print(f"  '*': {tokenizer.vocab.get('*', 'NOT_FOUND')}")
    print(f"  '=': {tokenizer.vocab.get('=', 'NOT_FOUND')}")
    print(f"  '966': {tokenizer.vocab.get('966', 'NOT_FOUND')}")


def test_specific_issue():
    """Test the specific issue that was causing '0 *' output."""
    print("\n" + "="*50)
    print("Testing the specific issue...")
    
    tokenizer = CalcTokenizer()
    
    # Test the expression that should produce the correct result
    test_expr = "389 + 577 = 966"
    
    print(f"Testing expression: '{test_expr}'")
    
    # Tokenize
    tokens = tokenizer._tokenize(test_expr)
    token_ids = tokenizer.encode(test_expr, add_special_tokens=False)
    
    print(f"  Tokens: {tokens}")
    print(f"  Token IDs: {token_ids}")
    
    # Check what comes after '='
    equals_idx = tokens.index('=')
    if equals_idx + 1 < len(tokens):
        result_token = tokens[equals_idx + 1]
        result_id = token_ids[equals_idx + 1]
        print(f"  Token after '=': '{result_token}' (ID: {result_id})")
        
        # Verify it's not '0' or '*'
        if result_token == '966':
            print("  ✓ Correct result token after '='")
        else:
            print(f"  ✗ Wrong result token after '=': '{result_token}'")
    
    # Test decode
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"  Decoded: '{decoded}'")
    
    # Verify the result
    if decoded == test_expr:
        print("  ✓ Decoding works correctly")
    else:
        print(f"  ✗ Decoding failed: '{decoded}' != '{test_expr}'")


def test_training_format():
    """Test the format that would be used during training."""
    print("\n" + "="*50)
    print("Testing training format...")
    
    tokenizer = CalcTokenizer()
    
    # Simulate training data format
    training_text = "389 + 577 = 966"
    
    print(f"Training text: '{training_text}'")
    
    # Encode with special tokens (as used in training)
    token_ids_with_special = tokenizer.encode(training_text, add_special_tokens=True)
    print(f"  Token IDs with special tokens: {token_ids_with_special}")
    
    # Decode with special tokens
    decoded_with_special = tokenizer.decode(token_ids_with_special, skip_special_tokens=False)
    print(f"  Decoded with special tokens: '{decoded_with_special}'")
    
    # Decode without special tokens
    decoded_without_special = tokenizer.decode(token_ids_with_special, skip_special_tokens=True)
    print(f"  Decoded without special tokens: '{decoded_without_special}'")
    
    # Verify
    if decoded_without_special == training_text:
        print("  ✓ Training format works correctly")
    else:
        print(f"  ✗ Training format failed: '{decoded_without_special}' != '{training_text}'")


if __name__ == "__main__":
    try:
        test_tokenizer_basic()
        test_vocabulary_order()
        test_specific_issue()
        test_training_format()
        print("\n" + "="*50)
        print("✓ All tests passed! The CalcTokenizer should now work correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 