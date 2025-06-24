#!/usr/bin/env python3
"""
Debug script to check what's happening in the pretraining processor.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.calc_tokenizer import CalcTokenizer


def debug_pretrain_processor():
    """Debug the pretraining processor logic."""
    print("Debugging pretraining processor...")
    
    # Initialize tokenizer
    tokenizer = CalcTokenizer()
    
    # Simulate the data that would come from the dataset
    # Based on the log, it looks like multiple expressions are concatenated
    sample_data = {
        "_prompt": [
            [{"content": "20 * 7 = 140 126 / 18 = 7 697 - 107 = 590"}],
            [{"content": "214 - 12 = 202 24 * 39 = 936 399 / 21 = 19"}],
            [{"content": "23 * 6 = 138 36 / 36 = 1 611 - 217 = 394"}]
        ]
    }
    
    print("Sample data structure:")
    for i, messages in enumerate(sample_data["_prompt"]):
        print(f"  {i}: {messages[0]['content']}")
    
    # Simulate the pretraining processor logic
    print(f"\n--- SIMULATING PRETRAIN PROCESSOR ---")
    
    # Step 1: Extract content and add EOS token
    eos_token = "<eos>"  # Assuming not using llama3 template
    text_examples = [messages[0]["content"] + eos_token for messages in sample_data["_prompt"]]
    
    print("After adding EOS token:")
    for i, text in enumerate(text_examples):
        print(f"  {i}: '{text}'")
    
    # Step 2: Add BOS token if needed
    if getattr(tokenizer, "add_bos_token", False):
        text_examples = [tokenizer.bos_token + example for example in text_examples]
        print(f"\nAfter adding BOS token:")
        for i, text in enumerate(text_examples):
            print(f"  {i}: '{text}'")
    
    # Step 3: Tokenize with add_special_tokens=False
    print(f"\n--- TOKENIZATION STEP ---")
    result = tokenizer(
        text_examples, 
        add_special_tokens=False, 
        truncation=True, 
        max_length=512  # Example cutoff length
    )
    
    print("Tokenization result:")
    for i, (text, input_ids) in enumerate(zip(text_examples, result["input_ids"])):
        print(f"\nExample {i}:")
        print(f"  Original text: '{text}'")
        print(f"  Token IDs: {input_ids}")
        
        # Decode to see what we get back
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"  Decoded: '{decoded}'")
        
        # Check for <unk> tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        unk_count = tokens.count("<unk>")
        if unk_count > 0:
            print(f"  ⚠️  Found {unk_count} <unk> tokens!")
            # Find positions of <unk> tokens
            unk_positions = [j for j, token in enumerate(tokens) if token == "<unk>"]
            print(f"  <unk> positions: {unk_positions}")
        else:
            print(f"  ✓ No <unk> tokens found")


def debug_specific_expression():
    """Debug a specific expression that might be causing issues."""
    print(f"\n" + "="*60)
    print("DEBUGGING SPECIFIC EXPRESSION")
    print("="*60)
    
    tokenizer = CalcTokenizer()
    
    # Test the expression from the log
    test_expr = "20 * 7 = 140 126 / 18 = 7 697 - 107 = 590"
    
    print(f"Testing expression: '{test_expr}'")
    
    # Tokenize step by step
    tokens = tokenizer._tokenize(test_expr)
    print(f"Tokens: {tokens}")
    
    # Check each token
    print(f"\nChecking each token:")
    for i, token in enumerate(tokens):
        if token in tokenizer.vocab:
            token_id = tokenizer.vocab[token]
            print(f"  '{token}': ID {token_id}")
        else:
            print(f"  '{token}': NOT IN VOCABULARY!")
    
    # Full tokenization
    token_ids = tokenizer.encode(test_expr, add_special_tokens=False)
    print(f"\nFull tokenization:")
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"Decoded: '{decoded}'")
    
    # Check if any tokens became <unk>
    tokens_from_ids = tokenizer.convert_ids_to_tokens(token_ids)
    unk_positions = [i for i, token in enumerate(tokens_from_ids) if token == "<unk>"]
    if unk_positions:
        print(f"⚠️  <unk> tokens found at positions: {unk_positions}")
    else:
        print(f"✓ No <unk> tokens")


def check_vocabulary_coverage():
    """Check if all expected tokens are in the vocabulary."""
    print(f"\n" + "="*60)
    print("CHECKING VOCABULARY COVERAGE")
    print("="*60)
    
    tokenizer = CalcTokenizer()
    
    # Test all operators
    operators = ["+", "-", "*", "/", "="]
    print("Checking operators:")
    for op in operators:
        if op in tokenizer.vocab:
            print(f"  '{op}': ID {tokenizer.vocab[op]}")
        else:
            print(f"  '{op}': MISSING!")
    
    # Test some key numbers
    test_numbers = ["20", "7", "140", "126", "18", "697", "107", "590"]
    print(f"\nChecking key numbers:")
    for num in test_numbers:
        if num in tokenizer.vocab:
            print(f"  '{num}': ID {tokenizer.vocab[num]}")
        else:
            print(f"  '{num}': MISSING!")
    
    # Check vocabulary size
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Expected: 4 (special) + 5 (operators) + 1000 (numbers) = 1009")


if __name__ == "__main__":
    debug_pretrain_processor()
    debug_specific_expression()
    check_vocabulary_coverage() 