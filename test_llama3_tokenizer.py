#!/usr/bin/env python3
"""
Test script to verify the tokenizer works with llama3 template.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.calc_tokenizer import CalcTokenizer


def test_llama3_tokenizer():
    """Test the tokenizer with llama3 template."""
    print("="*80)
    print("TESTING LLAMA3 TOKENIZER")
    print("="*80)
    
    # Initialize tokenizer with llama3 template
    tokenizer = CalcTokenizer(
        eos_token="<|end_of_text|>",
        bos_token="<bos>",
        pad_token="<pad>",
        unk_token="<unk>"
    )
    
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"EOS token: '{tokenizer.eos_token}'")
    print(f"BOS token: '{tokenizer.bos_token}'")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Check key tokens
    print(f"\nKey token IDs:")
    for token in ['<|end_of_text|>', '0', '1', '260', '966', '+', '-', '*', '/', '=']:
        if token in tokenizer.vocab:
            print(f"  '{token}': ID {tokenizer.vocab[token]}")
        else:
            print(f"  '{token}': NOT FOUND!")
    
    # Test the problematic case from your data
    test_cases = [
        "277 - 17 = 260<|end_of_text|>",
        "926 + 40 = 966<|end_of_text|>",
        "123 - 105 = 18<|end_of_text|>",
        "277 - 17 = 260 926 + 40 = 966 123 - 105 = 18<|end_of_text|>"
    ]
    
    print(f"\n" + "="*60)
    print("TESTING TOKENIZATION")
    print("="*60)
    
    for i, test_text in enumerate(test_cases):
        print(f"\nTest case {i+1}: '{test_text}'")
        
        # Tokenize
        tokens = tokenizer._tokenize(test_text)
        token_ids = tokenizer.encode(test_text, add_special_tokens=False)
        
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        
        # Check for <unk> tokens
        unk_count = tokens.count("<unk>")
        if unk_count > 0:
            print(f"  ⚠️  Found {unk_count} <unk> tokens!")
            unk_positions = [j for j, token in enumerate(tokens) if token == "<unk>"]
            print(f"  <unk> positions: {unk_positions}")
        else:
            print(f"  ✓ No <unk> tokens")
        
        # Decode
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"  Decoded: '{decoded}'")
        
        # Verify round-trip
        if decoded == test_text.replace("<|end_of_text|>", "").strip():
            print(f"  ✓ Round-trip successful")
        else:
            print(f"  ✗ Round-trip failed")


def test_pretrain_processor_simulation():
    """Simulate the pretrain processor with llama3 template."""
    print(f"\n" + "="*80)
    print("SIMULATING PRETRAIN PROCESSOR WITH LLAMA3")
    print("="*80)
    
    tokenizer = CalcTokenizer(
        eos_token="<|end_of_text|>",
        bos_token="<bos>",
        pad_token="<pad>",
        unk_token="<unk>"
    )
    
    # Simulate the data structure
    examples = {
        "_prompt": [
            [{"content": "277 - 17 = 260 926 + 40 = 966 123 - 105 = 18"}],
            [{"content": "253 - 2 = 251 801 + 185 = 986 754 - 678 = 76"}]
        ]
    }
    
    print("Original examples:")
    for i, messages in enumerate(examples["_prompt"]):
        print(f"  {i}: '{messages[0]['content']}'")
    
    # Step 1: Extract content and add EOS token (llama3 template)
    eos_token = "<|end_of_text|>"  # llama3 template
    text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]
    
    print(f"\nAfter adding EOS token:")
    for i, text in enumerate(text_examples):
        print(f"  {i}: '{text}'")
    
    # Step 2: Tokenize
    print(f"\nTokenizing...")
    result = tokenizer(
        text_examples, 
        add_special_tokens=False, 
        truncation=True, 
        max_length=512
    )
    
    # Check results
    for i, (text, input_ids) in enumerate(zip(text_examples, result["input_ids"])):
        print(f"\nExample {i}:")
        print(f"  Original: '{text}'")
        print(f"  Token IDs: {input_ids}")
        
        # Convert back to tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        unk_count = tokens.count("<unk>")
        if unk_count > 0:
            print(f"  ⚠️  Found {unk_count} <unk> tokens!")
        else:
            print(f"  ✓ No <unk> tokens")
        
        # Decode
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"  Decoded: '{decoded}'")


if __name__ == "__main__":
    test_llama3_tokenizer()
    test_pretrain_processor_simulation() 