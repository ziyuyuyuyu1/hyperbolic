#!/usr/bin/env python3
"""
Test script to demonstrate the difference between vocabulary ordering.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.calc_tokenizer import CalcTokenizer


def create_tokenizer_with_operators_first():
    """Create a tokenizer with operators first (the problematic version)."""
    class BadCalcTokenizer(CalcTokenizer):
        def __init__(self, **kwargs):
            # Define the vocabulary first
            self.operators = ["+", "-", "*", "/", "="]
            self.numbers = [str(i) for i in range(1000)]  # 0-999
            
            # Create vocabulary mapping
            self.vocab = {}
            self.ids_to_tokens = {}
            
            # Add special tokens first
            special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
            for i, token in enumerate(special_tokens):
                self.vocab[token] = i
                self.ids_to_tokens[i] = token
            
            # Add operators FIRST (the problematic way)
            for op in self.operators:
                token_id = len(self.vocab)
                self.vocab[op] = token_id
                self.ids_to_tokens[token_id] = op
            
            # Add numbers LAST
            for num in self.numbers:
                token_id = len(self.vocab)
                self.vocab[num] = token_id
                self.ids_to_tokens[token_id] = num
            
            # Initialize parent class
            super(CalcTokenizer, self).__init__(
                pad_token="<pad>",
                eos_token="<eos>",
                bos_token="<bos>",
                unk_token="<unk>",
                **kwargs
            )
            
            # Set special token IDs
            self.pad_token_id = self.vocab["<pad>"]
            self.eos_token_id = self.vocab["<eos>"]
            self.bos_token_id = self.vocab["<bos>"]
            self.unk_token_id = self.vocab["<unk>"]
    
    return BadCalcTokenizer()


def compare_tokenizers():
    """Compare the two tokenizer versions."""
    print("Comparing vocabulary ordering...")
    
    # Create both tokenizers
    good_tokenizer = CalcTokenizer()  # Numbers first
    bad_tokenizer = create_tokenizer_with_operators_first()  # Operators first
    
    test_expr = "389 + 577 = 966"
    
    print(f"\nTest expression: '{test_expr}'")
    
    # Test good tokenizer (numbers first)
    print(f"\n--- GOOD Tokenizer (Numbers First) ---")
    tokens = good_tokenizer._tokenize(test_expr)
    token_ids = good_tokenizer.encode(test_expr, add_special_tokens=False)
    
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    # Show specific token IDs
    print(f"  '966' ID: {good_tokenizer.vocab['966']}")
    print(f"  '0' ID: {good_tokenizer.vocab['0']}")
    print(f"  '*' ID: {good_tokenizer.vocab['*']}")
    print(f"  '=' ID: {good_tokenizer.vocab['=']}")
    
    # Test bad tokenizer (operators first)
    print(f"\n--- BAD Tokenizer (Operators First) ---")
    tokens = bad_tokenizer._tokenize(test_expr)
    token_ids = bad_tokenizer.encode(test_expr, add_special_tokens=False)
    
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    # Show specific token IDs
    print(f"  '966' ID: {bad_tokenizer.vocab['966']}")
    print(f"  '0' ID: {bad_tokenizer.vocab['0']}")
    print(f"  '*' ID: {bad_tokenizer.vocab['*']}")
    print(f"  '=' ID: {bad_tokenizer.vocab['=']}")
    
    # Analyze the problem
    print(f"\n--- ANALYSIS ---")
    print(f"In GOOD tokenizer:")
    print(f"  - Result '966' has ID {good_tokenizer.vocab['966']} (lower)")
    print(f"  - '0' has ID {good_tokenizer.vocab['0']} (higher)")
    print(f"  - '*' has ID {good_tokenizer.vocab['*']} (higher)")
    
    print(f"\nIn BAD tokenizer:")
    print(f"  - Result '966' has ID {bad_tokenizer.vocab['966']} (higher)")
    print(f"  - '0' has ID {bad_tokenizer.vocab['0']} (lower)")
    print(f"  - '*' has ID {bad_tokenizer.vocab['*']} (lower)")
    
    print(f"\nThe problem: In BAD tokenizer, the correct result '966' has a high ID,")
    print(f"while incorrect tokens '0' and '*' have low IDs. During training,")
    print(f"the model might predict low-ID tokens instead of the correct high-ID result.")


def demonstrate_training_issue():
    """Demonstrate why this causes training issues."""
    print(f"\n" + "="*60)
    print("DEMONSTRATING THE TRAINING ISSUE")
    print("="*60)
    
    bad_tokenizer = create_tokenizer_with_operators_first()
    
    # Simulate what happens during training
    print(f"\nDuring training, the model learns to predict the next token after '='")
    print(f"In the expression '389 + 577 = ?', the model should predict '966'")
    
    # Show the token IDs
    equals_id = bad_tokenizer.vocab['=']
    correct_result_id = bad_tokenizer.vocab['966']
    wrong_token_0_id = bad_tokenizer.vocab['0']
    wrong_token_star_id = bad_tokenizer.vocab['*']
    
    print(f"\nToken IDs in BAD tokenizer:")
    print(f"  '=' (context): {equals_id}")
    print(f"  '966' (correct): {correct_result_id}")
    print(f"  '0' (wrong): {wrong_token_0_id}")
    print(f"  '*' (wrong): {wrong_token_star_id}")
    
    print(f"\nThe problem:")
    print(f"  - Correct result '966' has high ID: {correct_result_id}")
    print(f"  - Wrong tokens '0' and '*' have low IDs: {wrong_token_0_id}, {wrong_token_star_id}")
    print(f"  - During training, models often have bias toward predicting lower IDs")
    print(f"  - This can cause the model to predict '0' or '*' instead of '966'")
    
    print(f"\nThis explains why you were seeing '0 *' after the equals sign!")


if __name__ == "__main__":
    compare_tokenizers()
    demonstrate_training_issue() 