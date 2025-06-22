#!/usr/bin/env python3
"""
Test script to verify CalcTokenizer registration with AutoTokenizer.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import CalcTokenizer to ensure it's registered
from llamafactory.model.calc_tokenizer import CalcTokenizer

# Test if it can be loaded by AutoTokenizer
from transformers import AutoTokenizer

def test_tokenizer_registration():
    print("Testing CalcTokenizer registration...")
    
    # Test 1: Direct instantiation
    try:
        tokenizer = CalcTokenizer()
        print("✅ Direct instantiation successful")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Direct instantiation failed: {e}")
        return False
    
    # Test 2: AutoTokenizer loading by class name
    try:
        # Create a temporary directory with tokenizer files
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the tokenizer
            tokenizer.save_pretrained(temp_dir)
            
            # Try to load it with AutoTokenizer
            loaded_tokenizer = AutoTokenizer.from_pretrained(temp_dir)
            print("✅ AutoTokenizer loading successful")
            print(f"   Loaded vocab size: {loaded_tokenizer.vocab_size}")
            
            # Test tokenization
            test_text = "2 + 3 ="
            tokens = loaded_tokenizer.tokenize(test_text)
            print(f"   Test tokenization: '{test_text}' -> {tokens}")
            
    except Exception as e:
        print(f"❌ AutoTokenizer loading failed: {e}")
        return False
    
    print("✅ All tests passed! CalcTokenizer is properly registered.")
    return True

if __name__ == "__main__":
    success = test_tokenizer_registration()
    sys.exit(0 if success else 1) 