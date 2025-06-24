# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List, Optional, Union, Dict, Any
from transformers import PreTrainedTokenizer
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy


class CalcTokenizer(PreTrainedTokenizer):
    """
    A customized tokenizer for mathematical expressions.
    Tokenizes only the operators +, -, *, /, =, and numbers 0-999.
    """
    
    def __init__(
        self,
        pad_token: str = "<pad>",
        eos_token: str = "<|end_of_text|>",
        bos_token: str = "<bos>",
        unk_token: str = "<unk>",
        **kwargs
    ):
        # Define the vocabulary first
        self.operators = ["+", "-", "*", "/", "="]
        self.numbers = [str(i) for i in range(1000)]  # 0-999
        
        # Create vocabulary mapping
        self.vocab = {}
        self.ids_to_tokens = {}
        
        # Add special tokens
        special_tokens = [pad_token, eos_token, bos_token, unk_token]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.ids_to_tokens[i] = token
        
        # Add operators
        for i, op in enumerate(self.operators):
            token_id = len(self.vocab)
            self.vocab[op] = token_id
            self.ids_to_tokens[token_id] = op
        
        # Add numbers
        for num in self.numbers:
            token_id = len(self.vocab)
            self.vocab[num] = token_id
            self.ids_to_tokens[token_id] = num
        
        # Initialize parent class
        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            bos_token=bos_token,
            unk_token=unk_token,
            **kwargs
        )
        
        # Set special token IDs
        self.pad_token_id = self.vocab[pad_token]
        self.eos_token_id = self.vocab[eos_token]
        self.bos_token_id = self.vocab[bos_token]
        self.unk_token_id = self.vocab[unk_token]
    
    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text into tokens.
        Splits on whitespace and handles operators and numbers.
        """
        # Remove extra whitespace and split
        text = re.sub(r'\s+', ' ', text.strip())
        
        if not text:
            return []
        
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in self.operators:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        # Post-process tokens to handle special tokens that might be concatenated
        processed_tokens = []
        for token in tokens:
            # Check if token contains special tokens
            if self.eos_token in token and token != self.eos_token:
                # Split token that contains EOS
                parts = token.split(self.eos_token)
                for i, part in enumerate(parts):
                    if part:  # Add non-empty part
                        processed_tokens.append(part)
                    if i < len(parts) - 1:  # Add EOS token (except after last part)
                        processed_tokens.append(self.eos_token)
            elif self.bos_token in token and token != self.bos_token:
                # Split token that contains BOS
                parts = token.split(self.bos_token)
                for i, part in enumerate(parts):
                    if i == 0 and part:  # Add first part if non-empty
                        processed_tokens.append(part)
                    if i > 0:  # Add BOS token and remaining parts
                        processed_tokens.append(self.bos_token)
                        if part:
                            processed_tokens.append(part)
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its ID."""
        return self.vocab.get(token, self.unk_token_id)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token."""
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def encode(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, Any]] = None,
        **kwargs
    ) -> Union[List[int], List[List[int]], "torch.Tensor"]:
        """Encode text to token IDs."""
        if isinstance(text, str):
            tokens = self._tokenize(text)
        else:
            tokens = text
        
        # Convert tokens to IDs
        token_ids = self.convert_tokens_to_ids(tokens)
        
        # Add special tokens if requested
        if add_special_tokens:
            if self.bos_token_id is not None:
                token_ids = [self.bos_token_id] + token_ids
            if self.eos_token_id is not None:
                token_ids = token_ids + [self.eos_token_id]
        
        return token_ids
    
    def decode(
        self,
        token_ids: Union[int, List[int], "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token]
        
        # Join tokens with spaces
        text = " ".join(tokens)
        
        if clean_up_tokenization_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the tokenizer to a directory."""
        import json
        import os
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save tokenizer config
        tokenizer_config = {
            "tokenizer_class": "CalcTokenizer",
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            "unk_token": self.unk_token,
            "vocab_size": self.vocab_size,
        }
        
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load the tokenizer from a directory."""
        import json
        import os
        
        # Load vocabulary
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        if not os.path.exists(vocab_file):
            raise ValueError(f"Vocabulary file not found at {vocab_file}")
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Load tokenizer config
        config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Create tokenizer instance
        tokenizer = cls(
            pad_token=config.get("pad_token", "<pad>"),
            eos_token=config.get("eos_token", "<eos>"),
            bos_token=config.get("bos_token", "<bos>"),
            unk_token=config.get("unk_token", "<unk>"),
            **kwargs
        )
        
        # Restore vocabulary
        tokenizer.vocab = vocab
        tokenizer.ids_to_tokens = {v: k for k, v in vocab.items()}
        
        return tokenizer

    @property
    def vocab_size(self):
        return len(self.vocab)


# Register the tokenizer with AutoTokenizer
from transformers import AutoTokenizer

# Add the CalcTokenizer to the AutoTokenizer mapping
AutoTokenizer.register("CalcTokenizer", CalcTokenizer)
AutoTokenizer.register("calc", CalcTokenizer)

# Also register it in the global namespace for direct import
__all__ = ["CalcTokenizer"]
