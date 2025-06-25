import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# --- Import your custom model registration code ---
import sys
import os

# Add src to sys.path so we can import the custom model registration
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from llamafactory.model import hyperbolic_utils  # This runs the registration code

# Path to your saved model and tokenizer
# MODEL_PATH = "saves/poincare_log_exp_all_wo_norm/checkpoint-5000"  # Try earlier checkpoint
MODEL_PATH = "saves/qwen2_large_sft"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
num_added_tokens = tokenizer.add_tokens(new_tokens=["<|eom_id|>"], special_tokens=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

# Debug: Check model weights for NaN values
print("=== Checking Model Weights for NaN ===")
nan_found = False
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"WARNING: NaN found in {name}")
        print(f"  Shape: {param.shape}")
        print(f"  NaN count: {torch.isnan(param).sum().item()}")
        print(f"  Min/Max: {param.min().item()}, {param.max().item()}")
        nan_found = True

if not nan_found:
    print("No NaN values found in model parameters")

# Debug: Check model buffers for NaN values
print("\n=== Checking Model Buffers for NaN ===")
nan_found = False
for name, buffer in model.named_buffers():
    if torch.isnan(buffer).any():
        print(f"WARNING: NaN found in {name}")
        print(f"  Shape: {buffer.shape}")
        print(f"  NaN count: {torch.isnan(buffer).sum().item()}")
        print(f"  Min/Max: {buffer.min().item()}, {buffer.max().item()}")
        nan_found = True

if not nan_found:
    print("No NaN values found in model buffers")

# Debug: Check specific components
print("\n=== Checking Specific Components ===")
if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
    embed_weight = model.model.embed_tokens.weight
    if torch.isnan(embed_weight).any():
        print(f"WARNING: NaN found in embedding weights!")
        print(f"  NaN count: {torch.isnan(embed_weight).sum().item()}")
        print(f"  Min/Max: {embed_weight.min().item()}, {embed_weight.max().item()}")
    else:
        print("Embedding weights are OK")

if hasattr(model, 'lm_head'):
    if hasattr(model.lm_head, 'weight'):
        lm_weight = model.lm_head.weight
        if torch.isnan(lm_weight).any():
            print(f"WARNING: NaN found in lm_head weights!")
            print(f"  NaN count: {torch.isnan(lm_weight).sum().item()}")
            print(f"  Min/Max: {lm_weight.min().item()}, {lm_weight.max().item()}")
        else:
            print("LM head weights are OK")
    
    if hasattr(model.lm_head, 'tangent_transform'):
        tangent_weight = model.lm_head.tangent_transform.weight
        tangent_bias = model.lm_head.tangent_transform.bias
        if torch.isnan(tangent_weight).any():
            print(f"WARNING: NaN found in tangent_transform weights!")
            print(f"  NaN count: {torch.isnan(tangent_weight).sum().item()}")
            print(f"  Min/Max: {tangent_weight.min().item()}, {tangent_weight.max().item()}")
        else:
            print("Tangent transform weights are OK")
        
        if torch.isnan(tangent_bias).any():
            print(f"WARNING: NaN found in tangent_transform bias!")
            print(f"  NaN count: {torch.isnan(tangent_bias).sum().item()}")
            print(f"  Min/Max: {tangent_bias.min().item()}, {tangent_bias.max().item()}")
        else:
            print("Tangent transform bias is OK")

# List of test equations with expected answers
test_data = [
    # ("2 + 3 =", "5"),
    # ("15 * 7 =", "105"),
    # ("100 - 45 =", "55"),
    # ("81 / 9 =", "9"),
    # ("999 + 1 =", "1000")
    ("<bos> <unk> 2 + 3 <|eom_id|> <unk>", "5"),
    ("<bos> <unk> 15 * 7 <|eom_id|> <unk>", "105"),
    ("<bos> <unk> 100 - 45 <|eom_id|> <unk>", "55"),
    ("<bos> <unk> 81 / 9 <|eom_id|> <unk>", "9"),
    ("<bos> <unk> 999 + 1 <|eom_id|> <unk>", "1000"),
]

for eq, expected_answer in test_data:
    print(f"\nInput: {eq}")
    print(f"Expected answer: {expected_answer}")
    
    # Tokenize input
    inputs = tokenizer(eq, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Generate output (set max_new_tokens as needed)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=3,  # Adjust as needed for your output length
            do_sample=False    # Greedy decoding for deterministic output
        )

    # Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(f"Model output: {output_text}")

    # Extract predicted answer (after the '=')
    predicted_answer = ""
    if '=' in output_text:
        predicted_answer = output_text.split('=')[-1].strip()
        print(f"Predicted answer: {predicted_answer}")
    
    # Calculate loss for the expected answer
    # Create target sequence: input + expected answer
    target_text = eq + expected_answer
    target_inputs = tokenizer(target_text, return_tensors="pt")
    target_ids = target_inputs["input_ids"]
    
    # Calculate loss using model forward pass
    with torch.no_grad():
        outputs = model(target_ids, labels=target_ids)
        loss = outputs.loss.item()
    
    print(f"Loss for expected answer '{expected_answer}': {loss:.4f}")
    
    # Also calculate perplexity
    perplexity = torch.exp(torch.tensor(loss)).item()
    print(f"Perplexity: {perplexity:.4f}")
