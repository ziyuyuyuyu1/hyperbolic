import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Import your custom model registration code ---
import sys
import os

# Add src to sys.path so we can import the custom model registration
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from llamafactory.model import hyperbolic_utils  # This runs the registration code

# Path to your saved model and tokenizer
MODEL_PATH = "saves/poincare_log_exp_all_wo_norm"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

# List of test equations
test_equations = [
    "2 + 3 =",
    "15 * 7 =",
    "100 - 45 =",
    "81 / 9 =",
    "999 + 1 ="
]

for eq in test_equations:
    print(f"\nInput: {eq}")
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
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Model output: {output_text}")

    # Optionally, print only the predicted answer (after the '=')
    if '=' in output_text:
        answer = output_text.split('=')[-1].strip()
        print(f"Predicted answer: {answer}")
