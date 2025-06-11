from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Qwen2Config, Qwen2ForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess

# def project_to_poincare_ball(v, eps=1e-5):
#     norm = torch.norm(v, dim=-1, keepdim=True)
#     max_norm = 1 - eps
#     scale = torch.clamp(max_norm / (norm + 1e-10), max=1.0)
#     return v * scale

# def poincare_distance(x, y, eps=1e-5):
#     """
#     Compute the Poincare distance between points x and y in the Poincare ball.
#     Args:
#         x: Tensor of shape (..., dim), with norm less than 1.
#         y: Tensor of shape (..., dim), with norm less than 1.
#         eps: Small epsilon to prevent division by zero or log of small values.
#     Returns:
#         Tensor of shape (...) with distances.
#     """

#     x = project_to_poincare_ball(x, eps)
#     y = project_to_poincare_ball(y, eps)
    
#     x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)  # (..., 1)
#     y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)  # (..., 1)
#     diff_norm_sq = torch.sum((x - y)**2, dim=-1, keepdim=True)  # (..., 1)

#     denom = (1 - x_norm_sq) * (1 - y_norm_sq)
#     argument = 1 + 2 * diff_norm_sq / (denom + eps)

#     # Clamp argument to be >= 1 to avoid NaNs in arccosh
#     argument = torch.clamp(argument, min=1 + eps)

#     return torch.acosh(argument).squeeze(-1)

    
# class HyperbolicDistanceHead(nn.Module):
#     def __init__(self, embedding_weight, eps=1e-5):
#         super().__init__()
#         print('---------------- HyperbolicDistanceHead initialized ---------------')
#         self.weight = embedding_weight  # shape (vocab_size, hidden_dim)
#         self.eps = eps  # Small epsilon to prevent numerical issues

#     def forward(self, hidden_states):
#         # compute the Poincare distance
#         batch_size, seq_len, hidden_dim = hidden_states.size()
#         hidden_states = hidden_states.view(-1, hidden_dim)
#         distances = poincare_distance(hidden_states.unsqueeze(1), self.weight.unsqueeze(0), eps=self.eps) # (batch * seq_len, vocab_size)
#         # print("Distances shape:", distances.shape, "hidden_states shape:", hidden_states.shape, "weight shape:", self.weight.shape)
#         # assert False
#         # distances: (batch, vocab_size)
#         logits = -distances  # Negate to convert distance to similarity
#         logits = logits.view(batch_size, seq_len, -1)  # Reshape back to (batch, seq_len, vocab_size)
#         print("Logits shape:", logits.shape, "batch_size:", batch_size, "seq_len:", seq_len, "vocab_size:", self.weight.size(0))
#         return logits
    

# class CustomConfig(Qwen2Config):
#     """
#     Custom configuration for Qwen2 model with hyperbolic distance head.
#     This is a placeholder if you want to register a custom config class.
#     """
#     model_type = "custom_qwen2_config"

# class CustomModelForCausalLM(Qwen2ForCausalLM):
#     config_class = CustomConfig

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.lm_head = HyperbolicDistanceHead(self.get_input_embeddings().weight)

# AutoConfig.register("custom_qwen2_config", CustomConfig)
# AutoModelForCausalLM.register(CustomConfig, CustomModelForCausalLM)

if __name__ == "__main__":
    # llamafactory-cli train examples/train_lora/llama3_lora_sft_qwen.yaml 
    subprocess.run(["llamafactory-cli", "train", "examples/train_lora/llama3_lora_pretrain_qwen_hyperbolic.yaml"], check=True)
