from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Qwen2ForCausalLM, Qwen2Config
import torch
import torch.nn as nn
import torch.nn.functional as F

def project_to_poincare_ball(v, eps=1e-5):
    norm = torch.norm(v, dim=-1, keepdim=True)
    max_norm = 1 - eps
    scale = torch.clamp(max_norm / (norm + 1e-10), max=1.0)
    return v * scale

def poincare_distance(x, y, eps=1e-5):
    """
    Compute the Poincare distance between points x and y in the Poincare ball.
    Args:
        x: Tensor of shape (..., dim), with norm less than 1.
        y: Tensor of shape (..., dim), with norm less than 1.
        eps: Small epsilon to prevent division by zero or log of small values.
    Returns:
        Tensor of shape (...) with distances.
    """

    x = project_to_poincare_ball(x, eps)
    y = project_to_poincare_ball(y, eps)
    
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)  # (..., 1)
    y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)  # (..., 1)
    diff_norm_sq = torch.sum((x - y)**2, dim=-1, keepdim=True)  # (..., 1)

    denom = (1 - x_norm_sq) * (1 - y_norm_sq)
    argument = 1 + 2 * diff_norm_sq / (denom + eps)

    # Clamp argument to be >= 1 to avoid NaNs in arccosh
    argument = torch.clamp(argument, min=1 + eps)

    return torch.acosh(argument).squeeze(-1)

class CosineSimHead(nn.Module):
    def __init__(self, embedding_weight):
        super().__init__()
        self.weight = embedding_weight  # shape (vocab_size, hidden_dim)

    def forward(self, hidden_states):
        return torch.matmul(hidden_states, self.weight.t())
        # Normalize both
        # print("Hidden states shape:", hidden_states.shape, "Weight shape:", self.weight.shape)
        hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
        embed_norm = F.normalize(self.weight, p=2, dim=-1)
        # Cosine similarity: (batch, seq_len, vocab_size)
        out = torch.matmul(hidden_norm, embed_norm.t())
        # print("Output shape:", out.shape, "Batch size:", hidden_states.size(0), "Seq len:", hidden_states.size(1), "Vocab size:", self.weight.size(0))
        return out
    
class HyperbolicDistanceHead(nn.Module):
    def __init__(self, embedding_weight, eps=1e-5):
        super().__init__()
        self.weight = embedding_weight  # shape (vocab_size, hidden_dim)
        self.eps = eps  # Small epsilon to prevent numerical issues

    def forward(self, hidden_states):
        # compute the Poincare distance
        batch_size, seq_len, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)
        distances = poincare_distance(hidden_states.unsqueeze(1), self.weight.unsqueeze(0), eps=self.eps) # (batch * seq_len, vocab_size)
        # print("Distances shape:", distances.shape, "hidden_states shape:", hidden_states.shape, "weight shape:", self.weight.shape)
        # assert False
        # distances: (batch, vocab_size)
        logits = -distances  # Negate to convert distance to similarity
        logits = logits.view(batch_size, seq_len, -1)  # Reshape back to (batch, seq_len, vocab_size)
        print("Logits shape:", logits.shape, "batch_size:", batch_size, "seq_len:", seq_len, "vocab_size:", self.weight.size(0))
        return logits

class CustomConfig(Qwen2Config):
    """
    Custom configuration for Qwen2 model with hyperbolic distance head.
    This is a placeholder if you want to register a custom config class.
    """
    model_type = "custom_qwen2_config"

class CustomModelForCausalLM(Qwen2ForCausalLM):
    config_class = CustomConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the cosine similarity head with the embedding weights
        # self.lm_head = CosineSimHead(self.get_input_embeddings().weight)
        self.lm_head = HyperbolicDistanceHead(self.get_input_embeddings().weight)

# AutoConfig.register("custom_qwen2_config", Qwen2Config)
# AutoModelForCausalLM.register(Qwen2Config, CustomModelForCausalLM)
# AutoModelForCausalLM.register(Qwen2ForCausalLM.config_class, CustomModelForCausalLM)
AutoModelForCausalLM.register(CustomConfig, CustomModelForCausalLM)

# model_name = "Qwen/Qwen2.5-0.5B"
# model_name = "test_save"
model_name = "hyperbolic_head"


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
model = CustomModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print(model.lm_head)

# # Get embedding weights (usually tied with input embeddings)
# embedding_weight = model.get_input_embeddings().weight  # shape (vocab_size, hidden_dim)

# # Create cosine sim head
# cosine_head = CosineSimHead(embedding_weight)

# # Create hyperbolic distance head
# hyperbolic_head = HyperbolicDistanceHead(embedding_weight)

# # Replace
# model.lm_head = hyperbolic_head
# # model.lm_head = cosine_head

# model.save_pretrained("hyperbolic_head")
# tokenizer.save_pretrained("hyperbolic_head")


# # # print model layer containing embedding
# # # for name, param in model.named_parameters():
# # #     if "embed" in name or "lm_head" in name:
# # #         print(name, param.shape)

# # # print(model.lm_head)

# # # prepare the model input
# # prompt = "Give me a short introduction to large language models."
# # messages = [
# #     {"role": "user", "content": prompt}
# # ]
# # text = tokenizer.apply_chat_template(
# #     messages,
# #     tokenize=False,
# #     add_generation_prompt=True,
# #     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
# # )
# # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# # # conduct text completion
# # generated_ids = model.generate(
# #     **model_inputs,
# #     max_new_tokens=32768
# # )
# # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# # # the result will begin with thinking content in <think></think> tags, followed by the actual response
# # print(tokenizer.decode(output_ids, skip_special_tokens=True))
