import torch
from torch import nn
from einops import rearrange

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM
)

def lift_to_lorentz(x, c=1.0):
    norm_sq = torch.sum(x * x, dim=-1, keepdim=True)  # (..., 1)
    time = torch.sqrt(1. / c + norm_sq)               # (..., 1)
    lifted_x = torch.cat((time, x), dim=-1)           # (..., d+1)
    return lifted_x

def lorentz_inner_product_batch(x, y):
    """
    x: (B, d+1)
    y: (V, d+1)
    Returns:
        inner product: (B, V)
    """
    x0 = x[:, 0:1]               # (B, 1)
    y0 = y[:, 0].unsqueeze(0)    # (1, V)
    xy_spatial = torch.matmul(x[:, 1:], y[:, 1:].T)  # (B, V)
    ip = -x0 @ y0 + xy_spatial   # (B, V)
    return ip  # (B, V)

def lorentz_distance(x, y, eps=1e-5, c=1.0):
    """
    x: (B, d), y: (V, d)
    Returns: distance matrix (B, V)
    """
    x_lifted = lift_to_lorentz(x, c)  # (B, d+1)
    y_lifted = lift_to_lorentz(y, c)  # (V, d+1)

    ip = lorentz_inner_product_batch(x_lifted, y_lifted)  # (B, V)
    argument = torch.clamp(-ip * c, min=1.0 + eps)        # (B, V)

    dist = torch.acosh(argument) * torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))  # (B, V)
    return dist

def project_to_poincare_ball(v, eps=1e-5, print_norm=False):
    norm = torch.norm(v, dim=-1, keepdim=True)
    max_norm = 0.96 - eps
    scale = torch.clamp(max_norm / (norm + 1e-5), max=1.0)
    if print_norm:
        print(norm)
    # # assert False
    # if torch.any(torch.norm(v*scale, dim=-1) >= 1 - eps):
    #     print("Warning: projected vector norm >= 1 - eps, clamping to avoid NaNs.")
    #     print("Projected vector norms:", torch.norm(v*scale, dim=-1)[torch.norm(v*scale, dim=-1) >= 1 - eps])
    #     print("v norms:", torch.norm(v, dim=-1)[torch.norm(v*scale, dim=-1) >= 1 - eps])
    #     print("scale:", scale[torch.norm(v*scale, dim=-1) >= 1 - eps])
    #     assert False
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
    
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)  # (..., 1)
    y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)  # (..., 1)
    # diff_norm_sq = torch.cdist(rearrange(x, 'b 1 d -> 1 b d'), y, p=2, compute_mode='use_mm_for_euclid_dist').squeeze(0).unsqueeze(-1)  # (..., 1)
    diff_norm_sq = torch.cdist(x.squeeze(1), y.squeeze(0), p=2, compute_mode='use_mm_for_euclid_dist').unsqueeze(-1)  # (..., 1)
    diff_norm_sq = torch.sum(diff_norm_sq * diff_norm_sq, dim=-1, keepdim=True)  # (..., 1)

    # return torch.acosh(diff_norm_sq - y_norm_sq + 1).squeeze(-1)

    denom = (1 - x_norm_sq) * (1 - y_norm_sq)
    argument = 1 + 2 * diff_norm_sq / (denom + eps)

    # check 1 - x_norm_sq > 0
    if torch.any(x_norm_sq >= 1 - eps):
        print("Warning: x_norm_sq >= 1 - eps, clamping to avoid NaNs.")
        print("x_norm_sq values:", x_norm_sq[x_norm_sq >= 1 - eps])
        assert False, "Norm of x is too large, should be less than 1 - eps."

    # Clamp argument to be >= 1 to avoid NaNs in arccosh
    argument = torch.clamp(argument, min=1 + eps)

    return torch.acosh(argument).squeeze(-1)

class LorentzDistanceHead(nn.Module):
    def __init__(self, embedding_weight, eps=1e-2, scale=False):
        super().__init__()
        print('---------------- LorentzDistanceHead initialized ---------------')
        self.weight = embedding_weight  # shape (vocab_size, hidden_dim)
        self.eps = eps  # Small epsilon to prevent numerical issues
        # trainable scale parameter
        self.use_scale = scale
        if scale:
            self.hidden_state_scale = nn.Parameter(torch.tensor(0.01))
            self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states):
        # compute the Lorentz distance
        batch_size, seq_len, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.use_scale:
            hidden_states = hidden_states * self.hidden_state_scale
        hidden_states = lift_to_lorentz(hidden_states)
        vocab_embeddings = lift_to_lorentz(self.weight)

        distances = lorentz_distance(
            hidden_states,
            vocab_embeddings,
            eps=self.eps
        )

        logits = -distances  # Negate to convert distance to similarity
        logits = logits.view(batch_size, seq_len, -1)  # Reshape back to (batch, seq_len, vocab_size)
        if self.use_scale:
            logits = logits * self.logit_scale  # Apply the scale parameter
        return logits

class HyperbolicDistanceHead(nn.Module):
    def __init__(self, embedding_weight, eps=1e-2, scale=False):
        super().__init__()
        print('---------------- HyperbolicDistanceHead initialized ---------------')
        self.weight = embedding_weight  # shape (vocab_size, hidden_dim)
        self.eps = eps  # Small epsilon to prevent numerical issues
        # trainable scale parameter
        self.use_scale = scale
        if scale:
            self.hidden_state_scale = nn.Parameter(torch.tensor(0.005))
            self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states):
        # compute the Poincare distance
        batch_size, seq_len, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.use_scale:
            hidden_states = hidden_states * self.hidden_state_scale
        hidden_states = project_to_poincare_ball(hidden_states, self.eps)
        vocab_embeddings = project_to_poincare_ball(self.weight, self.eps)

        distances = poincare_distance(
            hidden_states.unsqueeze(1),
            vocab_embeddings.unsqueeze(0),
            eps=self.eps
        )

        logits = -distances  # Negate to convert distance to similarity
        logits = logits.view(batch_size, seq_len, -1)  # Reshape back to (batch, seq_len, vocab_size)
        if self.use_scale:
            logits = logits * self.logit_scale  # Apply the scale parameter
        return logits
    

class PoinCareNormConfig(Qwen2Config):
    """
    Custom configuration for Qwen2 model with hyperbolic distance head.
    This is a placeholder if you want to register a custom config class.
    """
    model_type = "poincare_norm_config"

class PoincareNormForCausalLM(Qwen2ForCausalLM):
    config_class = PoinCareNormConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = HyperbolicDistanceHead(self.get_input_embeddings().weight, scale=True)

AutoConfig.register("poincare_norm_config", PoinCareNormConfig)
AutoModelForCausalLM.register(PoinCareNormConfig, PoincareNormForCausalLM)


class PoinCareWoNormConfig(Qwen2Config):
    """
    Custom configuration for Qwen2 model with hyperbolic distance head.
    This is a placeholder if you want to register a custom config class.
    """
    model_type = "poincare_wo_norm_config"

class PoincareWoNormForCausalLM(Qwen2ForCausalLM):
    config_class = PoinCareWoNormConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = HyperbolicDistanceHead(self.get_input_embeddings().weight, scale=True)
        self.model.norm = nn.Identity()

AutoConfig.register("poincare_wo_norm_config", PoinCareWoNormConfig)
AutoModelForCausalLM.register(PoinCareWoNormConfig, PoincareWoNormForCausalLM)


# class PoinCareWoNormProjConfig(Qwen2Config):
#     """
#     Custom configuration for Qwen2 model with hyperbolic distance head.
#     This is a placeholder if you want to register a custom config class.
#     """
#     model_type = "poincare_wo_norm_proj_config"

# class ProjectedEmbedding(nn.Embedding):
#     def __init__(self, num_embeddings, embedding_dim, proj_dim, **kwargs):
#         super().__init__(num_embeddings, embedding_dim, **kwargs)
#         self.embed_proj = nn.Linear(embedding_dim, proj_dim, bias=False)

#     def forward(self, input_ids):
#         embeds = super().forward(input_ids)  # (batch, seq, embed_dim)
#         projected = self.embed_proj(embeds)        # (batch, seq, proj_dim)
#         return projected

# class PoincareWoNormProjForCausalLM(Qwen2ForCausalLM):
#     config_class = PoinCareWoNormProjConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.model.embed_tokens = ProjectedEmbedding(
#             config.vocab_size, config.hidden_size, self.model.padding_idx
#         )
#         self.lm_head = HyperbolicDistanceHead(self.get_input_embeddings().weight, scale=True)
#         self.model.norm = nn.Identity()

# AutoConfig.register("poincare_wo_norm_proj_config", PoinCareWoNormProjConfig)
# AutoModelForCausalLM.register(PoinCareWoNormProjConfig, PoincareWoNormProjForCausalLM)

class LorentzWoNormConfig(Qwen2Config):
    """
    Custom configuration for Qwen2 model with Lorentz distance head.
    This is a placeholder if you want to register a custom config class.
    """
    model_type = "lorentz_wo_norm_config"

class LorentzWoNormForCausalLM(Qwen2ForCausalLM):
    config_class = LorentzWoNormConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = LorentzDistanceHead(self.get_input_embeddings().weight, scale=True)
        self.model.norm = nn.Identity()
        
AutoConfig.register("lorentz_wo_norm_config", LorentzWoNormConfig)
AutoModelForCausalLM.register(LorentzWoNormConfig, LorentzWoNormForCausalLM)

if __name__ == "__main__":
    # # Example save 
    # model_name = "Qwen/Qwen2.5-0.5B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = LorentzWoNormForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )

    # model.lm_head.hidden_state_scale = nn.Parameter(torch.tensor(0.01, dtype=model.dtype).to(model.device))
    # model.lm_head.logit_scale = nn.Parameter(torch.tensor(1.0, dtype=model.dtype).to(model.device))

    # tokenizer.save_pretrained("hyperbolic_model/poincare_wo_norm_proj_scale")

    # print(model.lm_head)
    # print(model.model.norm)
    # print(model.lm_head.hidden_state_scale)
    # print(model.lm_head.logit_scale)
    # print(model.lm_head.weight- model.get_input_embeddings().weight)
    # assert torch.allclose(model.lm_head.weight, model.get_input_embeddings().weight, atol=1e-5), "lm_head weight should be same as input embeddings weight"
    # print("Model loaded successfully with lorentz norm head.")

    # model.save_pretrained("hyperbolic_model/lorentz_wo_norm_scale")

    # ----------------------------------------

    # Example usage
    # model_name = "output/poincare_wo_norm_scale"
    # model_name = "hyperbolic_model/poincare_wo_norm"
    model_name = "Qwen/Qwen2.5-0.5B"
    # model_name = "saves/qwen-0.5b/only_embed_hyperbolic_lorentz_wo_norm_scale/pretrain"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = PoincareWoNormForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )
    # model = LorentzWoNormForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # model.lm_head.hidden_state_scale = nn.Parameter(torch.tensor(0.005, dtype=model.dtype).to(model.device))
    # model.lm_head.logit_scale = nn.Parameter(torch.tensor(1.0, dtype=model.dtype).to(model.device))

    # print(model.lm_head.weight - model.get_input_embeddings().weight)
    # print(model.lm_head.hidden_state_scale, model.lm_head.logit_scale)

    # prepare the model input
    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=100,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # the result will begin with thinking content in <think></think> tags, followed by the actual response
    print(tokenizer.decode(output_ids, skip_special_tokens=True))
