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

def project_to_poincare_ball(v, eps=1e-5):
    norm = torch.norm(v, dim=-1, keepdim=True)
    max_norm = 1 - eps
    scale = torch.clamp(max_norm / (norm + 1e-10), max=1.0)
    # print(v.shape, scale.shape)
    # assert False
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
    diff_norm_sq = torch.cdist(rearrange(x, 'b 1 d -> 1 b d'), y, p=2, compute_mode='use_mm_for_euclid_dist').squeeze(0).unsqueeze(-1)  # (..., 1)

    denom = (1 - x_norm_sq) * (1 - y_norm_sq)
    argument = 1 + 2 * diff_norm_sq / (denom + eps)

    # Clamp argument to be >= 1 to avoid NaNs in arccosh
    argument = torch.clamp(argument, min=1 + eps)

    return torch.acosh(argument).squeeze(-1)
    
class HyperbolicDistanceHead(nn.Module):
    def __init__(self, embedding_weight, eps=1e-5, scale=False):
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

if __name__ == "__main__":
    # Example save 
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PoincareWoNormForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    model.lm_head.hidden_state_scale = nn.Parameter(torch.tensor(0.005, dtype=model.dtype).to(model.device))
    model.lm_head.logit_scale = nn.Parameter(torch.tensor(1.0, dtype=model.dtype).to(model.device))

    tokenizer.save_pretrained("hyperbolic_model/poincare_wo_norm_scale")

    print(model.lm_head)
    print(model.model.norm)
    print(model.lm_head.hidden_state_scale)
    print(model.lm_head.logit_scale)
    print(model.lm_head.weight- model.get_input_embeddings().weight)
    assert torch.allclose(model.lm_head.weight, model.get_input_embeddings().weight, atol=1e-5), "lm_head weight should be same as input embeddings weight"
    print("Model loaded successfully with Poincare norm head.")

    model.save_pretrained("hyperbolic_model/poincare_wo_norm_scale")

    # # Example usage
    # model_name = "output/poincare_norm"
    # # model_name = "hyperbolic_model/poincare_wo_norm"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = PoincareWoNormForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )

    # print(model.lm_head.weight - model.get_input_embeddings().weight)

    # # prepare the model input
    # prompt = "Give me a short introduction to large language models."
    # messages = [
    #     {"role": "user", "content": prompt}
    # ]
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # # conduct text completion
    # generated_ids = model.generate(
    #     **model_inputs,
    #     max_new_tokens=32768
    # )
    # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # # the result will begin with thinking content in <think></think> tags, followed by the actual response
    # print(tokenizer.decode(output_ids, skip_special_tokens=True))
