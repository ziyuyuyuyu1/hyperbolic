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
    Qwen2ForCausalLM,
)

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

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


################

def poincare_log_map(x, c=1.0):
    """
    Logarithmic map from Poincaré ball to tangent space at origin.
    Args:
        x: Points in Poincaré ball, shape (..., dim)
        c: Curvature parameter (default: 1.0)
    Returns:
        Points in tangent space, shape (..., dim)
    """
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    # Handle zero norm case
    x_norm = torch.clamp(x_norm, min=1e-5)
    
    # Convert c to tensor if it's a float
    if isinstance(c, float):
        c = torch.tensor(c, device=x.device, dtype=x.dtype)
    
    log_map = torch.arctanh(torch.sqrt(c) * x_norm) * x / x_norm / torch.sqrt(c)
    
    return log_map

def poincare_exp_map(v, c=1.0):
    """
    Exponential map from tangent space to Poincaré ball.
    Args:
        v: Points in tangent space, shape (..., dim)
        c: Curvature parameter (default: 1.0)
    Returns:
        Points in Poincaré ball, shape (..., dim)
    """
    # Compute the norm of the tangent vector
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    # Handle zero norm case
    v_norm = torch.clamp(v_norm, min=1e-5)
    
    # Convert c to tensor if it's a float
    if isinstance(c, float):
        c = torch.tensor(c, device=v.device, dtype=v.dtype)
    
    exp_v = torch.tanh(torch.sqrt(c) * v_norm) * v / v_norm / torch.sqrt(c)
    
    return exp_v

def remove_all_norms(model):
    """
    Remove all normalization layers from the model and replace them with Identity.
    Args:
        model: The model to modify
    """
    for name, module in model.named_modules():
        if isinstance(module, (Qwen2RMSNorm)):
            # Replace with Identity
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, nn.Identity())
            else:
                setattr(model, child_name, nn.Identity())
            print(f"Replaced {name} with Identity")

class PoincareLogExpDistanceHead(nn.Module):
    def __init__(self, embedding_weight, eps=1e-2, scale=False, hidden_dim=None):
        super().__init__()
        print('---------------- PoincareLogExpDistanceHead initialized ---------------')
        self.weight = embedding_weight  # shape (vocab_size, hidden_dim)
        self.eps = eps  # Small epsilon to prevent numerical issues
        # trainable scale parameter
        self.use_scale = scale
        if scale:
            self.hidden_state_scale = nn.Parameter(torch.tensor(0.005))
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
        
        # Transformation layer in tangent space
        if hidden_dim is None:
            hidden_dim = embedding_weight.shape[1]
        self.tangent_transform = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize the transformation layer
        nn.init.xavier_uniform_(self.tangent_transform.weight)

    def forward(self, hidden_states):
        # Step 1: Apply scaling if enabled
        batch_size, seq_len, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.use_scale:
            hidden_states = hidden_states * self.hidden_state_scale
        
        # Step 2: Apply transformation in tangent space
        hidden_states_transformed = self.tangent_transform(hidden_states)

        # Step 3: Apply exponential map to Poincaré ball
        hidden_states_exp = poincare_exp_map(hidden_states_transformed)
        
        # Step 4: Project vocabulary embeddings to Poincaré ball
        vocab_embeddings_poincare = poincare_exp_map(self.weight)
        
        # Step 5: Compute Poincaré distances
        distances = poincare_distance(
            hidden_states_exp.unsqueeze(1),
            vocab_embeddings_poincare.unsqueeze(0),
            eps=self.eps
        )

        logits = -distances  # Negate to convert distance to similarity
        logits = logits.view(batch_size, seq_len, -1)  # Reshape back to (batch, seq_len, vocab_size)
        if self.use_scale:
            logits = logits * self.logit_scale  # Apply the scale parameter
        return logits

################


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

################

class PoincareLogExpConfig(Qwen2Config):
    """
    Custom configuration for Qwen2 model with Poincaré log-exp distance head.
    """
    model_type = "poincare_log_exp_config"

class PoincareLogExpForCausalLM(Qwen2ForCausalLM):
    config_class = PoincareLogExpConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = PoincareLogExpDistanceHead(self.get_input_embeddings().weight, scale=True)

AutoConfig.register("poincare_log_exp_config", PoincareLogExpConfig)
AutoModelForCausalLM.register(PoincareLogExpConfig, PoincareLogExpForCausalLM)

class PoincareLogExpWoNormConfig(Qwen2Config):
    """
    Custom configuration for Qwen2 model with Poincaré log-exp distance head without final norm.
    """
    model_type = "poincare_log_exp_wo_norm_config"

class PoincareLogExpWoNormForCausalLM(Qwen2ForCausalLM):
    config_class = PoincareLogExpWoNormConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = PoincareLogExpDistanceHead(self.get_input_embeddings().weight, scale=True)
        self.model.norm = nn.Identity()

AutoConfig.register("poincare_log_exp_wo_norm_config", PoincareLogExpWoNormConfig)
AutoModelForCausalLM.register(PoincareLogExpWoNormConfig, PoincareLogExpWoNormForCausalLM)

class PoincareLogExpAllWoNormConfig(Qwen2Config):
    """
    Custom configuration for Qwen2 model with Poincaré log-exp distance head and all norm layers removed.
    """
    model_type = "poincare_log_exp_all_wo_norm_config"

class PoincareLogExpAllWoNormForCausalLM(Qwen2ForCausalLM):
    config_class = PoincareLogExpAllWoNormConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = PoincareLogExpDistanceHead(self.get_input_embeddings().weight, scale=True)
        # Remove all normalization layers from the entire model
        remove_all_norms(self)

AutoConfig.register("poincare_log_exp_all_wo_norm_config", PoincareLogExpAllWoNormConfig)
AutoModelForCausalLM.register(PoincareLogExpAllWoNormConfig, PoincareLogExpAllWoNormForCausalLM)

################

if __name__ == "__main__":
    # Import the CalcTokenizer
    import sys
    import os
    # Add the src directory to the path to import calc_tokenizer
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from calc_tokenizer import CalcTokenizer
    
    # Initialize the CalcTokenizer
    tokenizer = CalcTokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Create a randomly initialized model configuration
    from transformers import Qwen2Config
    
    # Very small model configuration for simple arithmetic task
    config = Qwen2Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,  # Much smaller hidden size
        intermediate_size=256,  # Smaller intermediate size
        num_hidden_layers=4,  # Fewer layers
        num_attention_heads=4,  # Fewer attention heads
        num_key_value_heads=4,  # Required for Qwen2
        max_position_embeddings=64,  # Shorter sequences (a op b = c is short)
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,  # Important for hyperbolic models
        use_cache=False,  # Disable cache for training
        attention_dropout=0.0,  # No dropout for deterministic behavior
        hidden_dropout=0.0,  # No dropout for deterministic behavior
        rope_theta=10000.0,  # Default RoPE theta
        use_sliding_window=False,  # Disable sliding window attention
        sliding_window=4096,  # Default sliding window size
        attention_bias=False,  # No attention bias
        max_window_layers=0,  # No window layers
    )
    
    # Alternative: Even smaller model for testing
    # config = Qwen2Config(
    #     vocab_size=tokenizer.vocab_size,
    #     hidden_size=64,  # Very small
    #     intermediate_size=128,
    #     num_hidden_layers=2,  # Just 2 layers
    #     num_attention_heads=2,
    #     num_key_value_heads=2,
    #     max_position_embeddings=32,
    #     pad_token_id=tokenizer.pad_token_id,
    #     bos_token_id=tokenizer.bos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    #     tie_word_embeddings=True,
    #     use_cache=False,
    #     attention_dropout=0.0,
    #     hidden_dropout=0.0,
    #     rope_theta=10000.0,
    #     use_sliding_window=False,
    #     sliding_window=4096,
    #     attention_bias=False,
    #     max_window_layers=0,
    # )
    
    # Choose one of the following randomly initialized models:
    
    # Standard Poincaré log-exp model
    # model = PoincareLogExpForCausalLM(config)
    
    # Version without final norm only
    # model = PoincareLogExpWoNormForCausalLM(config)
    
    # Version with ALL norm layers removed (recommended for hyperbolic training)
    model = PoincareLogExpAllWoNormForCausalLM(config)
    
    print("Randomly initialized model created successfully with Poincaré log-exp distance head.")
    print(f"Model config: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
    print(f"Tangent transform layer: {model.lm_head.tangent_transform}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test the model with a simple calculation
    test_input = "2 + 3 ="
    print(f"\nTesting with input: '{test_input}'")
    
    # Tokenize the input
    inputs = tokenizer(test_input, return_tensors="pt", add_special_tokens=True)
    print(f"Tokenized input: {inputs.input_ids}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs.input_ids[0])}")
    
    # Test forward pass
    with torch.no_grad():
        try:
            outputs = model(**inputs)
            logits = outputs.logits
            print(f"Output logits shape: {logits.shape}")
            
            # Get the last token's logits
            last_logits = logits[0, -1, :]
            print(f"Last token logits shape: {last_logits.shape}")
            
            # Get top 5 predictions
            top_k = 5
            top_probs, top_indices = torch.topk(torch.softmax(last_logits, dim=-1), top_k)
            print(f"\nTop {top_k} predictions for next token:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = tokenizer.convert_ids_to_tokens(idx.item())
                print(f"  {i+1}. '{token}' (prob: {prob:.4f})")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            print("This might be due to model configuration issues.")
    
    # Example of how to save the model and tokenizer for training
    print(f"\nTo train with LLaMA Factory, save the model and tokenizer:")
    print(f"model.save_pretrained('hyperbolic_models/poincare_log_exp_all_wo_norm')")
    print(f"tokenizer.save_pretrained('hyperbolic_models/poincare_log_exp_all_wo_norm')")
    
    # Uncomment the following lines to actually save:
    model.save_pretrained('hyperbolic_models/poincare_log_exp_all_wo_norm')
    tokenizer.save_pretrained('hyperbolic_models/poincare_log_exp_all_wo_norm')
    print("Model and tokenizer saved to hyperbolic_models/poincare_log_exp_all_wo_norm")
    
    # Example training configuration for LLaMA Factory:
    print(f"\nExample LLaMA Factory training config:")
    print(f"model_name_or_path: hyperbolic_models/poincare_log_exp_all_wo_norm")
    print(f"dataset_path: data/calc_pretrain.json")
    print(f"dataset_name: calc_pretrain")
    print(f"template: identity")  # Since CalcTokenizer handles its own formatting
