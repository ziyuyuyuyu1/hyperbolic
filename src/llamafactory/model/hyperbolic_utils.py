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
from ..model import lorentz as L

def lift_to_lorentz(x, c=1.0):
    # Debug prints for input
    if torch.isnan(x).any():
        print("WARNING: NaN detected in lift_to_lorentz input x")
        print("x requires_grad:", x.requires_grad)
        print("x grad_fn:", x.grad_fn)
    if torch.isinf(x).any():
        print("WARNING: Inf detected in lift_to_lorentz input x")
    
    norm_sq = torch.sum(x * x, dim=-1, keepdim=True)  # (..., 1)
    
    # Debug prints for norm_sq
    if torch.isnan(norm_sq).any():
        print("WARNING: NaN detected in lift_to_lorentz norm_sq")
        print("norm_sq requires_grad:", norm_sq.requires_grad)
        print("norm_sq grad_fn:", norm_sq.grad_fn)
    if torch.isinf(norm_sq).any():
        print("WARNING: Inf detected in lift_to_lorentz norm_sq")
    
    time = torch.sqrt(1. / c + norm_sq)               # (..., 1)
    
    # Debug prints for time
    if torch.isnan(time).any():
        print("WARNING: NaN detected in lift_to_lorentz time")
        print("time requires_grad:", time.requires_grad)
        print("time grad_fn:", time.grad_fn)
    if torch.isinf(time).any():
        print("WARNING: Inf detected in lift_to_lorentz time")
    
    lifted_x = torch.cat((time, x), dim=-1)           # (..., d+1)
    
    # Debug prints for final output
    if torch.isnan(lifted_x).any():
        print("WARNING: NaN detected in lift_to_lorentz final output")
        print("lifted_x requires_grad:", lifted_x.requires_grad)
        print("lifted_x grad_fn:", lifted_x.grad_fn)
    if torch.isinf(lifted_x).any():
        print("WARNING: Inf detected in lift_to_lorentz final output")
    
    return lifted_x

def lorentz_inner_product_batch(x, y):
    """
    x: (B, d+1)
    y: (V, d+1)
    Returns:
        inner product: (B, V)
    """
    # Debug prints for input
    if torch.isnan(x).any():
        print("WARNING: NaN detected in lorentz_inner_product_batch input x")
        print("x requires_grad:", x.requires_grad)
        print("x grad_fn:", x.grad_fn)
    if torch.isinf(x).any():
        print("WARNING: Inf detected in lorentz_inner_product_batch input x")
    if torch.isnan(y).any():
        print("WARNING: NaN detected in lorentz_inner_product_batch input y")
        print("y requires_grad:", y.requires_grad)
        print("y grad_fn:", y.grad_fn)
    if torch.isinf(y).any():
        print("WARNING: Inf detected in lorentz_inner_product_batch input y")
    
    x0 = x[:, 0:1]               # (B, 1)
    y0 = y[:, 0].unsqueeze(0)    # (1, V)
    xy_spatial = torch.matmul(x[:, 1:], y[:, 1:].T)  # (B, V)
    
    # Debug prints for intermediate calculations
    if torch.isnan(xy_spatial).any():
        print("WARNING: NaN detected in lorentz_inner_product_batch xy_spatial")
        print("xy_spatial requires_grad:", xy_spatial.requires_grad)
        print("xy_spatial grad_fn:", xy_spatial.grad_fn)
    if torch.isinf(xy_spatial).any():
        print("WARNING: Inf detected in lorentz_inner_product_batch xy_spatial")
    
    ip = -x0 @ y0 + xy_spatial   # (B, V)
    
    # Debug prints for final output
    if torch.isnan(ip).any():
        print("WARNING: NaN detected in lorentz_inner_product_batch final output")
        print("ip requires_grad:", ip.requires_grad)
        print("ip grad_fn:", ip.grad_fn)
    if torch.isinf(ip).any():
        print("WARNING: Inf detected in lorentz_inner_product_batch final output")
    
    return ip  # (B, V)

def lorentz_distance(x, y, eps=1e-5, c=1.0):
    """
    x: (B, d), y: (V, d)
    Returns: distance matrix (B, V)
    """
    # Debug prints for input
    if torch.isnan(x).any():
        print("WARNING: NaN detected in lorentz_distance input x")
        print("x requires_grad:", x.requires_grad)
        print("x grad_fn:", x.grad_fn)
    if torch.isinf(x).any():
        print("WARNING: Inf detected in lorentz_distance input x")
    if torch.isnan(y).any():
        print("WARNING: NaN detected in lorentz_distance input y")
        print("y requires_grad:", y.requires_grad)
        print("y grad_fn:", y.grad_fn)
    if torch.isinf(y).any():
        print("WARNING: Inf detected in lorentz_distance input y")
    
    x_lifted = lift_to_lorentz(x, c)  # (B, d+1)
    y_lifted = lift_to_lorentz(y, c)  # (V, d+1)

    # Debug prints for lifted tensors
    if torch.isnan(x_lifted).any():
        print("WARNING: NaN detected in x_lifted")
        print("x_lifted requires_grad:", x_lifted.requires_grad)
        print("x_lifted grad_fn:", x_lifted.grad_fn)
    if torch.isnan(y_lifted).any():
        print("WARNING: NaN detected in y_lifted")
        print("y_lifted requires_grad:", y_lifted.requires_grad)
        print("y_lifted grad_fn:", y_lifted.grad_fn)
    
    ip = lorentz_inner_product_batch(x_lifted, y_lifted)  # (B, V)
    
    # Debug prints for inner product
    if torch.isnan(ip).any():
        print("WARNING: NaN detected in lorentz inner product")
        print("ip requires_grad:", ip.requires_grad)
        print("ip grad_fn:", ip.grad_fn)
    if torch.isinf(ip).any():
        print("WARNING: Inf detected in lorentz inner product")
    
    argument = torch.clamp(-ip * c, min=1.0 + eps)        # (B, V)
    
    # Debug prints for argument
    if torch.isnan(argument).any():
        print("WARNING: NaN detected in lorentz argument")
        print("argument requires_grad:", argument.requires_grad)
        print("argument grad_fn:", argument.grad_fn)
    if torch.isinf(argument).any():
        print("WARNING: Inf detected in lorentz argument")

    dist = torch.acosh(argument) * torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))  # (B, V)
    
    # Debug prints for final distance
    if torch.isnan(dist).any():
        print("WARNING: NaN detected in lorentz final distance")
        print("dist requires_grad:", dist.requires_grad)
        print("dist grad_fn:", dist.grad_fn)
    if torch.isinf(dist).any():
        print("WARNING: Inf detected in lorentz final distance")
    
    return dist

def project_to_poincare_ball(v, eps=1e-5, print_norm=False):
    # Debug prints for input
    if torch.isnan(v).any():
        print("WARNING: NaN detected in project_to_poincare_ball input v")
        print("v requires_grad:", v.requires_grad)
        print("v grad_fn:", v.grad_fn)
    if torch.isinf(v).any():
        print("WARNING: Inf detected in project_to_poincare_ball input v")
    
    # Ensure inputs are finite
    if not torch.isfinite(v).all():
        print("WARNING: Non-finite values detected in project_to_poincare_ball input!")
        return torch.zeros_like(v)
    
    norm = torch.norm(v, dim=-1, keepdim=True)
    
    # Debug prints for norm calculation
    if torch.isnan(norm).any():
        print("WARNING: NaN detected in project_to_poincare_ball norm calculation")
        print("norm requires_grad:", norm.requires_grad)
        print("norm grad_fn:", norm.grad_fn)
        return torch.zeros_like(v)
    if torch.isinf(norm).any():
        print("WARNING: Inf detected in project_to_poincare_ball norm calculation")
    
    # Use more conservative projection bounds
    max_norm = 0.96 - eps  
    scale = torch.clamp(max_norm / (norm + 1e-5), max=1.0)  # Better epsilon
    
    # Debug prints for scale calculation
    if torch.isnan(scale).any():
        print("WARNING: NaN detected in project_to_poincare_ball scale calculation")
        print("scale requires_grad:", scale.requires_grad)
        print("scale grad_fn:", scale.grad_fn)
    if torch.isinf(scale).any():
        print("WARNING: Inf detected in project_to_poincare_ball scale calculation")
    
    # Check for invalid scale
    if torch.isnan(scale).any() or torch.isinf(scale).any():
        print("WARNING: Invalid scale in project_to_poincare_ball, returning zeros")
        return torch.zeros_like(v)
    
    if print_norm:
        print(norm)
    
    result = v * scale
    
    # Debug prints for final result
    if torch.isnan(result).any():
        print("WARNING: NaN detected in project_to_poincare_ball final result")
        print("result requires_grad:", result.requires_grad)
        print("result grad_fn:", result.grad_fn)
    if torch.isinf(result).any():
        print("WARNING: Inf detected in project_to_poincare_ball final result")
    
    # Check for invalid output
    if torch.isnan(result).any() or torch.isinf(result).any():
        print("WARNING: Invalid final result in project_to_poincare_ball, returning zeros")
        return torch.zeros_like(v)
    
    return result

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
    
    # Debug prints for input
    if torch.isnan(x).any():
        print("WARNING: NaN detected in poincare_distance input x")
        print("x requires_grad:", x.requires_grad)
        print("x grad_fn:", x.grad_fn)
    if torch.isinf(x).any():
        print("WARNING: Inf detected in poincare_distance input x")
    if torch.isnan(y).any():
        print("WARNING: NaN detected in poincare_distance input y")
        print("y requires_grad:", y.requires_grad)
        print("y grad_fn:", y.grad_fn)
    if torch.isinf(y).any():
        print("WARNING: Inf detected in poincare_distance input y")
    
    # Ensure inputs are finite
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        print("WARNING: Non-finite values in poincare_distance inputs, returning inf")
        return torch.full_like(x[..., 0], float('inf'))
    
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)  # (..., 1)
    y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)  # (..., 1)
    
    # Debug prints for norm calculations
    if torch.isnan(x_norm_sq).any():
        print("WARNING: NaN detected in x_norm_sq calculation")
        print("x_norm_sq requires_grad:", x_norm_sq.requires_grad)
        print("x_norm_sq grad_fn:", x_norm_sq.grad_fn)
    if torch.isnan(y_norm_sq).any():
        print("WARNING: NaN detected in y_norm_sq calculation")
        print("y_norm_sq requires_grad:", y_norm_sq.requires_grad)
        print("y_norm_sq grad_fn:", y_norm_sq.grad_fn)
    
    # Check for NaN in norm calculations
    if torch.isnan(x_norm_sq).any() or torch.isnan(y_norm_sq).any():
        print("WARNING: NaN in norm calculations, returning inf")
        return torch.full_like(x[..., 0], float('inf'))
    
    # Compute squared Euclidean distance more safely
    diff_norm_sq = torch.cdist(x.squeeze(1), y.squeeze(0), p=2, compute_mode='use_mm_for_euclid_dist').unsqueeze(-1)  # (..., 1)
    diff_norm_sq = torch.sum(diff_norm_sq * diff_norm_sq, dim=-1, keepdim=True)  # (..., 1)
    
    # Debug prints for distance calculation
    if torch.isnan(diff_norm_sq).any():
        print("WARNING: NaN detected in diff_norm_sq calculation")
        print("diff_norm_sq requires_grad:", diff_norm_sq.requires_grad)
        print("diff_norm_sq grad_fn:", diff_norm_sq.grad_fn)
    
    # Check for NaN in distance calculation
    if torch.isnan(diff_norm_sq).any():
        print("WARNING: NaN in distance calculation, returning inf")
        return torch.full_like(x[..., 0], float('inf'))

    # Clamp norms to prevent numerical issues
    x_norm_sq = torch.clamp(x_norm_sq, max=0.95)  # Prevent norm from getting too close to 1
    y_norm_sq = torch.clamp(y_norm_sq, max=0.95)  # Prevent norm from getting too close to 1
    
    # Compute denominator with better numerical stability
    denom = (1 - x_norm_sq) * (1 - y_norm_sq)
    denom = torch.clamp(denom, min=eps)  # Ensure denominator is not too small
    
    # Debug prints for denominator
    if torch.isnan(denom).any():
        print("WARNING: NaN detected in denominator calculation")
        print("denom requires_grad:", denom.requires_grad)
        print("denom grad_fn:", denom.grad_fn)
    if torch.isinf(denom).any():
        print("WARNING: Inf detected in denominator calculation")
    
    # Compute argument with better bounds
    argument = 1 + 2 * diff_norm_sq / denom
    
    # Debug prints for argument
    if torch.isnan(argument).any():
        print("WARNING: NaN detected in argument calculation")
        print("argument requires_grad:", argument.requires_grad)
        print("argument grad_fn:", argument.grad_fn)
    if torch.isinf(argument).any():
        print("WARNING: Inf detected in argument calculation")
    
    # Clamp argument to valid range for arccosh
    argument = torch.clamp(argument, min=1.0 + eps, max=1e6)  # Upper bound to prevent overflow
    
    # Check for invalid arguments
    if torch.isnan(argument).any() or torch.isinf(argument).any():
        print("WARNING: Invalid argument for arccosh, returning inf")
        return torch.full_like(x[..., 0], float('inf'))
    
    # Compute distance
    distance = torch.acosh(argument).squeeze(-1)
    
    # Debug prints for final distance
    if torch.isnan(distance).any():
        print("WARNING: NaN detected in final distance calculation")
        print("distance requires_grad:", distance.requires_grad)
        print("distance grad_fn:", distance.grad_fn)
    if torch.isinf(distance).any():
        print("WARNING: Inf detected in final distance calculation")
    
    # Final check for valid output
    if torch.isnan(distance).any() or torch.isinf(distance).any():
        print("WARNING: Invalid final distance, returning inf")
        return torch.full_like(x[..., 0], float('inf'))
    
    return distance


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
    x_norm = torch.clamp(x_norm, min=1e-2)
    
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
    # Debug prints for input
    if torch.isnan(v).any():
        print("WARNING: NaN detected in poincare_exp_map input v")
        print("v requires_grad:", v.requires_grad)
        print("v grad_fn:", v.grad_fn)
    if torch.isinf(v).any():
        print("WARNING: Inf detected in poincare_exp_map input v")
    
    # Ensure inputs are finite
    if not torch.isfinite(v).all():
        print("WARNING: Non-finite values in poincare_exp_map input, returning zeros")
        return torch.zeros_like(v)
    
    # Compute the norm of the tangent vector
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    
    # Debug prints for norm calculation
    if torch.isnan(v_norm).any():
        print("WARNING: NaN detected in v_norm calculation")
        print("v_norm requires_grad:", v_norm.requires_grad)
        print("v_norm grad_fn:", v_norm.grad_fn)
    if torch.isinf(v_norm).any():
        print("WARNING: Inf detected in v_norm calculation")
    
    # Check for NaN in norm calculation
    if torch.isnan(v_norm).any():
        print("WARNING: NaN in v_norm calculation, returning zeros")
        return torch.zeros_like(v)
    
    # Handle zero norm case with better numerical stability
    v_norm = torch.clamp(v_norm, min=1e-2, max=1e2)
    
    # Convert c to tensor if it's a float
    if isinstance(c, float):
        c = torch.tensor(c, device=v.device, dtype=v.dtype)
    
    # Ensure c is positive and finite
    c = torch.clamp(c, min=1e-8, max=1e6)
    
    # Debug prints for curvature parameter
    if torch.isnan(c).any():
        print("WARNING: NaN detected in curvature parameter c")
        print("c requires_grad:", c.requires_grad)
        print("c grad_fn:", c.grad_fn)
    if torch.isinf(c).any():
        print("WARNING: Inf detected in curvature parameter c")
    
    # Compute sqrt(c) safely
    sqrt_c = torch.sqrt(c)
    if torch.isnan(sqrt_c).any() or torch.isinf(sqrt_c).any():
        print("WARNING: Invalid sqrt_c, returning zeros")
        return torch.zeros_like(v)
    
    # Compute tanh argument with bounds
    tanh_arg = sqrt_c * v_norm
    tanh_arg = torch.clamp(tanh_arg, min=-10.0, max=10.0)  # Prevent overflow in tanh
    
    # Debug prints for tanh argument
    if torch.isnan(tanh_arg).any():
        print("WARNING: NaN detected in tanh_arg calculation")
        print("tanh_arg requires_grad:", tanh_arg.requires_grad)
        print("tanh_arg grad_fn:", tanh_arg.grad_fn)
    if torch.isinf(tanh_arg).any():
        print("WARNING: Inf detected in tanh_arg calculation")
    
    # Check for invalid tanh argument
    if torch.isnan(tanh_arg).any() or torch.isinf(tanh_arg).any():
        print("WARNING: Invalid tanh argument, returning zeros")
        return torch.zeros_like(v)
    
    # Compute exponential map with better numerical stability
    exp_v = torch.tanh(tanh_arg) * v / v_norm / sqrt_c
    
    # Debug prints for final output
    if torch.isnan(exp_v).any():
        print("WARNING: NaN detected in final exp_v calculation")
        print("exp_v requires_grad:", exp_v.requires_grad)
        print("exp_v grad_fn:", exp_v.grad_fn)
    if torch.isinf(exp_v).any():
        print("WARNING: Inf detected in final exp_v calculation")
    
    # Check for invalid output
    if torch.isnan(exp_v).any() or torch.isinf(exp_v).any():
        print("WARNING: Invalid final exp_v, returning zeros")
        return torch.zeros_like(v)
    
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

def lorentz_loss_func(hidden_input, hidden_output):
    """
    Compute the Lorentz distance loss between hidden_input and hidden_output.
    """
    # Hyperbolic entailment loss: text should entail matching image.
    # TODO: make this a parameter
    _curv = 1.0
    _angle = L.oxy_angle(hidden_input, hidden_output, _curv)
    _aperture = L.half_aperture(hidden_input, _curv)
    entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()
    return entailment_loss

class PoincareLogExpDistanceHead(nn.Module):
    def __init__(self, embedding_weight, eps=1e-2, scale=False, hidden_dim=None):
        super().__init__()
        print('---------------- PoincareLogExpDistanceHead initialized ---------------')
        self.weight = embedding_weight  # shape (vocab_size, hidden_dim)
        self.eps = eps  # Small epsilon to prevent numerical issues
        self.use_scale = scale
        if scale:
            self.hidden_state_scale = nn.Parameter(torch.tensor(1.0))
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
        
        # Use hidden_dim from config if not provided
        if hidden_dim is None:
            # Try to get from embedding_weight, fallback to 128 (or raise error)
            if hasattr(embedding_weight, "shape") and len(embedding_weight.shape) == 2:
                hidden_dim = embedding_weight.shape[1]
            else:
                hidden_dim = 128  # or raise an error, or get from config
        
        # Initialize tangent transform with smaller weights
        self.tangent_transform = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.tangent_transform.weight)

    def forward(self, hidden_states):
        # Step 1: Apply scaling if enabled
        batch_size, seq_len, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Check for NaN in input
        if torch.isnan(hidden_states).any():
            # Create zero tensor that maintains gradient connection
            zero_logits = torch.zeros(batch_size, seq_len, self.weight.shape[0], device=hidden_states.device, requires_grad=True)
            # Connect to computation graph by adding 0 * hidden_states.sum()
            zero_logits = zero_logits + 0.0 * hidden_states.sum()
            return zero_logits
        
        if self.use_scale:
            hidden_states = hidden_states * self.hidden_state_scale
        
        # Step 2: Apply transformation in tangent space
        hidden_states_transformed = self.tangent_transform(hidden_states)
        
        # Check for NaN after transformation
        if torch.isnan(hidden_states_transformed).any():
            # Create zero tensor that maintains gradient connection
            zero_logits = torch.zeros(batch_size, seq_len, self.weight.shape[0], device=hidden_states.device, requires_grad=True)
            # Connect to computation graph by adding 0 * hidden_states_transformed.sum()
            zero_logits = zero_logits + 0.0 * hidden_states_transformed.sum()
            return zero_logits

        # Step 3: Apply exponential map to Poincaré ball
        hidden_states_exp = poincare_exp_map(hidden_states_transformed) # (batch_size * seq_len, hidden_dim)
        hidden_states_exp = project_to_poincare_ball(hidden_states_exp)
        
        # Check for NaN after exponential map
        if torch.isnan(hidden_states_exp).any():
            # Create zero tensor that maintains gradient connection
            zero_logits = torch.zeros(batch_size, seq_len, self.weight.shape[0], device=hidden_states.device, requires_grad=True)
            # Connect to computation graph by adding 0 * hidden_states_exp.sum()
            zero_logits = zero_logits + 0.0 * hidden_states_exp.sum()
            return zero_logits
        
        # Step 4: Project vocabulary embeddings to Poincaré ball
        vocab_embeddings_poincare = poincare_exp_map(self.weight) # (vocab_size, hidden_dim)
        vocab_embeddings_poincare = project_to_poincare_ball(vocab_embeddings_poincare)
        
        # Check for NaN in vocabulary embeddings
        if torch.isnan(vocab_embeddings_poincare).any():
            # Create zero tensor that maintains gradient connection
            zero_logits = torch.zeros(batch_size, seq_len, self.weight.shape[0], device=hidden_states.device, requires_grad=True)
            # Connect to computation graph by adding 0 * vocab_embeddings_poincare.sum()
            zero_logits = zero_logits + 0.0 * vocab_embeddings_poincare.sum()
            return zero_logits
        
        # Step 5: Compute Poincaré distances
        distances = poincare_distance(
            hidden_states_exp.unsqueeze(1),
            vocab_embeddings_poincare.unsqueeze(0),
            eps=self.eps
        )

        # Check for NaN in distances
        if torch.isnan(distances).any():
            # Create zero tensor that maintains gradient connection
            zero_logits = torch.zeros(batch_size, seq_len, self.weight.shape[0], device=hidden_states.device, requires_grad=True)
            # Connect to computation graph by adding 0 * distances.sum()
            zero_logits = zero_logits + 0.0 * distances.sum()
            return zero_logits

        logits = -distances  # Negate to convert distance to similarity
        logits = logits.view(batch_size, seq_len, -1)  # Reshape back to (batch, seq_len, vocab_size)
        if self.use_scale:
            logits = logits * self.logit_scale  # Apply the scale parameter
        return logits

################

def lorentz_exp_map(v, c=1.0):
    """
    v: (..., d)
    Returns:
        exp_v: (..., d)
    """
    base_point = torch.zeros_like(v)
    base_point = torch.cat((torch.ones_like(base_point[:, :1]), base_point), dim=-1)
    if isinstance(c, float):
        c = torch.tensor(c, device=v.device, dtype=v.dtype)
    base_point = base_point / torch.sqrt(c)
    v = torch.cat((torch.zeros_like(v[:, :1]), v), dim=-1)
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    v_norm = torch.clamp(v_norm, min=1e-3)
    exp_v = torch.cosh(torch.sqrt(c) * v_norm) * base_point + torch.sinh(torch.sqrt(c) * v_norm) * v / v_norm / torch.sqrt(c)
    exp_v = exp_v[:, 1:]
    return exp_v
    

def lorentz_log_map(v, c=1.0):
    """
    v: (..., d)
    Returns:
        log_v: (..., d)
    """
    v = lift_to_lorentz(v, c=c)
    base_point = torch.zeros_like(v)
    base_point = torch.cat((torch.ones_like(base_point[:, :1]), base_point), dim=-1)
    if isinstance(c, float):
        c = torch.tensor(c, device=v.device, dtype=v.dtype)
    base_point = base_point / torch.sqrt(c)
    cxz_product = torch.sum(v * base_point, dim=-1, keepdim=True) * c
    projz_x = v + cxz_product * base_point
    log_v = torch.acosh(- cxz_product) / torch.sqrt(cxz_product * cxz_product - 1) * projz_x
    log_v = log_v[:, 1:]
    return log_v

class LorentzLogExpDistanceHead(nn.Module):
    def __init__(self, embedding_weight, eps=1e-2, scale=False, hidden_dim=None):
        super().__init__()
        print('---------------- LorentzLogExpDistanceHead initialized ---------------')
        self.weight = embedding_weight  # shape (vocab_size, hidden_dim)
        self.eps = eps  # Small epsilon to prevent numerical issues
        self.use_scale = scale
        if scale:
            self.hidden_state_scale = nn.Parameter(torch.tensor(1.0))
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
        
        # Use hidden_dim from config if not provided
        if hidden_dim is None:
            # Try to get from embedding_weight, fallback to 128 (or raise error)
            if hasattr(embedding_weight, "shape") and len(embedding_weight.shape) == 2:
                hidden_dim = embedding_weight.shape[1]
            else:
                hidden_dim = 128  # or raise an error, or get from config
        
        # Initialize tangent transform with smaller weights
        self.tangent_transform = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.tangent_transform.weight)

    def forward(self, hidden_states):
        # Step 1: Apply scaling if enabled
        batch_size, seq_len, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Check for NaN in input
        if torch.isnan(hidden_states).any():
            # Create zero tensor that maintains gradient connection
            zero_logits = torch.zeros(batch_size, seq_len, self.weight.shape[0], device=hidden_states.device, requires_grad=True)
            # Connect to computation graph by adding 0 * hidden_states.sum()
            zero_logits = zero_logits + 0.0 * hidden_states.sum()
            return zero_logits
        
        if self.use_scale:
            hidden_states = hidden_states * self.hidden_state_scale
        
        # Step 2: Apply transformation in tangent space
        hidden_states_transformed = self.tangent_transform(hidden_states)
        
        # Check for NaN after transformation
        if torch.isnan(hidden_states_transformed).any():
            # Create zero tensor that maintains gradient connection
            zero_logits = torch.zeros(batch_size, seq_len, self.weight.shape[0], device=hidden_states.device, requires_grad=True)
            # Connect to computation graph by adding 0 * hidden_states_transformed.sum()
            zero_logits = zero_logits + 0.0 * hidden_states_transformed.sum()
            return zero_logits

        # Step 3: Apply exponential map to Poincaré ball
        hidden_states_exp = lorentz_exp_map(hidden_states_transformed) # (batch_size * seq_len, hidden_dim)
        
        # Check for NaN after exponential map
        if torch.isnan(hidden_states_exp).any():
            # Create zero tensor that maintains gradient connection
            zero_logits = torch.zeros(batch_size, seq_len, self.weight.shape[0], device=hidden_states.device, requires_grad=True)
            # Connect to computation graph by adding 0 * hidden_states_exp.sum()
            zero_logits = zero_logits + 0.0 * hidden_states_exp.sum()
            return zero_logits
        
        # Step 4: Project vocabulary embeddings to Poincaré ball
        vocab_embeddings_lorentz = lorentz_exp_map(self.weight) # (vocab_size, hidden_dim)
        
        # Check for NaN in vocabulary embeddings
        if torch.isnan(vocab_embeddings_lorentz).any():
            # Create zero tensor that maintains gradient connection
            zero_logits = torch.zeros(batch_size, seq_len, self.weight.shape[0], device=hidden_states.device, requires_grad=True)
            # Connect to computation graph by adding 0 * vocab_embeddings_poincare.sum()
            zero_logits = zero_logits + 0.0 * vocab_embeddings_lorentz.sum()
            return zero_logits
        
        # Step 5: Compute Poincaré distances
        distances = lorentz_distance(
            hidden_states_exp,
            vocab_embeddings_lorentz,
            eps=self.eps
        )

        # Check for NaN in distances
        if torch.isnan(distances).any():
            # Create zero tensor that maintains gradient connection
            zero_logits = torch.zeros(batch_size, seq_len, self.weight.shape[0], device=hidden_states.device, requires_grad=True)
            # Connect to computation graph by adding 0 * distances.sum()
            zero_logits = zero_logits + 0.0 * distances.sum()
            return zero_logits

        logits = -distances  # Negate to convert distance to similarity
        logits = logits.view(batch_size, seq_len, -1)  # Reshape back to (batch, seq_len, vocab_size)
        if self.use_scale:
            logits = logits * self.logit_scale  # Apply the scale parameter
        return logits

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
        nn.init.xavier_uniform_(self.model.embed_tokens.weight)

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
        nn.init.xavier_uniform_(self.model.embed_tokens.weight)

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
        nn.init.xavier_uniform_(self.model.embed_tokens.weight)

AutoConfig.register("poincare_log_exp_all_wo_norm_config", PoincareLogExpAllWoNormConfig)
AutoModelForCausalLM.register(PoincareLogExpAllWoNormConfig, PoincareLogExpAllWoNormForCausalLM)

class LorentzLogExpConfig(Qwen2Config):
    """
    Custom configuration for Qwen2 model with Lorentz log-exp distance head.
    """
    model_type = "lorentz_log_exp_config"

class LorentzLogExpForCausalLM(Qwen2ForCausalLM):
    config_class = LorentzLogExpConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = LorentzLogExpDistanceHead(self.get_input_embeddings().weight, scale=True)
        nn.init.xavier_uniform_(self.model.embed_tokens.weight)

AutoConfig.register("lorentz_log_exp_config", LorentzLogExpConfig)
AutoModelForCausalLM.register(LorentzLogExpConfig, LorentzLogExpForCausalLM)

class LorentzLogExpWoNormConfig(Qwen2Config):
    """
    Custom configuration for Qwen2 model with Lorentz log-exp distance head without final norm.
    """
    model_type = "lorentz_log_exp_wo_norm_config"

class LorentzLogExpWoNormForCausalLM(Qwen2ForCausalLM):
    config_class = LorentzLogExpWoNormConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = LorentzLogExpDistanceHead(self.get_input_embeddings().weight, scale=True)
        self.model.norm = nn.Identity()
        nn.init.xavier_uniform_(self.model.embed_tokens.weight)

AutoConfig.register("lorentz_log_exp_wo_norm_config", LorentzLogExpWoNormConfig)
AutoModelForCausalLM.register(LorentzLogExpWoNormConfig, LorentzLogExpWoNormForCausalLM)

class LorentzLogExpAllWoNormConfig(Qwen2Config):
    """
    Custom configuration for Qwen2 model with Lorentz log-exp distance head and all norm layers removed.
    """
    model_type = "lorentz_log_exp_all_wo_norm_config"

class LorentzLogExpAllWoNormForCausalLM(Qwen2ForCausalLM):
    config_class = LorentzLogExpAllWoNormConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_head = LorentzLogExpDistanceHead(self.get_input_embeddings().weight, scale=True)
        remove_all_norms(self)
        nn.init.xavier_uniform_(self.model.embed_tokens.weight)

AutoConfig.register("lorentz_log_exp_all_wo_norm_config", LorentzLogExpAllWoNormConfig)
AutoModelForCausalLM.register(LorentzLogExpAllWoNormConfig, LorentzLogExpAllWoNormForCausalLM)


################

if __name__ == "__main__":
    # import sys
    # import os
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    # from calc_tokenizer import CalcTokenizer
    # tokenizer = CalcTokenizer()
    # print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    # from transformers import Qwen2Config
    # config = Qwen2Config(
    #     vocab_size=tokenizer.vocab_size + 1,
    #     hidden_size=128,
    #     intermediate_size=512,
    #     num_hidden_layers=8,
    #     num_attention_heads=4,
    #     num_key_value_heads=4,
    #     max_position_embeddings=64,
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
    # model = Qwen2ForCausalLM(config)
    # model.save_pretrained('hyperbolic_models/qwen2_base')
    # tokenizer.save_pretrained('hyperbolic_models/qwen2_base')
    # print("Model and tokenizer saved to hyperbolic_models/qwen2_base")

    # # --- Old wo_norm block commented out ---
    
    # import sys
    # import os
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    # from calc_tokenizer import CalcTokenizer
    # tokenizer = CalcTokenizer()
    # print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    # from transformers import Qwen2Config
    # config = PoincareLogExpAllWoNormConfig(
    #     vocab_size=tokenizer.vocab_size + 1,
    #     hidden_size=128,
    #     intermediate_size=512,
    #     num_hidden_layers=8,
    #     num_attention_heads=4,
    #     num_key_value_heads=4,
    #     max_position_embeddings=64,
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
    # model = PoincareLogExpAllWoNormForCausalLM(config)
    # model.save_pretrained('hyperbolic_models/poincare_log_exp_all_wo_norm')
    # tokenizer.save_pretrained('hyperbolic_models/poincare_log_exp_all_wo_norm')
    # print("Model and tokenizer saved to hyperbolic_models/poincare_log_exp_all_wo_norm")
    

    # # --- New with-norm block ---
    # import sys
    # import os
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    # from calc_tokenizer import CalcTokenizer
    # tokenizer = CalcTokenizer()
    # print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    # from transformers import Qwen2Config
    # config = PoincareLogExpConfig(
    #     vocab_size=tokenizer.vocab_size + 1,
    #     hidden_size=128,
    #     intermediate_size=512,
    #     num_hidden_layers=8,
    #     num_attention_heads=4,
    #     num_key_value_heads=4,
    #     max_position_embeddings=64,
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
    # model = PoincareLogExpForCausalLM(config)
    # # Ensure model_type is set correctly before saving
    # model.config.model_type = "poincare_log_exp"
    # model.save_pretrained('hyperbolic_models/poincare_log_exp')
    # tokenizer.save_pretrained('hyperbolic_models/poincare_log_exp')
    # print("Model and tokenizer saved to hyperbolic_models/poincare_log_exp")

    # # --- wo_norm block ---
    
    # import sys
    # import os
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    # from calc_tokenizer import CalcTokenizer
    # tokenizer = CalcTokenizer()
    # print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    # from transformers import Qwen2Config
    # config = PoincareLogExpWoNormConfig(
    #     vocab_size=tokenizer.vocab_size + 1,
    #     hidden_size=128,
    #     intermediate_size=512,
    #     num_hidden_layers=8,
    #     num_attention_heads=4,
    #     num_key_value_heads=4,
    #     max_position_embeddings=64,
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
    # model = PoincareLogExpWoNormForCausalLM(config)
    # model.save_pretrained('hyperbolic_models/poincare_log_exp_wo_norm')
    # tokenizer.save_pretrained('hyperbolic_models/poincare_log_exp_wo_norm')
    # print("Model and tokenizer saved to hyperbolic_models/poincare_log_exp_wo_norm")
    
    # # --- LorentzWoNormLogExp block ---
    
    # import sys
    # import os
    # sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    # from calc_tokenizer import CalcTokenizer
    # tokenizer = CalcTokenizer()
    # print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    # from transformers import Qwen2Config
    # config = LorentzLogExpWoNormConfig(
    #     vocab_size=tokenizer.vocab_size + 1,
    #     hidden_size=128,
    #     intermediate_size=512,
    #     num_hidden_layers=8,
    #     num_attention_heads=4,
    #     num_key_value_heads=4,
    #     max_position_embeddings=64,
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
    # model = LorentzLogExpWoNormForCausalLM(config)
    # model.save_pretrained('hyperbolic_models/lorentz_log_exp_wo_norm')
    # tokenizer.save_pretrained('hyperbolic_models/lorentz_log_exp_wo_norm')
    # print("Model and tokenizer saved to hyperbolic_models/lorentz_log_exp_wo_norm")


    # --- LorentzLogExpAllWoNorm block ---

    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from calc_tokenizer import CalcTokenizer
    tokenizer = CalcTokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    from transformers import Qwen2Config
    config = LorentzLogExpAllWoNormConfig(
        vocab_size=tokenizer.vocab_size + 1,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
        use_cache=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        attention_bias=False,
        max_window_layers=0,
    )
    model = LorentzLogExpAllWoNormForCausalLM(config)
    model.save_pretrained('hyperbolic_models/lorentz_log_exp_all_wo_norm')
    tokenizer.save_pretrained('hyperbolic_models/lorentz_log_exp_all_wo_norm')
    print("Model and tokenizer saved to hyperbolic_models/lorentz_log_exp_all_wo_norm")