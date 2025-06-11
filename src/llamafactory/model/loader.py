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

import os
from typing import TYPE_CHECKING, Any, Optional, TypedDict
import torch.nn as nn
from einops import rearrange

import torch
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
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from ..extras.packages import is_transformers_version_greater_than
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model

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
    def __init__(self, embedding_weight, eps=1e-5):
        super().__init__()
        print('---------------- HyperbolicDistanceHead initialized ---------------')
        self.weight = embedding_weight  # shape (vocab_size, hidden_dim)
        self.eps = eps  # Small epsilon to prevent numerical issues

    def forward(self, hidden_states):
        # compute the Poincare distance
        batch_size, seq_len, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)
        hidden_states = project_to_poincare_ball(hidden_states, self.eps)
        vocab_embeddings = project_to_poincare_ball(self.weight, self.eps)

        distances = poincare_distance(
            hidden_states.unsqueeze(1),
            vocab_embeddings.unsqueeze(0),
            eps=self.eps
        )

        logits = -distances  # Negate to convert distance to similarity
        logits = logits.view(batch_size, seq_len, -1)  # Reshape back to (batch, seq_len, vocab_size)
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
        self.lm_head = HyperbolicDistanceHead(self.get_input_embeddings().weight)

AutoConfig.register("custom_qwen2_config", CustomConfig)
AutoModelForCausalLM.register(CustomConfig, CustomModelForCausalLM)


if is_transformers_version_greater_than("4.46.0"):
    from transformers import AutoModelForImageTextToText


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)
    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        patch_processor(processor, tokenizer, model_args)
    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                load_class = AutoModelForVision2Seq
            elif (
                is_transformers_version_greater_than("4.46.0")
                and type(config) in AutoModelForImageTextToText._model_mapping.keys()
            ):  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen2_5_omni
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                model = load_class.from_pretrained(**init_kwargs)
                if getattr(model.config, "model_type", None) == "qwen2_5_omni":
                    model = model.thinker  # use part of Omni model

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
