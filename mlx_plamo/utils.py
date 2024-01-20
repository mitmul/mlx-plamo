import glob
import json
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
from loguru import logger
from mlx_lm.utils import get_model_path, linear_class_predicate
from mlx_plamo.models.plamo import Model, ModelArgs
from transformers import AutoTokenizer, PreTrainedTokenizer


# From: https://github.com/ml-explore/mlx-examples/blob/61297f547b57417672738e398852265c58670102/llms/mlx_lm/utils.py#L145
def load(path_or_hf_repo: str) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """
    Load the model from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (str): The path or the huggingface repository to load the model from.

    Returns:
        Tuple[nn.Module, PreTrainedTokenizer]: The loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
            quantization = config.get("quantization", None)
    except FileNotFoundError:
        logger.error(f"Config file not found in {model_path}")
        raise
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        logger.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_args = ModelArgs.from_dict(config)
    model = Model(model_args)

    if quantization is not None:
        nn.QuantizedLinear.quantize_module(
            model,
            **quantization,
            linear_class_predicate=linear_class_predicate,
        )

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer
