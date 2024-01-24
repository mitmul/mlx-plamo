# From: https://github.com/ml-explore/mlx-examples/blob/527cea4027974b6a44d3d16d62b385f10dcbcb65/lora/fuse.py

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_plamo.convert import make_shards, upload_to_hub
from mlx_plamo.models.plamo import LoRALinear
from mlx_plamo.utils import load
from transformers import PretrainedConfig, PreTrainedTokenizer


def save_model(save_dir: str, weights: mx.array, tokenizer: PreTrainedTokenizer, config: PretrainedConfig) -> None:
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    shards = make_shards(weights)
    for i, shard in enumerate(shards):
        # TODO use HF file name scheme for simplicity
        mx.save_safetensors(str(save_dir_path / f"weights.{i:02d}.safetensors"), shard)
    tokenizer.save_pretrained(save_dir)
    with open(save_dir_path / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--save-path",
        default="lora_fused_model",
        help="The path to save the fused model.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Path to the trained adapter weights (npz or safetensors).",
    )
    parser.add_argument(
        "--hf-path",
        help=(
            "Path to the original Hugging Face model. This is " "required for upload if --model is a local directory."
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--upload-name",
        help="The name of model to upload to Hugging Face MLX Community",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="Number of layers to fine-tune",
    )
    parser.add_argument(
        "--lora-bias",
        action="store_true",
        default=False,
        help="Use bias for LoRA linear layers",
    )
    parser.add_argument(
        "--lora-scale",
        type=int,
        default=128,
        help="Number of layers to fine-tune",
    )

    print("Loading pretrained model")
    args = parser.parse_args()

    with open(Path(args.model) / "config.json", "r") as f:
        config = json.load(f)
    model, tokenizer = load(args.model)

    # Load adapters and get number of LoRA layers
    adapters = list(mx.load(args.adapter_file).items())
    lora_layers = len([m for m in adapters if "q_proj.lora_a" in m[0]])

    # Freeze all layers other than LORA linears
    model.freeze()
    for layer in model.model.layers.layers[-lora_layers:]:
        layer.self_attn.q_proj = LoRALinear.from_linear(
            layer.self_attn.q_proj, args.lora_rank, args.lora_bias, args.lora_scale
        )
        layer.self_attn.v_proj = LoRALinear.from_linear(
            layer.self_attn.v_proj, args.lora_rank, args.lora_bias, args.lora_scale
        )

    model.update(tree_unflatten(adapters))
    fused_linears = [(n, m.to_linear()) for n, m in model.named_modules() if isinstance(m, LoRALinear)]

    model.update_modules(tree_unflatten(fused_linears))
    weights = dict(tree_flatten(model.parameters()))
    save_model(args.save_path, weights, tokenizer, config)

    if args.upload_name is not None:
        hf_path = args.hf_path
        if not Path(args.model).exists():
            # If the model path doesn't exist, assume it's an HF repo
            hf_path = args.model
        elif hf_path is None:
            raise ValueError("Must provide original Hugging Face repo to upload local model.")
        upload_to_hub(args.save_path, args.upload_name, hf_path)
