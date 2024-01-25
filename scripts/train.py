# From: https://github.com/ml-explore/mlx-examples/blob/527cea4027974b6a44d3d16d62b385f10dcbcb65/lora/lora.py

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Generator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from loguru import logger
from mlx.utils import tree_flatten
from mlx_plamo.generate import generate_step
from mlx_plamo.models.plamo import LoRALinear, Model
from mlx_plamo.utils import load as load_model
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    # Generation args
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=256,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument("--temp", type=float, default=0.8, help="The sampling temperature")
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt for generation",
        default=None,
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
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
    parser.add_argument("--batch-size", type=int, default=16, help="Minibatch size.")
    parser.add_argument("--iters", type=int, default=2000, help="Iterations to train for.")
    parser.add_argument(
        "--val-batches",
        type=int,
        default=25,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Adam learning rate.")
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=200,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        default=None,
        help="Load path to resume training with the given adapter weights.",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for results.")
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text") -> None:
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(line) for line in fid]
        self._key = key

    def __getitem__(self, idx: int) -> str:
        if self._data:
            datum: dict[str, str] = self._data[idx]
            return datum[self._key]
        else:
            raise IndexError(f"Index {idx} out of bounds")

    def __len__(self) -> int:
        if self._data is not None:
            return len(self._data)
        else:
            return 0


def load(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset]:
    names = ("train", "valid", "test")
    train, valid, test = (Dataset(Path(args.data) / f"{n}.jsonl") for n in names)
    if args.train and len(train) == 0:
        raise ValueError("Training set not found or empty. Must provide training set for fine-tuning.")
    if args.train and len(valid) == 0:
        raise ValueError("Validation set not found or empty. Must provide validation set for fine-tuning.")
    if args.test and len(test) == 0:
        raise ValueError("Test set not found or empty. Must provide test set for evaluation.")
    return train, valid, test


def loss(model: Model, inputs: mx.array, targets: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
    # Run model on inputs
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(
    dset: Dataset, tokenizer: PreTrainedTokenizer, batch_size: int, train: bool = False
) -> Generator[mx.array, Any, None]:
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [tokenizer.encode(dset[indices[i + j]]) for j in range(batch_size)]
            lengths = [len(x) for x in batch]

            # Check if any sequence is longer than 2048 tokens
            if max(lengths) > 2048:
                logger.warning(
                    "[WARNING] Some sequences are longer than 2048 tokens. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the max length
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch_mx = mx.array(batch_arr)
            yield (batch_mx[:, :-1], batch_mx[:, 1:], mx.array(lengths))

        if not train:
            break


# def evaluate(
#     model: Model, dataset: Dataset, loss: Callable, tokenizer: PreTrainedTokenizer, batch_size: int, num_batches: int
# ) -> float:
#     all_losses = []
#     ntokens = 0
#     for it, batch in zip(
#         range(num_batches),
#         iterate_batches(dataset, tokenizer, batch_size),
#     ):
#         losses, toks = loss(model, *batch)
#         all_losses.append((losses * toks).item())
#         ntokens += toks.item()

#     mean_loss: float = np.sum(all_losses) / ntokens
#     return mean_loss


def train(
    model: Model,
    train_set: Dataset,
    val_set: Dataset,
    optimizer: optim.Optimizer,
    loss: Callable,
    tokenizer: PreTrainedTokenizer,
    args: argparse.Namespace,
    writer: SummaryWriter,
) -> None:
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(range(args.iters), iterate_batches(train_set, tokenizer, args.batch_size, True)):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Model update
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, lvalue)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks.item()

        # Report training loss if needed
        if (it + 1) % args.steps_per_report == 0:
            train_loss: float = np.mean(losses)

            stop = time.perf_counter()
            logger.info(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {args.steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            writer.add_scalar("loss/train", train_loss, it)  # type: ignore
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # # Report validation loss if needed
        # if it == 0 or (it + 1) % args.steps_per_eval == 0:
        #     stop = time.perf_counter()
        #     val_loss = evaluate(model, val_set, loss, tokenizer, args.batch_size, args.val_batches)
        #     logger.info(f"Iter {it + 1}: " f"Val loss {val_loss:.3f}, " f"Val took {(time.perf_counter() - stop):.3f}s")

        #     start = time.perf_counter()

        # Save adapter weights if needed
        if (it + 1) % args.save_every == 0:
            out_fn = os.path.join(args.output_dir, f"adapters_{it + 1}.npz")
            mx.savez(out_fn, **dict(tree_flatten(model.trainable_parameters())))
            logger.info(f"Iter {it + 1}: Saved adapter weights to {out_fn}.")


def generate(model: Model, prompt: str, tokenizer: PreTrainedTokenizer, args: argparse.Namespace) -> None:
    logger.info(prompt, end="", flush=True)

    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    for token, n in zip(
        generate_step(prompt, model, args.temp),
        range(args.max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        logger.info(s[skip:], end="", flush=True)
        skip = len(s)
    logger.info(tokenizer.decode(tokens)[skip:], flush=True)
    logger.info("=" * 10)
    if len(tokens) == 0:
        logger.info("No tokens generated for this prompt")
        return


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    logger.add(os.path.join(args.output_dir, "log.txt"))
    logger.info(f"args: {json.dumps(vars(args), indent=4)}")
    writer = SummaryWriter(log_dir=args.output_dir)  # type: ignore

    np.random.seed(args.seed)

    logger.info("Loading pretrained model")
    model, tokenizer = load_model(args.model)

    # Freeze all layers other than LORA linears
    model.freeze()
    for layer in model.model.layers.layers[len(model.model.layers.layers) - args.lora_layers :]:
        # q_proj
        layer.self_attn.q_proj = LoRALinear.from_linear(
            layer.self_attn.q_proj, args.lora_rank, args.lora_bias, args.lora_scale
        )
        # k_proj
        layer.self_attn.k_proj = LoRALinear.from_linear(
            layer.self_attn.k_proj, args.lora_rank, args.lora_bias, args.lora_scale
        )
        # v_proj
        layer.self_attn.v_proj = LoRALinear.from_linear(
            layer.self_attn.v_proj, args.lora_rank, args.lora_bias, args.lora_scale
        )
        # o_proj
        layer.self_attn.o_proj = LoRALinear.from_linear(
            layer.self_attn.o_proj, args.lora_rank, args.lora_bias, args.lora_scale
        )
        # gate_proj
        layer.mlp.gate_proj = LoRALinear.from_linear(
            layer.mlp.gate_proj, args.lora_rank, args.lora_bias, args.lora_scale
        )
        # up_proj
        layer.mlp.up_proj = LoRALinear.from_linear(
            layer.mlp.up_proj, args.lora_rank, args.lora_bias, args.lora_scale
        )
        # down_proj
        layer.mlp.down_proj = LoRALinear.from_linear(
            layer.mlp.down_proj, args.lora_rank, args.lora_bias, args.lora_scale
        )

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    logger.info(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    logger.info(f"Trainable parameters {p:.3f}M")

    logger.info("Loading datasets")
    train_set, valid_set, test_set = load(args)

    # Resume training the given adapters.
    if args.resume_adapter_file is not None:
        logger.info(f"Loading pretrained adapters from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    if args.train:
        logger.info("Training")
        opt = optim.Adam(learning_rate=args.learning_rate)

        # Train model
        train(model, train_set, valid_set, opt, loss, tokenizer, args, writer)

        # Save adapter weights
        out_fn = os.path.join(args.output_dir, "final.npz")
        mx.savez(out_fn, **dict(tree_flatten(model.trainable_parameters())))

    # Load the LoRA adapter weights which we assume should exist by this point
    last_adapter_file = os.path.join(args.output_dir, args.last_adapter_file)
    if not Path(last_adapter_file).is_file():
        raise ValueError(
            f"Adapter file {last_adapter_file} missing. " "Use --train to learn and save the adapters.npz."
        )
    model.load_weights(last_adapter_file, strict=False)

    if args.test:
        logger.info("Testing")

        # test_loss = evaluate(
        #     model,
        #     test_set,
        #     loss,
        #     tokenizer,
        #     args.batch_size,
        #     num_batches=args.test_batches,
        # )
        # test_ppl = math.exp(test_loss)

        # logger.info(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    if args.prompt is not None:
        logger.info("Generating")
        # generate(model, args.prompt, tokenizer, args)
