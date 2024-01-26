# From: https://github.com/ml-explore/mlx-examples/blob/61297f547b57417672738e398852265c58670102/llms/mlx_lm/generate.py

import argparse
import time

import mlx.core as mx
from loguru import logger
from mlx_lm.utils import generate_step
from mlx_plamo.utils import load

DEFAULT_MODEL_PATH = "mlx_model"
DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMP = 0.7
DEFAULT_SEED = 0


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Message to be processed by the model")
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
    parser.add_argument("--instruct", action="store_true")
    return parser


# From: https://huggingface.co/pfnet/plamo-13b-instruct
def generate_prompt(messages: list) -> str:
    sep = "\n\n### "
    prompt = [
        "以下はタスクを説明する指示で、文脈を説明した入力とペアになっています。",
        "要求を適切に補完するよう応答を書いてください。",
    ]
    roles = {"instruction": "指示", "response": "応答", "input": "入力"}
    for msg in messages:
        prompt.append(sep + roles[msg["role"]] + ":\n" + msg["content"])
    prompt.append(sep + roles["response"] + ":\n")
    return "".join(prompt)


def generate(args: argparse.Namespace) -> None:
    mx.random.seed(args.seed)
    model, tokenizer = load(args.model)

    instruction_base = [
        {
            "role": "input",
            "content": args.prompt,
        },
    ]
    if args.instruct:
        prompt = generate_prompt(instruction_base)
    else:
        prompt = args.prompt

    tokens = tokenizer.encode(prompt)
    tokens_arr = mx.array(tokens)
    tic = time.time()
    tokens = []
    skip = 0
    for token, n in zip(generate_step(tokens_arr, model, args.temp), range(args.max_tokens)):
        if token == tokenizer.eos_token_id:
            break
        if n == 0:
            prompt_time = time.time() - tic
            tic = time.time()
        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        print(s[skip:], end="", flush=True)
        skip = len(s)
    print(tokenizer.decode(tokens)[skip:], flush=True)
    gen_time = time.time() - tic
    logger.info("=" * 10)
    if len(tokens) == 0:
        logger.info("No tokens generated for this prompt")
        return
    prompt_tps = tokens_arr.size / prompt_time
    gen_tps = (len(tokens) - 1) / gen_time
    logger.info(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
    logger.info(f"Generation: {gen_tps:.3f} tokens-per-sec")


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    generate(args)
