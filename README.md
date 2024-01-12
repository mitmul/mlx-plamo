# mlx-plamo

An example of generating text with [PLaMo-13b](https://tech.preferred.jp/en/blog/llm-plamo/) using MLX.

### Setup

If your Python environment doesn't have poetry, install it first:

```bash
pip install poetry
```

Then, install all the dependencies:

```bash
poetry install
```

Next, download and convert the model.

Convert the weights with:

```bash
poetry run python -m mlx_plamo.convert --hf-path pfnet/plamo-13b-instruct-nc
```

By default, the conversion script will make the directory `mlx_model` and save
the converted `weights.npz`, `tokenizer.model`, and `config.json` there.

### Run

Once you've converted the weights to MLX format, you can interact with the PLaMo model:

```bash
poetry run python -m mlx_plamo.generate --instruct --prompt "コンピュータ科学とは何ですか？"
```

You will see the output like this:

```
[INFO] Loading model from plamo-13b-instruct-nc-bf16/weights.*.npz.
------
コンピュータ科学(コンピュータサイエンスまたはCSとも呼ばれる)は、コンピューターの動作原理と、コンピューターソフトウェアやハードウェアの設計と開発を扱う分野です。
------
[INFO] Prompt processing: 1.215 s
[INFO] Full generation: 4.098 s
```
