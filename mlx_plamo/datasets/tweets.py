import argparse
import json

from datasets import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--out-hf-path", type=str)
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    with open(args.out_path, "w") as f:
        for datum in data:
            text = datum["tweet"]["full_text"]
            print(json.dumps({"text": text}), file=f)

    dataset = Dataset.from_text(args.out_path)
    dataset.save_to_disk(args.out_hf_path)
