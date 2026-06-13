import json
import os
import random
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KTO training data by mixing SFT and negative data")
    parser.add_argument("--task", type=str, required=True, choices=["steam", "amazon"],
                        help="Task name: steam or amazon")
    parser.add_argument("--sft_path", type=str, default=None,
                        help="Path to SFT data. Default: tuning_data/sft/{task}.json")
    parser.add_argument("--neg_path", type=str, default=None,
                        help="Path to negative data. Default: tuning_data/neg/{task}_sft_neg_2000.json")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for KTO data. Default: tuning_data/kto/{task}.json")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if args.sft_path is None:
        args.sft_path = os.path.join(project_root, "tuning_data", "sft", f"{args.task}.json")
    if args.neg_path is None:
        args.neg_path = os.path.join(project_root, "tuning_data", "neg", f"{args.task}_sft_neg_2000.json")
    if args.output_path is None:
        args.output_path = os.path.join(project_root, "tuning_data", "kto", f"{args.task}.json")

    for len_thresh in [1]:
        for neg_thresh in [1]:
            with open(args.neg_path, "r") as f:
                neg_traj = json.load(f)

            with open(args.sft_path, "r") as f:
                pos_traj = json.load(f)

            combined_traj = []

            for item in pos_traj:
                combined_traj.append({
                    "input": item['input'],
                    "completion": item['target'],
                    "label": True,
                    "length": item["length"]
                })

            for item in neg_traj:
                combined_traj.append({
                    "input": item['input'],
                    "completion": item['neg_target'],
                    "label": False,
                    "length": item["length"]
                })

            random.seed(args.seed)
            random.shuffle(combined_traj)

            kto_dataset = {
                "prompt": [],
                "completion": [],
                "label": [],
                "length": []
            }

            for item in combined_traj:
                kto_dataset["prompt"].append(item["input"])
                kto_dataset["completion"].append(item["completion"])
                kto_dataset["label"].append(item["label"])
                kto_dataset["length"].append(item["length"])

            save_path = os.path.dirname(args.output_path)
            os.makedirs(save_path, exist_ok=True)
            with open(args.output_path, "w") as f:
                json.dump(kto_dataset, f)

            pos_count = sum(1 for item in combined_traj if item["label"])
            neg_count = sum(1 for item in combined_traj if not item["label"])
            print(f"[KTO] {args.task}: generated {len(combined_traj)} samples (pos={pos_count}, neg={neg_count})")
            print(f"[KTO] Saved to {args.output_path}")
