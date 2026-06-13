import json
import re
import os
import argparse

def counting_no_same_item(traj_by_line):
    log = traj_by_line
    NULL_action = 0
    recommendations = {}
    current_action = None
    for entry in log:
        try:
            if entry.startswith('Action'):
                parts = entry.split(': recommend[')
                current_action = parts[0].strip()
                recommended_item = parts[1].rstrip(']')
                recommendations[current_action] = recommended_item
            elif entry.startswith('Observation') and 'instead, recommend' in entry:
                parts = entry.split('instead, recommend[')
                new_recommendation = parts[1].rstrip(']')
                if current_action:
                    recommendations[current_action] = new_recommendation
        except:
            NULL_action += 1
    recommend_item_list = list(recommendations.values())
    for i in range(len(recommend_item_list)):
        if recommend_item_list[i] in recommend_item_list[:i]:
            return i
    return len(recommend_item_list)


def construct_training_data(traj_by_line):
    cleaned_lines = []
    i = 0
    while i < len(traj_by_line):
        line = traj_by_line[i]
        if i + 2 < len(traj_by_line) and "Episode finished, User Stop, reward=-1000.000" in traj_by_line[i + 2]:
            cleaned_lines.pop()
            break
        elif i + 1 < len(traj_by_line) and "Episode finished, User Stop, reward=-1000.000" in traj_by_line[i + 1]:
            break
        elif i + 1 < len(traj_by_line) and "Invalid Action" in traj_by_line[i + 1]:
            break
        elif line.startswith("Action") and i + 2 < len(traj_by_line) and traj_by_line[i + 2].startswith("Observation"):
            action = line
            observation = traj_by_line[i + 1]
            obs_item = observation.split("recommend[")[-1].split("]")[0]
            modified_action = action.replace(action.split("[")[-1].split("]")[0], obs_item)
            cleaned_lines.append(modified_action)
            i += 2
        elif line.startswith("Action") or line.startswith("Observation") or line.startswith("The user"):
            cleaned_lines.append(line)
            i += 1
        else:
            i += 1
    training_data = []
    no_same_len = counting_no_same_item(traj_by_line)

    def replace_recommendation(text):
        pattern = r'recommend\[(.*?)\]'
        if re.search(pattern, text):
            replaced_text = re.sub(pattern, r'recommend[ \1 ]', text)
            return replaced_text
        else:
            return text

    import copy
    cleaned_lines_orig = copy.deepcopy(cleaned_lines)
    cleaned_lines_new = [replace_recommendation(line) for line in cleaned_lines_orig]
    cleaned_lines = cleaned_lines_new
    for i, line in enumerate(cleaned_lines):
        if line.startswith("Action"):
            input_data = "\n".join(cleaned_lines[:i])
            target_data = line
            training_data.append((input_data, target_data))
            if i >= no_same_len * 2 - 1:
                break
    return training_data


def clean_traj_by_line_sampled(data, count=2000):
    cleaned_data = {}
    temp_count = 0
    for key, value in data.items():
        traj_by_line = value['traj_by_line']
        cleaned_traj_by_line = construct_training_data(traj_by_line)
        cleaned_data[key] = {
            "userid": value["userid"],
            "prompt": value["prompt"],
            "traj": value["traj"],
            "traj_by_line": value['traj_by_line'],
            "sft_data_orig": cleaned_traj_by_line
        }
        temp_count += 1
        if temp_count == count:
            break
    return cleaned_data


PROMPT_PREFIX = """Solve a recommendation task with interleaving Action, Observation steps. \nAction can be the following types: \n(1) recommend[item], which recommend an item to user based on user's interest. Your goal is to meet the user's interest as much as possible and make recommendations to users as many times as possible. Note that if the user is not satisfied with your recommendations, he will quit and not accept new recommendations\n\nYou may take as many steps as necessary.\nHere are some examples:\n\nQuestion: The user's viewing history is ['Pretty in Pink', "One Flew Over the Cuckoo's Nest", 'Ransom', 'Saving Private Ryan', 'X-Men', 'Coyote Ugly', 'The Patriot', 'Me, Myself and Irene', 'Gone in 60 Seconds', 'The Perfect Storm', 'Titanic', 'The Haunting', 'Bedknobs and Broomsticks', 'Clerks', 'The Matrix', 'The Shawshank Redemption', 'Vacation', 'Father of the Bride', 'Wallace & Gromit: The Best of Aardman Animation', 'Back to the Future', 'Fight Club'], please recommend item for this user\nAction 1: recommend[Forrest Gump]\nObservation 1: Episode continue, reward=0.30697035862193367\nAction 2: recommend[The Godfather]\nObservation 2: Episode finished, reward=0.49717313011547304\n\n(END OF EXAMPLES)\nQuestion: """


def construct_prompt(traj, k):
    training_data_list = []
    for traj_id in traj.keys():
        if len(traj[traj_id]["sft_data_orig"]) > k:
            for (input, target) in traj[traj_id]["sft_data_orig"]:
                training_data_list.append({
                    "input": PROMPT_PREFIX + input,
                    "target": target,
                    "length": len(traj[traj_id]["sft_data_orig"])
                })
    return training_data_list


def get_training_data(cleaned_traj, traj_len_thresh):
    pattern = r"(Thought \d+:|Action \d+:)"
    sft_traj = construct_prompt(cleaned_traj, traj_len_thresh)
    processed_data = []
    for item in sft_traj:
        matches = re.findall(pattern, item['target'])[0]
        if len(item['target'].split(matches)[1][1:]) != 0:
            processed_data.append({
                "input": item['input'] + "\n" + matches,
                "target": item['target'].split(matches)[1][1:],
                "length": item["length"]
            })
    return processed_data


def get_new_samples_between_thresholds(cleaned_traj, new_thresh, old_thresh, count):
    new_data = get_training_data(cleaned_traj, new_thresh)
    old_keys = {
        (item["input"], item["target"], item["length"])
        for item in get_training_data(cleaned_traj, old_thresh)
    }
    added = []
    for item in new_data:
        key = (item["input"], item["target"], item["length"])
        if key not in old_keys:
            added.append(item)
            if len(added) == count:
                break
    return added


def print_preview_samples(samples, new_thresh, old_thresh):
    print(f"[SFT] preview: first {len(samples)} samples added by thresh={new_thresh} vs thresh={old_thresh}")
    for idx, item in enumerate(samples, 1):
        prompt = item["input"].split("Question: ")[-1]
        prompt_tail = prompt[-500:]
        print(f"\n--- added sample {idx} | length={item['length']} ---")
        print(f"input_tail:\n{prompt_tail}")
        print(f"target:\n{item['target']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--task", type=str, required=True, choices=["steam", "amazon"],
                        help="Task name: steam or amazon")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Path to trajectory JSON file. Default: data_preprocess/orig_traj_react_format/{task}_traj_2000.json")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for SFT data. Default: tuning_data/sft/{task}.json")
    parser.add_argument("--count", type=int, default=2000,
                        help="Number of trajectories to sample")
    parser.add_argument("--traj_len_thresh", type=int, default=1,
                        help="Keep trajectories whose generated SFT length is greater than this value. Default keeps old behavior (1)")
    parser.add_argument("--preview_new_count", type=int, default=0,
                        help="Print the first N samples added by --traj_len_thresh compared with --compare_traj_len_thresh")
    parser.add_argument("--compare_traj_len_thresh", type=int, default=1,
                        help="Baseline threshold used by --preview_new_count")
    parser.add_argument("--preview_output_path", type=str, default=None,
                        help="Optional JSON path for samples shown by --preview_new_count")
    args = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if args.input_path is None:
        args.input_path = os.path.join(project_root, "data_preprocess", "orig_traj_react_format", f"{args.task}_traj_2000.json")
    if args.output_path is None:
        args.output_path = os.path.join(project_root, "tuning_data", "sft", f"{args.task}.json")

    with open(args.input_path, "r") as f:
        train_traj = json.load(f)

    cleaned_traj = clean_traj_by_line_sampled(train_traj, args.count)

    saving_path = os.path.dirname(args.output_path)
    os.makedirs(saving_path, exist_ok=True)

    sft_traj = get_training_data(cleaned_traj, args.traj_len_thresh)
    print(f"[SFT] {args.task}: generated {len(sft_traj)} samples (traj_len_thresh={args.traj_len_thresh})")
    with open(args.output_path, "w") as f:
        json.dump(sft_traj, f, ensure_ascii=False, indent=4)
    print(f"[SFT] Saved to {args.output_path}")

    if args.preview_new_count > 0:
        preview_samples = get_new_samples_between_thresholds(
            cleaned_traj,
            new_thresh=args.traj_len_thresh,
            old_thresh=args.compare_traj_len_thresh,
            count=args.preview_new_count,
        )
        print_preview_samples(preview_samples, args.traj_len_thresh, args.compare_traj_len_thresh)
        if args.preview_output_path:
            preview_dir = os.path.dirname(args.preview_output_path)
            if preview_dir:
                os.makedirs(preview_dir, exist_ok=True)
            with open(args.preview_output_path, "w") as f:
                json.dump(preview_samples, f, ensure_ascii=False, indent=4)
            print(f"[SFT] Preview saved to {args.preview_output_path}")
