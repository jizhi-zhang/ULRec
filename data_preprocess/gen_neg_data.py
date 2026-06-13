import json
import re
import os
import copy
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


def construct_negative_training_data(traj_by_line):
    cleaned_lines = []
    i = 0
    while i < len(traj_by_line):
        line = traj_by_line[i]
        if line.startswith("Action") and i + 2 < len(traj_by_line) and traj_by_line[i + 2].startswith("Observation"):
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
    if "Episode finished, User Stop, reward=-1000.000" in cleaned_lines[-1]:
        _ = cleaned_lines.pop()
        neg_action = cleaned_lines.pop()
        input_data = "\n".join(cleaned_lines)
        return (input_data, neg_action), int(len(cleaned_lines) + 1) / 2
    else:
        return [], 0


def clean_traj_by_line_sampled(data, count=2000):
    cleaned_data = {}
    temp_count = 0
    for key, value in data.items():
        traj_by_line = value['traj_by_line']
        cleaned_traj_by_line, length = construct_negative_training_data(traj_by_line)
        if length != 0:
            cleaned_data[key] = {
                "userid": value["userid"],
                "prompt": value["prompt"],
                "traj": value["traj"],
                "traj_by_line": value['traj_by_line'],
                "sft_data_orig": cleaned_traj_by_line,
                "traj_length": length
            }
        temp_count += 1
        if temp_count == count:
            break
    return cleaned_data


PROMPT_PREFIX = """Solve a recommendation task with interleaving Action, Observation steps. \nAction can be the following types: \n(1) recommend[item], which recommend an item to user based on user's interest. Your goal is to meet the user's interest as much as possible and make recommendations to users as many times as possible. Note that if the user is not satisfied with your recommendations, he will quit and not accept new recommendations\n\nYou may take as many steps as necessary.\nHere are some examples:\n\nQuestion: The user's viewing history is ['Pretty in Pink', "One Flew Over the Cuckoo's Nest", 'Ransom', 'Saving Private Ryan', 'X-Men', 'Coyote Ugly', 'The Patriot', 'Me, Myself and Irene', 'Gone in 60 Seconds', 'The Perfect Storm', 'Titanic', 'The Haunting', 'Bedknobs and Broomsticks', 'Clerks', 'The Matrix', 'The Shawshank Redemption', 'Vacation', 'Father of the Bride', 'Wallace & Gromit: The Best of Aardman Animation', 'Back to the Future', 'Fight Club'], please recommend item for this user\nAction 1: recommend[Forrest Gump]\nObservation 1: Episode continue, reward=0.30697035862193367\nAction 2: recommend[The Godfather]\nObservation 2: Episode finished, reward=0.49717313011547304\n\n(END OF EXAMPLES)\nQuestion: """


def construct_prompt(traj, k):
    training_data_list = []
    for traj_id in traj.keys():
        if traj[traj_id]["traj_length"] > k:
            input, neg_target = traj[traj_id]["sft_data_orig"]
            training_data_list.append({
                "input": PROMPT_PREFIX + input,
                "neg_target": neg_target,
                "length": traj[traj_id]["traj_length"]
            })
    return training_data_list


def get_training_data(cleaned_traj, traj_len_thresh):
    pattern = r"(Thought \d+:|Action \d+:)"
    sft_traj = construct_prompt(cleaned_traj, traj_len_thresh)
    processed_data = []
    for item in sft_traj:
        matches = re.findall(pattern, item['neg_target'])[0]
        if len(item['neg_target'].split(matches)[1][1:]) != 0:
            processed_data.append({
                "input": item['input'] + "\n" + matches,
                "neg_target": item['neg_target'].split(matches)[1][1:],
                "length": item["length"]
            })
    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate negative training data for KTO")
    parser.add_argument("--task", type=str, required=True, choices=["steam", "amazon"],
                        help="Task name: steam or amazon")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Path to trajectory JSON file. Default: data_preprocess/orig_traj_react_format/{task}_traj_2000.json")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for negative data. Default: tuning_data/neg/{task}_sft_neg_2000.json")
    parser.add_argument("--count", type=int, default=2000,
                        help="Number of trajectories to sample")
    parser.add_argument("--traj_len_thresh", type=int, default=1,
                        help="Keep negative trajectories whose length is greater than this value. Default keeps old behavior (1)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if args.input_path is None:
        args.input_path = os.path.join(project_root, "data_preprocess", "orig_traj_react_format", f"{args.task}_traj_2000.json")
    if args.output_path is None:
        args.output_path = os.path.join(project_root, "tuning_data", "neg", f"{args.task}_sft_neg_2000.json")

    with open(args.input_path, "r") as f:
        train_traj = json.load(f)

    cleaned_traj = clean_traj_by_line_sampled(train_traj, args.count)

    saving_path = os.path.dirname(args.output_path)
    os.makedirs(saving_path, exist_ok=True)

    sft_traj = get_training_data(cleaned_traj, args.traj_len_thresh)
    print(f"[NEG] {args.task}: generated {len(sft_traj)} negative samples (traj_len_thresh={args.traj_len_thresh})")
    with open(args.output_path, "w") as f:
        json.dump(sft_traj, f, ensure_ascii=False, indent=4)
    print(f"[NEG] Saved to {args.output_path}")
