import json
import re
import numpy as np  


def eval_performance(data):
    all_reward_list = []
    case1 = []
    case2 = []
    rewards_list = []
    turns_list = []
    for i, key in enumerate(data.keys()):
        if key == 'reflections':
            continue
        traj = data[key]['traj']
        if "No Suitable film" in traj:
            case1.append(i)
        elif "User Stop" in traj:
            case2.append(i)
        
        rewards = re.findall(r"reward=([\d\.]+)", traj)
        reward = 0
        for temp in rewards:
            reward = reward + float(temp)
            all_reward_list.append(float(temp))
        rewards_list.append(reward)
        turn = len(rewards)
        turns_list.append(turn)


    print(f"全局平均reward:{np.mean(np.array(rewards_list))}")
    print(f"平均每次推荐reward:{np.mean(np.array(all_reward_list))}")
    print(f"全局平均turn:{np.mean(np.array(turns_list))}")

    return np.mean(np.array(turns_list)), np.mean(np.array(all_reward_list)), np.mean(np.array(rewards_list))

def eval_performance_no_same_item(data):
    all_reward_list = []
    case1 = []
    case2 = []
    rewards_list = []
    turns_list = []

    
    for i, key in enumerate(data.keys()):

        history_item_list = []
        if key == 'reflections':
            continue
        traj = data[key]['traj']
        if "No Suitable film" in traj:
            case1.append(i)
        elif "User Stop" in traj:
            case2.append(i)
        
        rewards = re.findall(r"reward=([\d\.]+)", traj)
        reward = 0

        log = data[key]['traj_by_line']
        NULL_action=0
        recommendations={}
        for entry in log:
            try:
                if entry.startswith('Action'):
                    # 提取当前的Action编号和推荐项
                    parts = entry.split(': recommend[')
                    current_action = parts[0].strip()
                    recommended_item = parts[1].rstrip(']')
                    recommendations[current_action] = recommended_item

                elif entry.startswith('Observation') and 'instead, recommend' in entry:
                    # 如果Observation中有替换推荐项，则进行替换
                    parts = entry.split('instead, recommend[')
                    new_recommendation = parts[1].rstrip(']')
                    if current_action:
                        recommendations[current_action] = new_recommendation
            except:
                    NULL_action += 1
                    print("NULL act:" + str(NULL_action))


        # 推荐相同的item时候进行退出
        recommend_item_list = list(recommendations.values())
        for i in range(len(recommend_item_list)):
            if recommend_item_list[i] in recommend_item_list[:i]:
                rewards = rewards[:i]
                break
        


        # recommend_item_list = re.findall(r"(?<=recommend\[)[^\]]+(?=\])", traj)


        for temp in rewards:
            reward = reward + float(temp)
            all_reward_list.append(float(temp))
        rewards_list.append(reward)
        turn = len(rewards)
        turns_list.append(turn)


    print(f"全局平均reward:{np.mean(np.array(rewards_list))}")
    print(f"平均每次推荐reward:{np.mean(np.array(all_reward_list))}")
    print(f"全局平均turn:{np.mean(np.array(turns_list))}")

    return np.mean(np.array(turns_list)), np.mean(np.array(all_reward_list)), np.mean(np.array(rewards_list))

# path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agentamber/billp_result/amazon_no_same_item_bs_10/trajs_agent/amazon_test_agent_a2c_0_100_Meta-Llama-3-8B-Instruct__0.5_2025-03-26-21-12-26.json'
# with open(path, "r") as f:
#     data = json.load(f)

# temp = {}
# for key in list(data.keys())[:100]:
#     temp[key] = data[key]

# eval_performance_no_same_item(data)