import json

import os

from eval_performance_no_same_item_from_eval import eval_performance_no_same_item

os.environ['OPENAI_API_KEY'] = 'dummy_key'
os.environ['OPENAI_API_BASE']="dummy_key"

tiktoken_cache_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agentamber/tiktoken/"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir




import argparse
import concurrent.futures
import random
import time
import logging
logging.getLogger().setLevel(logging.ERROR)
from functools import partial
from  Agents.agent_action_only_space import ActOnlyAgent
from Agents.agent_react import ReactReflectAgent
from models.llama_vllm_para import LlamaInterface

from tasks import get_task
from env import get_envs, get_groundingmodel
# from tools import call_tools
# from tools.search import search_save
from datetime import datetime
# import re
import os
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def get_fewshot_prompt(promptpath, task=None, chatgpt_format=False):
    if len(promptpath) == 0:
        return [] if chatgpt_format else ""
    elif promptpath == "default" and task is not None:
        return task.get_prompt()
    if not chatgpt_format:
        with open(f"./prompts/{promptpath}.txt", "r") as fin:
            prompt = fin.read() 
        return prompt
    else:
        with open(f"./prompts/{promptpath}.json", "r") as fin:
            prompt = json.load(fin)
        return prompt

def prepare_prompt(question):
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"

def prune_thought(prompt):
    if prompt.startswith("Thought:"):
        return prompt[len("Thought:"):].strip()
    return prompt

def save_info(infos, outfilename, args):
    os.makedirs(f"{args.save_path}/trajs_agent", exist_ok=True)
    os.makedirs(f"{args.save_path}/reflections", exist_ok=True)
    os.makedirs(f"{args.save_path}/memory", exist_ok=True)
    os.makedirs(f"{args.save_path}/critic_memory", exist_ok=True)
    if 'trajs' in infos:
        traj_file_name = f"{args.save_path}/trajs_agent/{outfilename}.json"
        with open(traj_file_name, "w") as fout:
            json.dump(infos['trajs'], fout, indent=2)
    
    if 'reflections' in infos:
        reflection_file_name = f'{args.save_path}/reflections/{outfilename}.txt'
        with open(reflection_file_name, 'w', encoding='utf-8') as file:
            for item in infos['reflections']:
                file.write(str(item) + '\n')
    
    if 'Q_table' in infos:
        memory_file_name = f'{args.save_path}/memory/{outfilename}.json'
        with open(memory_file_name, "w", encoding='utf-8') as fout:
            json.dump(infos['Q_table'], fout, indent=2, cls=NpEncoder)
            
    if 'actor_memory' in infos:
        memory_file_name = f'{args.save_path}/memory/{outfilename}.json'
        with open(memory_file_name, "w", encoding='utf-8') as fout:
            json.dump(infos['actor_memory'], fout, indent=2, cls=NpEncoder)
    
    if 'critic_memory' in infos:
        critic_memory_file_name = f'{args.save_path}/critic_memory/{outfilename}.json'
        with open(critic_memory_file_name, "w", encoding='utf-8') as fout:
            json.dump(infos['critic_memory'], fout, indent=2, cls=NpEncoder)

      
def load_info(input_file_name, args):
    if input_file_name == None:
        return None, None, None
    
    reflection_file_name = f'{args.save_path}/reflections/{input_file_name}.txt'
    if os.path.exists(reflection_file_name):
        reflections = []
        with open(reflection_file_name, "r", encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # 去除行尾的换行符和空白字符
                reflections.append(line)
    else:
        reflections = None
        
    memory_file_name = f'{args.save_path}/memory/{input_file_name}.json'
    if os.path.exists(memory_file_name):
        with open(memory_file_name, 'r', encoding='utf-8') as file:
            Q_Memory = json.load(file)
    else:
        Q_Memory = None
    
    critic_memory_file_name = f'{args.save_path}/critic_memory/{input_file_name}.json'
    if os.path.exists(critic_memory_file_name):
        with open(critic_memory_file_name, 'r', encoding='utf-8') as file:
            Critic_Memory = json.load(file)
    else:
        Critic_Memory = None
    
    return reflections, Q_Memory, Critic_Memory
    
'''
inital prompt -> agents
while true
    action, action_type = agents.run(observation, reward)
    observation, reward = env.step()
'''



def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='gpt-4')
    args.add_argument('--agent_name', type=str, default='reflexion', required=True)
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True)
    args.add_argument('--task_split', type=str, default='train')
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=100)

    args.add_argument('--evaluate', action='store_true')
    args.add_argument('--add_lora', action='store_true')
    args.add_argument('--random', action='store_true')
    args.add_argument('--alpaca_format', action='store_true')
    args.add_argument('--chatgpt_format', action='store_true')
    args.add_argument('--question_prefix', type=str, default='')

    args.add_argument('--modelpath', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/LLM_model/Llama-2-7b-chat-hf/')
    args.add_argument('--peftpath', type=str, default='')
    args.add_argument('--promptpath', type=str, default='')
    args.add_argument('--env_path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agentamber/BiLLP-main/env')
    args.add_argument('--grounding_model_path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/LLM_model/Llama-2-7b-chat-hf/')
    
    args.add_argument('--env', type=str, required=True)
    args.add_argument('--env_window_length', type=int, default=5)
    args.add_argument('--env_threshold', type=float, default=-1)
    
    args.add_argument('--Max_Iteration', type=int, default=11)
    args.add_argument("--tp", type=int, default=2)
    args.add_argument('--Max_Reflections', type=int, default=2)
    args.add_argument('--batch_size', type=int, default=5)
    args.add_argument('--traj', action='store_true')
    args.add_argument('--change_examples', action='store_true')
    args.add_argument('--input_file_name', default=None)
    args.add_argument('--save_path', default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agentamber/results/")
    args.add_argument("--infer_model_self_define", type=str, default=None)
    
    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task = get_task(args.task, args.task_split)
    
    random.seed(0)

    model_name_path_dict = {
        "self_define": args.infer_model_self_define
    }

    if args.backend in model_name_path_dict.keys():
        args.modelpath = model_name_path_dict[args.backend]
    modelname = args.backend

    if args.backend in model_name_path_dict.keys():
        pathname = args.peftpath.replace('/', '_') if args.add_lora else args.modelpath.replace('/', '_')
        modelname += f"_{pathname}"
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if args.backend in model_name_path_dict.keys():
        model_name_temp = args.modelpath.split("/")[-2] + "_" + args.modelpath.split("/")[-1]
        outfilename = f"{args.task}_{args.task_split}_{args.agent_name}_{args.task_start_index}_{args.task_end_index}_{model_name_temp}_{args.temperature}_{time_str}"
    else:
        outfilename = f"{args.task}_{args.task_split}_{args.agent_name}_{args.task_start_index}_{args.task_end_index}_{modelname}_{args.temperature}_{time_str}"
    print(outfilename)
    print(args.agent_name)
    
    idxs_all = list(range(len(task)))
    if args.random:
        random.Random(233).shuffle(idxs_all)
    idxs = idxs_all[args.task_start_index:args.task_end_index]

    envs = get_envs(args.env, args, args.task_split)
    

    if args.backend in model_name_path_dict.keys():
        print(args.modelpath)
        llama = LlamaInterface(args.modelpath, args)
        model = partial(llama.generate_responses_from_llama, temperature=args.temperature, stop=['\n', 'Action', 'Observation', 'Thought', 'Observation ', 'Action ', 'Thought '])

    grounding_model = get_groundingmodel(args.env, args.grounding_model_path, args, args.task_split) 

    os.makedirs(args.save_path, exist_ok=True)
    reflections, Q_Memory, Critic_Memory = load_info(args.input_file_name, args)
    

    if args.agent_name == "agent_act_only":
        agent = ActOnlyAgent(task, idxs, args, envs, grounding_model, max_steps=args.Max_Iteration, react_llm=model)
    elif args.agent_name == "agent_reflection":
        agent = ReactReflectAgent(task, idxs, args, envs, grounding_model, max_steps=args.Max_Iteration, react_llm=model, reflect_llm=model, reflections_memory=reflections)

    infos = agent.run(outfilename=outfilename)
    
    save_info(infos, outfilename, args)
    
    
    avg_turn, avg_action_reward, avg_traj_reward = eval_performance_no_same_item(infos['trajs'])

    result_dict = {
        "performance":{
            "avg_turn":avg_turn,
            "avg_action_reward" : avg_action_reward,
            "avg_traj_reward" : avg_traj_reward},
    }
    print(result_dict)

    os.makedirs(f'{args.save_path}/result_value/', exist_ok= True)

    # with open(f'{args.save_path}/result_value/{outfilename}_no_same_item_from_eval.json', "w") as f:
    with open(f'{args.save_path}/result_value/no_same_item_from_eval.json', "w") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
