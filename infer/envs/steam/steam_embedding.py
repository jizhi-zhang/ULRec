from transformers import GenerationConfig, AutoModel, AutoTokenizer
import transformers
import torch
import os
import math
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

base_model = "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/LLM_model/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModel.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)


model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.eval()

tokenizer.padding_side = "left"


import json

with open("./datamaps.json") as f:
    data = json.load(f)
    id2item = data['id2item_dict']
    item2id = data['item2id_dict']
    items = list(item2id.keys())
    tasks = [f'The type of {item} is' for item in items]

print(len(data['id2item_dict']))
print(len(data['item2id_dict']))

from tqdm import tqdm
import numpy as np 
import json

# with open("./test_id_list.json", "r") as file:
#     json_str = file.read()
# test_id_list = json.loads(json_str)

def batch(list, list2, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield (list[batch_size * i: batch_size * (i + 1)], list2[batch_size * i: batch_size * (i + 1)])


movie_embeddings = []
id_indexs = []
for i, batch_i in tqdm(enumerate(batch(tasks, items, 16))):
    batch_input = batch_i[0]
    batch_item = batch_i[1]
    input = tokenizer(batch_input, return_tensors="pt", padding=True)
    input = input.to('cuda')
    input_ids = input.input_ids
    attention_mask = input.attention_mask
    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    movie_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
    id_indexs.append(torch.Tensor([item2id[it] for it in batch_item]))

data = {}    
embeddings = torch.cat(movie_embeddings, dim=0)
indexs = torch.cat(id_indexs, dim=0)
print(embeddings.shape)
print(indexs.shape)

# indices = torch.LongTensor([i for i, val in enumerate(indexs) if val.item() in test_id_list])
indices = torch.LongTensor([i for i, val in enumerate(indexs)])
data['embeddings'] = embeddings[indices]
data['indexs'] = indexs[indices]
print(data['embeddings'].shape)

torch.save(data, './steam_embedding_task.pt')