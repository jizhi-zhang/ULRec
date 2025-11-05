import os
import json
import torch
import argparse
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from typing import List, Tuple

# 设置环境变量以禁用tokenizers的并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


class CustomDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=8192):
        tokenizer = Tokenizer(tokenizer)
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        for value in data:
            input_data = value["input"]
            target_data = value["target"]

            # 对input进行标记化
            # tokenized_input = tokenizer.encode(input_data, add_special_tokens=True, truncation=True, max_length=block_size)
            tokenized_input = tokenizer.encode(input_data, bos=True, eos=False)
            # 对target进行标记化
            # tokenized_target = tokenizer.encode(target_data, add_special_tokens=True, truncation=True, max_length=block_size)
            tokenized_target = tokenizer.encode(target_data, bos=False, eos=True)

            # 创建input_ids和labels，注意labels部分需要与input部分对齐
            input_ids = tokenized_input + tokenized_target
            labels = [-100] * len(tokenized_input) + tokenized_target  # 用-100来忽略input部分的损失计算

            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def main():
    parser = argparse.ArgumentParser(description="SFT with llama2-7b-chat-hf")
    parser.add_argument("--model_name_or_path", type=str, default="/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/LLM_model/Meta-Llama-3-8B-Instruct", help="Path to pre-trained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_path", type=str,default="/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agentamber/sft_data/diff_reward_metric_steam/sft/no_same_item_from_eval/llama3_70b_steam_10000_step_by_step_thresh_5_diff_reward_metric_action_only_no_rand_cat.json", help="Path to the training dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agentamber/sft_debug/", help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save checkpoint every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--bf16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--deepspeed", type=str, default="deepspeed_config.json", help="Path to DeepSpeed config file")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=2e-5)

    args = parser.parse_args()

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    # 加载数据集
    train_dataset = CustomDataset(tokenizer, args.dataset_path)

    # 定义自定义的data_collator
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_collator = DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        dataloader_num_workers=4,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="no",
        learning_rate=args.learning_rate,
        weight_decay=0.,
        warmup_ratio=0.1,
    )

    # 定义Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 开始训练
    trainer.train()

    # 保存模型和tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
