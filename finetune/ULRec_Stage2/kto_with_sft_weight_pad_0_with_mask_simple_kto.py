import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
import json
import copy
from custmized_trainer.mask_kto_trainer_simple_beta import Mask_KTOTrainer_simple_beta
from trl.trainer.kto_config import KTOConfig


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    # pos_beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for kto loss"})
    # neg_beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for kto loss"})
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for kto loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/LLM_model/Qwen2.5-3B-Instruct",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "the lr scheduler type"})
    warmup_ratio: Optional[int] = field(default=0.1, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0., metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_hf", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    # lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    # lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    # lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=65536, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=65536, metadata={"help": "the maximum sequence length"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    # save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    # eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agentamber/debug/dpo_epoch1/", metadata={"help": "the output directory"})
    data_path: Optional[str] = field(default="", metadata={"help": "the training data directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    loss_type: Optional[str] = field(default="kto", metadata={"help": "loss type"})
    # load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )
    deepspeed_json: Optional[str] = field(default="deepspeed_config_3.json")


    # instrumentation
    # report_to: Optional[str] = field(
    #     default="wandb",
    #     metadata={
    #         "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
    #         '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
    #         'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
    #     },
    # )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    sft_weight: Optional[float] = field(
        default=1, metadata={"help": "Weight for sft"}
    )
    # pos_length_alpha: Optional[float] = field(
    #     default=0, metadata={"help": "Length Regulation"}
    # )
    # neg_length_alpha: Optional[float] = field(
    #     default=0, metadata={"help": "Length Regulation"}
    # pos_reward_normalizer: Optional[float] = field(default=1, metadata={"help": "Length normalization"})
    # neg_reward_normalizer: Optional[float] = field(default=1, metadata={"help": "Length normalization"})
    # pos_reward_tau: Optional[float] = field(default=0.1, metadata={"help": "Length dynamic"})
    # neg_reward_tau: Optional[float] = field(default=-0.1, metadata={"help": "Length dynamic"})


# def get_stack_exchange_paired(
#     data_dir: str = "data/rl",
#     cache_dir: Optional[str] = None,
#     num_proc=24,
# ) -> Dataset:
#     """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

#     The dataset is converted to a dictionary with the following structure:
#     {
#         'prompt': list[str],
#         'chosen': list[str],
#         'rejected': list[str],
#     }

#     Prompts are structured as follows:
#       "Question: " + <prompt> + "\n\nAnswer: "
#     """
#     dataset = load_dataset(
#         "lvwerra/stack-exchange-paired",
#         split="train",
#         cache_dir=cache_dir,
#         data_dir=data_dir,
#         verification_mode="no_checks",
#     )
#     original_columns = dataset.column_names

#     def return_prompt_and_responses(samples) -> dict[str, str]:
#         return {
#             "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
#             "chosen": samples["response_j"],
#             "rejected": samples["response_k"],
#         }

#     return dataset.map(
#         return_prompt_and_responses,
#         batched=True,
#         num_proc=num_proc,
#         remove_columns=original_columns,
#     )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        # low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False
    
    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        # low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map={"": Accelerator().local_process_index},
    )
    model_ref.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference


    # 2. Load the Stack-exchange paired dataset
    # train_dataset = get_stack_exchange_paired(data_dir="data/rl")
    with open(script_args.data_path,"r") as f:
        train_dataset = json.load(f)
    train_dataset = Dataset.from_dict(train_dataset)
    
    # train_dataset = train_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length,
    #     num_proc=script_args.num_proc,
    # )

    # # 3. Load evaluation dataset
    # eval_dataset = get_stack_exchange_paired(data_dir="data/evaluation")
    # eval_dataset = eval_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length,
    #     num_proc=script_args.num_proc,
    # )

    # 4. initialize training arguments:
    training_args = KTOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        # per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        num_train_epochs = script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        # save_steps=script_args.save_steps,
        save_strategy="no",
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        # eval_strategy="steps",
        # eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        # report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="kto_llama",
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
        beta=script_args.beta,
        max_prompt_length=script_args.max_prompt_length,
        loss_type=script_args.loss_type,
        max_length=script_args.max_length,
        deepspeed=script_args.deepspeed_json,
    )


    # peft_config = LoraConfig(
    #     r=script_args.lora_r,
    #     lora_alpha=script_args.lora_alpha,
    #     lora_dropout=script_args.lora_dropout,
    #     target_modules=[
    #         "q_proj",
    #         "v_proj",
    #         "k_proj",
    #         "out_proj",
    #         "fc_in",
    #         "fc_out",
    #         "wte",
    #     ],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # 5. initialize the DPO trainer

    # counting mean reward
    reward_list = {"pos":[], "neg":[]}
    for i in range(len(train_dataset["label"])):
        if train_dataset["label"][i]:
            reward_list["pos"] += [train_dataset["length"][i]]
        else:
            reward_list["neg"] += [train_dataset["length"][i]]
    
    import numpy as np
    
    mean_pos_reward = np.mean(reward_list["pos"])
    mean_neg_reward = np.mean(reward_list["neg"])

    kto_trainer = Mask_KTOTrainer_simple_beta(
        model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        use_chat_template=False,
        # eval_dataset=eval_dataset,
        sft_weight=script_args.sft_weight,
        processing_class=tokenizer,
        # pos_length_alpha=script_args.pos_length_alpha,
        # neg_length_alpha=script_args.neg_length_alpha,
        # peft_config=peft_config,
        ignore_id_for_kto=[67689, 58, 60, 2331, 128009],
        # pos_beta = script_args.pos_beta,
        # neg_beta = script_args.neg_beta
        # pos_reward_normalizer=script_args.pos_reward_normalizer * mean_pos_reward,
        # neg_reward_normalizer=script_args.neg_reward_normalizer * mean_neg_reward,
        # pos_reward_tau=script_args.pos_reward_tau,
        # neg_reward_tau=script_args.neg_reward_tau
    )

    # 6. train
    kto_trainer.train()
    kto_trainer.save_model(script_args.output_dir)

    # 7. save
    # output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    # dpo_trainer.model.save_pretrained(script_args.output_dir)
    # tokenizer.save_pretrained(script_args.output_dir)