pip3 install accelerate==1.2.1 
pip3 install deepspeed==0.15.4
pip3 install seaborn==0.13.2

epoch=3
thresh=1
lr=5e-6
runs_id=1
tp=4

output_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agent_clean/infer/debug_result/very_orig_data_rerun/amazon/sft_result/epoch_${epoch}_thresh_${thresh}_lr_${lr}_${runs_id}
dataset_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agent_clean/tuning_data/SFT/amazon.json



TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=${tp} llama_sft.py \
    --model_name_or_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/LLM_model/Meta-Llama-3-8B-Instruct/ \
    --dataset_path $dataset_path \
    --output_dir $output_path \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 1 \
    --logging_steps 1 \
    --save_steps 10000 \
    --save_total_limit 1 \
    --bf16 \
    --deepspeed /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agentamber/BiLLP-hope/agent_sft/deepspeed_config.json \
    --gradient_accumulation_steps 16 \
    --learning_rate ${lr}

# ### Inference and eval

#!/bin/bash
pip3 install -U openai -i http://pypi.sankuai.com/simple
python ../../infer/agent_infer.py    \
 --task amazon     --backend self_define  --tp 1  \
 --infer_model_self_define $output_path \
 --promptpath cot_movie_upper     --evaluate  \
 --random     --task_split test     \
 --temperature 0.5   --task_end_index 100   \
 --env amazon_no_same_item  --env_threshold 15  --env_window_length 4 \
 --Max_Iteration 100 --agent_name agent_act_only --batch_size 100 --save_path ${output_path}/debug_result_amazon/100_eval_result/


