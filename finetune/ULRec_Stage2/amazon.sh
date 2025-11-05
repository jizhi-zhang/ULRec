pip3 install fire==0.7.0 
pip3 install accelerate==1.2.1 
pip3 install deepspeed==0.15.4  
pip3 install seaborn==0.13.2 
pip3 install trl==0.13.0 
pip3 install transformers==4.48 


tp=4

len_thresh=1

lr=5e-6
beta=$1

sft_weight=$2

epoch=3
runs=1

# echo $thresh
output_path=./amazon/ULRec_result/beta_${beta}_sft_${sft_weight}/
dataset_path=kto_data_path


TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=${tp} /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agentamber/BiLLP-hope/agent_sft/dpo/kto_with_sft_weight_pad_0_with_mask_simple_kto.py \
      --model_name_or_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agent_clean/infer/debug_result/sft_result/amazon/epoch_3_thresh_1_lr_5e-6_1 \
      --output_dir $output_path \
      --num_train_epochs ${epoch}  \
      --data_path $dataset_path \
      --learning_rate $lr \
      --beta $beta \
      --sft_weight $sft_weight \
      --loss_type "kto" \


### Inference and eval

#!/bin/bash
pip3 install -U openai 
python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agent_clean/infer/agent_infer.py    \
 --task amazon     --backend self_define  --tp 1  \
 --infer_model_self_define $output_path \
 --promptpath cot_movie_upper     --evaluate  \
 --random     --task_split test     \
 --temperature 0.5   --task_end_index 100   \
 --env amazon_no_same_item  --env_threshold 15  --env_window_length 4 \
 --Max_Iteration 100 --agent_name agent_act_only --batch_size 100 --save_path ${output_path}/100_eval_result/


