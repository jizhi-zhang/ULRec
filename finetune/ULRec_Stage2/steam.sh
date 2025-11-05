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
runs=2

# echo $thresh
output_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agent_clean/infer/debug_result_2/kto_result_very_orig/llama3_70b/diff_metric_steam/naive_kto_dynamic_beta_epoch_${epoch}_from_base_3_epoch_same_param/beta_${beta}/len_thresh_${len_thresh}_desirable_weight_${desirable_weight}_undesirable_weight_${undesirable_weight}_lr_${lr}_sft_weight_${sft_weight}_runs_${runs}
dataset_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agent_clean/tuning_data/KTO/steam.json


TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=${tp} kto_with_sft_weight_pad_0_with_mask_simple_kto.py \
      --model_name_or_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangjizhi/jizhizhang_zhangjizhi/fairness/echo_chamber/agent_clean/infer/debug_result/sft_result_very_orig/steam/epoch_3_thresh_1_lr_5e-6_0 \
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
 --task steam     --backend self_define  --tp 1  \
 --infer_model_self_define $output_path \
 --promptpath cot_movie_upper     --evaluate  \
 --random     --task_split test     \
 --temperature 0.5   --task_end_index 100   \
 --env steam_no_same_item  --env_threshold 50  --env_window_length 4 \
 --Max_Iteration 100 --agent_name agent_act_only --Max_Reflections 2 --batch_size 100 --save_path ${output_path}/100_eval_result/


