# ULRec

This is the source code for ULRec.

The "requirements" file contains the versions of packages we used.

We provide the training data for ULRec in Link https://pan.ustc.edu.cn/share/index/c0b4725da1c849e9aad4 to facilitate result reproduction. Please copy the tuning_data folder from the link to the ULRec code directory.

To learn from positive feedback and improve long-term recommendation performance (corresponding to the ULRec-P variant), run the bash script in finetune/ULRec-P_Stage1/.

After completing ULRec-P, further learning from negative feedback can be achieved by running the bash script in finetune/ULRec_Stage2/ based on the previous results. In our paper experiments, we found that setting beta=0.01 and SFT weight (alpha)=0.1 yielded good performance, though the model is sensitive to these parameters. When using different data or python environment, you may also change these parameters within the range of 0-1 or explore larger ranges.
