# ULRec

This is the source code for ULRec.

To generate training data, call `agent_react` in `agent_infer.py` and use a 70B Llama3 model to produce trajectory data, and then use the scripts in `data_preprocess/` to convert the trajectories into data for subsequent training stages. The interaction environment data is available at https://drive.google.com/file/d/1cWOJh6IotokY7gOWdZVbqIR6V0jWzKYu/view?usp=drive_link.

To learn from positive feedback and improve long-term recommendation performance (corresponding to the ULRec-P variant), run the bash script in `finetune/ULRec-P_Stage1/`.

After completing ULRec-P, further learning from unpaired feedback can be achieved by running the bash script in `finetune/ULRec_Stage2/` based on the previous results. We use `beta=0.01` and SFT weight (`alpha`) `=0.1` as the default parameters, though you may also change these parameters within the range of 0-1 or explore larger ranges.

Upon revisiting our released codebase, we discovered a bug inherited from the simulation environment of BiLLP: the reward threshold check was only performed when comparing against items that had already been recommended in the current episode, and was therefore inadvertently skipped on the first recommendation, when no such item was yet available. We have corrected this in our codebase and re-ran experiments. The updated default-setting results are shown below, with the same overall trend:

| Method  | Steam Len | Steam Traj Reward | Amazon Len | Amazon Traj Reward |
|---------|----------:|------------------:|-----------:|-------------------:|
| Base    | 2.21      | 9.95              | 1.71       | 7.60               |
| ULRec-P | 9.78      | 45.16             | 7.06       | 31.30              |
| ULRec   | 15.42     | 71.06             | 9.50       | 42.64              |