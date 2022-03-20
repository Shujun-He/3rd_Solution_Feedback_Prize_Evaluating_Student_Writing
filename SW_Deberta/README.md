# Code to reproduce SW Deberta model

1. ```python write_scripts.py``` to create necessary sh scripts to run 8 fold training concurrently. If you don't have 8 gpus, you would need to remove the nohup commands and run training one fold at a time. You GPU will need to have 48 gb of vram for training to fit on a single gpu.

2. ```bash run_folds.sh``` to run training

3. Following training, do ```python get_best_weights.py``` to check cv without stacking for each fold. They should be around 0.69-0.7

3. ```bash make_oofs.sh``` to generate oofs used for stacking
