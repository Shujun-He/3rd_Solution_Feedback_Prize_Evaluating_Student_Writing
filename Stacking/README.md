# Code to reproduce stacking models for span predictions

1. ```bash make_seq_datasets.sh``` to create pickled datasets for stacking model training

2. run ```Stacking_data_prep_v2-Copy2.ipynb``` to prepare data in a suitable format

3. run ```Stacking_shujun_7322.ipynb``` to training stacking models. The xgb/lightgbm models will be saved  to a folder called ```gbm_models```
