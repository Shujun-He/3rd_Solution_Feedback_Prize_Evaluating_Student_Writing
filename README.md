# 3rd Place Solution of Feedback-Prize-Evaluating-Student-Writing

Hello!

Below you can find a outline of how to reproduce our solution for the Feedback-Prize-Evaluating-Student-Writing competition.
If you run into any trouble with the setup/code or have any questions please contact me at shujun@tamu.edu

## ARCHIVE CONTENTS
```Longformer``` code to reproduce training of Longformer NER models

```SW_Deberta``` code to reproduce training of Sliding window Deberta-xl NER models

```Stacking``` code to reproduce span prediction gbm models using out of fold predictions

```3rd_solution.pdf``` has a pdf describing our solution with references

## HARDWARE: (The following specs were used to create the original solution)
We used a compute server with 8 x Nvidia RTX A6000, and stacking was run locally on a rtx 3090.

## SOFTWARE:
Python 3.8.10
CUDA 11.3
nvidia drivers v.460.56
For the rest of required packages see ```requirements.txt```
-- Equivalent Dockerfile for the GPU installs: ```Dockerfile.tmpl```

## DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)

Download the following datasets and put it in ```../input```. The commands needed are as follows:

```kaggle datasets download -d cdeotte/py-bigbird-v26```
```kaggle datasets download -d shujun717/deberta-xlarge```
```kaggle datasets download -d shujun717/pytorch_longformer_large```

unzip these with their dataset names as folder names and put them in ```../input``` (for instance contents of ```shujun717/deberta-xlarge``` should be in ```../input/deberta-xlarge```)


## DATA PROCESSING
Words are tokenized in real time during training/inference. See ```SW_Deberta/Dataset.py``` and ```Longformer/Dataset.py``` for details.

## MODEL BUILD

1. Training SW Deberta-xl: see detailed instructions in ```SW_Deberta```
2. Training Longformer: see detailed instructions in ```Longformer```
3. Stacking: see detailed instructions in ```Stacking```. Note that you should have finished running ```SW_Deberta``` and ```Longformer``` before running ```Stacking```.

## MODEL BUILD

Our inference notebook with all datasets (model weights, etc.) made public can be accessed at: https://www.kaggle.com/code/aerdem4/xgb-lgb-feedback-prize-cv-0-7322/notebook.
