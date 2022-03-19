import os

os.system('mkdir scripts')

nfolds=10
with open('run_folds.sh','w+') as f:
    for i in range(8):
        with open(f'scripts/{i}.sh','w+') as f2:
            f2.write(f'python run.py --gpu_id {i} --fold {i} --nfolds 8 --weight_decay 5e-7 --lr_scale 0.75')
        f.write(f'nohup bash scripts/{i}.sh > {i}.out&\n')


with open('run_folds_rnn.sh','w+') as f:
    for i in range(8):
        with open(f'scripts/{i}_rnn.sh','w+') as f2:
            f2.write(f'python run_rnn.py --gpu_id {i} --fold {i} --nfolds 8 --weight_decay 5e-7 --lr_scale 1')
        f.write(f'nohup bash scripts/{i}_rnn.sh > {i}.out&\n')


with open('run_folds_cnn.sh','w+') as f:
    for i in range(8):
        with open(f'scripts/{i}_cnn.sh','w+') as f2:
            f2.write(f'python run_cnn.py --gpu_id {i} --fold {i} --nfolds 8 --weight_decay 5e-7 --lr_scale 1')
        f.write(f'nohup bash scripts/{i}_cnn.sh > {i}.out&\n')
