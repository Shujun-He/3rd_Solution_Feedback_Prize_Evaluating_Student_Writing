import os

os.system('mkdir scripts')


with open('run_folds.sh','w+') as f:
    for i in range(8):
        with open(f'scripts/{i}.sh','w+') as f2:
            f2.write(f'python run.py --gpu_id {i} --fold {i} --nfolds 8 --weight_decay 0 --lr_scale 0.2')
        f.write(f'nohup bash scripts/{i}.sh > {i}.out&\n')
