import pandas as pd
import os


for i in range(8):
    df=pd.read_csv(f"logs/fold{i}.csv")
    best_epoch=df['epoch'].iloc[df['val_score'].argmax()]
    print(f"for original best epoch: {best_epoch} with score: {df['val_score'].iloc[best_epoch]} for fold {i}")
    #os.system(f'cp models/fold{i}_epoch{best_epoch}.pt best_weights/fold{i}.pt')
    #df=pd.read_csv(f"logs/fold{i}_rnn.csv")
    #best_epoch=df['epoch'].iloc[df['val_score'].argmax()]
    #print(f"for RNN :    best epoch: {best_epoch} with score: {df['val_score'].iloc[best_epoch]} for fold {i}")

    #df=pd.read_csv(f"logs/fold{i}_cnn.csv")
    #best_epoch=df['epoch'].iloc[df['val_score'].argmax()]
    #print(f"for CNN :    best epoch: {best_epoch} with score: {df['val_score'].iloc[best_epoch]} for fold {i}")
#for i in range(8):
