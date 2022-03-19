from transformers import *
import torch.nn as nn
import torch.nn.functional as F

class ResidualLSTM(nn.Module):

    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.downsample=nn.Linear(d_model,d_model//2)
        self.LSTM=nn.GRU(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        self.dropout1=nn.Dropout(0.2)
        self.norm1= nn.LayerNorm(d_model//2)
        self.linear1=nn.Linear(d_model//2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)
        self.dropout2=nn.Dropout(0.2)
        self.norm2= nn.LayerNorm(d_model)

    def forward(self, x):
        res=x
        x=self.downsample(x)
        x, _ = self.LSTM(x)
        x=self.dropout1(x)
        x=self.norm1(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=self.dropout2(x)
        x=res+x
        return self.norm2(x)

class SlidingWindowTransformerModel(nn.Module):
    def __init__(self,DOWNLOADED_MODEL_PATH, window_size=512):
        super(SlidingWindowTransformerModel, self).__init__()
        config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH+'/config.json')

        self.backbone=AutoModel.from_pretrained(
                           DOWNLOADED_MODEL_PATH+'/pytorch_model.bin',config=config_model)

        self.lstm=ResidualLSTM(1024)
        self.classification_head=nn.Linear(1024,15)
        self.window_size=window_size
        #self.head=nn.Sequential(nn.Linear(1024,15))

        # self.downsample=nn.Sequential(nn.Linear(1024,256))
        # self.conv1d=nn.Sequential(nn.Conv1d(256,256,3,padding=0),
        #                           nn.ReLU(),
        #                           nn.LayerNorm(256),
        #                           nn.Conv1d(256,256,3,padding=1),
        #                           nn.ReLU(),
        #                           nn.LayerNorm(256))

        #self.BIO_head=nn.Sequential(nn.Linear(1024,3))

    def forward(self,x,attention_mask):

        B,L=x.shape

        # print(L)
        # exit()

        if L>self.window_size:
            #print(x.shape)
            #print(L//self.window_size)
            x=x.reshape(B,L//self.window_size,self.window_size).reshape(-1,self.window_size)
            attention_mask=attention_mask.reshape(B,L//self.window_size,self.window_size).reshape(-1,self.window_size)
            # print(x.shape)
            # exit()
        x=self.backbone(input_ids=x,attention_mask=attention_mask,return_dict=False)[0]
        x=x.reshape(B,L,-1)
        x=self.lstm(x.permute(1,0,2)).permute(1,0,2)
        x=self.classification_head(x)

        # x=x.permute(0,2,1)
        # x=self.conv1d(x)
        # print(x.shape)
        # exit()
        # classification_output=self.classification_head(x)
        #BIO_output=self.BIO_head(x[0])
        # print(x.shape)
        # exit()
        return x#, BIO_output
