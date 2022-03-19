from transformers import *
import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualLSTM(nn.Module):

    def __init__(self, d_model, rnn='GRU'):
        super(ResidualLSTM, self).__init__()
        self.downsample=nn.Linear(d_model,d_model//2)
        if rnn=='GRU':
            self.LSTM=nn.GRU(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        else:
            self.LSTM=nn.LSTM(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        self.dropout1=nn.Dropout(0.2)
        self.norm1= nn.LayerNorm(d_model//2)
        self.linear1=nn.Linear(d_model//2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)
        self.dropout2=nn.Dropout(0.2)
        self.norm2= nn.LayerNorm(d_model)

    def forward(self, x):
        x=x.permute(1,0,2)
        res=x
        x=self.downsample(x)
        x, _ = self.LSTM(x)
        x=self.dropout1(x)
        x=self.norm1(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=self.dropout2(x)
        x=res+x
        return self.norm2(x).permute(1,0,2)

def noop(x): return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding="same", use_bn=False):
        super().__init__()

        self.idconv = noop if in_channels == out_channels \
            else nn.Conv1d(in_channels, out_channels, 1, stride=1)

        if padding == "same":
            padding = kernel_size // 2 * dilation

        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.idconv(x))


class ResNet(nn.Module):
    def __init__(self, use_msd=False,
                 cnn_dim=512, input_dim=1024, kernel_sizes=[3,5,7,9], use_bn=False):
        super().__init__()
        self.use_msd = use_msd

        self.cnn = nn.Sequential(
            ResBlock(input_dim, cnn_dim, kernel_size=kernel_sizes[0], use_bn=use_bn),
            ResBlock(cnn_dim, cnn_dim, kernel_size=kernel_sizes[1], use_bn=use_bn),
            # ResBlock(cnn_dim, cnn_dim, kernel_size=kernel_sizes[2], use_bn=use_bn),
            # ResBlock(cnn_dim, cnn_dim, kernel_size=kernel_sizes[3], use_bn=use_bn),
        )

        self.logits = nn.Linear(cnn_dim, 1024)

        self.high_dropout = nn.Dropout(p=0.5)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)


    def forward(self, x):

        x = x.permute(0,2,1)
        features = self.cnn(self.dropout1(x)).permute(0, 2, 1) # [Bs x T x nb_ft]
#         print(f'features: {features.shape}')

        #if self.use_msd and self.training:
        features = torch.mean(
            torch.stack(
                [self.high_dropout(features) for _ in range(5)],
                dim=0,
                ),
            dim=0,
        )
        features=self.logits(features)
        # else:
        #     logits = self.logits(self.dropout2(features))

#         print(f'logits: {logits.shape}')

        return features

class TransformerModel(nn.Module):
    def __init__(self,DOWNLOADED_MODEL_PATH, rnn='GRU'):
        super(TransformerModel, self).__init__()
        config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH+'/config.json')

        self.backbone=AutoModel.from_pretrained(
                           DOWNLOADED_MODEL_PATH+'/pytorch_model.bin',config=config_model)

        if rnn=='GRU' or rnn=='LSTM':
            self.lstm=ResidualLSTM(1024,rnn)
        else:
            self.lstm=ResNet()
        self.classification_head=nn.Linear(1024,15)
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
        x=self.backbone(input_ids=x,attention_mask=attention_mask,return_dict=False)[0]

        x=self.lstm(x)
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
