from sklearn.model_selection import StratifiedKFold

import os, sys
# DECLARE HOW MANY GPUS YOU WISH TO USE.
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE

import argparse
def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--disc_type', type=int, default=0,  help='disc_type')
    parser.add_argument('--fold', type=int, default=0,  help='fold')
    parser.add_argument('--gpu_id', type=str, default='0',  help='gpu_id')
    opts = parser.parse_args()
    return opts

args=get_args()

os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id) #0,1,2,3 for four gpu


out_dir='seqclassifiers_v3'
os.system(f'mkdir {out_dir}')

# VERSION FOR SAVING MODEL WEIGHTS
VER=26

# IF VARIABLE IS NONE, THEN NOTEBOOK COMPUTES TOKENS
# OTHERWISE NOTEBOOK LOADS TOKENS FROM PATH
LOAD_TOKENS_FROM = '../../input/py-bigbird-v26'

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
LOAD_MODEL_FROM = 'models'

# Use the entire ensemble.
ENSEMBLE_IDS = [args.fold]

# Setting Fold = None leaves out an arbitrary 10% of the dataset for sequence classifier training.
# Setting Fold to one of [0,1,2,3,4] leaves out the portion of the dataset not trained on by the corresponding ensemble model.
# 'half' leaves out an arbitrary 50%.
FOLD = args.fold

# print(FOLD)
# exit()
# IF FOLLOWING IS NONE, THEN NOTEBOOK
# USES INTERNET AND DOWNLOADS HUGGINGFACE
# CONFIG, TOKENIZER, AND MODEL
DOWNLOADED_MODEL_PATH = '../../input/longformer_large'

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = 'allenai/longformer-large-4096'

# Tune the probability threshold for sequence classifiers to maximize F1
TRAIN_SEQ_CLASSIFIERS = False

N_FEATURES=16


KAGGLE_CACHE = 'cache' #location of valid_pred files

cache = 'cache' #save location of valid_seqds files
cacheExists = os.path.exists(cache)
if not cacheExists:
  os.makedirs(cache)

print(ENSEMBLE_IDS)

# In[90]:


# skopt optimizer has a bug when scipy is installed with its default version
if TRAIN_SEQ_CLASSIFIERS:
    os.system('pip install --no-dependencies scipy==1.5.2 ')


# In[91]:


from torch import cuda
config = {'model_name': MODEL_NAME,
         'max_length': 2048,
         'train_batch_size':4,
         'valid_batch_size':1,
         'epochs':5,
         'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
         'max_grad_norm':10,
         'device': 'cuda' if cuda.is_available() else 'cpu'}


# # How To Submit PyTorch Without Internet
# Many people ask me, how do I submit PyTorch models without internet? With HuggingFace Transformer, it's easy. Just download the following 3 things (1) model weights, (2) tokenizer files, (3) config file, and upload them to a Kaggle dataset. Below shows code how to get the files from HuggingFace for Google's BigBird-base. But this same code can download any transformer, like for example roberta-base.

# In[92]:


from transformers import *
if DOWNLOADED_MODEL_PATH == 'model':
    os.mkdir('model')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    tokenizer.save_pretrained('model')

    config_model = AutoConfig.from_pretrained(MODEL_NAME)
    config_model.num_labels = 15
    config_model.save_pretrained('model')

    backbone = AutoModelForTokenClassification.from_pretrained(MODEL_NAME,
                                                               config=config_model)
    backbone.save_pretrained('model')


# # Load Data and Libraries
# In addition to loading the train dataframe, we will load all the train and text files and save them in a dataframe.

# In[93]:


import numpy as np, os
from scipy import stats
import pandas as pd, gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW


from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score
from torch.cuda import amp


# In[94]:


train_df = pd.read_csv('../../input/feedback-prize-2021/train.csv')
print( train_df.shape )
train_df.head()


# In[95]:


# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, test_texts = [], []
for f in list(os.listdir('../../input/feedback-prize-2021/test')):
    test_names.append(f.replace('.txt', ''))
    test_texts.append(open('../../input/feedback-prize-2021/test/' + f, 'r').read())
test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})

test_texts['len']=test_texts['text'].apply(lambda x:len(x.split()))
test_texts=test_texts.sort_values(by=['len']).reset_index()

test_texts

SUBMISSION = False
if len(test_names) > 5:
      SUBMISSION = True

test_texts.head()


# In[96]:


# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, train_texts = [], []
for f in tqdm(list(os.listdir('../../input/feedback-prize-2021/train'))):
    test_names.append(f.replace('.txt', ''))
    train_texts.append(open('../../input/feedback-prize-2021/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})
train_text_df.head()


# # Convert Train Text to NER Labels
# We will now convert all text words into NER labels and save in a dataframe.

# In[97]:


if not LOAD_TOKENS_FROM:
    all_entities = []
    for ii,i in enumerate(train_text_df.iterrows()):
        if ii%100==0: print(ii,', ',end='')
        total = i[1]['text'].split().__len__()
        entities = ["O"]*total
        for j in train_df[train_df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            for k in list_ix[1:]: entities[k] = f"I-{discourse}"
        all_entities.append(entities)
    train_text_df['entities'] = all_entities
    train_text_df.to_csv('train_NER.csv',index=False)

else:
    from ast import literal_eval
    train_text_df = pd.read_csv(f'{LOAD_TOKENS_FROM}/train_NER.csv')
    # pandas saves lists as string, we must convert back
    train_text_df.entities = train_text_df.entities.apply(lambda x: literal_eval(x) )

print( train_text_df.shape )
train_text_df.head()


# In[98]:


# CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER
output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim',
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}
disc_type_to_ids = {'Evidence':(11,12),'Claim':(5,6),'Lead':(1,2),'Position':(3,4),'Counterclaim':(7,8),'Rebuttal':(9,10),'Concluding Statement':(13,14)}


# In[99]:


labels_to_ids


# # Define the dataset function
# Below is our PyTorch dataset function. It always outputs tokens and attention. During training it also provides labels. And during inference it also provides word ids to help convert token predictions into word predictions.
#
# Note that we use `text.split()` and `is_split_into_words=True` when we convert train text to labeled train tokens. This is how the HugglingFace tutorial does it. However, this removes characters like `\n` new paragraph. If you want your model to see new paragraphs, then we need to map words to tokens ourselves using `return_offsets_mapping=True`. See my TensorFlow notebook [here][1] for an example.
#
# Some of the following code comes from the example at HuggingFace [here][2]. However I think the code at that link is wrong. The HuggingFace original code is [here][3]. With the flag `LABEL_ALL` we can either label just the first subword token (when one word has more than one subword token). Or we can label all the subword tokens (with the word's label). In this notebook version, we label all the tokens. There is a Kaggle discussion [here][4]
#
# [1]: https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-617
# [2]: https://huggingface.co/docs/transformers/custom_datasets#tok_ner
# [3]: https://github.com/huggingface/transformers/blob/86b40073e9aee6959c8c85fcba89e47b432c4f4d/examples/pytorch/token-classification/run_ner.py#L371
# [4]: https://www.kaggle.com/c/feedback-prize-2021/discussion/296713

# In[100]:


# Return an array that maps character index to index of word in list of split() words
def split_mapping(unsplit):
    splt = unsplit.split()
    offset_to_wordidx = np.full(len(unsplit),-1)
    txt_ptr = 0
    for split_index, full_word in enumerate(splt):
        while unsplit[txt_ptr:txt_ptr + len(full_word)] != full_word:
            txt_ptr += 1
        offset_to_wordidx[txt_ptr:txt_ptr + len(full_word)] = split_index
        txt_ptr += len(full_word)
    return offset_to_wordidx

def iter_split(data,labels,fold,nfolds=5,seed=2020):
    splits = StratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=True)
    splits = list(splits.split(data,labels))
    # splits = np.zeros(len(data)).astype(np.int)
    # for i in range(nfolds): splits[splits[i][1]] = i
    # indices=np.arange(len(data))
    train_indices=splits[fold][0]
    val_indices=splits[fold][1]
    return train_indices, val_indices


# In[101]:


class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len, get_wids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids # for validation

  def __getitem__(self, index):
        # GET TEXT AND WORD LABELS
        text = self.data.text[index]
        word_labels = self.data.entities[index] if not self.get_wids else None

        # TOKENIZE TEXT
        encoding = self.tokenizer(text,
                             return_offsets_mapping=True,
                             padding=False,
                             truncation=True,
                             max_length=self.max_len)

        word_ids = encoding.word_ids()
        split_word_ids = np.full(len(word_ids),-1)
        offset_to_wordidx = split_mapping(text)
        offsets = encoding['offset_mapping']

        # CREATE TARGETS AND MAPPING OF TOKENS TO SPLIT() WORDS
        label_ids = []
        # Iterate in reverse to label whitespace tokens until a Begin token is encountered
        for token_idx, word_idx in reversed(list(enumerate(word_ids))):

            if word_idx is None:
                if not self.get_wids: label_ids.append(-100)
            else:
                if offsets[token_idx][0] != offsets[token_idx][1]:
                    #Choose the split word that shares the most characters with the token if any
                    split_idxs = offset_to_wordidx[offsets[token_idx][0]:offsets[token_idx][1]]
                    split_index = stats.mode(split_idxs[split_idxs != -1]).mode[0] if len(np.unique(split_idxs)) > 1 else split_idxs[0]

                    if split_index != -1:
                        if not self.get_wids: label_ids.append( labels_to_ids[word_labels[split_index]] )
                        split_word_ids[token_idx] = split_index
                    else:
                        # Even if we don't find a word, continue labeling 'I' tokens until a 'B' token is found
                        if label_ids and label_ids[-1] != -100 and ids_to_labels[label_ids[-1]][0] == 'I':
                            split_word_ids[token_idx] = split_word_ids[token_idx + 1]
                            if not self.get_wids: label_ids.append(label_ids[-1])
                        else:
                            if not self.get_wids: label_ids.append(-100)
                else:
                    if not self.get_wids: label_ids.append(-100)

        encoding['labels'] = list(reversed(label_ids))

        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids:
            item['wids'] = torch.as_tensor(split_word_ids)

        return item

  def __len__(self):
        return self.len

class CustomCollate:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer

    def __call__(self,data):
        """
        need to collate: input_ids, attention_mask, labels
        input_ids is padded with 1, attention_mask 0, labels -100

        """

        bs=len(data)
        lengths=[]
        for i in range(bs):
            lengths.append(len(data[i]['input_ids']))
            # print(data[i]['input_ids'].shape)
            # print(data[i]['attention_mask'].shape)
            # print(data[i]['labels'].shape)
        max_len=max(lengths)

        #always pad the right side
        input_ids, attention_mask, labels, BIO_labels, discourse_labels,wids=[],[],[],[],[],[]
        #if np.random.uniform()>0.5:
        for i in range(bs):
            input_ids.append(torch.nn.functional.pad(data[i]['input_ids'],(0,max_len-lengths[i]),value=self.tokenizer.pad_token_id))
            attention_mask.append(torch.nn.functional.pad(data[i]['attention_mask'],(0,max_len-lengths[i]),value=0))
            wids.append(torch.nn.functional.pad(data[i]['wids'],(0,max_len-lengths[i]),value=-1))
            #BIO_labels.append(torch.nn.functional.pad(data[i]['BIO_labels'],(0,max_len-lengths[i]),value=-100))
            #discourse_labels.append(torch.nn.functional.pad(data[i]['discourse_labels'],(0,max_len-lengths[i]),value=-100))
        # else:
        #     for i in range(bs):
        #         input_ids.append(torch.nn.functional.pad(data[i]['input_ids'],(max_len-lengths[i],0),value=1))
        #         attention_mask.append(torch.nn.functional.pad(data[i]['attention_mask'],(max_len-lengths[i],0),value=0))
        #         labels.append(torch.nn.functional.pad(data[i]['labels'],(max_len-lengths[i],0),value=-100))

        input_ids=torch.stack(input_ids)
        attention_mask=torch.stack(attention_mask)
        wids=torch.stack(wids)
       # labels=torch.stack(labels)
        #BIO_labels=torch.stack(BIO_labels)
        #discourse_labels=torch.stack(discourse_labels)
        #exit()

        return {"input_ids":input_ids,"attention_mask":attention_mask,
        "labels":labels,"BIO_labels":BIO_labels,"discourse_labels":discourse_labels,"wids":wids}


# # Create Train and Validation Dataloaders
# We will use the same train and validation subsets as my TensorFlow notebook [here][1]. Then we can compare results. And/or experiment with ensembling the validation fold predictions.
#
# [1]: https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-617

# In[102]:


# CHOOSE VALIDATION INDEXES (that match my TF notebook)
IDS = train_df.id.unique()


np.random.seed(42)

if FOLD == 'half':
    train_idx = np.random.choice(np.arange(len(IDS)),int(0.5*len(IDS)),replace=False)
    valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)

elif FOLD == 'full':
    train_idx = np.random.choice(np.arange(len(IDS)),int(0.5*len(IDS)),replace=False)
    valid_idx = np.arange(len(IDS))

# elif FOLD is not None:
#     print('There are',len(IDS),'train texts. We will split 93% 7% for ensemble training.')
#     shuffled_ids = np.arange(len(IDS))
#     np.random.shuffle(shuffled_ids)
#
#     valid_len = int(.07 * len(IDS))
#     valid_idx = shuffled_ids[FOLD*valid_len:(FOLD+1)*valid_len]
#     train_idx = np.setdiff1d(np.arange(len(IDS)),valid_idx)

else:
    print('There are',len(IDS),'train texts. We will split 90% 10% for ensemble training.')
    #train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
    #valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
    train_idx, valid_idx= iter_split(np.arange(len(IDS)),np.ones(len(IDS)),args.fold,nfolds=8)

TRAIN_IDS=IDS[train_idx]
VAL_IDS=IDS[valid_idx]

# print(len(valid_idx))
# exit()
# print(VAL_IDS)
# exit()

np.random.seed(None)


# In[103]:


# CREATE TRAIN SUBSET AND VALID SUBSET
data = train_text_df[['id','text', 'entities']]
train_dataset = data.loc[data['id'].isin(IDS[train_idx]),['text', 'entities']].reset_index(drop=True)
test_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)

print(test_dataset.id)
# # print(VAL_IDS)
#exit()

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

tokenizer = AutoTokenizer.from_pretrained(DOWNLOADED_MODEL_PATH)
training_set = dataset(train_dataset, tokenizer, config['max_length'], False)
testing_set = dataset(test_dataset, tokenizer, config['max_length'], True)


# In[111]:


# TRAIN DATASET AND VALID DATASET
train_params = {'batch_size': config['train_batch_size'],
                'shuffle': True,
                'num_workers': 2,
                'pin_memory':True
                }

test_params = {'batch_size': config['valid_batch_size'],
                'shuffle': False,
                'num_workers': 2,
                'pin_memory':True
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params,collate_fn=CustomCollate(tokenizer))

# TEST DATASET
test_texts_set = dataset(test_texts, tokenizer, config['max_length'], True)
test_texts_loader = DataLoader(test_texts_set, **test_params,collate_fn=CustomCollate(tokenizer))

#exit()


# In[112]:


from transformers import *
import torch.nn as nn
import torch.nn.functional as F
rearrange_indices=[14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
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


class ConvLSTMHead(nn.Module):
    def __init__(self):
        super(ConvLSTMHead, self).__init__()
        self.downsample=nn.Sequential(nn.Linear(1024,256))
        self.conv1=  nn.Sequential(nn.Conv1d(256,256,3,padding=1),
                                  nn.ReLU())
        self.norm1 = nn.LayerNorm(256)
        self.conv2=  nn.Sequential(nn.Conv1d(256,256,3,padding=1),
                                  nn.ReLU())
        self.norm2 = nn.LayerNorm(256)
        #self.lstm=nn.LSTM(256,256,2,bidirectional=True)
        self.lstm=ResidualLSTM(256)
        self.upsample=nn.Sequential(nn.Linear(256,1024),nn.ReLU())
        self.classification_head=nn.Sequential(nn.Linear(1024,15))


    def forward(self,x):

        x=self.downsample(x)
        res=x
        x=self.conv1(x.permute(0,2,1))
        x=self.norm1(x.permute(0,2,1)).permute(0,2,1)
        x=self.conv2(x)
        x=self.norm1(x.permute(0,2,1))
        x=x+res
        x=self.lstm(x.permute(1,0,2))
        x=x.permute(1,0,2)
        x=self.upsample(x)
        x=self.classification_head(x)
        #print(x.shape)
        #exit()
        return x


class TransformerModel(nn.Module):
    def __init__(self,DOWNLOADED_MODEL_PATH, rnn='LSTM', no_backbone=False):
        super(TransformerModel, self).__init__()
        config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH+'/config.json')

        self.backbone=AutoModel.from_pretrained(
                           DOWNLOADED_MODEL_PATH+'/pytorch_model.bin',config=config_model)

        self.lstm=ResidualLSTM(1024,rnn)
        self.classification_head=nn.Linear(1024,15)
        self.no_backbone=no_backbone


    def forward(self,x,attention_mask,return_transformer_hidden_states=False):
        if self.no_backbone==True:
            pass
        else:
            x=self.backbone(input_ids=x,attention_mask=attention_mask,return_dict=False)[0]
            if return_transformer_hidden_states:
                transformer_hidden_states=x
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
        if return_transformer_hidden_states:
            return [x[:,:,rearrange_indices]],transformer_hidden_states
        else:
            return [x[:,:,rearrange_indices]]#, BIO_output
model = TransformerModel(DOWNLOADED_MODEL_PATH,'GRU').to(config['device'])



import warnings

warnings.filterwarnings('ignore', '.*__floordiv__ is deprecated.*',)






# Returns per-word, mean class prediction probability over all tokens corresponding to each word
# Returns per-word, mean class prediction probability over all tokens corresponding to each word
def inference(data_loader, model_ids):

    gc.collect()
    torch.cuda.empty_cache()

    ensemble_preds = np.zeros((len(data_loader.dataset), config['max_length'], len(labels_to_ids)), dtype=np.float32)
    wids = np.full((len(data_loader.dataset), config['max_length']), -100)
    for model_i, model_id in enumerate(model_ids):

        model.load_state_dict(torch.load(f'{LOAD_MODEL_FROM}/fold{model_id}.pt', map_location=config['device']))

        # put model in training mode
        model.eval()
        for batch_i, batch in tqdm(enumerate(data_loader),total=len(data_loader)):

            if model_i == 0: wids[batch_i*config['valid_batch_size']:(batch_i+1)*config['valid_batch_size'],:batch['wids'].shape[1]] = batch['wids'].numpy()

            # MOVE BATCH TO GPU AND INFER
            ids = batch["input_ids"].to(config['device'])
            mask = batch["attention_mask"].to(config['device'])
            with torch.no_grad():
                with amp.autocast():
                    outputs, hidden_states = model(ids, attention_mask=mask,return_transformer_hidden_states=True)
            all_preds = torch.nn.functional.softmax(outputs[0], dim=2).cpu().detach().numpy()
            #all_preds/=2
            ensemble_preds[batch_i*config['valid_batch_size']:(batch_i+1)*config['valid_batch_size'],:all_preds.shape[1]] += all_preds

            del ids
            del mask
            del outputs
            del all_preds

        gc.collect()
        torch.cuda.empty_cache()

    ensemble_preds /= len(model_ids)
    predictions = []
    # INTERATE THROUGH EACH TEXT AND GET PRED
    for text_i in range(ensemble_preds.shape[0]):
        token_preds = ensemble_preds[text_i]

        prediction = []
        previous_word_idx = -1
        prob_buffer = []
        word_ids = wids[text_i][wids[text_i] != -100]
        for idx,word_idx in enumerate(word_ids):
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:
                if prob_buffer:
                    prediction.append(np.mean(prob_buffer, dtype=np.float32, axis=0))
                    prob_buffer = []
                prob_buffer.append(token_preds[idx])
                previous_word_idx = word_idx
            else:
                prob_buffer.append(token_preds[idx])
        prediction.append(np.mean(prob_buffer, dtype=np.float32, axis=0))
        predictions.append(prediction)

    gc.collect()
    torch.cuda.empty_cache()
    return predictions




import pickle
valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

print('Predicting with BigBird...')
if not SUBMISSION:
    try:
        with open( KAGGLE_CACHE + f"/valid_preds_fold{args.fold}.p", "rb" ) as validFile:
            valid_word_preds = pickle.load( validFile )
        print("preds loaded")
    except:
        valid_word_preds = inference(testing_loader, ENSEMBLE_IDS)
        with open( cache + f"/valid_preds_fold{args.fold}.p", "wb+" ) as validFile:
            pickle.dump( valid_word_preds, validFile )
else: valid_word_preds = []

#test_word_preds = inference(test_texts_loader, ENSEMBLE_IDS)


print('Done.')

uniqueValidGroups = range(len(valid_word_preds))
uniqueSubmitGroups = range(5)
