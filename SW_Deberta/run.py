import os
import torch.nn as nn
import torch
import warnings
import argparse
from Logger import *
import pickle
from Dataset import *
warnings.filterwarnings("ignore")
from Functions import *
from Network import *
import pandas as pd
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight dacay used in optimizer')
    parser.add_argument('--ntoken', type=int, default=4, help='number of tokens to represent DNA nucleotides (should always be 4)')
    parser.add_argument('--nclass', type=int, default=919, help='number of classes from the linear decoder')
    parser.add_argument('--ninp', type=int, default=512, help='ninp for transformer encoder')
    parser.add_argument('--nhead', type=int, default=8, help='nhead for transformer encoder')
    parser.add_argument('--nhid', type=int, default=2048, help='nhid for transformer encoder')
    parser.add_argument('--nlayers', type=int, default=6, help='nlayers for transformer encoder')
    parser.add_argument('--save_freq', type=int, default=1, help='saving checkpoints per save_freq epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--warmup_steps', type=int, default=3200, help='training schedule warmup steps')
    parser.add_argument('--lr_scale', type=float, default=1, help='learning rate scale')
    parser.add_argument('--nmute', type=int, default=18, help='number of mutations during training')
    parser.add_argument('--kmers', type=int, nargs='+', default=[7], help='k-mers to be aggregated')
    #parser.add_argument('--kmer_aggregation', type=bool, default=True, help='k-mers to be aggregated')
    parser.add_argument('--kmer_aggregation', dest='kmer_aggregation', action='store_true')
    parser.add_argument('--no_kmer_aggregation', dest='kmer_aggregation', action='store_false')
    parser.set_defaults(kmer_aggregation=True)
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    opts = parser.parse_args()
    return opts



#def train_fold():

args=get_args()


# DECLARE HOW MANY GPUS YOU WISH TO USE.
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id #0,1,2,3 for four gpu

if torch.cuda.device_count() > 1:
    DATAPARALLEL=True
else:
    DATAPARALLEL=False

# print(torch.cuda.device_count())
#
# exit()

# VERSION FOR SAVING MODEL WEIGHTS
VER=26

# IF VARIABLE IS NONE, THEN NOTEBOOK COMPUTES TOKENS
# OTHERWISE NOTEBOOK LOADS TOKENS FROM PATH
LOAD_TOKENS_FROM = '../../input/py-bigbird-v26'

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
#LOAD_MODEL_FROM = '../input/whitespace'
LOAD_MODEL_FROM = None


# IF FOLLOWING IS NONE, THEN NOTEBOOK
# USES INTERNET AND DOWNLOADS HUGGINGFACE
# CONFIG, TOKENIZER, AND MODEL
DOWNLOADED_MODEL_PATH = "../../input/deberta-xlarge"

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = 'microsoft/deberta-xlarge'
#MODEL_NAME = "google/bigbird-roberta-large"

from torch import cuda
config = {'model_name': MODEL_NAME,
         'max_length': 1536,
         'train_batch_size':2,
         'valid_batch_size':1,
         'epochs':7,
         'learning_rates': [2.5e-5, 2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-6, 2.5e-7],
         'max_grad_norm':1,
         'device': 'cuda' if cuda.is_available() else 'cpu'}

config['learning_rates']=[lr*args.lr_scale for lr in config['learning_rates']]

print('learning_rates:')

print(config['learning_rates'])

#lr_scale

# THIS WILL COMPUTE VAL SCORE DURING COMMIT BUT NOT DURING SUBMIT
COMPUTE_VAL_SCORE = True
if len( os.listdir('../../input/test') )>5:
      COMPUTE_VAL_SCORE = False

from transformers import *
if DOWNLOADED_MODEL_PATH == 'model':
    os.system('mkdir model')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    tokenizer.save_pretrained('model')

    config_model = AutoConfig.from_pretrained(MODEL_NAME)
    config_model.num_labels = 15
    config_model.save_pretrained('model')

    backbone = AutoModelForTokenClassification.from_pretrained(MODEL_NAME,
                                                               config=config_model)
    backbone.save_pretrained('model')

#load data and libraries
import numpy as np, os
from scipy import stats
import pandas as pd, gc
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score


train_df = pd.read_csv('../../input/train.csv')
print( train_df.shape )
train_df.head()

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, test_texts = [], []
for f in list(os.listdir('../../input/test')):
    test_names.append(f.replace('.txt', ''))
    test_texts.append(open('../../input/test/' + f, 'r').read())
test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})
print(test_texts.head())


# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, train_texts = [], []
for f in tqdm(list(os.listdir('../../input/train'))):
    test_names.append(f.replace('.txt', ''))
    train_texts.append(open('../../input/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})
print(train_text_df.head())

#convert train to text labels
if not LOAD_TOKENS_FROM:
    all_entities = []
    for ii,i in tqdm(enumerate(train_text_df.iterrows())):
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
print(train_text_df.head())


# CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER
output_labels = ['B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim',
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement','O',]




labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}




# CHOOSE VALIDATION INDEXES (that match my TF notebook)
IDS = train_df.id.unique()
print('There are',len(IDS),'train texts. We will split 90% 10% for validation.')

# TRAIN VALID SPLIT 90% 10%
#np.random.seed(42)
train_idx, valid_idx= iter_split(np.arange(len(IDS)),np.ones(len(IDS)),args.fold,nfolds=args.nfolds)
#np.random.seed(None)

# CREATE TRAIN SUBSET AND VALID SUBSET
data = train_text_df[['id','text', 'entities']]
train_dataset = data.loc[data['id'].isin(IDS[train_idx]),['text', 'entities']].reset_index(drop=True)
test_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

tokenizer = AutoTokenizer.from_pretrained(DOWNLOADED_MODEL_PATH)
training_set = FeedbackDataset(train_dataset, tokenizer, config['max_length'], False, labels_to_ids, ids_to_labels)
testing_set = FeedbackDataset(test_dataset, tokenizer, config['max_length'], True, labels_to_ids, ids_to_labels)
#exit()
# TRAIN DATASET AND VALID DATASET
train_params = {'batch_size': config['train_batch_size'],
                'shuffle': True,
                'num_workers': config['train_batch_size'],
                'pin_memory':True
                }

test_params = {'batch_size': config['valid_batch_size'],
                'shuffle': False,
                'num_workers': config['valid_batch_size'],
                'pin_memory':True
                }

training_loader = DataLoader(training_set, **train_params, collate_fn=CustomCollate(tokenizer,512))
testing_loader = DataLoader(testing_set, **test_params, collate_fn=CustomCollate(tokenizer,512))

# TEST DATASET
test_texts_set = FeedbackDataset(test_texts, tokenizer, config['max_length'], True, labels_to_ids, ids_to_labels)
test_texts_loader = DataLoader(test_texts_set, **test_params, collate_fn=CustomCollate(tokenizer,512))

os.system('mkdir logs')
os.system('mkdir models')
os.system('mkdir oofs')

columns=['epoch','train_loss','train_acc','val_score']
logger=CSVLogger(columns,f"logs/fold{args.fold}.csv")

from torch.cuda.amp import GradScaler

scaler = GradScaler()

# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    #tr_preds, tr_labels = [], []
    criterion=nn.CrossEntropyLoss(reduction='none')
    #criterion=DiceLoss(reduction='none')
    #criterion=DiceLoss(square_denominator=True, with_logits=True, index_label_position=True,
                        #smooth=1, ohem_ratio=0, alpha=0.01, reduction="none")
    # put model in training mode
    model.train()
    bar=tqdm(enumerate(training_loader),total=len(training_loader))
    for idx, batch in bar:


        ids = batch['input_ids'].to(config['device'], dtype = torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype = torch.long)
        labels = batch['labels'].to(config['device'], dtype = torch.long)
        #BIO_labels = batch['BIO_labels'].to(config['device'], dtype = torch.long)
        #discourse_labels = batch['discourse_labels'].to(config['device'], dtype = torch.long)
        #loss_mask=BIO_labels!=-100
        if np.random.uniform()<0.5:
            cut=0.25
            perm=torch.randperm(ids.shape[0]).cuda()
            rand_len=int(ids.shape[1]*cut)
            start=np.random.randint(ids.shape[1]-int(ids.shape[1]*cut))
            ids[:,start:start+rand_len]=ids[perm,start:start+rand_len]
            mask[:,start:start+rand_len]=mask[perm,start:start+rand_len]
            labels[:,start:start+rand_len]=labels[perm,start:start+rand_len]
        # print(labels.shape)
        # exit()


        # print(BIO_labels.shape)
        # print(discourse_labels.shape)
        # exit()


        with torch.autocast(device_type="cuda"):
            output = model(ids,mask)
            # print(classification_output.shape)
            # print(BIO_output.shape)
            # exit()
            # print(classification_output.shape)
            # print(loss_mask.shape)
            # exit()
            # classification_output=torch.masked_select(classification_output,loss_mask.unsqueeze(-1))
            # discourse_labels=torch.masked_select(discourse_labels,loss_mask)
            # BIO_output=torch.masked_select(BIO_output,loss_mask.unsqueeze(-1))
            # BIO_labels=torch.masked_select(BIO_labels,loss_mask)

            # print(classification_output.shape)
            # print(discourse_labels.shape)
            # print(BIO_output.shape)
            # print(BIO_labels.shape)
            # exit()
            # print(classification_output.shape)
            # classification_output=classification_output.reshape(-1,8)
            # BIO_output=BIO_output.reshape(-1,3)
            # BIO_labels=BIO_labels.reshape(-1)
            # discourse_labels=discourse_labels.reshape(-1)
            # loss_mask=BIO_labels!=-100
            #
            # print(classification_output.shape)
            # classification_output=torch.masked_select(classification_output,loss_mask)
            # discourse_labels=torch.masked_select(discourse_labels,loss_mask)
            # BIO_output=torch.masked_select(BIO_output,loss_mask)
            # BIO_labels=torch.masked_select(BIO_labels,loss_mask)
            #print(BIO_labels)

            output=output.reshape(-1,15)
            labels=labels.reshape(-1)
            #discourse_labels=discourse_labels.reshape(-1)
            loss_mask=labels!=-100
            labels[labels==-100]=0
            #discourse_labels[discourse_labels==-100]=0
            #print(BIO_labels)
            #exit(0)
            loss=criterion(output,labels)
            #BIO_loss=criterion(BIO_output,BIO_labels)
            # print(classification_loss)
            # print(BIO_loss)
            # print(BIO_loss.shape)
            # exit()
            loss=torch.masked_select(loss,loss_mask).mean()



        if DATAPARALLEL:
            loss=loss.mean()

        tr_loss += loss.item()

        bar.set_postfix({'train_loss': tr_loss/(idx+1)})

        nb_tr_steps += 1
        # nb_tr_examples += labels.size(0)
        #
        # # if idx % 200==0:
        # #     loss_step = tr_loss/nb_tr_steps
        # #     print(f"Training loss after {idx:04d} training steps: {loss_step}")
        #
        # # compute training accuracy
        # flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        # if DATAPARALLEL:
        #     active_logits = tr_logits.view(-1, model.module.num_labels)
        # else:
        #     active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        # flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        #
        # # only compute accuracy at active labels
        # active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        # #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        #
        # labels = torch.masked_select(flattened_targets, active_accuracy)
        # predictions = torch.masked_select(flattened_predictions, active_accuracy)
        #
        # tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        # tr_accuracy += tmp_tr_accuracy

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=config['max_grad_norm']
        )



        scaler.step(optimizer)
        scaler.update()
        #loss.backward()
        #optimizer.step()
        #break

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = 0
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")
    return epoch_loss, tr_accuracy

# CREATE MODEL
config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH+'/config.json')
# model = AutoModelForTokenClassification.from_pretrained(
#                    DOWNLOADED_MODEL_PATH+'/pytorch_model.bin',config=config_model)
model = SlidingWindowTransformerModel(DOWNLOADED_MODEL_PATH)
model.to(config['device'])
if DATAPARALLEL:
    model=nn.DataParallel(model)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['learning_rates'][0], weight_decay=args.weight_decay)

def inference(batch):

    # MOVE BATCH TO GPU AND INFER
    ids = batch["input_ids"].to(config['device'])
    mask = batch["attention_mask"].to(config['device'])

    with torch.no_grad():
        outputs = F.softmax(model(ids, mask),-1)
    # = torch.argmax(outputs[0], axis=-1).cpu().numpy()
    max_values, all_preds = outputs.max(-1)#.cpu().numpy()
    # print(all_preds.shape)
    # print(max_values.shape)
    #all_preds[max_values<0.5]=0
    all_preds=all_preds.cpu().numpy()
    max_values=max_values.cpu().numpy()
    #exit()
    #all_preds[torch.max(outputs[0], axis=-1)]


    # INTERATE THROUGH EACH TEXT AND GET PRED
    predictions = []
    word_max_values=[]
    for k,text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]

        prediction = []
        word_ids = batch['wids'][k].numpy()

        previous_word_idx = -1
        gather_indices=[]
        for idx,word_idx in enumerate(word_ids):
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:
                prediction.append(token_preds[idx])
                gather_indices.append(idx)
                previous_word_idx = word_idx
        #print(max_values.shape)
        word_max_values_sample=max_values[k][gather_indices]
        word_max_values.append(word_max_values_sample)
        # print(word_max_values_sample.shape)
        # exit()
        predictions.append(prediction)

    return predictions, word_max_values

# https://www.kaggle.com/zzy990106/pytorch-ner-infer
# code has been modified from original
def get_predictions(df=test_dataset, loader=testing_loader):
    proba_thresh = {
        "Lead": 0.7,
        "Position": 0.55,
        "Evidence": 0.65,
        "Claim": 0.55,
        "Concluding Statement": 0.7,
        "Counterclaim": 0.5,
        "Rebuttal": 0.55,
    }
    # put model in training mode
    model.eval()

    # GET WORD LABEL PREDICTIONS
    y_pred2 = []
    max_values=[]
    for batch in tqdm(loader):
        # print(batch)
        # exit()
        labels, word_max_values = inference(batch)
        y_pred2.extend(labels)
        max_values.extend(word_max_values)


    final_preds2 = []
    for i in range(len(df)):

        idx = df.id.values[i]
        #pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
        pred = y_pred2[i] # Leave "B" and "I"
        preds = []
        j = 0
        while j < len(pred):
            cls = pred[j]
            # The commented out line below appears to be a bug.
#             if cls == 'O': j += 1
            if cls == 'O': pass
            else: cls = cls.replace('B','I') # spans start with B
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1

            if cls != 'O' and cls != '' and end - j > 2:
                # print(max_values[i][j:end].mean())
                # print(proba_thresh[cls.split('-')[1]])
                th=proba_thresh[cls.split('-')[1]]
                #th=0
                if max_values[i][j:end].mean()>th:
                    final_preds2.append((idx, cls.replace('I-',''),
                                         ' '.join(map(str, list(range(j, end))))))

            j = end

    oof = pd.DataFrame(final_preds2)
    if len(oof)>0:
        oof.columns = ['id','class','predictionstring']

    return oof, y_pred2, max_values






# LOOP TO TRAIN MODEL (or load model)
best_score=0
for epoch in range(config['epochs']):

    print(f"### Training epoch: {epoch + 1}")
    for g in optimizer.param_groups:
        g['lr'] = config['learning_rates'][epoch]
    lr = optimizer.param_groups[0]['lr']
    print(f'### LR = {lr}\n')

    train_loss,train_acc=train(epoch)
    torch.cuda.empty_cache()
    gc.collect()

    valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

    # OOF PREDICTIONS
    oof, y_pred2, max_values = get_predictions(test_dataset, testing_loader)
    if len(oof)>0:
        oof = link_evidence(oof)
        oof['len'] = oof['predictionstring'].apply(lambda x:len(x.split()))
        oof = threshold(oof)
        oof=oof[['id','class','predictionstring']]
        # COMPUTE F1 SCORE
        f1s = []
        CLASSES = oof['class'].unique()
        print()
        for c in CLASSES:
            pred_df = oof.loc[oof['class']==c].copy()
            gt_df = valid.loc[valid['discourse_type']==c].copy()
            f1 = score_feedback_comp(pred_df, gt_df)
            print(c,f1)
            f1s.append(f1)
        val_score=np.mean(f1s)
    else:
        val_score=0

    if val_score>best_score:
         best_score=val_score
         torch.save(model.state_dict(), f'models/fold{args.fold}.pt')
         with open(f"oofs/fold{args.fold}.p",'wb+') as f:
             pickle.dump([y_pred2, max_values],f)


    print()
    print(f'Val score for epoch {epoch}',val_score)
    print()

    logger.log([epoch,train_loss,train_acc,val_score])
