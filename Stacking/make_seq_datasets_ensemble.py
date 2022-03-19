from sklearn.model_selection import StratifiedKFold

import os, sys
# DECLARE HOW MANY GPUS YOU WISH TO USE.
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE

import argparse
def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--disc_type', type=int, default=0,  help='disc_type')
    parser.add_argument('--fold', type=int, default=0,  help='fold')
    opts = parser.parse_args()
    return opts

args=get_args()

os.environ["CUDA_VISIBLE_DEVICES"]= '0'#str(args.fold) #0,1,2,3 for four gpu


# VERSION FOR SAVING MODEL WEIGHTS
VER=26

# IF VARIABLE IS NONE, THEN NOTEBOOK COMPUTES TOKENS
# OTHERWISE NOTEBOOK LOADS TOKENS FROM PATH
LOAD_TOKENS_FROM = True #'../../input/py-bigbird-v26'

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

N_FEATURES=34


KAGGLE_CACHE = ['../SW_Deberta/cache/', '../Longformer/cache/'] #, 'fp78_cache', deberta_overlap_cache location of valid_pred files

cache = 'seq_datasets' #save location of valid_seqds files
cacheExists = os.path.exists(cache)
if not cacheExists:
  os.makedirs(cache)

print(ENSEMBLE_IDS)

from transformers import *

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


train_df = pd.read_csv('../../input/train.csv')
print( train_df.shape )
train_df.head()


# In[95]:


# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, test_texts = [], []
for f in list(os.listdir('../../input/test')):
    test_names.append(f.replace('.txt', ''))
    test_texts.append(open('../../input/test/' + f, 'r').read())
test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})

test_texts['len']=test_texts['text'].apply(lambda x:len(x.split()))
test_texts=test_texts.sort_values(by=['len']).reset_index()

test_texts

SUBMISSION = False
if len(test_names) > 5:
      SUBMISSION = True

test_texts.head()




# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, train_texts = [], []
for f in tqdm(list(os.listdir('../../input/train'))):
    test_names.append(f.replace('.txt', ''))
    train_texts.append(open('../../input/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})
train_text_df.head()



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
    train_text_df.to_csv('../train_NER.csv',index=False)

else:
    from ast import literal_eval
    train_text_df = pd.read_csv(f'../../input/py-bigbird-v26/train_NER.csv')
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




# CHOOSE VALIDATION INDEXES (that match my TF notebook)
IDS = train_df.id.unique()
print(IDS[:4])


np.random.seed(42)

print('There are',len(IDS),f'train texts. We will split it into {8} folds with seed {2020} and use {args.fold} oofs to make {N_FEATURES} features')
#train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
#valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
#train_idx, valid_idx= iter_split(np.arange(len(IDS)),np.ones(len(IDS)),args.fold,nfolds=8)
#print(train_idx)
split_df=pd.read_csv("../shujun_8_fold_split_seed_2020.csv")
train_idx=np.where(split_df['kfold']!=args.fold)[0]
#print(train_idx)
valid_idx=np.where(split_df['kfold']==args.fold)[0]
#exit()


TRAIN_IDS=IDS[train_idx]
VAL_IDS=IDS[valid_idx]


np.random.seed(None)



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

import warnings

warnings.filterwarnings('ignore', '.*__floordiv__ is deprecated.*',)


# from Rob Mulla @robikscube
# https://www.kaggle.com/robikscube/student-writing-competition-twitch
def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter/ len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id','discourse_type','predictionstring']]         .reset_index(drop=True).copy()
    pred_df = pred_df[['id','class','predictionstring']]         .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id','class'],
                           right_on=['id','discourse_type'],
                           how='outer',
                           suffixes=('_pred','_gt')
                          )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])


    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1','overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP')         .sort_values('max_overlap', ascending=False)         .groupby(['id','predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    #calc microf1
    my_f1_score = TP / (TP + 0.5*(FP+FN))
    return my_f1_score

def calc_overlap_shujun(pred, gt):
    """
    Calculates if the overlap between prediction and
    ground truth is enough fora potential True positive
    """
    try:
        g1=pred[1]+1-gt[0]
        g2=gt[1]+1-pred[0]
        l1=pred[1]-pred[0]+1
        l2=gt[1]-gt[0]+1
        #print(g1,g2)
        if g1*g2>=0:
            #g1=abs(g1)+1
            #g2=abs(g2)+1
            inter=min((g1,g2,l1,l2))#/max((g1,g2,l1,l2))
            overlap_1=inter/l1
            overlap_2=inter/l2
            return overlap_1 >= 0.5 and overlap_2 >= 0.5
        else:
            return False
    except:
        return False



def score_feedback_comp_micro_shujun(pred_df, gt_df, discourse_type):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df.loc[gt_df['discourse_type'] == discourse_type,
                      ['id', 'predictionstring']].reset_index(drop=True)
    pred_df = pred_df.loc[pred_df['class'] == discourse_type,
                      ['id', 'predictionstring']].reset_index(drop=True)
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    pred_df['predictionstring'] = [(int(pred.split(' ')[0]),int(pred.split(' ')[-1])) for pred in pred_df['predictionstring']]
    gt_df['predictionstring'] = [(int(pred.split(' ')[0]),int(pred.split(' ')[-1])) for pred in gt_df['predictionstring']]


#     print(pred_df[pred_df['predictionstring']!=pred_df['predictionstring']])
#     exit()
    #gt_strings=

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on='id',
                           right_on='id',
                           how='outer',
                           suffixes=('_pred','_gt')
                          )
    overlaps = [calc_overlap_shujun(*args) for args in zip(list(joined.predictionstring_pred),
                                                     list(joined.predictionstring_gt))]

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    # we don't need to compute the match to compute the score
    TP = joined.loc[overlaps]['gt_id'].nunique()

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    TPandFP = len(pred_df)
    TPandFN = len(gt_df)

    #calc microf1
    my_f1_score = 2*TP / (TPandFP + TPandFN)
    return my_f1_score

def score_feedback_comp_shujun(pred_df, gt_df, return_class_scores=False):
    class_scores = {}
    for discourse_type in gt_df.discourse_type.unique():
        class_score = score_feedback_comp_micro_shujun(pred_df, gt_df, discourse_type)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


# Aggregate the per-word, mean class probability predictions from BigBird for validation and submit sets.

# In[117]:


import pickle
valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

#print('Predicting with BigBird...')
if not SUBMISSION:
    with open( KAGGLE_CACHE[0] + f"/valid_preds_fold{args.fold}.p", "rb" ) as validFile:
        valid_word_preds = pickle.load( validFile )
    with open( KAGGLE_CACHE[1] + f"/valid_preds_fold{args.fold}.p", "rb" ) as validFile:
        valid_word_preds2 = pickle.load( validFile )
    print("preds loaded")

ensemble_weights = {"Rebuttal": 0.6 + 0.05,
                    "Counterclaim": 0.7 + 0.05,
                    "Concluding Statement": 0.5 + 0.05,
                    "Claim": 0.6 + 0.05,
                    "Evidence": 0.55 + 0.05,
                    "Position": 0.7 + 0.05,
                    "Lead": 0.65 + 0.05
                    }

preds={}
for dis in disc_type_to_ids:
    dis_preds={}
    for i in range(len(test_dataset)):
        id=test_dataset.id[i]
        dis_preds[id]=valid_word_preds[i]
        w=ensemble_weights[dis]
        for j in range(len(valid_word_preds[i])):
            dis_preds[id][j]=w*dis_preds[id][j]+(1-w)*valid_word_preds2[i][j]

    preds[dis]=dis_preds

#exit()

with open(f"{cache}/preds_hash_fold{args.fold}.p",'wb+') as f:
    pickle.dump(preds,f)
#exit()
#test_word_preds = inference(test_texts_loader, ENSEMBLE_IDS)


#print('Done.')

uniqueValidGroups = range(len(valid_word_preds))
uniqueSubmitGroups = range(5)


# # Sequence Datasets
# We will create datasets that, instead of describing individual words or tokens, describes sequences of words. Within some heuristic constraints, every possible sub-sequence of words in a text will converted to a dataset sample with the following attributes:
#
# * features- sequence length, position, and various kinds of class probability predictions/statistics
# * labels- whether the sequence matches exactly a discourse instance
# * truePos- whether the sequence matches a discourse instance by competition criteria for true positive
# * groups- the integer index of the text where the sequence is found
# * wordRanges- the start and end word index of the sequence in the text
#
# Sequence datasets are generated for each discourse type and for validation and submission datasets.

#

# In[118]:


from collections import Counter
from bisect import bisect_left

# Percentile code taken from https://www.kaggle.com/vuxxxx/tensorflow-longformer-ner-postprocessing
# Thank Vu!
#
# Use 99.5% of the distribution of lengths for a disourse type as maximum.
# Increasing this constraint makes this step slower but generally increases performance.
MAX_SEQ_LEN = {}
train_df['len'] = train_df['predictionstring'].apply(lambda x:len(x.split()))
max_lens = train_df.groupby('discourse_type')['len'].quantile(.995)
for disc_type in disc_type_to_ids:
    MAX_SEQ_LEN[disc_type] = int(max_lens[disc_type])

#The minimum probability prediction for a 'B'egin class for which we will evaluate a word sequence
MIN_BEGIN_PROB = {
    'Claim': .35*0.8,
    'Concluding Statement': .15*1.0,
    'Counterclaim': .04*1.25,
    'Evidence': .1*0.8,
    'Lead': .32*1.0,
    'Position': .25*0.8,
    'Rebuttal': .01*1.25,
}

class SeqDataset(object):

    def __init__(self, ids, features, labels, groups, wordRanges, truePos):

        self.features = np.array(features, dtype=np.float32)
        self.labels = np.array(labels)
        self.groups = np.array(groups, dtype=np.int16)
        self.wordRanges = np.array(wordRanges, dtype=np.int16)
        self.truePos = np.array(truePos)
        self.ids=ids

# Adapted from https://stackoverflow.com/questions/60467081/linear-interpolation-in-numpy-quantile
# This is used to prevent re-sorting to compute quantile for every sequence.
def sorted_quantile(array, q):
    array = np.array(array)
    n = len(array)
    index = (n - 1) * q
    left = np.floor(index).astype(int)
    fraction = index - left
    right = left
    right = right + (fraction > 0).astype(int)
    i, j = array[left], array[right]
    return i + (j - i) * fraction

def seq_dataset(disc_type, pred_indices=None, submit=False):
    begin_class_ids = [0, 1, 3, 5, 7, 9, 11, 13]

    w = ensemble_weights[disc_type]
    word_preds = list(valid_word_preds)

    for i in range(len(valid_word_preds)):
        if len(valid_word_preds[i]) <= len(valid_word_preds2[i]):
            for j in range(len(valid_word_preds[i])):
                word_preds[i][j] = w*valid_word_preds[i][j] + (1-w)*valid_word_preds2[i][j]
        else:
            for j in range(len(valid_word_preds2[i])):
                word_preds[i][j] = w*valid_word_preds[i][j] + (1-w)*valid_word_preds2[i][j]

    window = pred_indices if pred_indices else range(len(word_preds))
    # print(window)
    # exit()
    X = np.empty((int(1e6),N_FEATURES), dtype=np.float32)
    X_ind = 0
    y = []
    truePos = []
    wordRanges = []
    groups = []
    ids =[]
    #df_array = []
    #df_columns=['id',"true_pos","exact_match","pred_start","pred_end"]+[f"feature_{i}" for i in range(13)]
    #df=pd.DataFrame(columns=df_columns)
    for text_i in tqdm(window):
        current_id=test_dataset.id.values[text_i]
        #print(id)
        text_preds = np.array(word_preds[text_i])
        num_words = len(text_preds)
        # with open(f'../input/train/{current_id}.txt','r') as f:
        #     text=f.read().split()
        # print("####")
        # print(len(text))
        # print(num_words)

        global_features, global_locs = [], []

        for dt in disc_type_to_ids:
            disc_begin, disc_inside = disc_type_to_ids[dt]

            gmean = (text_preds[:, disc_begin] + text_preds[:, disc_inside]).mean()
            global_features.append(gmean)
            global_locs.append(np.argmax(text_preds[:, disc_begin])/float(num_words))


        disc_begin, disc_inside = disc_type_to_ids[disc_type]

        # The probability that a word corresponds to either a 'B'-egin or 'I'-nside token for a class
        #prob_or = lambda word_preds: (1-(1-word_preds[:,disc_begin]) * (1-word_preds[:,disc_inside]))
        prob_or = lambda word_preds: word_preds[:,disc_begin] + word_preds[:,disc_inside]



        if not submit:
            gt_idx = set()
            gt_arr = np.zeros(num_words, dtype=int)
            text_gt = valid.loc[valid.id == test_dataset.id.values[text_i]]
            disc_gt = text_gt.loc[text_gt.discourse_type == disc_type]

            # Represent the discourse instance locations in a hash set and an integer array for speed
            for row_i, row in enumerate(disc_gt.iterrows()):
                splt = row[1]['predictionstring'].split()
                start, end = int(splt[0]), int(splt[-1]) + 1
                gt_idx.add((start, end))
                gt_arr[start:end] = row_i + 1
            gt_lens = np.bincount(gt_arr)

        # Iterate over every sub-sequence in the text
        quants = np.linspace(0,1,7)
        prob_begins = np.copy(text_preds[:,disc_begin])
        min_begin = MIN_BEGIN_PROB[disc_type]
        for pred_start in range(num_words):
            prob_begin = prob_begins[pred_start]
            if prob_begin > min_begin:
                begin_or_inside = []
                for pred_end in range(pred_start+1,min(num_words+1, pred_start+MAX_SEQ_LEN[disc_type]+1)):

                    new_prob = prob_or(text_preds[pred_end-1:pred_end])
                    insert_i = bisect_left(begin_or_inside, new_prob)
                    begin_or_inside.insert(insert_i, new_prob[0])

                    # Generate features for a word sub-sequence

                    # The length and position of start/end of the sequence
                    features = [pred_end - pred_start, pred_start / float(num_words), pred_end / float(num_words)]

                    # 7 evenly spaced quantiles of the distribution of relevant class probabilities for this sequence
                    features.extend(list(sorted_quantile(begin_or_inside, quants)))

                    # The probability that words on either edge of the current sub-sequence belong to the class of interest
                    features.append(prob_or(text_preds[pred_start-1:pred_start])[0] if pred_start > 0 else 0)
                    features.append(prob_or(text_preds[pred_end:pred_end+1])[0] if pred_end < num_words else 0)
                    features.append(prob_or(text_preds[pred_start-2:pred_start-1])[0] if pred_start > 1 else 0)
                    features.append(prob_or(text_preds[pred_end+1:pred_end+2])[0] if pred_end < (num_words-1) else 0)

                    # The probability that the first word corresponds to a 'B'-egin token
                    features.append(text_preds[pred_start,disc_begin])
                    features.append(text_preds[pred_start-1,disc_begin])

                    if pred_end < num_words:
                        features.append(text_preds[pred_end, begin_class_ids].sum())
                    else:
                        features.append(1.0)


                    s = prob_or(text_preds[pred_start:pred_end])
                    features.append(np.argmax(s)/features[0]) # maximum point location on sequence
                    features.append(np.argmin(s)/features[0]) # minimum point location on sequence
                    instability = 0
                    if len(s) > 1:
                        instability = (np.diff(s)**2).mean()
                    features.append(instability)

                    features.extend(list(global_features))
                    features.extend(list([loc - features[1] for loc in global_locs]))

                    exact_match = (pred_start, pred_end) in gt_idx if not submit else None

                    if not submit:
                        true_pos = False
                        for match_cand, count in Counter(gt_arr[pred_start:pred_end]).most_common(2):
                            if match_cand != 0 and count / float(pred_end - pred_start) >= .5 and float(count) / gt_lens[match_cand] >= .5: true_pos = True
                    else: true_pos = None

                    # For efficiency, use a numpy array instead of a list that doubles in size when full to conserve constant "append" time complexity
                    if X_ind >= X.shape[0]:
                        new_X = np.empty((X.shape[0]*2,N_FEATURES), dtype=np.float32)
                        new_X[:X.shape[0]] = X
                        X = new_X
                    X[X_ind] = features
                    X_ind += 1

                    y.append(exact_match)
                    truePos.append(true_pos)
                    wordRanges.append((np.int16(pred_start), np.int16(pred_end)))
                    groups.append(np.int16(text_i))
                    ids.append(current_id)
                    #df_row=[current_id,true_pos,exact_match,pred_start,pred_end]+features
                    #df=df.append({k:v for k,v in zip(df_columns,df_row)},ignore_index=True)
                    #df_array.append(df_row)
                    # print(np.int16(text_i))
                    # exit()
                    #feature_0: pred_end - pred_start so length of span -1
                    #feature_1: normalized start position (normalized by number of words)
                    #feature_2: normalized end position (normalized by number of words)
                    #feature_3-9: 7 evenly spaced quantiles of the distribution of relevant class probabilities for this sequence
                    #feature_10-11: The probability that words on either edge of the current sub-sequence belong to the class of interest
                    #feature_12: The probability that the first word corresponds to a 'B'-egin token
                    # print(features)
                    # print(len(features))
                    # exit()
    #df_array=np.array(df_array)
    #df.to_csv(f"{cache}/oof_{args.fold}.csv")
    #df
    #features, labels, groups, wordRanges, truePos
    return SeqDataset(ids, X[:X_ind], y, groups, wordRanges, truePos)


# In[119]:


from joblib import Parallel, delayed
from multiprocessing import Manager

manager = Manager()


def sequenceDataset(disc_type, submit=False):
    if not submit: validSeqSets[disc_type] = seq_dataset(disc_type) if not SUBMISSION else None
    else: submitSeqSets[disc_type] = seq_dataset(disc_type, submit=True)

# try:
#     with open( KAGGLE_CACHE + f"/valid_seqds_fold{args.fold}.p", "rb" ) as validFile:
#         validSeqSets = pickle.load( validFile )
# except:
print('Making validation sequence datasets...')
validSeqSets = manager.dict()
Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(sequenceDataset)(disc_type, False)
       for disc_type in disc_type_to_ids
    )
print('Done.')
with open( cache + f"/valid_seqds_fold{args.fold}.p", "wb+" ) as validFile:
    pickle.dump( dict(validSeqSets), validFile )
