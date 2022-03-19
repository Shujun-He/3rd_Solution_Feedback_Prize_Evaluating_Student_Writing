from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

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


class CustomCollate:
    def __init__(self,tokenizer,sliding_window=None):
        self.tokenizer=tokenizer
        self.sliding_window=sliding_window

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
        if self.sliding_window is not None and max_len > self.sliding_window:
            max_len= int((np.floor(max_len/self.sliding_window-1e-6)+1)*self.sliding_window)
        #always pad the right side
        input_ids, attention_mask, labels, BIO_labels, discourse_labels=[],[],[],[],[]
        #if np.random.uniform()>0.5:
        #print(data[0].keys())
        if 'wids' in data[0]:
            get_wids=True
        else:
            get_wids=False
        #print(get_wids)
        wids = []
            #wids.append(torch.nn.functional.pad(data[i]['wids'],(0,max_len-lengths[i]),value=-1))
        for i in range(bs):
            input_ids.append(torch.nn.functional.pad(data[i]['input_ids'],(0,max_len-lengths[i]),value=self.tokenizer.pad_token_id))
            attention_mask.append(torch.nn.functional.pad(data[i]['attention_mask'],(0,max_len-lengths[i]),value=0))
            labels.append(torch.nn.functional.pad(data[i]['labels'],(0,max_len-lengths[i]),value=-100))
            #print(labels[-1].shape)
            BIO_labels.append(torch.nn.functional.pad(data[i]['BIO_labels'],(0,max_len-lengths[i]),value=-100))
            discourse_labels.append(torch.nn.functional.pad(data[i]['discourse_labels'],(0,max_len-lengths[i]),value=-100))
            if get_wids:
                wids.append(torch.nn.functional.pad(data[i]['wids'],(0,max_len-lengths[i]),value=-1))
            #print(labels[-1].shape)
        # else:
        #     for i in range(bs):
        #         input_ids.append(torch.nn.functional.pad(data[i]['input_ids'],(max_len-lengths[i],0),value=1))
        #         attention_mask.append(torch.nn.functional.pad(data[i]['attention_mask'],(max_len-lengths[i],0),value=0))
        #         labels.append(torch.nn.functional.pad(data[i]['labels'],(max_len-lengths[i],0),value=-100))

        input_ids=torch.stack(input_ids)
        attention_mask=torch.stack(attention_mask)
        labels=torch.stack(labels)
        BIO_labels=torch.stack(BIO_labels)
        discourse_labels=torch.stack(discourse_labels)
        if get_wids:
            wids=torch.stack(wids)
        #exit()
        if get_wids:
            return {"input_ids":input_ids,"attention_mask":attention_mask,
            "labels":labels,"BIO_labels":BIO_labels,"discourse_labels":discourse_labels,
            "wids":wids}
        else:
            return {"input_ids":input_ids,"attention_mask":attention_mask,
            "labels":labels,"BIO_labels":BIO_labels,"discourse_labels":discourse_labels}



def custom_collate_train(data):
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
    input_ids, attention_mask, labels=[],[],[]
    #if np.random.uniform()>0.5:
    for i in range(bs):
        input_ids.append(torch.nn.functional.pad(data[i]['input_ids'],(0,max_len-lengths[i]),value=tokenizer.pad_token_id))
        attention_mask.append(torch.nn.functional.pad(data[i]['attention_mask'],(0,max_len-lengths[i]),value=0))
        labels.append(torch.nn.functional.pad(data[i]['labels'],(0,max_len-lengths[i]),value=-100))
    # else:
    #     for i in range(bs):
    #         input_ids.append(torch.nn.functional.pad(data[i]['input_ids'],(max_len-lengths[i],0),value=1))
    #         attention_mask.append(torch.nn.functional.pad(data[i]['attention_mask'],(max_len-lengths[i],0),value=0))
    #         labels.append(torch.nn.functional.pad(data[i]['labels'],(max_len-lengths[i],0),value=-100))

    input_ids=torch.stack(input_ids)
    attention_mask=torch.stack(attention_mask)
    labels=torch.stack(labels)
    #exit()

    return {"input_ids":input_ids,"attention_mask":attention_mask,"labels":labels}


def iter_split(data,labels,fold,nfolds=5,seed=2020):
    splits = StratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=True)
    splits = list(splits.split(data,labels))
    # splits = np.zeros(len(data)).astype(np.int)
    # for i in range(nfolds): splits[splits[i][1]] = i
    # indices=np.arange(len(data))
    train_indices=splits[fold][0]
    val_indices=splits[fold][1]
    return train_indices, val_indices


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
    gt_df = gt_df[['id','discourse_type','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id','class','predictionstring']] \
        .reset_index(drop=True).copy()
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
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id','predictionstring_gt']).first()['pred_id'].values

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

def threshold(df):
    map_clip = {'Lead':9, 'Position':5, 'Evidence':14, 'Claim':3, 'Concluding Statement':11,
                 'Counterclaim':6, 'Rebuttal':4}
    df = df.copy()
    for key, value in map_clip.items():
    # if df.loc[df['class']==key,'len'] < value
        index = df.loc[df['class']==key].query(f'len<{value}').index
        df.drop(index, inplace = True)
    return df

from tqdm import tqdm
def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])

def link_evidence(oof):
  if not len(oof):
    return oof

  def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])

  thresh = 1
  idu = oof['id'].unique()
  eoof = oof[oof['class'] == "Evidence"]
  neoof = oof[oof['class'] != "Evidence"]
  eoof.index = eoof[['id', 'class']]
  for thresh2 in range(26, 27, 1):
    retval = []
    for idv in tqdm(idu, desc='link_evidence', leave=False):
      for c in ['Evidence']:
        q = eoof[(eoof['id'] == idv)]
        if len(q) == 0:
          continue
        pst = []
        for r in q.itertuples():
          pst = [*pst, -1,  *[int(x) for x in r.predictionstring.split()]]
        start = 1
        end = 1
        for i in range(2, len(pst)):
          cur = pst[i]
          end = i
          if  ((cur == -1) and ((pst[i + 1] > pst[end - 1] + thresh) or (pst[i + 1] - pst[start] > thresh2))):
            retval.append((idv, c, jn(pst, start, end)))
            start = i + 1
        v = (idv, c, jn(pst, start, end + 1))
        retval.append(v)
    roof = pd.DataFrame(retval, columns=['id', 'class', 'predictionstring'])
    roof = roof.merge(neoof, how='outer')
    return roof
