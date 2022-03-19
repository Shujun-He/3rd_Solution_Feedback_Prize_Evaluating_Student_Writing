import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Functions import *
from scipy import stats

BIO_labels=['B','I','O']
discourse_labels=["Lead","Position","Claim","Counterclaim","Rebuttal","Evidence","Concluding Statement","O"]

class FeedbackDataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len, get_wids, labels_to_ids, ids_to_labels):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids # for validation
        self.labels_to_ids = labels_to_ids
        self.ids_to_labels=ids_to_labels
        self.mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

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
                if offsets[token_idx][0]!=offsets[token_idx][1]:
                    #Choose the split word that shares the most characters with the token if any
                    split_idxs = offset_to_wordidx[offsets[token_idx][0]:offsets[token_idx][1]]
                    split_index = stats.mode(split_idxs[split_idxs != -1]).mode[0] if len(np.unique(split_idxs)) > 1 else split_idxs[0]

                    if split_index != -1:
                        if not self.get_wids: label_ids.append( self.labels_to_ids[word_labels[split_index]] )
                        split_word_ids[token_idx] = split_index
                    else:
                        # Even if we don't find a word, continue labeling 'I' tokens until a 'B' token is found
                        if label_ids and label_ids[-1] != -100 and self.ids_to_labels[label_ids[-1]][0] == 'I':
                            split_word_ids[token_idx] = split_word_ids[token_idx + 1]
                            if not self.get_wids: label_ids.append(label_ids[-1])
                        else:
                            if not self.get_wids: label_ids.append(-100)
                else:
                    if not self.get_wids: label_ids.append(-100)

        encoding['labels'] = list(reversed(label_ids))

        encoding['discourse_labels'] = [i//2 if i!=-100 else -100 for i in encoding['labels']]
        encoding['BIO_labels'] = []#[i%2 if i!=-100 else -100 for i in encoding['labels']]
        for i in encoding['labels']:
            if i!=-100 and i!=14:
                encoding['BIO_labels'].append(i%2)
            elif i==14:
                encoding['BIO_labels'].append(2)
            elif i==-100:
                encoding['BIO_labels'].append(-100)
        # print(encoding['BIO_labels'])
        # exit()
        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids:
            item['wids'] = torch.as_tensor(split_word_ids)

        if not self.get_wids:
            ix = torch.rand(size=(len(item['input_ids']),)) < 0.15
            #ids = torch.tensor(input_ids, dtype=torch.long)
            # print(item['input_ids'].shape)
            # print(ix.shape)
            item['input_ids'][ix] = self.mask_token

        return item

  def __len__(self):
        return self.len
