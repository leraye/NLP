#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 22:44:13 2019

@author: heqingye
"""

import os
from io import open
import torch
import numpy as np

class Dictionary(object):
    def __init__(self):
        self.word2idx = {} #consisting of (word, id) pairs
        self.idx2word = [] #list containing all the words encountered
        self.tag2idx = {}
        self.idx2tag = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def add_tag(self, tag):
        if tag not in self.tag2idx:
            self.idx2tag.append(tag)
            self.tag2idx[tag] = len(self.idx2tag) - 1

    def vocab_len(self):
        return len(self.idx2word)
    
    def tag_len(self):
        return len(self.idx2tag)

class Corpus(object):
    def __init__(self, path, w=1):
        self.dictionary = Dictionary()
        self.w = w
        self.train = self.tokenize(os.path.join(path, 'tweets-train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'tweets-dev.txt'))
        self.test = self.tokenize(os.path.join(path, 'tweets-devtest.txt'))
 

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                line = line.strip('\n')
                if line:
                    wrd, lab = line.split('\t')
                    self.dictionary.add_word(wrd)
                    self.dictionary.add_tag(lab)

        data = []
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = []
            labs = []
            for line in f:
                line = line.strip('\n')
                if line:
                    wrd, lab = line.split('\t')
                    ids.append(self.dictionary.word2idx[wrd])
                    labs.append(self.dictionary.tag2idx[lab])
                else:
                    
                    data.append((torch.tensor(ids).type(torch.int64), torch.tensor(labs).type(torch.int64)))
                    ids = []
                    labs = []
        return data
    
    def load_pretrained(self,path,unk='UUUNKKK'):
        assert os.path.exists(path)
        emd_dict = {}
        with open(path, 'r') as f:
            for line in f:
                symbols = line.split()
                word = symbols[0]
                emd_dict[word] = np.array(symbols[1:], dtype='float32')
        emsize = len(emd_dict[unk])
        emdmat = np.random.uniform(-0.1, 0.1, (self.dictionary.dict_len(), emsize))
        for w, wid in self.dictionary.word2idx.items():
            if w in emd_dict:
                emdmat[wid] = emd_dict[w]
            else:
                emdmat[wid] = emd_dict[unk]
        return torch.from_numpy(emdmat).float()