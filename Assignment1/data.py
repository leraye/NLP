#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:31:07 2019

@author: heqingye
"""

from io import open
import numpy as np
from collections import Counter, defaultdict
from math import sqrt
from scipy.stats import spearmanr
import heapq

class Corpus:
    
    def __init__(self, winsize, words, context_words, sentences, matrix=False):
        self.word_vocab = self.build_vocab(words)
        self.context_vocab = self.build_vocab(context_words)
        self.w = winsize
        self.corpora = defaultdict(Counter)
        self.build_matrix(sentences)
        self.ppmi = {}
        self.compute_ppmi()
        if matrix:
            m, n = len(self.word_vocab), len(self.context_vocab)
            pmimat = np.zeros((m, n))
            self.word_map = {}
            for i, w in enumerate(list(self.word_vocab)):
                self.word_map[w] = i
                for j, c in enumerate(list(self.context_vocab)):
                    if w in self.ppmi and c in self.ppmi[w]:
                        pmimat[i][j] = self.ppmi[w][c]
            self.U, _, _ = np.linalg.svd(pmimat, full_matrices=False)
        
    def build_vocab(self, path):
        ans = set()
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                l = line.strip('\n')
                if l:
                    ans.add(l)
        return ans
    
    def build_matrix(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                if line.strip('\n'):
                    l = ['<s>'] + line.strip('\n').split() + ['</s>']
                    for i in range(1, len(l) - 1):
                        if l[i] in self.word_vocab:
                            for j in range(max(0, i - self.w), min(1 + i + self.w, len(l))):
                                if l[j] in self.context_vocab:
                                    self.corpora[l[i]][l[j]] += 1
                                
    def cosine_similarity(self, w1, w2, table):
        dot_prod = 0.0
        for k in table[w1]:
            if k in table[w2]:
                dot_prod += table[w1][k] * table[w2][k]
        len1, len2 = sqrt(sum(x ** 2 for x in table[w1].values())), sqrt(sum(x ** 2 for x in table[w2].values()))
        return dot_prod / (len1 * len2)
                                
    def eval_ws(self, path, matrix='rc'):
        gold_standard = []
        computed = []
        if matrix == 'rc':
            table = self.corpora
        else:
            table = self.ppmi
        with open(path, 'r', encoding="utf8") as f:
            next(f)
            for line in f:
                l = line.strip('\n')
                if l:
                    w1, w2, score = l.split()
                    gold_standard.append(float(score))
                    if w1 not in table or w2 not in table:
                        computed.append(0.0)
                    else:
                        computed.append(self.cosine_similarity(w1, w2, table))
        rho, pval = spearmanr(np.array(gold_standard), np.array(computed))
        return rho
    
    def compute_ppmi(self):
        s = sum(sum(self.corpora[k].values()) for k in self.corpora)
        context_sum = Counter()
        word_sum = Counter()
        for k in self.corpora:
            for w in self.corpora[k]:
                context_sum[w] += self.corpora[k][w]
            word_sum[k] = sum(self.corpora[k].values())
        for k in self.corpora:
            self.ppmi[k] = {}
            for w in self.corpora[k]:
                self.ppmi[k][w] = max(np.log2(self.corpora[k][w]) + np.log2(s) - np.log2(context_sum[w]) - np.log2(word_sum[k]), 0)
                
    def knn(self, query, k):
        q = []
        for w in self.corpora:
            if w != query:
                score = self.cosine_similarity(w, query, self.ppmi)
                if len(q) >= k:
                    if score > q[0][0]:
                        heapq.heappop(q)
                        heapq.heappush(q, (score, w))
                else:
                    heapq.heappush(q, (score, w))
        return q
    
    def truncated_svd(self, k, path):
        gold_standard = []
        computed = []
        U = self.U[:,:k]
        with open(path, 'r', encoding="utf8") as f:
            next(f)
            for line in f:
                w1, w2, score = line.strip('\n').split()
                gold_standard.append(float(score))
                if w1 not in self.word_vocab or w2 not in self.word_vocab:
                    computed.append(0.0)
                else:
                    i, j = self.word_map[w1], self.word_map[w2]
                    u, v = U[i], U[j]
                    computed.append(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        rho, pval = spearmanr(np.array(gold_standard), np.array(computed))
        return rho
    
    