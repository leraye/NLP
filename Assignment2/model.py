#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 23:17:06 2019

@author: heqingye
"""
import os
from io import open
from collections import Counter
from random import shuffle
import numpy as np
import operator

class Corpus:
    def __init__(self, path):
        self.train = []
        ptr = os.path.join(path, 'sst3.train')
        with open(ptr, 'r', encoding="utf8") as f:
            for line in f:
                l = line.strip('\n')
                lable = int(l[-1])
                l = l[:-1]
                self.train.append((l, lable))
        pdev = ptr.replace('.train', '.dev')
        self.dev = []
        with open(pdev, 'r', encoding="utf8") as f:
            for line in f:
                l = line.strip('\n')
                lable = int(l[-1])
                l = l[:-1]
                self.dev.append((l, lable))
        pte = ptr.replace('.train', '.devtest')
        self.test = []
        with open(pte, 'r', encoding="utf8") as f:
            for line in f:
                l = line.strip('\n')
                lable = int(l[-1])
                l = l[:-1]
                self.test.append((l, lable))

class Unigram:
    def __init__(self, corpora, cutoff=1, seed=42):
        self.dict = Counter()
        self.weights = {}
        rng = np.random.default_rng()
        for text, lable in corpora:
            for w in text.split():
                self.dict[(w, lable)] += 1
                if (w, lable) not in self.weights:
#                    self.weights[(w, lable)] = 0.0
                    self.weights[(w, lable)] = rng.uniform(-0.1,0.1,1)[0]
        if cutoff > 1:
            to_delete = []
            for k in self.dict:
                if self.dict[k] < cutoff:
                    to_delete.append(k)
            for k in to_delete:
                del self.dict[k]
                del self.weights[k]
        print(len(self.dict))
        
    def evaluate(self, corpora):
        total, correct = 0, 0
        for text, lable in corpora:
            scores = [0.0] * 3
            for w in text.split():
                if (w, lable) in self.weights:
                    scores[lable] += self.weights[(w, lable)]
                for l in range(3):
                    if l != lable and (w, l) in self.weights:
                        scores[l] += self.weights[(w, l)]
            yhat = np.argmax(scores)
            correct += int(lable == yhat)
            total += 1
        return 100 * correct / total
        
    def train(self, train_corpora, eval_corpora, test_corpora, niter=20, bsize=1, lr=0.01, hinge=False):
        best_eval, test = float('-inf'), None
        for k in range(niter):
            shuffle(train_corpora)
            for i, p in enumerate(train_corpora):
                text, lable = p
                scores = [0.0] * 3
                for w in text.split():
                    if (w, lable) in self.weights:
                        scores[lable] += self.weights[(w, lable)]
                    for l in range(3):
                        if l != lable and (w, l) in self.weights:
                            if hinge:
                                scores[l] += 1
                            scores[l] += self.weights[(w, l)]
                yhat = np.argmax(scores)
                if yhat != lable:
                    for w in text.split():
                        if (w, lable) in self.weights:
                            self.weights[(w, lable)] += lr
                        if (w, yhat) in self.weights:
                            self.weights[(w, yhat)] -= lr
                if i % 20000 == 0:
                    eval_acc = self.evaluate(eval_corpora)
                    test_acc = self.evaluate(test_corpora)
#                    print("Epoch {} Round {}: Evaluation Accuracy: {}".format(k + 1, i, eval_acc))
                    if eval_acc > best_eval:
                        best_eval = eval_acc
                        test = test_acc
#                    print("Epoch {} Round {}: Test Accuracy: {}".format(k + 1, i, test_acc))
            eval_acc = self.evaluate(eval_corpora)
            test_acc = self.evaluate(test_corpora)
            print("Epoch {}: Evaluation Accuracy: {}".format(k + 1, eval_acc))
            if eval_acc > best_eval:
                best_eval = eval_acc
                test = test_acc
            print("Epoch {}: Test Accuracy: {}".format(k + 1, test_acc))
        print("Best Evaluation Accuracy: {}".format(best_eval))
        print("Test Accuracy: {}".format(test))
        
    def weight_inspection(self, topk=10):
        l0, l1, l2 = [], [], []
        lst = sorted(self.weights.items(), key=operator.itemgetter(1), reverse=True)
        i = 0
        while len(l0) < topk or len(l1) < topk or len(l2) < topk:
            if lst[i][0][1] == 0 and len(l0) < topk:
                l0.append(lst[i][0][0])
            if lst[i][0][1] == 1 and len(l1) < topk:
                l1.append(lst[i][0][0])
            if lst[i][0][1] == 2 and len(l2) < topk:
                l2.append(lst[i][0][0])
            i += 1
        print("Top {} words with the highest weights for lable 0:".format(topk))
        print(l0)
        print("Top {} words with the highest weights for lable 1:".format(topk))
        print(l1)
        print("Top {} words with the highest weights for lable 2:".format(topk))
        print(l2)
                
data = Corpus('./sst3')
unigram = Unigram(data.train)
unigram.train(data.train, data.dev, data.test, hinge=True)
unigram.weight_inspection()