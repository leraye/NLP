#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 22:51:10 2019

@author: heqingye
"""

from io import open
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from random import shuffle

import data
import model

parser = argparse.ArgumentParser(description='LSTM Tagger: Main Function')
parser.add_argument('--data', type=str, default='./tweet-pos',
                    help='location of the data corpus')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of hidden layers')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained embeddings')
parser.add_argument('--bidirectional', action='store_true',
                    help='use bidirectional LSTM')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

torch.manual_seed(args.seed)

corpus = data.Corpus(args.data)
if args.pretrained:
    emd_path = 'embeddings-twitter.txt'
    pre_tr = corpus.load_pretrained(emd_path)
else:
    pre_tr = None

ntokens = corpus.dictionary.vocab_len()
ntags = corpus.dictionary.tag_len()

tagger = model.RNNTagger(ntokens, args.emsize, args.nhid, args.nlayers, ntags, args.bidirectional, pre_tr)
criterion = nn.CrossEntropyLoss()
if args.pretrained:
    parameters = filter(lambda p: p.requires_grad, tagger.parameters())
else:
    parameters = tagger.parameters()
optimizer = optim.Adagrad(parameters, lr=0.1, lr_decay=1e-5, weight_decay=1e-5)

def evaluate(src):
    # Turn on evaluation mode which disables dropout.
    tagger.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for seq, lable in src:
            hidden = tagger.init_hidden()
            output, _ = tagger(seq, hidden)
            output_flat = output.view(-1, ntags)
            yhat = output_flat.argmax(1)
            total += len(yhat)
            correct += torch.sum(torch.eq(lable, yhat)).item()
    return correct / total


def train():
    tagger.train()
    total_loss = 0.
    shuffle(corpus.train)
    for seq, lable in corpus.train:
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
#        model.zero_grad()
        hidden = tagger.init_hidden()
        optimizer.zero_grad()
        output, _ = tagger(seq, hidden)
        loss = criterion(output.view(-1, ntags), lable)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
    return total_loss

best_val_acc = None
epochs = args.epochs

try:
    for epoch in range(1, epochs+1):
        tr_loss = train()
        val_acc = evaluate(corpus.valid)
        print('-' * 80)
        print('| end of epoch {:3d} | valid accuracy {:5.3f} | '
                'training loss {:8.3f}'.format(epoch, val_acc, tr_loss))
        print('-' * 80)
        # Save the model if the validation accuracy is the best we've seen so far.
        if not best_val_acc or val_acc > best_val_acc:
            with open(args.save, 'wb') as f:
                torch.save(tagger, f)
            best_val_acc = val_acc
except KeyboardInterrupt:
    print('-' * 80)
    print('Exiting from training early')

with open(args.save, 'rb') as f:
    best_model = torch.load(f)

test_acc = evaluate(corpus.test)
print('=' * 80)
print('| End of training | test accuracy {:5.3f}'.format(test_acc))
print('=' * 80)