#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 22:48:42 2019

@author: heqingye
"""

import torch.nn as nn

class RNNTagger(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, nemd, nhid, nlayers, ntag, bi=False,pretrained=None):
        super(RNNTagger, self).__init__()
        
        if pretrained is None:
            self.emd = nemd
            self.encoder = nn.Embedding(ntoken, self.emd)
            self.init_emd_weights()
        else:
            self.emd = pretrained.size(1)
            self.encoder = nn.Embedding(ntoken, self.emd)
            self.encoder.weight.data.copy_(pretrained)
            self.encoder.weight.requires_grad = False
        
        self.rnn = nn.LSTM(self.emd, nhid, nlayers,bidirectional=bi)
        if bi:
            self.ndir = 2
        else:
            self.ndir = 1
        self.decoder = nn.Linear(nhid * self.ndir, ntag)

        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_emd_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb.unsqueeze(1), hidden)
        decoded = self.decoder(output.contiguous().view(output.size(0)*output.size(1), output.size(2)))
        return decoded, hidden

    def init_hidden(self):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers * self.ndir, 1, self.nhid),
                weight.new_zeros(self.nlayers * self.ndir, 1, self.nhid))