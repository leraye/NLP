#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 18:18:05 2019

@author: heqingye
"""

from data import Corpus
import argparse

parser = argparse.ArgumentParser(description='TTIC 31190 Assignment 1')

parser.add_argument('--word_vocab', type=str, default="./files/vocab-wordsim.txt",
                    help='vocabulary for the target words')
parser.add_argument('--context_vocab', type=str, default="./files/vocab-25k.txt",
                    help='vocabulary for the context words')
parser.add_argument('--corpus', type=str, default="./wiki-1percent.txt",
                    help='corpus')
parser.add_argument('--wsize', type=int, default=3,
                    help='size of the context window')
parser.add_argument('--query', type=str,
                    help='query word')
parser.add_argument('--knn', type=int, default=10,
                    help='number of nearest neighbors')
parser.add_argument('--svdk', type=int, default=10,
                    help='truncated SVD')
parser.add_argument('--matrix', action='store_true',
                    help='If true, then compute the full word-context matrix. For computing truncated SVD only')
parser.add_argument('--similarity', type=str, default="",
                    help='file containing the similarities of word pairs')
args = parser.parse_args()

data = Corpus(args.wsize, args.word_vocab, args.context_vocab, args.corpus, args.matrix)
#f1 = "./files/men.txt"
#f2 = "./files/simlex-999.txt"
print(data.eval_ws(args.similarity, 'ppmi'))
#print(data.eval_ws(f2, 'ppmi'))
#print(data.copora["monster"])
#print(data.knn(args.query, args.knn))
print(data.truncated_svd(args.svdk, args.similarity))