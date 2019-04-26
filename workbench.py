#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import logging
import time

from brain import Brain
from corpus import Corpus
from huffman import Huffman
from neuralnetwork import NeuralNetwork
import numpy as np


def execution():
    
    LOG_FORMAT = "%(asctime)s - %(module)s - %(levelname)s - %(lineno)d - %(message)s"
    logging.basicConfig(filename='PoorW2V.log', level=logging.INFO, format=LOG_FORMAT)
    print "Please find PoorW2V.log to get log information"
    
    corpus_dir = "/Users/retire2053/Desktop/lkx_14645"
    stopword_file = "/Users/retire2053/Desktop/lkx_stopwords.txt"
    user_dict_file = "/Users/retire2053/Desktop/medical-general-vocab.txt"
    
    corpus = Corpus(corpus_dir, stopword_file, user_dict_file)
    corpus.parse()

    tree = Huffman(corpus)
    tree.build()
    
    neuralnetwork = NeuralNetwork(corpus, tree)

    brain = Brain(corpus, tree, neuralnetwork)
    brain.work()
    print "Program has been executed successfully"


execution()


