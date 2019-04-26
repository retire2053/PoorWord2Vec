#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import math

from numpy import random as nr

from corpus import Corpus
from huffman import Huffman
from neuralnetwork import NeuralNetwork
import numpy as np


class Brain:
    
    def __init__(self, corpus, huffman, neuralnetwork):
        self.__corpus = corpus
        self.__huffman = huffman
        self.__network = neuralnetwork

    
    def __print_synonym(self, word):
        print "=====Synonym Test====="
        print "| word = "+word
        word_list = self.__network.get_most_close_word2(word)    
        if not word_list is None and len(word_list)>0:
            for a_word in word_list:
                print "| \t["+a_word[0]+"] = "+str(a_word[1])
        else:
            print "word not found"
        print "=====Synonym Test Finish====="

    def work(self):
        while True:
            print ""
            print "type (run) to run another epoch"
            print "type (synonym word) to find synonym"
            print "type (no) to exit"
            text_input = raw_input("START>")
            if text_input =="no" : break
            elif text_input.startswith("run" ):
                elements = text_input.split(" ") 
                if len(elements)>=2 and not elements[1] is None and elements[1].isdigit():
                    loops_count = int(elements[1])
                    self.__network.regression(loops_count)
                else:
                    self.__network.regression(1)
                continue
            elif text_input == "inspect" :
                self.inspect()
                continue
            elif text_input.startswith("synonym"):
                elements = text_input.split(" ")
                if len(elements)>1 and len(elements[1].strip())>0 :
                    word = elements[1]
                    self.__print_synonym(word)
                else:
                    print "illegal input, SHOULD attached with a word"
                    continue
            else:
                continue
        
