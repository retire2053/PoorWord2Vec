#!/usr/bin/python
# -*- coding: utf-8 -*-

from corpus import Corpus
import logging

class LeafNode:
    
    def __init__(self, item, freq):
        self.word = item
        self.freq = freq
        self.code = 0
    
    def getFreq(self):
        return self.freq
    
    def travel_and_set_value(self, code, valueMap):
        self.code = code
        if not valueMap is None: valueMap[self.word] = code
        
    def debug(self):
        logging.debug("[Leaf].[%s][freq=%d]=%s", self.word, self.freq, str(self.code))


class CombineNode:
    
    def __init__(self, leftNode, rightNode):
        self.leftNode = leftNode
        self.rightNode = rightNode
        
        self.freq = leftNode.getFreq() + rightNode.getFreq()
        
    def getFreq(self):
        return self.freq
    
    def travel_and_set_value(self, code, valueMap):
        self.leftNode.travel_and_set_value(code+"1", valueMap)
        self.rightNode.travel_and_set_value(code+"0", valueMap)
        
    def debug(self):
        self.leftNode.debug()
        self.rightNode.debug()

class Huffman:
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.root = None
        self.valueMap = None

    def build(self):
        logging.debug("start to build huffman tree.")
        self.valueMap = {}
        node_list =[ LeafNode(w[0],w[1]) for w in sorted(self.corpus.words_freq.items(), key=lambda x: x[1], reverse=True) ]
        pcount = 0
        while len(node_list)>2:
            cn = CombineNode(node_list[-2], node_list[-1])
            del node_list[-2]   # remove the last 2 in this order
            del node_list[-1]

            position_to_insert = 0
            for p in range(len(node_list)-2):
                if node_list[p].getFreq() > cn.getFreq(): position_to_insert = p+1
                else: break
            node_list.insert(position_to_insert, cn)
            
            pcount = pcount + 1
            if pcount%1000==0: logging.debug("%d words have been processed in huffman", pcount+1)
            
        if len(node_list)==2:
            self.root = CombineNode(node_list[0], node_list[1])
            self.root.travel_and_set_value("", self.valueMap)
            #self.root.debug()
            
        logging.debug("finish building huffman tree.")
    
    def get_word_code(self, word):
        if not self.valueMap is None and self.valueMap.has_key(word):
            code = self.valueMap[word]
            return code
        else:
            return None
        