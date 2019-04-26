#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import os
import re

import jieba


class Corpus:
    
    def __init__(self, corpus_dir, stop_file_path, user_dict_file):
        self.corpus_dir = corpus_dir
        
        self.words_sequences = []
        self.user_dict = user_dict_file
                
        self.words_count_unique = 0
        self.words_count_total = 0
        self.words_freq = {}
        self.one_hot_index = {}
        
        if not stop_file_path is None:
            self.stopword = Stopwords(stop_file_path)
        
        rawtexts = self.__read_content_from_dir(self.corpus_dir) 
        self.rawtexts = [w for w in rawtexts if len(w)>3 ] 
        
        
    def __read_content_from_dir(self, basepath):
        logging.debug("start to read documents from %s", basepath)
        filelist = os.listdir(basepath)
        rawtexts = []
        for count in range(len(filelist)):
            file_full_path = basepath+"/"+filelist[count]
            target = open(file_full_path,"rb")
            text = target.read()
            target.close()
            rawtexts.append(text)
            logging.info("read content from %s", file_full_path)
            if count%1000==0 and count>0:
                print "\tread "+str(count)+" paragraphs from "+basepath
        return rawtexts
    
    def parse(self):
        logging.debug("start to parse text")

        if not self.user_dict is None: jieba.load_userdict(self.user_dict)
        numberpattern = self.__get_number_pattern()
        for text in self.rawtexts:
            words_sequence = []
            segs = jieba.cut(text)
            for seg in segs:
                w = seg.encode("UTF-8")
                if (self.stopword is None or not self.stopword.isStopword(w)) and not self.__is_number(w, numberpattern) :
                    words_sequence.append(w)
                    self.words_count_total = self.words_count_total + 1
                    if not w in self.words_freq: 
                        self.words_freq[w] = 1
                        self.words_count_unique = self.words_count_unique + 1
                    else:
                        self.words_freq[w] = self.words_freq[w] +1
            self.words_sequences.append(words_sequence)
            
        logging.info("total distinct words count (V) is %d", self.words_count_unique)
        logging.info("total words sequence count (m) is %d", self.words_count_total)
        
        counter_list = sorted(self.words_freq.items(), key=lambda x: x[1], reverse=True)
        for p in range(len(counter_list)):
            word = counter_list[p][0]
            self.one_hot_index[word] = p
            if p<20: logging.info("Word frequency No.%d is \"%s\", freq = %d", p+1, word, counter_list[p][1])        
        
        logging.info("parse text finished.")
        
    def __get_number_pattern(self):
        pattern = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        return pattern

    def __is_number(self, word, numberpattern):
        if word.isdigit() or numberpattern.match(word):
            return True
        else:
            return False
       
class Stopwords:
    
    def __init__(self, filepath):
        stopfile = open(filepath, "rb")
        lines = stopfile.read().split()
        stopfile.close()
    
        self.stop_words_list = set()
        for line in lines:
            self.stop_words_list.add(line)
            self.stop_words_list.add(" ")
        for item in self.__get_non_printable_charset():
            self.stop_words_list.add(item)
    
    def __get_non_printable_charset(self):
        nonprint = set()
        nonprint.add("\x00")
        nonprint.add("\x01")
        nonprint.add("\x02")
        nonprint.add("\x03")
        nonprint.add("\x04")
        nonprint.add("\x05")
        nonprint.add("\x06")
        nonprint.add("\x07")
        nonprint.add("\x08")
        nonprint.add("\x09")
        nonprint.add("\x0A")
        nonprint.add("\x0B")
        nonprint.add("\x0C")
        nonprint.add("\x0D")
        nonprint.add("\x0D\x0A")
        return nonprint
    
    def isStopword(self, word):
        return word in self.stop_words_list



