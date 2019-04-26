#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime 
import logging
import math
import time

from numpy import random as nr

import numpy as np


class NeuralNetwork:
    
    def __init__(self, corpus, huffman):
        self.__corpus = corpus
        self.__huffman = huffman
        self.V = corpus.words_count_unique
        self.m = corpus.words_count_total
        
        self.SSWINDOW = 10    #single side window width
        self.DIMENSION = 200  #target word2vec dimension
        self.step_param = 0.01
        
        self.word_vec_map = {}
        for word in corpus.one_hot_index.keys():
            self.word_vec_map[word] = np.random.uniform(-1,1,(self.DIMENSION, 1)) 
            
        self.thetas = {}
        self.d_row = {}
        self.EPOCH = 0
        for word in self.__corpus.one_hot_index.keys():
            code = self.__huffman.get_word_code(word)
            path_length = len(code)
            self.thetas[word] = np.random.uniform(-1,1,(self.DIMENSION, path_length))
            d_value_row = []
            for p in range(path_length): d_value_row.append(int(code[p]))
            self.d_row[word] = np.array(d_value_row)
                
    def get_context_vector(self, index1, index2):
        neighbor_words_vector = {}
        for t in range(index2-self.SSWINDOW, index2+self.SSWINDOW+1, 1):
            if t>=0 and t<>index2 and t<len(self.__corpus.words_sequences[index1]): 
                a_context_word = self.__corpus.words_sequences[index1][t]
                neighbor_words_vector[a_context_word] = self.word_vec_map[a_context_word]
        return neighbor_words_vector
    
    def sigma_function(self, x): #σ
        return 1.0/(math.e**(-1.0*x)+1)
    
    def regression(self, new_epoch_count):

        for p in range(new_epoch_count):
            logging.info("=================Start Epoch No.%d regression==================", self.EPOCH+p+1)
            logging.info("total word count is %d, unique word count is %d", self.m, self.V)
            logging.info("word vec dimension size is %d, words window size is +-%d", self.DIMENSION, self.SSWINDOW)
            start_time =datetime.now() 
            self.__regression()
            
            end_time = datetime.now()
            logging.info(">>>>>Finish Epoch No.%d regression, time elasped = %d seconds ", self.EPOCH+p+1, (end_time-start_time).seconds)
            logging.info("")
            print "Finish Epoch No."+str(self.EPOCH+p+1)+" regression"
        self.EPOCH = self.EPOCH + new_epoch_count
    
    def __regression(self):

        likelihood_total = 0.0
        likelihood_count = 0
        word_counter = 0 
        s_time = time.time()
        
        for p1 in range(len(self.__corpus.words_sequences)): #each sub corpus
            sub_corpus = self.__corpus.words_sequences[p1]
            if len(sub_corpus)<=2: continue
            for p2 in range(len(sub_corpus)):
                
                word_counter = word_counter + 1
                if word_counter %1000==0 and word_counter >0: print "\t "+str(word_counter)+"/"+str(self.m)+" words have been processed"
                
                word = sub_corpus[p2]
                code = self.__huffman.get_word_code(word)
                path_length = len(code)
                theta = self.thetas[word]
                
                context_word_list = []
                context_vector = np.zeros((self.DIMENSION, 1))
                for t in range(p2-self.SSWINDOW, p2+self.SSWINDOW+1, 1):
                    if t>=0 and t<>p2 and t<len(sub_corpus): 
                        a_context_word = self.__corpus.words_sequences[p1][t]
                        context_vector = np.add(context_vector, self.word_vec_map[a_context_word])
                        context_word_list.append(a_context_word)
                context_vector = context_vector / len(context_word_list)
                
                logging.debug("No.%d/%d word \"%s\"(%s)", word_counter, self.m, word, code)
                q = self.sigma_function(np.dot(context_vector.T, theta))
                d = self.d_row[word]
                g = self.step_param * (np.ones((1, path_length)) - d - q)
                E = np.dot(g, theta.T)
                
                self.thetas[word] = np.add(theta, np.dot(context_vector, g))
                likelihood = (q**(1-d))*((1-q)**d)
                likelihood_total = likelihood_total+ np.sum(np.reshape(likelihood,(likelihood.size,)))
                likelihood_count = likelihood_count + likelihood.shape[0]*likelihood.shape[1]
                
                for t in range(len(context_word_list)):
                    a_context_word = context_word_list[t]
                    self.word_vec_map[a_context_word] = np.add(self.word_vec_map[a_context_word] , E.T)
                
                logging.debug("-- \tnorm of E(%s) is %f, likelyhood=%f", word, np.linalg.norm(E), np.linalg.norm(likelihood))
                
                if word_counter %1000==0 and word_counter>0:
                    e_time = time.time()
                    logging.debug("-- time elapse for %d vector processing is %f", word_counter ,e_time-s_time)

        logging.info("finish calculating, now start to validate norm and likelihood")        
        likelihood_average = likelihood_total / likelihood_count
        average_vector = np.zeros((self.DIMENSION, 1))
        for vec in self.word_vec_map.values(): average_vector = np.add(average_vector, vec)
        logging.info("average norm of original vector = %f", np.linalg.norm(average_vector))
        logging.info("average likelihood = %f", likelihood_average)

    
    def get_most_close_word2(self, word):
        if word in self.word_vec_map.keys():
            word_vec = self.word_vec_map[word]
            distance_dict = {}
            for a_word in self.__corpus.one_hot_index.keys():
                if a_word<>word:
                    vector2 = self.word_vec_map[a_word]
                    divided = np.linalg.norm(word_vec)*(np.linalg.norm(vector2))
                    if divided==0 : continue
                    else:
                        dist=np.dot(word_vec.T,vector2)[0][0]/(divided)
                        distance_dict[a_word] = dist
            ordered = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)
            if len(ordered)>20: ordered = ordered[0:20]
            return ordered   
        else:
            return None
            
