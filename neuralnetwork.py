#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
from numpy import random as nr
import logging
from datetime import datetime 

class NeuralNetwork:
    
    def __init__(self, corpus, huffman):
        self.__corpus = corpus
        self.__huffman = huffman
        self.V = corpus.words_count_unique
        self.m = corpus.words_count_total
        
        self.SSWINDOW = 20    #single side window width
        self.DIMENSION = 300  #target word2vec dimension
        self.step_param = 0.01
        
        self.word_vec_map = {}
        for word in corpus.one_hot_index.keys():
            self.word_vec_map[word] = np.random.uniform(-1,1,(self.DIMENSION, 1)) 
            
        self.context_vec_map = {}
        
        self.thetas = {}
        self.EPOCH = 0
        for word in self.__corpus.one_hot_index.keys():
            code = self.__huffman.get_word_code(word)
            path_length = len(code)
            self.thetas[word] = [np.random.uniform(-1,1,(self.DIMENSION, 1)) for loop_item in range(path_length) ]

                
    def get_context_vector(self, index1, index2):
        neighbor_words_vector = []
        for t in range(index2-self.SSWINDOW, index2+self.SSWINDOW+1, 1):
            if t>=0 and t<>index2 and t<len(self.__corpus.words_sequences[index1]): 
                a_context_word = self.__corpus.words_sequences[index1][t]
                neighbor_words_vector.append(self.word_vec_map[a_context_word])
        X_of_word = np.zeros((self.DIMENSION, 1))
        for neighbor in neighbor_words_vector: 
            X_of_word = np.add(X_of_word , np.array(neighbor))
        return X_of_word
    
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

        __buffer_word_vec_map = {}
        for word in self.__corpus.one_hot_index.keys():
            __buffer_word_vec_map[word] = np.zeros((self.DIMENSION, 1))
        
        likelihood_total = 0.0
        word_counter = 0 
        for p1 in range(len(self.__corpus.words_sequences)): #each sub corpus
            sub_corpus = self.__corpus.words_sequences[p1]
            for p2 in range(len(sub_corpus)):
                word_counter = word_counter + 1
                if word_counter %1000==0 and word_counter >0: print "\t "+str(word_counter)+"/"+str(self.m)+" words have been processed"
                
                word = sub_corpus[p2]
                code = self.__huffman.get_word_code(word)
                path_length = len(code)
                X_of_word = self.get_context_vector(p1, p2)   
                theta_list_for_path = self.thetas[word]
                
                E = np.zeros((self.DIMENSION, 1))
                logging.debug("No.%d/%d word \"%s\"(%s), norm of X_of_word is %f", word_counter, self.m, word, code, np.linalg.norm(X_of_word))
                
                likelihood = 1.0
                for j in range(path_length):
                    d = int(code[j])
                    X_multiply_theta = np.array(np.dot(X_of_word.T, theta_list_for_path[j]))
                    sigma_value = X_multiply_theta[0][0]
                    q = self.sigma_function(sigma_value)
                    g = self.step_param*(1 - q - d)
                    E = np.add(E , g* theta_list_for_path[j])
                    likelihood = likelihood * (q**(1-d))*((1-q)**d)
                    
                    theta_list_for_path[j] = np.add(theta_list_for_path[j] , g* X_of_word)
                    logging.debug("|        X.T*theta(%s)(%d) = %s, sigma = %s",word, j+1, str(sigma_value),str(q))
                
                likelihood_total = likelihood_total + likelihood
                logging.debug("-- \tnorm of E(%s) is %f, likelyhood=%f", word, np.linalg.norm(E), likelihood)
                for t in range(p2-self.SSWINDOW, p2+self.SSWINDOW+1, 1):
                    if t>=0 and t<>p2 and t<len(sub_corpus): 
                        context_word = sub_corpus[t]
                        __buffer_word_vec_map[context_word] = np.add(__buffer_word_vec_map[context_word] , E)

        for word in self.__corpus.one_hot_index.keys():
            self.word_vec_map[word] = np.add(self.word_vec_map[word], __buffer_word_vec_map[word])

        logging.info("finish calculating, now start to validate norm and likelihood")        
        likelihood_average = likelihood_total / self.m
        average_vector = np.zeros((self.DIMENSION, 1))
        for vec in self.word_vec_map.values(): average_vector = np.add(average_vector, vec)
        logging.info("average norm of original vector = %f", np.linalg.norm(average_vector))
        logging.info("average likelihood = %f", likelihood_average)
    
    
        logging.info("generate context word vector list for synonym")
        self.context_vec_map.clear()
        for p1 in range(len(self.__corpus.words_sequences)-1, -1, -1):
            sub_corpus = self.__corpus.words_sequences[p1]
            for p2 in range(len(sub_corpus)-1, -1, -1):
                word = sub_corpus[p2]
                if not self.context_vec_map.has_key(word): self.context_vec_map[word] = self.get_context_vector(p1, p2)   
        logging.info("finish generate context word vector list for synonym")
    
    def get_most_close_word(self, word):
        if word in self.context_vec_map.keys():
            word_vec = self.context_vec_map[word]
            distance_dict = {}
            for a_word in self.__corpus.one_hot_index.keys():
                if a_word<>word:
                    vector2 = self.context_vec_map[a_word]
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
