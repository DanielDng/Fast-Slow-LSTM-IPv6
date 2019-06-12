"""
description: this file helps to load raw file and gennerate batch x,y
author:luchi
date:22/11/2016
"""
import numpy as np
import pickle as pkl
import tkinter as tk

#file path
dataset_path='data/subj0.pkl'

#train_filename='data/training_mini.csv'
#test_filename='data/test_mini.csv'
#train_filename='data/training.csv'
################################################
################################################
################################################
#设置选择的文件(train/test)路径
train_filename='data/UNSW_NB15_training-set_n3.csv'

# test_filename='data/UNSW_NB15_testing-set_n3.csv'
#设置test文件的路径
test_filename='#'

#train_filename='data/UNSW_NB15_training-set_ae25.csv'
#test_filename='data/UNSW_NB15_testing-set_ae25.csv'
#train_filename='data/UNSW_NB15_training-set_p.csv'
#test_filename='data/UNSW_NB15_testing-set_p.csv'
valid_portion=0.2


def set_dataset_path(path):
    dataset_path=path


##############################################################################################
def load_data():
    fr = open(train_filename)
    print ('load training data from %s',train_filename)
    lines = fr.readlines()
    line_nums = len(lines)
    #line_nums = 25000
    #para_num = 25 #ae30
    para_num = 42 #UNSW-NB15
    train_set = np.zeros((line_nums, para_num+1))  #Create a matrix of line_nums rows and para_num columns
    for i in range(line_nums):
        line = lines[i].strip()
        train_set[i, :] = line.split(',')
    fr.close()
    #return train_set, class_label

    ##############################################################################
    ##############################################################################
    #修改if-else
    if test_filename=='#':
        tk.messagebox.showwarning(title='Warning',message='You have not select file to detect!\n'+
            'Please select the file to next step!')
    else:
        fr = open(test_filename)
        print ('load testing data from %s',test_filename)
        lines = fr.readlines()
        line_nums = len(lines)
        #line_nums = 20000
        #para_num = 25 #ae30
        para_num = 42 #UNSW-NB15
        test_set = np.zeros((line_nums, para_num+1))  # Create a matrix of line_nums rows and para_num columns
        for i in range(line_nums):
            #line = lines[i].strip()
            line = lines[i].strip()
            test_set[i, :] = line.split(',')
        fr.close()

    #train_set length
    n_samples= len(train_set)
    #shuffle and generate train and valid dataset
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set = [train_set[s] for s in sidx[n_train:]]
    train_set = [train_set[s] for s in sidx[:n_train]]

    return train_set, valid_set, test_set

def load_data2():
    fr = open(train_filename)
    print ('load training data from %s',train_filename)
    lines = fr.readlines()
    line_nums = len(lines)
    #line_nums = 25000
    para_num = 42
    train_set = np.zeros((line_nums, para_num+1))  #Create a matrix of line_nums rows and para_num columns
    for i in range(line_nums):
        line = lines[i].strip()
        train_set[i, :] = line.split(',')
    fr.close()
    #return train_set, class_label

    #train_set length
    n_samples= len(train_set)
    #shuffle and generate train and valid dataset
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set = [train_set[s] for s in sidx[n_train:]]
    train_set = [train_set[s] for s in sidx[:n_train]]
    return train_set,valid_set    
