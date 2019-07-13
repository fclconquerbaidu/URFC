__author__ = 'victor'

import numpy as np
import glob
import os
import math
from scipy.stats import norm 
import matplotlib.pyplot as plt 
import cv2
import csv
import params
# from keras.utils.np_utils import to_categorical
import time
import datetime
import pandas as pd


def window_stat(np_path, window,out_folder):
    if os.path.exists(out_folder)==0:
        os.mkdir(out_folder)
    print('succ1')
    dirs = sorted(os.listdir(np_path))
    data = []
    count=0
    num_file=len(dirs)
    for file in dirs:
        visit = np.load(os.path.join(np_path, file))
        visit = visit.transpose(1,0,2) # from 7x26x24 to 26x7x24
        visit = visit.flatten() # only 1-D array
        visit = visit.reshape(-1, window)
        visit_mean = np.mean(visit, axis =0)
        #visit_median = np.median(visit, axis =1)
        #visit_sum = np.sum(visit, axis =1)
        visit_std = np.std(visit, axis =0)
        visit=(visit-visit_mean)/visit_std
        np.save(os.path.join(out_folder,file), visit)
        #visit_ptp = np.ptp(visit, axis =1) #max-min for every time window
       # visit_ptp_day = np.ptp(visit) #max - min for each day
    
        #visit_all = np.concatenate((visit_mean,visit_median,visit_sum,visit_std,visit_ptp), axis = 0).tolist()
        #visit_all.append(visit_ptp_day)
        #areaID = file.split("_")[0].split(".")[0]
        #classID = int(file.split("_")[-1].split(".")[0])
        #data.append([areaID, visit_all, classID])
        count=count+1
        print('processing {}/{}:'.format(count,num_file))
    print("loading complete 1:") 

    #df_stat_window = pd.DataFrame(data)
    #df_stat_window.columns = ['areaID', 'visit_stat', 'classID']
    #df_stat.columns = ['areaID', 'total_visitor', 'unique_visit_ratio', 'avg_duration', 'classID']
    #print("loading complete 2:")
    #save_path = '/media/peijun/PJ-ubuntu-ntfs/semi-final/save_df_semifinal/df_stat_window'+str(window)+'.pkl'
    #df_stat_window.to_pickle(save_path)
    #return df_stat

if __name__ == '__main__':
   # train_npy_path = "../semi-final-data/npy/train_visit/000"
    train_npy_path=r'C:\URFC_data\semi_final\train\visit_npy'
    out_folder=r'C:\URFC_data\semi_final\train\visit_npy_normalzed4368'
    # test_npy_path = "../semi-final-data/npy/test_visit/"
    # if not os.path.exists(test_npy_path):
    #     os.makedirs(test_npy_path)
    # if not os.path.exists(train_npy_path):
    #     os.makedirs(train_npy_path)
    window = 1
    window_stat(train_npy_path, window,out_folder)

    # for i in range(0,10):
    #     train_visit_path = "../semi-final-data/final_train_visit_"+str(i)+"/train_part/"+str(i)+"/"
    #     print(train_visit_path)
    #     window_stat(train_visit_path, 6)
    # 