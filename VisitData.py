"""
this code is used to process the Visit data in the Urban classification Contest2 2019.
extract the features:
1)	Total visitor (the number of records)
2)	24/12 hours, the number of visits. 
3)	12 months, the number of visits. 
4)	The return times of each visitor. 
5)	The stay time of each visitor at each visit. 
6)	The size of data of the record.
"""

__author__ = 'Changlin'
__version__ = 0.1

import numpy as np
import glob
import os
import math
from scipy.stats import norm 
import matplotlib.pyplot as plt 
import cv2
import params
from keras.utils.np_utils import to_categorical

def GetVisitFeatures(file_path,output_path):
    total_visitor=0
    hour_visitor=[0]*12
    month_visitor=[0]*12
    retention=0
    avg_duration=0
    fsize=0 
    # ######
    fsize = os.path.getsize(file_path)
    fsize = fsize/float(1024*1024)

    fp = open(file_path)
    lines = fp.readlines()
    total_visitor=len(lines)
    fp.close()
    for line in lines:
        line = line.strip('/n')
        data=line.split()[1]
        data=data.split(',')
        years=[]
        days=[]
        months=[0]*12
        hours=[0]*12
        stays=[]
        time_hour_record=[[]]
        for sub_data in data:   ###20181221&09|10|11|12|13|14|15
            yearday_data=sub_data.split('&')[0]
            hours_data=sub_data.split('&')[1]
            years.append(yearday_data[:4])
            month_id=int(yearday_data[4:6])-1
            months[month_id]=months[month_id]+1
            day_id=int(yearday_data[-2:])
            days.append(day_id)

            #### hours
            hours_data=hours_data.split('|')
            stays.append(len(hours_data))
            hour_record=[]
            for hour_ in hours_data:
                hour_id=math.floor(int(hour_)/2.0)
                hours[hour_id]=hours[hour_id]+1
                hour_record.append(int(hour_))

            ### need to know if it is continued day.

            # for i in range(len(days)):
            #     day=days[i]
            # if i-1>=0:
            #     day_1=days[i-1]
            #     if day-day_1==1:
            #         day_last_hour=hour_record[0]
            #         last_day_hours=time_hour_record[i-1]
            #         day_1_last_hour=last_day_hours[len(hour_record)]
        avg_duration=avg_duration+np.mean(stays)
        hour_visitor=np.sum([hour_visitor,hours], axis = 0)
        month_visitor=np.sum([month_visitor,months], axis = 0)
        
    avg_duration=avg_duration/total_visitor
    #normalization (0-1)
    total_visitor=total_visitor
    hour_visitor=hour_visitor/sum(hour_visitor)
    avg_duration=avg_duration
    fsize=fsize
    output_path
    with open(output_path, 'w') as f:
        f.write(str(total_visitor)+',')
        for i in range(12):
            f.write(str(hour_visitor[i])+',')
        f.write(str(avg_duration)+',')
        f.write(str(fsize)+'\n')
        f.close()





def ConvertVisit2Features(data_folder, out_folder):
    if os.path.exists(out_folder)==0:
        os.mkdir(out_folder)
    glob_path=os.path.join(data_folder,'*.txt')
    files=glob.glob(glob_path)
    num_file=len(files)
    processed_=0
    print('there are total {} file:'.format(num_file))
    for file in files:
        # if processed_<24000:
        #     processed_=processed_+1
        #     continue
        file_name=os.path.split(file)[-1]
        out_path=os.path.join(out_folder,file_name)
        GetVisitFeatures(file,out_path)
        processed_=processed_+1
        print('processing {} %:'.format(processed_/num_file))



def DataNormalization(data_folder):
    features=[]
    glob_path=os.path.join(data_folder,'*.txt')
    files=glob.glob(glob_path)
    for file in files:
        fp = open(file)
        lines = fp.readlines()
        fp.close()
        featur=[]
        for line in lines:
            line = line.strip('\n')
            numbers=line.split(',')
            for num in numbers:
                featur.append(float(num))
        features.append(featur)
    features=np.array(features)
    with open('mean_std.txt', 'w') as f:
        for i in range(features.shape[1]):
                data=features[:,i]
                mean,std=norm.fit(data)
                f.write(str(mean)+',')
                f.write(str(std)+'\n')

                plt.hist(data, bins=100, normed=True) 
                xmin, xmax = plt.xlim() 
                x = np.linspace(xmin, xmax, 100) 
                y = norm.pdf(x, mean, std) 
                plt.plot(x, y) 
                plt.show() 
        f.close()

def DataSampleAnalysis(data_folder):
    folders = os.listdir(data_folder)
    for folder in folders:
        sub_folder=os.path.join(data_folder,folder)
        if os.path.isdir(sub_folder):
            txt_path=os.path.join(data_folder,folder+'.txt')
            f=open(txt_path, 'w')
            glob_path=os.path.join(sub_folder,'*.jpg')
            files=glob.glob(glob_path)
            for img in files:
                img_name=os.path.split(img)[-1]
                img_path=os.path.join(folder,img_name)
                f.write(img_path+'\n')
            f.close()

def load_class_balanced_files(data_folder,label_folder='',text_files=[],vali_ratio=0.1,max_samples=-1,using_visit_data=True):
    
    #balanced_sample_number=1600
    visit_feature_foler=params.visit_feature_folder
    #visit_feature_foler=r'G:\DataSet\UrbanClassification\data\train_visit_features'
    imgs = []
    gts=[]
    extras=[]

    imgs_v=[]
    gts_v=[]
    extras_v=[]
    if len(text_files)<1: ##load all txt recording all the images.
        glob_path=os.path.join(data_folder,'*.txt')
        files=glob.glob(glob_path)
        for txt in files:
            text_files.append(txt)
    clasee_samples=[]
    num_class_sample=[]
    for i in range(len(text_files)):
        fp = open(text_files[i])
        lines = fp.readlines()
        fp.close()
        ids=[]
        for line in lines:
            line = line.strip('\n')
            ids.append(line)
        clasee_samples.append(ids)
        num_class_sample.append(len(ids))

    #extrac the validation data first
    num_class_sample_train=[]
    val_class_ids=[]
    train_class_ids=[]
    for i in range(len(text_files)):
        val_num=int((num_class_sample[i])*vali_ratio)
        idx = np.random.permutation(num_class_sample[i])
        val_ids=idx[0:val_num]
        train_ids=idx[val_num:num_class_sample[i]]
        val_class_ids.append(val_ids)
        train_class_ids.append(train_ids)
        num_class_sample_train.append(len(train_ids))

    if max_samples<1:
        max_samples=max(num_class_sample_train)
    min_samples=min(num_class_sample_train)
    record_sampels=0
    train_class_ids_balanced=[]
    for i in range(len(text_files)):
        train_class_ids_balanced.append([])
    batch_star=[0]*len(text_files)
    while record_sampels<max_samples:
        
        
        for i in range(len(text_files)):
            batch_ids=[]
            class_ids=train_class_ids[i]
            batch_end=batch_star[i]+min_samples
            if batch_end>=num_class_sample_train[i]:
                num_plu=batch_end-num_class_sample_train[i]
                batch_end=num_class_sample_train[i]
                batch_range1=range(batch_star[i],batch_end)
                batch_ids.extend((class_ids[ind] for ind in batch_range1))
                batch_range2=range(0,num_plu)
                batch_ids.extend((class_ids[ind] for ind in batch_range2))
            else:
                batch_range=np.array(range(batch_star[i],batch_end))
                batch_star[i]=batch_end
                batch_ids.extend(class_ids[ind] for ind in batch_range)

            train_class_ids_balanced[i].extend(batch_ids)
        record_sampels=record_sampels+min_samples

    for i in range(len(text_files)):
        ids=train_class_ids_balanced[i]
        files=clasee_samples[i]
        for id in ids:
            img_path=files[id]
            imgs.append(os.path.join(data_folder,img_path))
            gts.append(i)
            if using_visit_data:
                img_name=os.path.split(img_path)[-1]
                extra_path=os.path.join(visit_feature_foler,img_name.replace('jpg','txt'))
                extras.append(os.path.join(data_folder,extra_path))
    for i in range(len(text_files)):
        ids=val_class_ids[i]
        files=clasee_samples[i]
        for id in ids:
            img_path=files[id]
            imgs_v.append(os.path.join(data_folder,img_path))
            gts_v.append(i)
            if using_visit_data:
                img_name=os.path.split(img_path)[-1]
                extra_path=os.path.join(visit_feature_foler,img_name.replace('jpg','txt'))
                extras_v.append(os.path.join(data_folder,extra_path))        

    if using_visit_data:
        return imgs,gts,extras,  imgs_v, gts_v, extras_v
    else:
        return imgs, gts, imgs_v,  gts_v
def ImgNormalization(img):
    img_return=img/125.0-1
    return img_return
def VisitNormalization(feature):
    feature[:,0]=feature[:,0]/1400-1;
    feature[:,1:13]=feature[:,1:13]/0.085-1;
    feature[:,13]=feature[:,13]/3.57-1;
    feature[:,14]=feature[:,14]/0.3-1;
    feature[feature<-1]=-1;
    feature[feature>1]=1;
    return feature
def LoadVisitFeatures(txt_path):
    if not os.path.exists(txt_path):
        print('no such visit txt file: ',txt_path)
    fp = open(txt_path)
    lines = fp.readlines()
    fp.close()
    featur=[]
    for line in lines:
        line = line.strip('\n')
        numbers=line.split(',')
        for num in numbers:
            featur.append(float(num))
    return featur
def input_load_train_data(img_file_t, lable_t,visit_file_t):
    if len(img_file_t)!= len(visit_file_t) or len(img_file_t)!= len(lable_t):
        print('the input data have not equal length!!!')
    imgs=[]
    visits=[]
    labels=[]
    visit_=False
    if len(visit_file_t)>1:
        visit_=True

    for i in range(len(img_file_t)):
        imgs.append(cv2.imread(img_file_t[i]))
        
        labels.append(lable_t[i])
        if visit_:
            visits.append(LoadVisitFeatures(visit_file_t[i]))
            
    imgs=np.array((imgs))
    imgs=ImgNormalization(imgs)
    labels = to_categorical(labels, params.num_labels)
    if visit_:
        visits=np.array(visits)
        visits=VisitNormalization(visits)
        return [imgs, visits], labels
    else:
        return imgs, labels




if __name__ == "__main__":
   # data_folder='G:/DataSet/UrbanClassification/data/train_visit'
    #outfolder="G:/DataSet/UrbanClassification/data/train_visit_features"
    #img_folder=r'G:\DataSet\UrbanClassification\data\train_img'
    data_folder='G:/DataSet/UrbanClassification/data/test_visit'
    outfolder="G:/DataSet/UrbanClassification/data/test_visit_features"
    ConvertVisit2Features(data_folder,outfolder)
    #DataSampleAnalysis(img_folder)
    ##DataNormalization(outfolder)
