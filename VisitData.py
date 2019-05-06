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
        if processed_<24000:
            processed_=processed_+1
            continue
        file_name=os.path.split(file)[-1]
        out_path=os.path.join(out_folder,file_name)
        GetVisitFeatures(file,out_path)
        processed_=processed_+1
        print('processing {} %:'.format(processed_/num_file))






if __name__ == "__main__":
    data_folder='G:/DataSet/UrbanClassification/data/train_visit'
    outfolder="G:/DataSet/UrbanClassification/data/train_visit_features"
    ConvertVisit2Features(data_folder,outfolder)
