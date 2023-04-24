#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Sampling VICON ground truth camera pose. (absolute pose)

In EuRoC MAV dataset, the vicon motion capture system (Leica MS50) record 
data with 100Hz.  (All pose in vicon seems to be global pose, which is
the pose related to first camera pose.)

Because VINet prediction trajectory  with the frequency equal to image 
frame rate, the "answer" of the training need to be in the same frequency.

My quick workaround is to find the nearest timestamp in vicon/data.csv based 
on the timestamp of cam0/.

"""


from PIL import Image
import os
import sys
import errno
from subprocess import call
import csv

def align_data(dataset_dir):


    ## Get image list
    img_data = []
    with open(dataset_dir + '/cam0/data.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            img_data.append(row)
            
    img_data = img_data[1:]
    # img_data = os.listdir(dataset_dir + '/cam0/data')  
    # img_data.sort()

    # for i in range(len(img_data)):
    #     img_data[i] = img_data[i][0:-4]
    
    ## Get Ground Truth data
    GT_data = []
    with open(dataset_dir + '/state_groundtruth_estimate0/data.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            GT_data.append(row)
            
    GT_data = GT_data[1:]

    ## Get IMU original data
    IMU_data = []
    with open(dataset_dir + '/imu0/data.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            IMU_data.append(row)
            
    IMU_data = IMU_data[1:]

    ## Find matched data
    img_start_idx = 0
    img_end_idx = 0
    img_curr_idx = 0
    gt_start_idx = 0
    gt_end_idx = 0
    gt_curr_idx = 0
    imu_start_idx = 0
    imu_end_idx = 0
    imu_curr_idx = 0

    while True:
        if IMU_data[imu_curr_idx][0] < GT_data[gt_curr_idx][0]:
            imu_curr_idx += 1
        elif IMU_data[imu_curr_idx][0] > GT_data[gt_curr_idx][0]:
            gt_curr_idx +=1
        else:
            imu_start_idx = imu_curr_idx
            gt_start_idx = gt_curr_idx
            break
    
    print("Filtered ", imu_start_idx, "IMU data points and ", gt_start_idx, "GT data points at the beginning")
    IMU_data = IMU_data[imu_start_idx:]
    GT_data = GT_data[gt_start_idx:]

    imu_curr_idx = 0
    gt_curr_idx = 0
    while True:
        if imu_curr_idx < len(IMU_data) and\
            gt_curr_idx < len(GT_data) and\
            abs(int(IMU_data[imu_curr_idx][0]) - int(GT_data[gt_curr_idx][0])) <= 500:
            imu_curr_idx += 1
            gt_curr_idx += 1
        else:
            IMU_data = IMU_data[:imu_curr_idx]
            GT_data = GT_data[:gt_curr_idx]
            print(len(GT_data), " data points remaining for IMU and GT")
            break
    
    gt_curr_idx = 0
    img_curr_idx = 0

    while True:
        if abs(int(img_data[img_curr_idx][0]) - int(GT_data[gt_curr_idx][0])) <= 500:
            img_data = img_data[img_curr_idx:]
            GT_data = GT_data[gt_curr_idx:]
            IMU_data = IMU_data[gt_curr_idx:]
            print("Filtered ", img_curr_idx, "img data points and ", gt_start_idx, "GT data points at the beginning")
            break
        elif int(img_data[img_curr_idx][0]) < int(GT_data[gt_curr_idx][0]):
            img_curr_idx += 1 
        elif int(img_data[img_curr_idx][0]) > int(GT_data[gt_curr_idx][0]):
            gt_curr_idx += 1
    
    gt_curr_idx = len(GT_data) - 1
    img_curr_idx = len(img_data) - 1
    while True:
        if abs(int(img_data[img_curr_idx][0]) - int(GT_data[gt_curr_idx][0])) <= 500:
            img_data = img_data[:img_curr_idx+1]
            GT_data = GT_data[:gt_curr_idx+1]
            IMU_data = IMU_data[:gt_curr_idx+1]
            print(len(GT_data), " data points remaining for IMU and GT", len(img_data), " data points remaining for img")
            break
        elif int(img_data[img_curr_idx][0]) < int(GT_data[gt_curr_idx][0]):
            gt_curr_idx -= 1 
        elif int(img_data[img_curr_idx][0]) > int(GT_data[gt_curr_idx][0]):
            img_curr_idx -= 1
        
    with open(dataset_dir + '/state_groundtruth_estimate0/time_aligned.csv', 'w+') as f:
        for i in range(len(GT_data)):
            tmpStr = ",".join(GT_data[i])
            f.write(tmpStr + '\n')
    
    with open(dataset_dir + '/cam0/time_aligned.csv', 'w+') as f:
        for i in range(len(img_data)):
            tmpStr = ",".join(img_data[i])
            f.write(tmpStr + '\n')
        
    with open(dataset_dir + '/imu0/time_aligned.csv', 'w+') as f:
        for i in range(len(IMU_data)):
            tmpStr = ",".join(IMU_data[i])
            f.write(tmpStr + '\n')
    
    return
                

def main():
    align_data('./Data/MH_01_easy/mav0')
    align_data('./Data/MH_02_easy/mav0')
    align_data('./Data/MH_03_medium/mav0')
    align_data('./Data/MH_04_difficult/mav0')
    align_data('./Data/MH_05_difficult/mav0')

    #_get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V1_01_easy')
    # _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V1_02_medium')
    # _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V1_03_difficult')
    # _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V2_01_easy')
    # _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V2_02_medium')
    # _get_filenames_and_classes('/media/rvl/hddData1/dockerData/euroc/V2_03_difficult')
       
 

    
    
if __name__ == "__main__":
    main()
    
    
