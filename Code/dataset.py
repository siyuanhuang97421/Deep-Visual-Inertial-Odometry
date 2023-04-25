import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from numpy import asarray
import numpy as np
import csv
import random

class VIODataset(Dataset):
    def __init__(self, data_dir, image_frames, imu_reading, ground_truth, sequence_length=10):
        self.data_dir = data_dir
        self.image_frames = image_frames
        self.imu_reading = imu_reading
        self.ground_truth = ground_truth
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.image_frames)

    def __getitem__(self, i):
        if i < self.sequence_length:
            i = self.sequence_length

        i_start = i - self.sequence_length
        image_frames = []
        imu = []
        gt_image_frames = []
        gt_imu = []
        for frame_idx in range(i_start,i+1):
            frame_path = self.data_dir + "/cam0/data/{frameId}.png".format(frameId = self.image_frames[frame_idx])
            frame = Image.open(frame_path)
            image_frames.append(asarray(frame))
            gt_image_frames.append(self.ground_truth[(frame_idx)*10])
            
        
        for frame_idx in range(i_start,i):
            imu.append(self.imu_reading[(frame_idx)*10:(frame_idx+1)*10])
            gt_imu.append(self.ground_truth[(frame_idx)*10:(frame_idx+1)*10])

        return torch.tensor(np.array(image_frames)),\
                torch.tensor(np.array(gt_image_frames)),\
                torch.tensor(np.array(imu)),\
                torch.tensor(np.array(gt_imu))
    

data_dirs = ["./Data/MH_02_easy/mav0", "./Data/MH_03_medium/mav0","./Data/MH_04_difficult/mav0", "./Data/MH_05_difficult/mav0"]
datasets = []
dataloaders = []
for data_dir in data_dirs:
    image_frames = []
    with open(data_dir + '/cam0/time_aligned.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                image_frames.append(row[0])
    imu_file = data_dir + '/imu0/time_aligned.csv'
    imu_reading = np.genfromtxt(imu_file, delimiter=',')

    gt_file = data_dir + '/state_groundtruth_estimate0/time_aligned.csv'
    gt = np.genfromtxt(gt_file, delimiter=',')

    dataset = VIODataset(data_dir,image_frames,imu_reading,gt)
    datasets.append(dataset)
    dataloaders.append(DataLoader(dataset, batch_size=4, shuffle=True))

# frames, gt_image_frames, imu, gt_imu = dataset[10]
numDataLoaders = len(dataloaders)

frames, gt_image_frames, imu, gt_imu = next(iter(dataloaders[random.randint(0, numDataLoaders-1)]))
