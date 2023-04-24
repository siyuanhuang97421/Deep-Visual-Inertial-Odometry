import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import argparse
import glob
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pyquaternion import Quaternion as Qua

from dataset import *
from Network import *

import os
# from utils import tools
# from utils import se3qua

# import FlowNetC
# import flowlib

from PIL import Image
import numpy as np

import csv
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def loss_function(pred_f2f, gt_f2f):
    # criterion  = nn.L1Loss(size_average=False)
    # loss = criterion(pred_f2f, target_f2f) + criterion(pred_abs, target_abs)

    L2  = nn.L2Loss(size_average=False)

    alpha = 100
    loss_local_trans = L2(pred_f2f[:, :, :3], gt_f2f[:, :, :3])
    loss_local_angle = L2(pred_f2f[:, :, 3:], gt_f2f[:, :, 3:])
    loss_local = loss_local_trans + alpha * loss_local_angle

    # loss_global_trans = L2(pred_global[:, :, :3], gt_global[:, :, :3])
    # loss_global_angle = L2(pred_global[:, :, 3:], gt_global[:, :, 3:])
    # loss_global = loss_global_trans + alpha * loss_global_angle

    # loss = loss_local + loss_global
    
    return loss_local

def load_batch_data_vo(batch_size):
    '''
    Output:
    img_pairs: batch_size * sequence length * (2 images)
    imu: batch_size * num_imu_data * 6 (ax, ay, az, wx, wy, wz)
    gt_f2f: batch_size * sequence length * 7 (x, y, z, q1, q2, q3, q4)
            relative transform between two frames
    gt_global: batch_size * sequence length * 7 (x, y, z, q1, q2, q3, q4)
            absolute transform compared with the initial pose???
    ''' 

    # need to calculate quaternion difference between two frames

    # accumulate relative pose to global pose with last output
    # assume we know the initial pose?


    # calculate quaternion difference between each frame and first frame

    return img_pairs, imu, gt_f2f, gt_global

def loadModel(model, args):
    startIter = 0
    files = glob.glob(args.checkpoint_path + '*.ckpt')
    latest_ckpt_file = max(files, key=os.path.getctime) if files else None
    print(files)

    if latest_ckpt_file and args.load_checkpoint:
        print(latest_ckpt_file)
        latest_ckpt = torch.load(latest_ckpt_file, map_location=torch.device(device))
        startIter = latest_ckpt_file.replace(args.checkpoint_path,'').replace('model_','').replace('.ckpt','')
        startIter = int(startIter)
        model.load_state_dict(latest_ckpt['model_state_dict'])
        print(f"Loaded latest checkpoint from {latest_ckpt_file} ....")
    else:
        print('New model initialized....')
    
    return startIter

def train(args, dataloaders):
    # setup tensorboard
    writer = SummaryWriter(args.logs_path)
    traj_start = 5
    traj_end = 1000 # len(dataset)

    if args.network_type == "io":
        model = DeepIO()
    optimizer = optim.SGD(model.parameters(), lr=args.lrate, momentum=0.9)
    # startIter = loadModel(model, args)
    startIter = 0
    model.train() # put inside epoch loop in my previous code

    for epoch in tqdm(range(startIter, args.max_epochs)):
        # load a batch
        img_pairs, imu, gt_f2f, gt_global = load_batch_data_vo(args.batch_size)

        optimizer.zero_grad()

        pred_f2f = model(img_pairs, imu)

        # calculate accumulated abs pose

        loss = loss_function(pred_f2f, gt_f2f)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"epoch:{epoch}, loss_this_epoch:{loss}\n")
            writer.add_scalar('LossEveryIter', loss, epoch)
            writer.flush()

        # save checkpoint
        # if epoch % args.save_ckpt_iter == 0:
        print("Saved a checkpoint {}".format(epoch))
        if not (os.path.isdir(args.checkpoint_path)):
            os.makedirs(args.checkpoint_path)
        
        checkpoint_save_name =  args.checkpoint_path + os.sep + 'model_' + str(epoch) + '.ckpt'
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, checkpoint_save_name)

def test(args, dataloaders):
    ...
    # if args.network_type == "io":
    #     model = DeepIO()

    # loadModel(model, args)
    # model.eval()

    # traj_start = 5
    # traj_end = 1000 # len(dataset)

    # for i in range(traj_start, traj_end):
    #     # load this dataset section
    #     data, target_f2f, target_abs = load_data()
    #     if i == traj_start:
    #     # load first SE3 pose xyzQuaternion
    #         ...

    #     pred_f2f = model(data)
    #     # calculate accumulated abs pose
    #     pred_abs = pred_f2f # calculate it




def main(args):
    print("Loading data")
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
        dataloaders.append(DataLoader(dataset, batch_size=3, shuffle=True))

    if args.mode == "train":
        train(args, dataloaders)
    elif args.mode == "test":
        test(args, dataloaders)


def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Phase2/data/lego/",help="dataset path")
    parser.add_argument('--network_type',default="io",help="vo/io/vio")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--max_epochs',default=10000,help="number of max epochs for training")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--batch_size',default=8,help="batch size")
    parser.add_argument('--checkpoint_path',default="./Phase2/example_checkpoint/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main(args)