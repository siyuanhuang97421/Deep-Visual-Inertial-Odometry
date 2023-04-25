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

def loss_function(pred_f2f, gt_f2f, pred_global, gt_global):
    # criterion  = nn.L1Loss(size_average=False)
    # loss = criterion(pred_f2f, target_f2f) + criterion(pred_abs, target_abs)

    L2  = nn.MSELoss(size_average=False)
    alpha = 1.
    batch_size = pred_f2f.shape[0]
    # loss = []

    # for i in range(batch_size):
    loss_local_trans = L2(pred_f2f[:, :, :3], gt_f2f[:, :, :3])
    loss_local_angle = L2(pred_f2f[:, :, 3:], gt_f2f[:, :, 3:])
    loss_local = loss_local_trans + alpha * loss_local_angle
        # loss.append(loss_local)

    loss_global_trans = L2(pred_global[:, :, :3], gt_global[:, :, :3])
    loss_global_angle = L2(pred_global[:, :, 3:], gt_global[:, :, 3:])
    loss_global = loss_global_trans + alpha * loss_global_angle

    loss = loss_local + loss_global
    
    # return torch.stack(loss)
    return loss

def pose_diff(pre_pose, curr_pose):
    q_pre = Qua(array=pre_pose[3:])
    q_curr = Qua(array=curr_pose[3:])
    q_diff = q_curr * q_pre.inverse

    diff = np.zeros(7)
    diff[:3] = curr_pose[:3] - pre_pose[:3]
    diff[3:] = np.array([q_diff[0], q_diff[1], q_diff[2], q_diff[3]])

    return diff

def pose_accumulate(pre_pose, pose_diff):
    q_pre = Qua(array=pre_pose[3:])
    q_diff = Qua(array=pose_diff[3:])
    q_final = q_diff * q_pre

    pose_final = torch.zeros(7).to(device)
    pose_final[:3] = pre_pose[:3] + pose_diff[:3]
    pose_final[3:] = torch.tensor([q_final[0], q_final[1], q_final[2], q_final[3]])

    return pose_final

def load_batch_data_vo(dataloaders):
    '''
    Output:
    img_pairs: batch_size * sequence length * (2 images)
    imu: batch_size * num_imu_data * 6 (ax, ay, az, wx, wy, wz)
    gt_f2f: batch_size * sequence length * 7 (x, y, z, q1, q2, q3, q4)
            relative transform between two frames
    gt_global: batch_size * sequence length * 7 (x, y, z, q1, q2, q3, q4)
            absolute transform compared with the initial pose???
    ''' 
    numDataLoaders = len(dataloaders)
    frames, gt_image_frames, imu, gt_imu = next(iter(dataloaders[random.randint(0, numDataLoaders-1)]))
    # frames = frames.to(device)
    # gt_image_frames = gt_image_frames.to(device)
    # imu = imu.to(device)

    batch_size = frames.shape[0]
    seq_length = frames.shape[1]
    gt_pose = gt_image_frames[:, :, 1:8]
    img_pairs = []
    gt_f2f = []
    gt_global = []
    for i in range(batch_size):
        # img_seq = frames[i]
        # img_pair_seq = np.array([np.concatenate((img_seq[k, np.newaxis], img_seq[k + 1, np.newaxis]), axis=0)
        #                         for k in range(seq_length - 1)])
        # img_pairs.append(img_pair_seq)

        # gt_init = gt_pose[i, 0, :]
        gt_f2f.append(np.array([pose_diff(gt_pose[i, k, :], gt_pose[i, k + 1, :]) for k in range(seq_length - 1)]))
        # gt_global.append(np.array([gt_pose[i] for k in range(1, seq_length)]))
        gt_global.append(np.array([pose_diff(gt_pose[i, k, :], gt_pose[i, 0, :]) for k in range(1, seq_length)]))

    img_pairs = torch.tensor(np.array(img_pairs)).float().to(device)
    gt_f2f = torch.tensor(np.array(gt_f2f)).float().to(device)
    gt_global = torch.tensor(np.array(gt_global)).float().to(device)

    frames = frames[:, :, None].float()

    return frames.to(device), imu, gt_f2f, gt_global
    # return img_pairs, imu, gt_f2f, gt_global

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
    # traj_start = 5
    # traj_end = 1000 # len(dataset)

    if args.network_type == "io":
        model = DeepIO()
    if args.network_type == "vo":
        model = DeepVO()
    model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lrate, momentum=0.9)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    # startIter = loadModel(model, args)
    startIter = 0
    model.train() # put inside epoch loop in my previous code

    for epoch in tqdm(range(startIter, args.max_epochs)):
        # print("start epoch {}".format(epoch))
        # load a batch
        imgs, imu, gt_f2f, gt_global = load_batch_data_vo(dataloaders)
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        if args.network_type == "vo":
            pred_f2f = model(imgs)

        # calculate accumulated abs pose
        batch_size = pred_f2f.shape[0]
        seq_length = pred_f2f.shape[1]
        pred_global = torch.empty(batch_size, seq_length, 7).to(device)
        for i in range(batch_size):
            seq = pred_f2f[i]
            pred_global[i, 0] = seq[0]
            for j in range(1, len(seq)):
                pred_global[i, j] = pose_accumulate(pred_global[i, j-1], seq[i])
            # pred_global[i, 0] = pred_seq
        # pred_global = torch.tensor(np.array(pred_global)).to(device)

        loss = loss_function(pred_f2f, gt_f2f, pred_global, gt_global)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"epoch:{epoch}, loss_this_epoch:{loss}\n")
            writer.add_scalar('LossEveryIter', loss, epoch)
            writer.flush()

        # save checkpoint
        if epoch % args.save_ckpt_iter == 100:
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
        dataloaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=True))

    if args.mode == "train":
        train(args, dataloaders)
    elif args.mode == "test":
        test(args, dataloaders)


def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Phase2/data/lego/",help="dataset path")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--network_type',default="vo",help="vo/io/vio")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--max_epochs',default=1000,help="number of max epochs for training")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--batch_size',default=2,help="batch size")
    parser.add_argument('--checkpoint_path',default="./checkpoint/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main(args)