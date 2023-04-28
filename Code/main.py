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
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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

    # loss_global_trans = L2(pred_global[:, :, :3], gt_global[:, :, :3])
    # loss_global_angle = L2(pred_global[:, :, 3:], gt_global[:, :, 3:])
    # loss_global = loss_global_trans + alpha * loss_global_angle

    # loss = loss_local + loss_global
    loss = loss_local #/ (batch_size * pred_f2f.shape[1])
    
    # return torch.stack(loss)
    return loss

def io_loss_function(pred_f2f, gt_f2f, epoch=0):
    # criterion  = nn.L1Loss(size_average=False)
    # loss = criterion(pred_f2f, target_f2f) + criterion(pred_abs, target_abs)

    L2  = nn.MSELoss(size_average=False)
    alpha = 150.
    if epoch > 1000:
        alpha = 15
    elif epoch > 5000:
        alpha = 1
    batch_size = pred_f2f.shape[0]
    # loss = []

    # for i in range(batch_size):
    loss_local_trans = L2(pred_f2f[:, :3], gt_f2f[:, :3])
    # loss_local_angle =torch.abs(torch.sum(pred_f2f[:, 3:] * gt_f2f[:, 3:],dim=1))
    # loss_local_angle =torch.sum(torch.ones_like(loss_local_angle) - loss_local_angle)

    loss_local_angle = torch.stack([1-torch.abs(torch.dot(pred_f2f[i, 3:], gt_f2f[i, 3:])) for i in range(batch_size)])
    loss_local_angle = torch.sum(torch.abs(loss_local_angle))
    loss_local = loss_local_trans + alpha * loss_local_angle
    print(loss_local_trans.data, loss_local_angle.data)
    print(loss_local_trans.data / batch_size, loss_local_angle.data / batch_size)
        # loss.append(loss_local)

    # loss_global_trans = L2(pred_global[:, :, :3], gt_global[:, :, :3])
    # loss_global_angle = L2(pred_global[:, :, 3:], gt_global[:, :, 3:])
    # loss_global = loss_global_trans + alpha * loss_global_angle

    # loss = loss_local + loss_global
    loss = loss_local / batch_size #/ (batch_size * pred_f2f.shape[1])
    
    # return torch.stack(loss)
    return loss

def loss_test(pred_f2f, gt_f2f, pred_global, gt_global):
    # criterion  = nn.L1Loss(size_average=False)
    # loss = criterion(pred_f2f, target_f2f) + criterion(pred_abs, target_abs)

    L2  = nn.MSELoss(size_average=False)
    alpha = 1.
    batch_size = pred_f2f.shape[0]
    # loss = []

    # for i in range(batch_size):
    loss_local_trans = L2(pred_f2f[:, :, :3], gt_f2f[:, :, :3])  / (batch_size * pred_f2f.shape[1])
    loss_local_angle = L2(pred_f2f[:, :, 3:], gt_f2f[:, :, 3:])  / (batch_size * pred_f2f.shape[1])
    loss_local = loss_local_trans + alpha * loss_local_angle
        # loss.append(loss_local)

    loss_global_trans = L2(pred_global[:, :, :3], gt_global[:, :, :3])  / (batch_size * pred_f2f.shape[1])
    loss_global_angle = L2(pred_global[:, :, 3:], gt_global[:, :, 3:])  / (batch_size * pred_f2f.shape[1])
    loss_global = loss_global_trans + alpha * loss_global_angle

    # loss = loss_local + loss_global
    # loss = loss_local
    
    # return torch.stack(loss)
    # return loss
    print(loss_local_trans, loss_local_angle, loss_global_trans, loss_global_angle)

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

def load_batch_data_vo(dataloaders, mode = "train"):
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
    if mode == "test":
        frames, gt_image_frames, imu, gt_imu = next(iter(dataloaders))
    else:
        frames, gt_image_frames, imu, gt_imu = next(iter(dataloaders[random.randint(0, numDataLoaders-1)]))
    # frames = frames.to(device)
    # gt_image_frames = gt_image_frames.to(device)
    # imu = imu.to(device)

    batch_size = frames.shape[0]
    frame_seq_length = frames.shape[1]
    # imu_seq_lendth = imu.shape[1]
    gt_pose = gt_image_frames[:, :, 1:8]
    # gt_imu_pose = gt_imu[:,:,1:8]
    imu = imu[:,:,1:7]
    # img_pairs = []
    gt_f2f = []
    gt_global = []

    for i in range(batch_size):
        # img_seq = frames[i]
        # img_pair_seq = np.array([np.concatenate((img_seq[k, np.newaxis], img_seq[k + 1, np.newaxis]), axis=0)
        #                         for k in range(seq_length - 1)])
        # img_pairs.append(img_pair_seq)

        # gt_init = gt_pose[i, 0, :]
        gt_f2f.append(np.array([pose_diff(gt_pose[i, k, :], gt_pose[i, k + 1, :]) for k in range(frame_seq_length - 1)]))
        # gt_global.append(np.array([gt_pose[i] for k in range(1, seq_length)]))
        gt_global.append(np.array([pose_diff(gt_pose[i, k, :], gt_pose[i, 0, :]) for k in range(1, frame_seq_length)]))

    # img_pairs = torch.tensor(np.array(img_pairs)).float().to(device)
    gt_f2f = torch.tensor(np.array(gt_f2f)).float().to(device)
    gt_global = torch.tensor(np.array(gt_global)).float().to(device)

    frames = frames.float().squeeze()
    imu = imu.float()
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
        # startIter = latest_ckpt_file.replace(args.checkpoint_path,'').replace('model_','').replace('.ckpt','')
        # startIter = int(startIter)
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
        model = DeepIO(input_size=6, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2)
    if args.network_type == "vo":
        model = DeepVO()
    model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lrate, momentum=0.9)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(300, args.max_epochs, 300), gamma=0.5)
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
        elif args.network_type == "io":
            pred_f2f = model(imu)
        gt_f2f = gt_f2f[:,-1,:]
        # pred_f2f = normalize_quaternion(pred_f2f)

        # calculate accumulated abs pose
        # batch_size = pred_f2f.shape[0]
        # seq_length = pred_f2f.shape[1]
        # pred_global = torch.empty(batch_size, seq_length, 7).to(device)
        # if args.network_type == "vo":
        #     for i in range(batch_size):
        #         seq = pred_f2f[i]
        #         pred_global[i, 0] = seq[0]
        #         for j in range(1, len(seq)):
        #             pred_global[i, j] = pose_accumulate(pred_global[i, j-1], seq[j])

        if args.network_type == "vo":
            loss = io_loss_function(pred_f2f, gt_f2f, epoch) #, pred_global, gt_global)
        elif args.network_type == "io":
            loss = io_loss_function(pred_f2f, gt_f2f)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"epoch:{epoch}, loss_this_epoch:{loss}\n")
        writer.add_scalar('LossEveryIter', loss, epoch)
        writer.flush()

        # save checkpoint
        if (epoch + 1) % args.save_ckpt_iter == 0:
            print("Saved a checkpoint {}".format(epoch))
            if not (os.path.isdir(args.checkpoint_path)):
                os.makedirs(args.checkpoint_path)
            
            checkpoint_save_name =  args.checkpoint_path + os.sep + 'model_' + str(epoch) + '.ckpt'
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, checkpoint_save_name)

def testvo(args, dataloader):
    frames, gt_image_frames, imu, gt_imu = next(iter(dataloader))

    model = DeepVO()
    model.to(device)

    loadModel(model, args)
    model.eval()
    with torch.no_grad():
        imgs, imu, gt_f2f, gt_global = load_batch_data_vo(dataloader, "test")

        seq_length = imgs.shape[0]
        pred_f2f_all = []
        # segment_length = 10
        for i in range(seq_length - 1):
            imgs_segment = imgs[i:i+2, :]
            imgs_segment = imgs_segment[None, :]

            if args.network_type == "vo":
                pred_f2f = model(imgs_segment)
            pred_f2f_all.append(pred_f2f)
            # print(i, pred_f2f)
        # pred_f2f_all = torch.as_tensor(pred_f2f_all).to(device)
        # last_pose_f2f = pred_f2f_all[-1]
        pred_f2f_all = torch.stack(pred_f2f_all).squeeze()
        # pred_f2f_all = pred_f2f_all.view(seq_length - seq_length % segment_length, -1)
        # pred_f2f_all = torch.cat((pred_f2f_all, last_pose_f2f))

        # calculate accumulated abs pose
        # batch_size = pred_f2f.shape[0]
        # seq_length = pred_f2f.shape[1]
        pred_global = torch.empty(seq_length-1, 7).to(device)
        # for i in range(batch_size):
        # seq = pred_f2f_all[0]
        pred_global[0] = pred_f2f_all[0]
        for j in range(1, seq_length-1):
            pred_global[j] = pose_accumulate(pred_global[j-1], pred_f2f_all[j])

    loss = io_loss_function(pred_f2f_all, gt_f2f.squeeze())
    # print(loss)

    pred_global = np.squeeze(pred_global.detach().cpu().numpy())
    gt_global = np.squeeze(gt_global.detach().cpu().numpy())
    x = pred_global[:, 0]
    y = pred_global[:, 1]
    z = pred_global[:, 2]

    xt = gt_global[:, 0]
    yt = gt_global[:, 1]
    zt = gt_global[:, 2]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x, y, "*-", label="estimation")
    plt.plot(xt, yt, "*-", label="ground truth")
    plt.axis('equal')
    plt.legend()
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.subplot(1, 2, 2)
    plt.plot(x, z, "*-", label="estimation")
    plt.plot(xt, zt, "*-", label="ground truth")
    plt.axis('equal')
    plt.xlabel("x / m")
    plt.ylabel("z / m")
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection ='3d') 
    ax.plot3D(x, y, z, 'blue', label="estimation")
    ax.plot3D(xt, yt, zt, 'red', label="ground truth")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("z / m")
    ax.set_aspect('equal')
    ax.legend()
    # ax.set_title('3D line plot geeks for geeks')
    plt.show()

def testio(args, dataset):
    # frames, gt_image_frames, imu, gt_imu = next(iter(dataloader))
    dataset_len = dataset.__len__()

    model = DeepIO(input_size=6, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2)
    model.to(device)

    loadModel(model, args)
    model.eval()
    with torch.no_grad():
        # imgs, imu, gt_f2f, gt_global = load_batch_data_vo(dataloader, "test")

        # seq_length = imgs.shape[1]
        pred_f2f_all = []
        segment_length = 10
        for i in range(10, dataset_len):
            # imgs_segment = imgs[:, i:i+segment_length+1]
            _,_,imu_sequence,_ = dataset.__getitem__(i)
            imu_sequence = imu_sequence[:,1:7]
            imu_sequence = imu_sequence.float()
            imu_sequence = imu_sequence[None,:]
            pred_f2f = model(imu_sequence)
            pred_f2f_all.extend(pred_f2f)
        # pred_f2f_all = torch.as_tensor(pred_f2f_all).to(device)
        # last_pose_f2f = pred_f2f_all[-1]
        pred_f2f_all = torch.stack(pred_f2f_all)
        # pred_f2f_all = pred_f2f_all.view(seq_length - seq_length % segment_length, -1)
        # pred_f2f_all = torch.cat((pred_f2f_all, last_pose_f2f))

        # calculate accumulated abs pose
        # batch_size = pred_f2f.shape[0]
        # seq_length = pred_f2f.shape[1]
        pred_global = torch.empty(pred_f2f_all.shape[0], 7).to(device)
        # for i in range(batch_size):
        # seq = pred_f2f_all[0]
        pred_global[0] = pred_f2f_all[0]
        for j in range(1, pred_f2f_all.shape[0]):
            pred_global[j] = pose_accumulate(pred_global[j-1], pred_f2f_all[j])

    # loss = loss_test(pred_f2f_all.view(1, -1, 7), gt_f2f, pred_global.view(1, -1, 7), gt_global)
    # print(loss)

    pred_global = np.squeeze(pred_global.detach().cpu().numpy())
    x = pred_global[:, 0]
    y = pred_global[:, 1]
    z = pred_global[:, 2]

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.plot(x, y, "*-")
    # plt.subplot(1, 2, 2)
    # plt.plot(x, z, "*-")
    # plt.show()

    fig = plt.figure()
    ax = plt.axes(projection ='3d') 
    ax.plot3D(x, y, z, 'green')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_title('3D line plot geeks for geeks')
    plt.show()


def main(args):
    print("Loading data")

    if args.mode == "train":
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

            if args.network_type == "vo":
                dataset = VIODataset(data_dir,image_frames,imu_reading,gt, 1)
            else:
                dataset = VIODataset(data_dir,image_frames,imu_reading,gt)
            datasets.append(dataset)
            dataloaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=True))

        train(args, dataloaders)

    elif args.mode == "test":
        data_dir = "./Data/MH_01_easy/mav0"
        image_frames = []
        with open(data_dir + '/cam0/time_aligned.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                image_frames.append(row[0])
        imu_file = data_dir + '/imu0/time_aligned.csv'
        imu_reading = np.genfromtxt(imu_file, delimiter=',')

        gt_file = data_dir + '/state_groundtruth_estimate0/time_aligned.csv'
        gt = np.genfromtxt(gt_file, delimiter=',')

        # test_dataset = VIODataset(data_dir,image_frames,imu_reading,gt,100)
        if(args.network_type == "io"):
            test_dataset = VIODataset(data_dir,image_frames,imu_reading,gt)
            testio(args, test_dataset)
        else:
            test_dataset = VIODataset(data_dir,image_frames,imu_reading,gt,len(image_frames) - 1)
            dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            testvo(args, dataloader)


def configParser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path',default="./Phase2/data/lego/",help="dataset path")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--network_type',default="vo",help="vo/io/vio")
    parser.add_argument('--mode',default='test',help="train/test/val")
    parser.add_argument('--max_epochs',default=9000,help="number of max epochs for training")
    parser.add_argument('--lrate',default=1e-4,help="training learning rate")
    parser.add_argument('--batch_size',default=32,help="batch size")
    parser.add_argument('--checkpoint_path',default="./checkpoint_a150_512_1e4_dot_abs_nodropout/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=200,help="num of iteration to save checkpoint")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main(args)