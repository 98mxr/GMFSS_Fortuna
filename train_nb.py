import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from model.GMFSS_train_b import Model
from model.dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")

log_path = 'log'

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, local_rank):
    if local_rank == 0:
        writer = SummaryWriter('log/train')
    else:
        writer = None
    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train')
    train_data = DataLoader(dataset, batch_size=args.batch_size, shuffle =True, num_workers=6, pin_memory=True, drop_last=True)
    args.step_per_epoch = train_data.__len__()
    time_stamp = time.time()
    for epoch in range(args.epoch):
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(step)
            pred, flow, metric0, metric1, loss_l1, loss_lpips = model.update(imgs, gt, learning_rate, True, timestep, step, args.step_per_epoch)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 20 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', loss_l1, step)
                writer.add_scalar('loss/lpips', loss_lpips, step)
            if step % 100 == 1 and local_rank == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                metric = (torch.cat((metric0, metric1), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow = flow.permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(5):
                    imgs = np.concatenate((pred[i], gt[i]), 1)
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', flow2rgb(flow[i]), step, dataformats='HWC')
                    writer.add_image(str(i) + '/metric', metric[i], step, dataformats='HWC')
                writer.flush()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_lpips:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, loss_lpips))
            step += 1
        nr_eval += 1
        model.save_model(log_path, local_rank)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=44, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model()
    model.load_model('train_log')
    train(model, args.local_rank)
    