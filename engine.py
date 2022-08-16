import time
from tqdm import tqdm
from typing import Iterable

import torch

import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, update_freq=None):
    model.train()

    train_loss = utils.AverageMeter()

    for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader)):

        img, target = data
        batchsize = img.size(0)

        img = img.to(device)
        target = target.to(device)

        output = model(img)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), batchsize)

    lr = optimizer.state_dict()['param_groups'][0]['lr']
    return {'train_loss': train_loss.avg, 'epoch': epoch, 'lr': lr}



@torch.no_grad()
def evaluate(data_loader, model, device):

    model.eval()
    f1_score = utils.AverageMeter()

    for batch_id, data in enumerate(data_loader):

        img, target = data
        batchsize = img.size(0)

        img = img.to(device)
        target = target.to(device)

        output = model(img)
        f1_score.update(utils.calculateF1Measure(output, target), batchsize)
        

    return {'F1_score': f1_score}