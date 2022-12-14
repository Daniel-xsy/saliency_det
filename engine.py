import time
from tqdm import tqdm
from typing import Iterable

import torch
import torch.nn.functional as F

import utils


def train_one_epoch(G1: torch.nn.Module, G2: torch.nn.Module, D: torch.nn.Module,
                    data_loader: Iterable, optimizer_D: torch.optim.Optimizer, 
                    optimizer_G1: torch.optim.Optimizer, optimizer_G2: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args):
    G1.train()
    G2.train()
    D.train()

    loss_D_all = utils.AverageMeter()
    loss_G_all = utils.AverageMeter()
    loss_con = utils.AverageMeter()
    loss_data = utils.AverageMeter() 
    loss_totol = utils.AverageMeter()
    
    con_loss = utils.ConsistencyLoss()
    data_loss = utils.DataLoss(alpha=args.alpha)

    for batch_id, data in enumerate(data_loader):

        img, target = data
        batchsize = img.size(0)

        img = img.to(device)
        target = target.to(device)

        ## Train Generator
        optimizer_G1.zero_grad()
        optimizer_G2.zero_grad()

        fake1 = G1(img)
        fake2 = G2(img)
        logits1, feat1 = D(fake1)
        logits2, feat2 = D(fake2)
        logits_gt, _ = D(target)

        loss_G = args.loss_alpha2 * torch.mean(1 - torch.log(logits1) + 1 - torch.log(logits2))
        loss_c = con_loss(feat1, feat2)
        loss_d = args.loss_alpha1 * data_loss(fake1, fake2, target)

        loss_all = loss_G + loss_c + loss_d

        loss_G_all.update(loss_G.item(), batchsize/args.batch_size)
        loss_con.update(loss_c.item(), batchsize/args.batch_size)
        loss_data.update(loss_d.item(), batchsize/args.batch_size)
        loss_totol.update(loss_all.item(), batchsize/args.batch_size)

        loss_all.backward()
                    
        optimizer_G1.step()
        optimizer_G2.step()

        if batch_id % args.n_critic == 0:
            ## Train Discriminator
            optimizer_D.zero_grad()
            
            logits1, feat1 = D(G1(img))
            logits2, feat2 = D(G2(img))
            logits_gt, _ = D(target)
            
            loss_D = torch.mean(torch.log(logits1) + torch.log(logits2) - torch.log(logits_gt))
            loss_D_all.update(loss_D.item(), batchsize/args.batch_size)

            loss_D.backward()
            optimizer_D.step()

            
        if batch_id % 20 == 0:
            f1_score, _, _ = utils.calculateF1Measure((fake2 + fake1 / 2), target, 0.5)
            print('[Iteration %i/%i] loss_G: %f loss_D: %f loss_con: %f loss_data: %f loss_total: %f f1 score %f' % 
                  (batch_id, len(data_loader), loss_G_all.avg, loss_D_all.avg, loss_con.avg, loss_data.avg, loss_totol.avg, f1_score))

    lr = optimizer_G1.state_dict()['param_groups'][0]['lr']
    
    return {'loss_D': loss_D_all.avg, 'loss_G': loss_G_all.avg, 'loss_con': loss_con.avg,
            'loss_data': loss_data.avg, 'loss_totol': loss_totol.avg, 'epoch': epoch, 'lr': lr}



@torch.no_grad()
def evaluate(data_loader, model1, model2,  device, args):

    model1.eval()
    model2.eval()
    f1_score = utils.AverageMeter()
    recall = utils.AverageMeter()
    prec = utils.AverageMeter()

    for batch_id, data in enumerate(data_loader):

        img, target = data
        batchsize = img.size(0)

        img = img.to(device)
        target = target.to(device)

        output1 = model1(img)
        output2 = model2(img)
        output = (output1 + output2) / 2
        f1_score_, recall_, prec_ = utils.calculateF1Measure(output, target, 0.5)
        f1_score.update(f1_score_, batchsize/args.batch_size)
        recall.update(recall_, batchsize/args.batch_size)
        prec.update(prec_, batchsize/args.batch_size)
        
        # print('[%i/%i] average: f1_score: %f' % (batch_id, len(data_loader), f1_score.avg))
        

    return {'F1_score': f1_score.avg, 'recall': recall.avg, 'precision': prec.avg}