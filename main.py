import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

from dataset import DEFAULT_PATH, MDFADataset, SirstDataset
from model import CAN, Discriminator
from engine import train_one_epoch, evaluate
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(description='Model Traning')

    # Experiment Name
    parser.add_argument('--exp_name', type=str, default='test', metavar='N', 
                        help='experiment name')
    
    # Dataset
    parser.add_argument('--test-dataset', type=str, default='MDFA', choices=['MDFA', 'Sirst'])
    parser.add_argument('--root', type=str, default='')
    
    # Train Setting
    parser.add_argument('--input-size', type=int, default=112) 
    parser.add_argument('--emb-dim', type=int, default=512)  
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--loss_alpha1', type=float, default=10)
    parser.add_argument('--loss_alpha2', type=float, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-4, metavar='N', help='weight decay')
    parser.add_argument('--eval', action='store_true', default=False, help='Only for evaluation')
   
    return parser
    

def main(args):

    transforms = tfs.Compose([
        tfs.Resize((args.input_size, args.input_size)),
        tfs.CenterCrop((args.input_size, args.input_size)),
        tfs.Grayscale(num_output_channels=1),
        tfs.ToTensor(),
        tfs.Normalize(mean=0.5, std=0.5)
    ])
    target_transforms = tfs.Compose([
        tfs.Resize((args.input_size, args.input_size)),
        tfs.CenterCrop((args.input_size, args.input_size)),
        tfs.Grayscale(num_output_channels=1),
        tfs.ToTensor()
    ])

    root = DEFAULT_PATH['mdfa_train']
    train_dataset = MDFADataset(root=root, split='train', transform=transforms, target_transform=target_transforms)

    if args.test_dataset == 'MDFA':
        root = DEFAULT_PATH['mdfa_test']
        test_dataset = MDFADataset(root=root, split='test', transform=transforms, target_transform=target_transforms)
    elif args.test_dataset == 'Sirst':
        root = DEFAULT_PATH['sirst']
        test_dataset = SirstDataset(root=root, transform=transforms, target_transform=target_transforms)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    G1 = CAN(dilation_factors=[1,2,4,8,4,2,1], in_chans=1, emb_dims=128)
    G2 = CAN(dilation_factors=[1,2,4,8,16,8,4,2,1], in_chans=1, emb_dims=128)
    D = Discriminator(emb_dims=128)

    G1 = G1.to(device)
    G2 = G2.to(device)
    D = D.to(device)

    if args.eval:
        eval_stat = evaluate(testloader, G1, G2, device)
        print(eval_stat)
        exit(0)

    optimizer_G1 = optim.Adam(G1.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer_G2 = optim.Adam(G2.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, weight_decay=args.wd)

    # scheduler = utils.cosine_lr(optimizer, args.lr, args.warmup, args.epochs)

    con_loss = utils.ConsistencyLoss()
    data_loss = utils.DataLoss(alpha=args.alpha)

    for epoch in range(args.epochs):

        G1.train()
        G2.train()
        D.train()

        train_loss = utils.AverageMeter()

        for batch_id, data in tqdm(enumerate(trainloader), total=len(trainloader)):

            img, target = data
            batchsize = img.size(0)

            img = img.to(device)
            target = target.to(device)
        
            fake1 = G1(img)
            fake2 = G2(img)
            logits1, feat1 = D(fake1)
            logits2, feat2 = D(fake2)
            logits_gt, _ = D(target)

            ## Train Discriminator
            optimizer_D.zero_grad()

            loss_D = torch.mean(logits1) + torch.mean(logits2) - torch.mean(logits_gt)
            loss_D.backward()
            optimizer_D.step()

            if batch_id % args.n_critic == 0:
                ## Train Generator
                optimizer_G1.zero_grad()
                optimizer_G2.zero_grad()

                loss_G = -(torch.mean(logits1) + torch.mean(logits2)) + args.loss_alpha1 * con_loss(feat1, feat2) + \
                           args.loss_alpha2 * data_loss(fake1, feat2, target)

                loss_G.backward()
                optimizer_G1.step()
                optimizer_G2.step()

        #eval_stat = evaluate(testloader, G1, G2, device)
        #log_stat = dict(train_stat.items() + eval_stat.items())
        #print(log_stat)

        #if (epoch + 1) == args.epochs:
        #    torch.save(model, './checkpoint.pt')
        



if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)