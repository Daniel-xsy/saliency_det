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
from model import CAN, Discriminator, AttentionBlock
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
    parser.add_argument('--attn-dim', type=int, default=256)  
    parser.add_argument('--out-dim', type=int, default=64)  
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--clip_value', type=float, default=0.5)
    parser.add_argument('--loss_alpha1', type=float, default=100)
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
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    attn1 = AttentionBlock(args.attn_dim, args.out_dim)
    attn2 = AttentionBlock(args.attn_dim, args.out_dim)

    G1 = CAN(attention=attn1, dilation_factors=[1,2,4,8,4,2,1], attn_dims=args.out_dim, in_chans=1, emb_dims=args.emb_dim)
    G2 = CAN(attention=attn2, dilation_factors=[1,2,4,8,16,8,4,2,1], attn_dims=args.out_dim, in_chans=1, emb_dims=args.emb_dim)
    D = Discriminator(emb_dims=128)
    
    G1 = torch.nn.DataParallel(G1)
    G2 = torch.nn.DataParallel(G2)
    D = torch.nn.DataParallel(D)

    G1 = G1.to(device)
    G2 = G2.to(device)
    D = D.to(device)

    if args.eval:
        eval_stat = evaluate(testloader, G1, G2, device, args)
        print(eval_stat)
        exit(0)

    optimizer_G1 = optim.Adam(G1.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer_G2 = optim.Adam(G2.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr * 0.01)

    scheduler = utils.cosine_lr(optimizer_G1, args.lr, args.warmup, args.epochs)

    for epoch in range(args.epochs):

        scheduler(epoch)

        log_stat = train_one_epoch(G1, G2, D, trainloader, optimizer_D, optimizer_G1, optimizer_G2,device, epoch, args)
        print(log_stat)
        eval_stat = evaluate(testloader, G1, G2, device, args)
        print(eval_stat)

        if (epoch+1) % 5 == 0:
            torch.save(G1, './checkpoints/G1_%i.pt'%epoch)
            torch.save(G2, './checkpoints/G2_%i.pt'%epoch)



if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)