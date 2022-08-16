import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

from dataset import DEFAULT_PATH, MDFADataset, SirstDataset
from model import UNet
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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-4, metavar='N', help='weight decay')
    parser.add_argument('--eval', action='store_true', default=False, help='Only for evaluation')
   
    return parser


def main(args):

    transforms = tfs.Compose([
        tfs.Resize((args.input_size, args.input_size)),
        tfs.CenterCrop((args.input_size, args.input_size)),
        tfs.ToTensor(),
        tfs.Normalize(mean=0.5, std=0.5)
    ])
    target_transforms = tfs.Compose([
        tfs.Resize((args.input_size, args.input_size)),
        tfs.CenterCrop((args.input_size, args.input_size)),
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

    model = UNet(input_size=args.input_size, emb_dim=args.emb_dim)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = utils.cosine_lr(optimizer, args.lr, args.warmup, args.epochs)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):

        scheduler(epoch)
        train_stat = train_one_epoch(model, criterion, trainloader, optimizer, device, epoch)
        eval_stat = evaluate(testloader, model, device)
        log_stat = dict(train_stat.items() + eval_stat.items())
        print(log_stat)

        if (epoch + 1) == args.epochs:
            torch.save(model, './checkpoint.pt')
        



if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)