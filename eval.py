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
    
    # Eval Setting
    parser.add_argument('--emb-dim', type=int, default=512)  
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model1_ckp', type=str, default='checkpoints/G1_29.pt')
    parser.add_argument('--model2_ckp', type=str, default='checkpoints/G2_29.pt')
   
    return parser
    

def main(args):

    transforms = tfs.Compose([
        tfs.Grayscale(num_output_channels=1),
        tfs.ToTensor(),
        tfs.Normalize(mean=0.5, std=0.5)
    ])
    target_transforms = tfs.Compose([
        tfs.Grayscale(num_output_channels=1),
        tfs.ToTensor()
    ])


    if args.test_dataset == 'MDFA':
        root = DEFAULT_PATH['mdfa_test']
        test_dataset = MDFADataset(root=root, split='test', transform=transforms, target_transform=target_transforms)
    elif args.test_dataset == 'Sirst':
        root = DEFAULT_PATH['sirst']
        test_dataset = SirstDataset(root=root, transform=transforms, target_transform=target_transforms)

    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model1 = torch.load(args.model1_ckp)
    model2 = torch.load(args.model2_ckp)
    
    model1 = model1.module
    model2 = model2.module
    
    model1 = model1.to(device)
    model2 = model2.to(device)

    eval_stat = evaluate(testloader, model1, model1, device, args)
    print(eval_stat)



if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)