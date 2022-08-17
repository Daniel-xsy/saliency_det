
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        # input and output the same size
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_chans)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CANBlock(nn.Module):
    def __init__(self, in_chans, out_chans, dilation):
        super(CANBlock, self).__init__()
        # input and output the same size
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_chans)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CAN(nn.Module):
    def __init__(self, dilation_factors, in_chans, emb_dims):
        super().__init__()
        layers = []
        for i in range(len(dilation_factors)):
            if i == 0:
                layers.append(CANBlock(in_chans, emb_dims, dilation_factors[i]))
            else:
                layers.append(CANBlock(emb_dims, emb_dims, dilation_factors[i]))

        self.blocks = nn.Sequential(*layers)
        self.linear_head = nn.Conv2d(emb_dims, in_chans, kernel_size=1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.linear_head(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, emb_dims=128):
        super().__init__()
        self.pooling1 = nn.AdaptiveMaxPool2d(output_size=128)
        self.pooling2 = nn.AdaptiveMaxPool2d(output_size=64)
        self.conv_block = nn.Sequential(
            ConvBlock(in_chans=1, out_chans=16, kernel_size=3, stride=2, padding=1),  # [16, 32, 32]
            ConvBlock(in_chans=16, out_chans=32, kernel_size=3, stride=2, padding=1), # [32, 16, 16]
            ConvBlock(in_chans=32, out_chans=64, kernel_size=3, stride=2, padding=1), # [64, 8, 8]
            ConvBlock(in_chans=64, out_chans=emb_dims, kernel_size=3, stride=2, padding=1), # [emb_dims, 4, 4]
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4*4*emb_dims, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        x = self.pooling2(self.pooling1(x))
        x = self.conv_block(x)
        x = x.flatten(1,3)
        feat = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x, feat

"""
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(ResNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_chans)

        if not in_chans == out_chans:
            self.shortcut = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)
        x = F.relu(self.bn(self.conv(x)))

        return x + identity


class DownSampling(nn.Module):
    def __init__(self, k=2, s=2):
        super(DownSampling, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=s)
    def forward(self, x):
        return self.maxpool(x)


class UpSampling(nn.Module):
    def __init__(self, scale, in_chans, out_chans):
        super(UpSampling, self).__init__()
        self.scale = scale
        if in_chans == out_chans:
            self.conv = nn.Identity()
        else:
            self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
    def forward(self, x):
        out_size = [size * self.scale for size in x.size()[2:]]
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_size=224, emb_dim=512):
        super(UNet, self).__init__()

        self.position_embedding = nn.Parameter(torch.randn((input_size//8)**2, emb_dim))

        self.encoder_1 = ResNetBlock(in_chans=1, out_chans=64)
        self.encoder_2 = nn.Sequential(
            DownSampling(k=2, s=2),
            ResNetBlock(in_chans=64, out_chans=128))
        self.encoder_3 = nn.Sequential(
            DownSampling(k=2, s=2),
            ResNetBlock(in_chans=128, out_chans=256))
        self.encoder_4 = nn.Sequential(
            DownSampling(k=2, s=2),
            ResNetBlock(in_chans=256, out_chans=emb_dim))

        self.attn = Attention(dim=emb_dim, num_heads=8)
        self.upsampling = UpSampling(scale=2, in_chans=emb_dim, out_chans=256)

        self.decoder_1 = nn.Sequential(
            ResNetBlock(in_chans=256*2, out_chans=256),
            UpSampling(scale=2, in_chans=256, out_chans=128))
        self.decoder_2 = nn.Sequential(
            ResNetBlock(in_chans=128*2, out_chans=128),
            UpSampling(scale=2, in_chans=128, out_chans=64))
        self.decoder_3 = nn.Sequential(
            ResNetBlock(in_chans=64*2, out_chans=64))
        self.decoder_4 = ResNetBlock(in_chans=64, out_chans=1)

    def forward(self, x):

        feat_1 = self.encoder_1(x)  # orginal size
        feat_2 = self.encoder_2(feat_1)  # 1 / 2 size
        feat_3 = self.encoder_3(feat_2)  # 1 / 4 size
        feat_4 = self.encoder_4(feat_3)  # 1 / 8 size

        B, C, H, W = feat_4.size()
        x = feat_4.flatten(2, 3).permute(0, 2, 1)
        x = self.attn(x + self.position_embedding)
        x = x.reshape(B, H, W, C).permute(0, 3, 2, 1)

        x = self.upsampling(x)

        x = torch.cat((x, feat_3), dim=1)
        x = self.decoder_1(x)
        x = torch.cat((x, feat_2), dim=1)
        x = self.decoder_2(x)
        x = torch.cat((x, feat_1), dim=1)
        x = self.decoder_3(x)
        x = self.decoder_4(x)

        return x
"""