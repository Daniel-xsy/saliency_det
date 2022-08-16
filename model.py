
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.encoder_1 = ResNetBlock(in_chans=3, out_chans=64)
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



    