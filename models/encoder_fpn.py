import torch.nn as nn
from collections import OrderedDict

class Encoder_FPN(nn.Module):
    def __init__(self, encoder, fpn, levels):
        super().__init__()

        self.encoder = encoder
        self.fpn = fpn
        self.levels = levels

    def forward(self, x):
        feature_maps = self.encoder(x)

        x = OrderedDict()

        for level in self.levels:
            x[str(level)] = feature_maps[level]

        feature_maps_fpn = self.fpn(x)

        return [feature_maps_fpn[k] for k,v in feature_maps_fpn.items()]

import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, residual_kaiming, dropout):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.InstanceNorm2d(output_dim))

        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.InstanceNorm2d(output_dim))

        if residual_kaiming:
            self.residual = nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1))
            self.relu = nn.LeakyReLU(0.01)

        if dropout:
            self.dropout_layer = nn.Dropout(0.2)

        self.residual_kaiming = residual_kaiming
        self.dropout = dropout


    def forward(self, x):
        if self.residual_kaiming:
            res = self.residual(x)
        x = self.conv1(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.conv2(x)
        if self.dropout:
            x = self.dropout_layer(x)
        if self.residual_kaiming:
            x += res
            x = self.relu(x)
        return x

class SimpleEncoderDeep(nn.Module):
    def __init__(self, input_dim, feat=(32, 64, 128, 256, 512, 512, 512), residual_kaiming=False, dropout=False):
        super().__init__()

        self.enc_feat = [input_dim] + list(feat)

        self.encoder = nn.ModuleList(
            [EncoderBlock(self.enc_feat[i], self.enc_feat[i + 1], residual_kaiming, dropout) for i in range(len(self.enc_feat) - 1)])

        self.downsample = nn.MaxPool2d(2)

        if residual_kaiming:
            print('Using (kaiming) residuals')
        if dropout:
            print('Using dropout in CNN')

    def forward(self, x):

        residuals = []
        for i, enc_level in enumerate(self.encoder):
            x = enc_level(x)
            residuals.append(x)
            if i != 0: # no downsample after first block
                x = self.downsample(x)
        residuals.append(x)
        #print([r.shape for r in residuals])

        return residuals




