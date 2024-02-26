# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import models.vgg_ as models

# class BackboneBase_VGG(nn.Module):
#     def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
#         super().__init__()
#         features = list(backbone.features.children())
#         if return_interm_layers:
#             if name == 'vgg16_bn':
#                 self.body1 = nn.Sequential(*features[:13])
#                 self.body2 = nn.Sequential(*features[13:23])
#                 self.body3 = nn.Sequential(*features[23:33])
#                 self.body4 = nn.Sequential(*features[33:43])
#             else:
#                 self.body1 = nn.Sequential(*features[:9])
#                 self.body2 = nn.Sequential(*features[9:16])
#                 self.body3 = nn.Sequential(*features[16:23])
#                 self.body4 = nn.Sequential(*features[23:30])
#         else:
#             if name == 'vgg16_bn':
#                 self.body = nn.Sequential(*features[:44])  # 16x down-sample
#             elif name == 'vgg16':
#                 self.body = nn.Sequential(*features[:30])  # 16x down-sample

class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg19_bn':
                # VGG19_bn has more convolutional layers; adjust indices accordingly
                self.body1 = nn.Sequential(*features[:16])  # Up to the first max pooling
                self.body2 = nn.Sequential(*features[16:27])  # Up to the second max pooling
                self.body3 = nn.Sequential(*features[27:40])  # Up to the third max pooling
                self.body4 = nn.Sequential(*features[40:53])  # Up to the fourth max pooling
            else:
                # For VGG19 without batch normalization (not requested, but for completeness)
                self.body1 = nn.Sequential(*features[:4])
                self.body2 = nn.Sequential(*features[4:9])
                self.body3 = nn.Sequential(*features[9:14])
                self.body4 = nn.Sequential(*features[14:18])
        else:
            if name == 'vgg19_bn':
                # Include everything up to the fourth max pooling for x16 down-sample
                self.body = nn.Sequential(*features[:54])  # Adjusted for VGG19_bn
            elif name == 'vgg19':
                # For VGG19 without batch normalization (not requested, but for completeness)
                self.body = nn.Sequential(*features[:18])
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        out = []

        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)

        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, return_interm_layers: bool):
        if name == 'vgg16_bn':
            backbone = models.vgg16_bn(pretrained=True)
        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=True)
        elif name == 'vgg19_bn':
            backbone = models.vgg19_bn(pretrained=True)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


def build_backbone(args):
    backbone = Backbone_VGG(args.backbone, True)
    return backbone

if __name__ == '__main__':
    Backbone_VGG('vgg19_bn', True)