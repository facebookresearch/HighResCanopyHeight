# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torchvision

class RNet(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=13,
        n_pix=256,
        filters=(8, 16, 32, 64, 64, 128),
        pool=(2, 2),
        kernel_size=(3, 3),
        n_meta=0,
    ) -> None:
        super(RNet, self).__init__()

        def conv_block(in_filters, out_filters, kernel_size):
            layers = nn.Sequential(
                # first conv is across channels, size=1
                nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), padding="same"),
                nn.BatchNorm2d(out_filters),
                nn.ReLU(),
                nn.Conv2d(
                    out_filters, out_filters, kernel_size=kernel_size, padding="same"
                ),
            )
            return layers

        def fc_block(in_features, out_features):
            layers = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                #nn.BatchNorm1d(out_features),
                #nn.InstanceNorm1d(out_features),
                nn.ReLU(),
            )
            return layers

        self.pool = nn.MaxPool2d(2, 2)
        self.input_layer = conv_block(n_channels, filters[0], kernel_size)
        self.conv_block1 = conv_block(filters[0], filters[1], kernel_size)
        self.conv_block2 = conv_block(filters[1], filters[2], kernel_size)
        self.conv_block3 = conv_block(filters[2], filters[3], kernel_size)
        self.conv_block4 = conv_block(filters[3], filters[4], kernel_size)
        self.conv_block5 = conv_block(filters[4], filters[5], kernel_size)
        n_pool = 5
        self.fc1 = fc_block(in_features= int(filters[5] * (n_pix / 2**n_pool) ** 2), out_features=64)
        self.fc2 = fc_block(in_features=64 + n_meta, out_features=64)
        self.fc3 = fc_block(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=n_classes)

    def forward(self, x):
        x1 = self.pool(self.input_layer(x))
        x2 = self.pool(self.conv_block1(x1))
        x3 = self.pool(self.conv_block2(x2))
        x4 = self.pool(self.conv_block3(x3))
        x4b = self.pool(self.conv_block4(x4))
        x5 = self.conv_block5(x4b)
        x6 = torch.flatten(x5, 1)  # flatten all dimensions except batch
        x7 = self.fc1(x6)
        x9 = self.fc2(x7)
        x10 = self.fc3(x9)
        x11 = self.fc4(x10)
        return x11