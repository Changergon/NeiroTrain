import torch.nn as nn
from models.cbam import CBAM
from models.dropblock import DropBlock2D


def conv_block(in_channels, out_channels, use_batchnorm=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ]
    if use_batchnorm:
        # Вставляем BatchNorm после каждого Conv2d
        layers.insert(1, nn.BatchNorm2d(out_channels))
        layers.insert(4, nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes, use_cbam=False, cbam_ratio=16, cbam_kernel_size=7,
                 use_dropblock=False, drop_prob=0.1, block_size=5, use_batchnorm=False):
        super().__init__()

        self.use_cbam = use_cbam
        self.use_dropblock = use_dropblock
        self.use_batchnorm = use_batchnorm

        # Блоки сверток
        self.block1 = conv_block(3, 64, use_batchnorm)
        self.block2 = conv_block(64, 128, use_batchnorm)
        self.block3 = conv_block(128, 256, use_batchnorm)
        self.block4 = conv_block(256, 512, use_batchnorm)
        self.block5 = conv_block(512, 1024, use_batchnorm)
        self.block6 = conv_block(1024, 2048, use_batchnorm)

        # CBAM модули
        if use_cbam:
            self.cbam1 = CBAM(64, ratio=cbam_ratio, kernel_size=cbam_kernel_size)
            self.cbam2 = CBAM(128, ratio=cbam_ratio, kernel_size=cbam_kernel_size)
            self.cbam3 = CBAM(256, ratio=cbam_ratio, kernel_size=cbam_kernel_size)
            self.cbam4 = CBAM(512, ratio=cbam_ratio, kernel_size=cbam_kernel_size)
            self.cbam5 = CBAM(1024, ratio=cbam_ratio, kernel_size=cbam_kernel_size)
            self.cbam6 = CBAM(2048, ratio=cbam_ratio, kernel_size=cbam_kernel_size)

        # DropBlock слои
        if use_dropblock:
            self.drop1 = DropBlock2D(drop_prob=drop_prob, block_size=block_size)
            self.drop2 = DropBlock2D(drop_prob=drop_prob, block_size=block_size)
            self.drop3 = DropBlock2D(drop_prob=drop_prob, block_size=block_size)
            self.drop4 = DropBlock2D(drop_prob=drop_prob, block_size=block_size)
            self.drop5 = DropBlock2D(drop_prob=drop_prob, block_size=block_size)
            self.drop6 = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.apply_cbam_and_drop(x, 64)

        x = self.block2(x)
        x = self.apply_cbam_and_drop(x, 128)

        x = self.block3(x)
        x = self.apply_cbam_and_drop(x, 256)

        x = self.block4(x)
        x = self.apply_cbam_and_drop(x, 512)

        x = self.block5(x)
        x = self.apply_cbam_and_drop(x, 1024)

        x = self.block6(x)
        x = self.apply_cbam_and_drop(x, 2048)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def apply_cbam_and_drop(self, x, channels):
        # Применяем CBAM если он включен
        if self.use_cbam:
            if channels == 64:
                x = self.cbam1(x)
            elif channels == 128:
                x = self.cbam2(x)
            elif channels == 256:
                x = self.cbam3(x)
            elif channels == 512:
                x = self.cbam4(x)
            elif channels == 1024:
                x = self.cbam5(x)
            elif channels == 2048:
                x = self.cbam6(x)

        # Применяем DropBlock если он включен и мы в режиме обучения
        if self.use_dropblock and self.training:
            if channels == 64:
                x = self.drop1(x)
            elif channels == 128:
                x = self.drop2(x)
            elif channels == 256:
                x = self.drop3(x)
            elif channels == 512:
                x = self.drop4(x)
            elif channels == 1024:
                x = self.drop5(x)
            elif channels == 2048:
                x = self.drop6(x)

        return x