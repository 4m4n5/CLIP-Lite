from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


#########################################
# ===== Classifiers ===== #
#########################################

class LinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=10):
        super(LinearClassifier, self).__init__()

        self.net = nn.Linear(dim_in, n_label)

    def forward(self, x):
        return self.net(x)


class NonLinearClassifier(nn.Module):

    def __init__(self, dim_in, n_label=10, p=0.1):
        super(NonLinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 200),
            nn.Dropout(p=p),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, n_label),
        )

    def forward(self, x):
        return self.net(x)


class Conv4(nn.Module):
    def __init__(self, num_classes=100):
        super(Conv4, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.aap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, is_feat=False):
        f1 = F.relu(self.bn1(self.conv1(x)))
        f2 = F.relu(self.bn2(self.conv2(f1)))
        f3 = F.relu(self.bn3(self.conv3(f2)))
        f4 = F.relu(self.bn4(self.conv4(f3)))

        x = self.aap(f4)
        out = x.view(x.size(0), -1)

        return [f1, f2, f3, f4, out], out


class Conv4MP(nn.Module):
    def __init__(self, num_classes=100):
        super(Conv4MP, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.mp = nn.MaxPool2d(2)
        self.aap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, is_feat=False):
        f1 = F.relu(self.bn1(self.conv1(x)))
        x = self.mp(f1)

        f2 = F.relu(self.bn2(self.conv2(x)))
        x = self.mp(f2)

        f3 = F.relu(self.bn3(self.conv3(x)))
        x = self.mp(f3)

        f4 = F.relu(self.bn4(self.conv4(x)))
        x = self.mp(f4)

        x = self.aap(x)
        out = x.view(x.size(0), -1)

        return [f1, f2, f3, f4, out], out
