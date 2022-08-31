'''SENet in PyTorch.
SENet is the winner of ImageNet-2017. The paper is not released yet.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut


        # out=nn.GELU()(self.bn1(x))
        # shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # out = self.conv1(out)
        # #list_size['out_2']=(out.size())

        
        # # out=F.max_pool2d(out,out.size(2))
        # # out=nn.MaxPool2d(2)(out)
        # # list_size['out_maxpool']=(out.size())

        # #shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # #out=nn.Dropout(0.15)(out)
        # #out = self.conv2(F.relu(self.bn2(out)))
        # out = self.conv2(nn.GELU()(self.bn2(out)))
        # #list_size['out_3']=(out.size())

        # # out=nn.ConvTranspose2d(in_channels=out.size(1),out_channels=x.size(1),kernel_size=(x.size(0)-(out.size(2)-1)))(out)
        # # list_size['out_conv_transpose']=(out.size())


        # # Squeeze
        # w = F.avg_pool2d(out, out.size(2))
        # #w = F.relu(self.fc1(w))
        # w = nn.GELU()(self.fc1(w))
        # w = torch.sigmoid(self.fc2(w))
        # # Excitation
        # out = out * w

        #out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, layers, w, initial_image_size, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  int(round(64 * w[0])), layers[0], stride=1)
        self.layer2 = self._make_layer(block, int(round(128 * w[1])), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(round(256 * w[2])), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(round(512 * w[3])), layers[3], stride=2)
        size_fc = self.get_input_size_first_lin_layer(initial_image_size)
        self.linear = nn.Linear(size_fc, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_input_size_first_lin_layer(self, image_size):
        """
        :return: current_size
        """
        # current_size = self.image_size
        dsize = (1, 3, image_size, image_size)
        inputs = torch.randn(dsize)

        out = F.relu(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        current_size = out.shape[1]
        return current_size


def SENet18():
    return SENet(PreActBlock, [2,2,2,2])


def scaled_senet(d, w, initial_image_size):
    # d is a list of depth factors: 1 for each layer
    # w is a also a list of width factors: 1 for each layer
    assert len(d) == 4, 'length of d must be 4'
    assert len(w) == 4, 'length of w must be 4'
    layers = [2, 2, 2, 2]
    layers = [int(round(layers[i] * d[i])) for i in range(len(d))]
    return SENet(PreActBlock, layers, w, initial_image_size)


if __name__ == '__main__':

    net = scaled_senet([1, 2.3, 2, 3], [1.67, 0.25, 1, 1.2], 32)
    print(net)
