from torch import nn
from torch import randn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()
        self.linear0 = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.linear1 = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.linear0(x)
        out = self.linear1(out)
        if self.downsample: residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_classes = 10):
        super(ResNet, self).__init__()
        self.max_dimensions = 64
        self.LAYER0 = self._make_layer(block,  64, 3)
        self.LAYER1 = self._make_layer(block, 128, 3)
        self.LAYER2 = self._make_layer(block, 256, 3)
        self.LAYER3 = self._make_layer(block, 512, 3)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_layers):
        downsample = None
        if self.max_dimensions != planes: downsample = nn.Linear(
                in_features=self.max_dimensions, out_features=planes
                )
        layers = []
        layers.append(block(self.max_dimensions, planes, downsample))
        self.max_dimensions = planes
        for i in range(1, num_layers):
            layers.append(block(self.max_dimensions, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.LAYER0(x)
        x = self.LAYER1(x)
        x = self.LAYER2(x)
        x = self.LAYER3(x)
        x = self.fc(x)
        return x

in_channels = 32
out_channels = 16
r = ResidualBlock(in_channels=in_channels, out_channels=out_channels)
x = randn(size=(1,32))
linear0 = nn.Linear(in_features=in_channels, out_features=out_channels)
linear1 = nn.Linear(in_features=out_channels, out_features=out_channels)
residual = x
out = linear0(x)
out = linear1(out)
out += residual
