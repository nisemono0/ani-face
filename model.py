import torch
import torch.nn as nn

import config.config as cfg


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.arch = cfg.ARCH_CONFIG
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.arch)
        self.full_conn = self._create_full_conn_layers(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.full_conn(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels

        for a in arch:
            if type(a) == tuple:
                layers.append(
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=a[1],
                        kernel_size=a[0],
                        stride=a[2],
                        padding=a[3]
                    )
                )
                in_channels = a[1]
            elif type(a) == str:
                layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
            elif type(a) == list:
                conv1 = a[0]
                conv2 = a[1]
                repeats = a[2]

                for _ in range(repeats):
                    layers.append(
                        CNNBlock(
                            in_channels=in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    )
                    layers.append(
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    )
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_full_conn_layers(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        feature_size = 4096  # In the paper this is 4096
        dropout_rate = 0.5  # In the paper this is 0.5
        relu_slope = 0.1  # In the paper this is 0.1
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, feature_size),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(relu_slope),
            nn.Linear(feature_size, S * S * (C + B * 5)) # Will be later reshaped to (S, S, C+B*5): in this case (7, 7, 11) i think
        )


# # Simple sanity check test function to see if model is structured correctly
# def test(S=7, B=2, C=1):
#     model = YOLOv1(split_size=S, num_boxes=B, num_classes=C)
#     x = torch.randn((2, 3, 448, 448))
#     print(model(x).shape)
#     print(S*S*(C+B*5))
#     print(model(x).reshape(-1, S, S, C+B*5).shape)
# test()
