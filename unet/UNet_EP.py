import torch
import torch.nn as nn
import torch.nn.functional as F

#Build out the Convolutions + Batch Norm + ReLU class
#Used during the Contracting Path
class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#Build out the Contracting Path
#2 implementations of a 3x3 convolution followed by a 2x2 max pooling operation
#  with stride 2 for downsampling
class ContractingPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContractingPath, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace

#Building out the Expansive Path
#Upsampling of the feature map followed by a 2x2 "up-convolution" (we will use
#  transposed convolutions for this), a concatenation with the corresponding
#  cropped feature map from the Contracting Path, and 2 3x3 convolutions
#  each followed by ReLU
#TODO: not totally sure what stride for upconvr should be
#TODO: Not entirely sure we need upsampling and Transposed Convolution,
#  we may only just need one of them
class ExpandingPath(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size):
        super(ExpandingPath, self).__init__()

        self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2, 2), mode="bilinear")
        self.upconvr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = (2,2), stride = 2, padding = 0)
        # Crop + concat step between these 2
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)

    def _crop_concat(self, upsampled, bypass):
        """
         Crop y to the (h, w) of x and concat them.
         Used for the expansive path.
        Returns:
            The concatenated tensor
        """
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))

        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        x = self.upSample(x)
        x = self.upconvr(x)
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)
        return x

#UNet
class UNetOriginal(nn.Module):
    def __init__(self, in_shape):
        super(UNetOriginal, self).__init__()
        channels, height, width = in_shape

        self.down1 = ContractingPath(channels, 64)
        self.down2 = ContractingPath(64, 128)
        self.down3 = ContractingPath(128, 256)
        self.down4 = ContractingPath(256, 512)

        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=0),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=0)
        )

        self.up1 = ExpandingPath(in_channels=1024, out_channels=512, upsample_size=(56, 56))
        self.up2 = ExpandingPath(in_channels=512, out_channels=256, upsample_size=(104, 104))
        self.up3 = ExpandingPath(in_channels=256, out_channels=128, upsample_size=(200, 200))
        self.up4 = ExpandingPath(in_channels=128, out_channels=64, upsample_size=(392, 392))

        # 1x1 convolution at the last layer
        # Different from the paper is the output size here
        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)

    def forward(self, x):
        x, x_trace1 = self.down1(x)  # Calls the forward() method of each layer
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x = self.center(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)
        out = torch.squeeze(out, dim=1)
        return out
