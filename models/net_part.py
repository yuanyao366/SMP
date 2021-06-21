from torch import nn
from models import r3d,r21d

class conv3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(conv3d,self).__init__();
        self.conv3d =nn.Sequential(
            nn.Conv3d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),

        )

    def forward(self, x):
        x = self.conv3d(x)
        return x


class down3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(down3d,self).__init__();
        self.mpcon3d= nn.Sequential(
            conv3d(in_ch,out_ch),
            nn.MaxPool3d(kernel_size=2,stride=2)
        )
    def forward(self, x):
        x = self.mpcon3d(x)
        return x
class up3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(up3d,self).__init__()

        self.up3d = nn.ConvTranspose3d(in_ch,in_ch,2,stride=2)
        #self.up3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv3d = conv3d(in_ch,out_ch);

    def forward(self, x):
        x = self.up3d(x)
        x = self.conv3d(x)

        return x;
class inc(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(inc,self).__init__()
        self.conv = nn.Conv3d(in_ch,out_ch,kernel_size=3,padding=1)

class out(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(out,self).__init__()
        self.conv = nn.Conv3d(in_ch,out_ch,1);
    def forward(self,x):
        x = self.conv(x)

        return x

    
class res3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(res3d, self).__init__()
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride = 2 the input x
            self.downsampleconv = r3d.SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2 when producing the residual
            self.conv1 = r3d.SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            if in_channels != out_channels:
                self.downsampleconv = r3d.SpatioTemporalConv(in_channels, out_channels, 1, stride=1)
                self.downsamplebn = nn.BatchNorm3d(out_channels)
                self.downsample = True
            self.conv1 = r3d.SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        # self.relu1 = nn.ReLU()
        #
        # # standard conv->batchnorm->ReLU
        # self.conv2 = r3d.SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        # self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        # res = self.relu1(self.bn1(self.conv1(x)))
        # res = self.bn2(self.conv2(res))

        res = self.bn1(self.conv1(x))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)

class upr3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(upr3d,self).__init__()

        self.upr3d = nn.ConvTranspose3d(in_ch,in_ch,2,stride=2)
        #self.up3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convr3d = res3d(in_ch,out_ch,3)

    def forward(self, x):
        x = self.upr3d(x)
        x = self.convr3d(x)

        return x


class res21d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(res21d, self).__init__()
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride = 2 the input x
            self.downsampleconv = r21d.SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2 when producing the residual
            self.conv1 = r21d.SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            if in_channels != out_channels:
                self.downsampleconv = r21d.SpatioTemporalConv(in_channels, out_channels, 1, stride=1)
                self.downsamplebn = nn.BatchNorm3d(out_channels)
                self.downsample = True
            self.conv1 = r21d.SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        # self.relu1 = nn.ReLU()
        #
        # # standard conv->batchnorm->ReLU
        # self.conv2 = r21d.SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        # self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        # res = self.relu1(self.bn1(self.conv1(x)))
        # res = self.bn2(self.conv2(res))

        res = self.bn1(self.conv1(x))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)

class upr21d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(upr21d,self).__init__()

        self.upr3d = nn.ConvTranspose3d(in_ch,in_ch,2,stride=2)
        #self.up3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convr3d = res21d(in_ch,out_ch,3)

    def forward(self, x):
        x = self.upr3d(x)
        x = self.convr3d(x)

        return x