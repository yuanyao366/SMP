import torch
import torch.nn as nn
from models.c3d import C3D
from models.net_part import *

feature_sizes = {
    'c3d':{'c1':64, 'c2':128, 'c3':256, 'c4':512, 'c5':512},
    'r3d':{'c1':64, 'c2':64, 'c3':128, 'c4':256, 'c5':512},
    'r21d':{'c1':64, 'c2':64, 'c3':128, 'c4':256, 'c5':512},
}


class SSCN_OneClip(nn.Module):
    def __init__(self, model_name, base_network, with_classifier=False, num_classes=4, with_ClsEncoder=None):

        super(SSCN_OneClip, self).__init__()

        self.feature_size = feature_sizes[model_name]
        self.base_network = base_network
        self.with_classifier = with_classifier
        self.num_classes = num_classes
        self.with_ClsEncoder = with_ClsEncoder

        self.up1 = up3d(512, 256)
        self.up2 = up3d(256, 128)
        self.up3 = up3d(128, 64)
        self.up4_1 = nn.ConvTranspose3d(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.outc = conv3d(64, 3)

        if self.with_classifier:
            self.conv5_cls = conv3d(512, 512)
            self.conv5_rec = conv3d(512, 512)
            self.avgpool5 = nn.AdaptiveAvgPool3d(1)
            self.fc8_c5 = nn.Linear(512, self.num_classes)

            if self.with_ClsEncoder is not None:
                if 'c4' in self.with_ClsEncoder:
                    self.conv4_cls = conv3d(self.feature_size['c4'], self.feature_size['c4'])
                    self.avgpool4 = nn.AdaptiveAvgPool3d(1)
                    self.fc8_c4 = nn.Linear(self.feature_size['c4'], self.num_classes)

                if 'c3' in self.with_ClsEncoder:
                    self.conv3_cls = conv3d(self.feature_size['c3'], self.feature_size['c3'])
                    self.avgpool3 = nn.AdaptiveAvgPool3d(1)
                    self.fc8_c3 = nn.Linear(self.feature_size['c3'], self.num_classes)

                if 'c2' in self.with_ClsEncoder:
                    self.conv2_cls = conv3d(self.feature_size['c2'], self.feature_size['c2'])
                    self.avgpool2 = nn.AdaptiveAvgPool3d(1)
                    self.fc8_c2 = nn.Linear(self.feature_size['c2'], self.num_classes)

                if 'c1' in self.with_ClsEncoder:
                    self.conv1_cls = conv3d(self.feature_size['c1'], self.feature_size['c1'])
                    self.avgpool1 = nn.AdaptiveAvgPool3d(1)
                    self.fc8_c1 = nn.Linear(self.feature_size['c1'], self.num_classes)


    def forward(self, x):
        x1, x2, x3, x4, x5 = self.base_network(x)
        h = {}

        if self.with_classifier:
            x_cls = self.conv5_cls(x5)
            h_c5 = self.avgpool5(x_cls)
            h_c5 = h_c5.view(-1, h_c5.size(1))
            h_c5 = self.fc8_c5(h_c5)
            h['c5'] = h_c5
            x = self.conv5_rec(x5)

            if self.with_ClsEncoder is not None:
                if 'c4' in self.with_ClsEncoder:
                    x4_cls = self.conv4_cls(x4)
                    h_c4 = self.avgpool4(x4_cls)
                    h_c4 = h_c4.view(-1, h_c4.size(1))
                    h_c4 = self.fc8_c4(h_c4)
                    h['c4'] = h_c4

                if 'c3' in self.with_ClsEncoder:
                    x3_cls = self.conv3_cls(x3)
                    h_c3 = self.avgpool3(x3_cls)
                    h_c3 = h_c3.view(-1, h_c3.size(1))
                    h_c3 = self.fc8_c3(h_c3)
                    h['c3'] = h_c3

                if 'c2' in self.with_ClsEncoder:
                    x2_cls = self.conv2_cls(x2)
                    h_c2 = self.avgpool2(x2_cls)
                    h_c2 = h_c2.view(-1, h_c2.size(1))
                    h_c2 = self.fc8_c2(h_c2)
                    h['c2'] = h_c2

                if 'c1' in self.with_ClsEncoder:
                    x1_cls = self.conv1_cls(x1)
                    h_c1 = self.avgpool1(x1_cls)
                    h_c1 = h_c1.view(-1, h_c1.size(1))
                    h_c1 = self.fc8_c1(h_c1)
                    h['c1'] = h_c1

        x_r1 = self.up1(x)
        x_r2 = self.up2(x_r1)
        x_r3 = self.up3(x_r2)
        x_r4 = self.up4_1(x_r3)
        x_r4 = self.outc(x_r4)

        if self.with_classifier:
            return x_r4, h, x_cls
        else:
            return x_r4


if __name__ == '__main__':
    base = C3D(with_classifier=False);
    sscn = SSCN_OneClip(base, with_classifier=True, num_classes=4, with_ClsRecon=False, with_ClsCycle=True)

    learning_rate = 0.01
    model_params = []
    for key, value in dict(sscn.named_parameters()).items():
        if value.requires_grad:
            if 'fc8' in key:
                print(key)
                model_params += [{'params': [value], 'lr': 10 * learning_rate}]
            else:
                model_params += [{'params': [value], 'lr': learning_rate}]

    input_tensor = torch.autograd.Variable(torch.rand(4, 3, 16, 112, 112))
    # print(input_tensor)
    out, h, h_r = sscn(input_tensor)
    # m = nn.ConvTranspose3d(16,33,3,stride=2)
    # input_tensor = torch.autograd.Variable(torch.rand(20,16,10,50,100))
    # out = m(input_tensor)
    print(out.shape)
    print(h.shape)
    print(h_r.shape)
    print('aaaa')

# out = out[:,:,16,:,:]

