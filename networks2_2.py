import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

__all__ = ['DenseNet', 'DAGNet']


model_urls = {
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}


def DAGNet(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
                key = new_key
            if key[:20] == 'features.denseblock1':
                new1_key = 'DB1.denseblock1' + key[20:]
                new2_key = 'DB_att.denseblock1' + key[20:]
                state_dict[new1_key] = state_dict[key]
                state_dict[new2_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class in_attention(nn.Module):

    def __init__(self, in_c):
        super(in_attention, self).__init__()
        self.K1 = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.T1 = nn.ConvTranspose2d(in_c, in_c//2, kernel_size=2, stride=2, padding=0)
        self.K2 = nn.Conv2d(in_c//2, in_c//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.A = nn.Conv2d(in_c//2, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, d_ft, x):
        x_k1 = F.relu(self.K1(d_ft))
        x_t1 = self.T1(x_k1)
        x_k2 = F.relu(self.K2(x_t1))
        att_f = F.sigmoid(self.A(x_k2))
        att_f = F.interpolate(att_f, [x.size(-2), x.size(-1)], mode='bilinear')

        x_att = x * att_f

        return x_att, att_f


class loc_attention(nn.Module):

    def __init__(self, in_c):
        super(loc_attention, self).__init__()
        self.K1 = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.K2 = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.A = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.O = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, d_ft, x):
        att_f = F.relu(self.K1(d_ft))
        # x_k2 = F.relu(self.K2(att_f))

        att_f = F.sigmoid(self.A(att_f))
        x_att = att_f * x
        # x_att = self.O(x_att)

        return x_att, att_f


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.conv_att = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.i_att = in_attention(256)
        self.l_att = loc_attention(256)
        self.features = nn.Sequential(OrderedDict([]))

        # Each denseblock
        num_features = num_init_features
        num_layers = block_config[0]
        self.DB_att = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.DB1 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.TN1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        for i, num_layers in enumerate(block_config[1:]):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 2), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config[1:]) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 2), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.marginSM = nn.Linear(num_features, 256)
        self.cellclassifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        c1_att = self.conv_att(x)
        d1_att = self.DB_att(c1_att)
        x_att, att_f_i = self.i_att(d1_att, x)
        c1 = self.conv1(x_att)
        d1 = self.DB1(c1)
        d1, att_f_l = self.l_att(d1_att, d1)
        t1 = self.TN1(d1)
        features = self.features(t1)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out2 = self.marginSM(out)
        out1 = self.cellclassifier(out)
        return out1, out2, att_f_i, att_f_l
