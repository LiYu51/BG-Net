import math
import torch
from torch import nn
import torch.nn.functional as F

def gradconv(op_type):
    if op_type == 'cv':
        return F.conv2d
    
    elif op_type == 'gd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            list_x = []
            for i in range(weights.shape[1]):
                list_x.append(torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], device='cuda:0'))

            list_x = torch.stack(list_x, 0)

            list_y = []
            for i in range(weights.shape[1]):
                list_y.append(torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], device='cuda:0'))

            list_y = torch.stack(list_y, 0)
            weight_x = torch.mul(weights, list_x)
            weight_y = torch.mul(weights, list_y)

            input_x = F.conv2d(x, weight_x, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            input_y = F.conv2d(x, weight_y, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

            input_x = torch.mul(input_x, input_x)
            input_y = torch.mul(input_y, input_y)

            result = torch.add(input_x, input_y)
            result = result.sqrt()

            return result

        return func

    elif op_type == 'cygd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weight_x = (weights[:, :, [2, 0, 1, 5, 3, 4, 8, 6, 7]] - weights).view(shape)  # clock-wise
            weight_y = (weights[:, :, [6, 7, 8, 0, 1, 2, 3, 4, 5]] - weights).view(shape)  # clock-wise

            input_x = F.conv2d(x, weight_x, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            input_y = F.conv2d(x, weight_y, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

            input_x = torch.mul(input_x, input_x)
            input_y = torch.mul(input_y, input_y)

            result = torch.add(input_x, input_y)
            result = result.sqrt()

            return result

        return func
    elif op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc

        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            # print("weight.shape:", weights.shape)
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
 
        return func
    elif op_type == 'bam':
        def __init__(self, in_channels):
            super(BAM, self).__init__()
            self.in_channels = in_channels
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Linear(in_channels, 1)
        
        def forward(self, x):
            x = torch.randn(batch_size, in_channels, feature_size, feature_size)
            weights = torch.randn(in_channels, in_channels, 3, 3)
            dg_model = dg()
            Gf = dg_model.func(x, weights)
            pooled = self.max_pool(Gf)
            att_map = 1 / (1 + torch.exp(-pooled))
            att_feature = x * att_map.view(-1, self.in_channels, 1, 1)
            return att_feature, att_map
    
        def loss(self, att_map, target):
            loss_fn = nn.BCELoss()
            loss = loss_fn(att_map.squeeze(), target.float())
            
            return att_map
 
        return func       
        
    else:
        print('impossible to be here unless you force that')
        return None

class dg(nn.Module):
    def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
        assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
        assert padding == dilation, 'padding for ad_conv set wrong'

        list_x = []
        for i in range(weights.shape[1]):
            list_x.append(torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], device='cuda:0'))

        list_x = torch.stack(list_x, 0)

        list_y = []
        for i in range(weights.shape[1]):
            list_y.append(torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], device='cuda:0'))

        list_y = torch.stack(list_y, 0)
        weight_x = torch.mul(weights, list_x)
        weight_y = torch.mul(weights, list_y)

        input_x = F.conv2d(x, weight_x, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        input_y = F.conv2d(x, weight_y, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

        input_x = torch.mul(input_x, input_x)
        input_y = torch.mul(input_y, input_y)

        result = torch.add(input_x, input_y)
        Gf = result.sqrt()

        return Gf
       

class Conv2d(nn.Module):
    def __init__(self, gradconv, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.gradconv = gradconv

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.gradconv(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


nets = {
'gradconv': {
        'layer0': 'gd',
        'layer1': 'bam',
        'layer2': 'cygd',
        'layer3': 'bam',
        'layer4': 'gd',
        'layer5': 'bam',
        'layer6': 'cygd',
        'layer7': 'bam',
        'layer8': 'gd',
        'layer9': 'bam',
        'layer10': 'cygd',
        'layer11': 'bam',
        'layer12': 'gd',
        'layer13': 'bam',
    },
    
    'gradconv1': {
        'layer0': 'gd',
        'layer1': 'gd',
        'layer2': 'gd',
        'layer3': 'gd',
        'layer4': 'gd',
        'layer5': 'gd',
        'layer6': 'gd',
        'layer7': 'gd',
    },
    'cygradconv': {
        'layer0': 'cygd',
        'layer1': 'cygd',
        'layer2': 'cygd',
        'layer3': 'cygd',
        'layer4': 'cygd',
        'layer5': 'cygd',
        'layer6': 'cygd',
        'layer7': 'cygd',
    },
    'cygradconv_1': {
        'layer0': 'cygd',
        'layer1': 'gd',
        'layer2': 'cygd',
        'layer3': 'gd',
        'layer4': 'cygd',
        'layer5': 'gd',
        'layer6': 'cygd',
        'layer7': 'gd',
    },
    'layer_4': {
        'layer0':  'cd',
        'layer1':  'ad',
        'layer2':  'rd',
        'layer3':  'cv',
        'layer4':  'cd',
        'layer5':  'ad',
        'layer6':  'rd',
        'layer7':  'cv',
    },
    'layer_5': {
            'layer0':  'gd',
            'layer1':  'cd',
            'layer2':  'ad',
            'layer3':  'rd',
            'layer4':  'gd',
            'layer5':  'cd',
            'layer6':  'ad',
            'layer7':  'rd',
        }
}


def config_model(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(nets[model]))

    pdcs = []
    for i in range(8):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        pdcs.append(gradconv(op))

    return pdcs


class GradConvBlock(nn.Module):
    def __init__(self, gradconv, inplane, ouplane, stride=1):
        super(GradConvBlock, self).__init__()
        self.stride = stride

        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(gradconv, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y


class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """

    def __init__(self, channels):
        super(MapReduce, self).__init__()
        # self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        self.conv = nn.Conv2d(channels, 16, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class GradConvNet(nn.Module):
    def __init__(self, inplane, gradconvs):
        super(GradConvNet, self).__init__()

        self.fuseplanes = []

        self.inplane = inplane

        self.init_block = Conv2d(gradconvs[0], 3, self.inplane, kernel_size=3, padding=1)
        block_class = GradConvBlock

        self.block1_1 = block_class(gradconvs[1], self.inplane, self.inplane)
        self.block1_2 = block_class(gradconvs[2], self.inplane, self.inplane)
        self.block1_3 = block_class(gradconvs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(gradconvs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(gradconvs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(gradconvs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(gradconvs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 2C

        self.conv_reduces = nn.ModuleList()

        for i in range(2):
            self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        # self.classifier = nn.Conv2d(4, 1, kernel_size=1) # has bias 全连接分类
        # nn.init.constant_(self.classifier.weight, 0.25)
        # nn.init.constant_(self.classifier.bias, 0)

        print('initialization done')

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x_fuses = [x1, x2]

        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2]
        outputs = torch.cat(outputs, dim=1)

        # output = self.classifier(torch.cat(outputs, dim=1))
        # #if not self.training:
        # #    return torch.sigmoid(output)
        #
        # outputs.append(output)
        # outputs = [torch.sigmoid(r) for r in outputs]
        return outputs


def gradconvnet(model):
    gradconvs = config_model(model)
    return GradConvNet(60, gradconvs)
