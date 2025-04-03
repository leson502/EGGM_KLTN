import torch
import torch.nn as nn
import torch.nn.functional as F
from .moe import MoE, Gate
from copy import deepcopy

from src.eval_metrics import eval_crema
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, modality, num_classes=1000, pool='avgpool', zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.modality = modality
        self.pool = pool
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if modality == 'audio':
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif modality == 'visual':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            raise NotImplementedError('Incorrect modality, should be audio or visual but got {}'.format(modality))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # if self.pool == 'avgpool':
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)  # 8192

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        # if self.modality == 'visual':
        #     (B, C, T, H, W) = x.size()
        #     x = x.permute(0, 2, 1, 3, 4).contiguous()
        #     x = x.view(B * T, C, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = x

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def _resnet(arch, block, layers, modality, progress, **kwargs):
    model = ResNet(block, layers, modality, **kwargs)
    return model


def resnet18(modality, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], modality, progress,
                   **kwargs)


class CREMADModel(nn.Module):
    def __init__(self):
        super(CREMADModel, self).__init__()

        self.n_classes = 6
        
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        
        self.classifier = Classifier(512, self.n_classes)

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        out_a, gates_a, load_a = self.classifier(a, 0)
        out_v, gates_v, load_v = self.classifier(v, 1)

        gates = [gates_a, gates_v]
        load = [load_a, load_v]
     
        hs = [a.clone().detach(), v.clone().detach()]
        out = out_a + out_v

        return out, hs, [gates, load]

def cv_squared(x):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.
    Args:
    x: a `Tensor`.
    Returns:
    a `Scalar`.
    """
    eps = 1e-10
    # if only num_experts = 1

    if x.shape[0] == 1:
        return torch.tensor([0], device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean()**2 + eps)

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_expert=16, num_mod=2, k=12):
        super(Classifier, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_expert = num_expert
        self.num_mod = num_mod
        self.k = k

        self.moe = MoE(in_dim, in_dim, num_expert, in_dim // 4, True)
        self.gatting_network = nn.ModuleList(Gate(in_dim, num_expert, k=k) for _ in range(num_mod))
        self.out_layer = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, modality):
        gates, load = self.gatting_network[modality](x, modality)
        
        out = self.moe(x, gates)
        x = F.relu(out) + x
        x = self.out_layer(x)
        return x, gates, load


class ClassifierGuided(nn.Module):
    def __init__(self, num_mod, proj_dim=1024):
        super(ClassifierGuided, self).__init__()
        # Classifiers
        self.num_mod = num_mod
        self.classifers = nn.ModuleList([
            Classifier(512, 6)
            for _ in range(self.num_mod)
        ])
    
    def init_classifier(self, cls):
        for i in range(self.num_mod):
            with torch.no_grad():
                for param_A, param_B in zip(self.classifers[i].parameters(), cls.parameters()):
                    param_A.data.copy_(param_B.data)


    def cal_coeff(self, y, cls_res):
        acc_list = list()
        for i in range(self.num_mod):
            acc = eval_crema(y, cls_res[i])
            acc_list.append(acc)

        return acc_list

    def forward(self, x):
        self.cls_res = list()
        gatting = 0
        for i in range(len(x)):
            out, _, _ = self.classifers[i](x[i], i)
            self.cls_res.append(out)
        return self.cls_res


if __name__ == "__main__":
    model = CREMADModel()
    print(model)
    audio = torch.randn(1, 1, 256, 128)
    visual = torch.randn(1, 3, 224, 224)
    out, hs = model(audio, visual)
    print(out.size())
    print(len(hs))
    print(hs[0].size())
    print(hs[1].size())