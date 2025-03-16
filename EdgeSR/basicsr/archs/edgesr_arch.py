import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY

class SeqConv3x3(nn.Module):
    ''' SeqConv3x3 '''
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super().__init__()

        self.type = seq_type
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.mid_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            conv1 = torch.nn.Conv2d(self.mid_planes,
                                    self.out_planes,
                                    kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        ''' forward '''
        if self.type == 'conv1x1-conv3x3':
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            y1 = F.conv2d(input=y0,
                          weight=self.scale * self.mask,
                          bias=self.bias,
                          stride=1,
                          groups=self.out_planes)
        return y1

    def rep_params(self):
        ''' rep_params '''
        device = self.k0.get_device()
        if device < 0:
            device = None
        if self.type == 'conv1x1-conv3x3':
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            RB = torch.ones(1, self.mid_planes, 3, 3,
                            device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1, ) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3),
                             device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            RB = torch.ones(1, self.out_planes, 3, 3,
                            device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class TDE_block(nn.Module):

    def __init__(self,
                 inp_planes,
                 out_planes,
                 depth_multiplier):
        super().__init__()
        self.n_feat = inp_planes
        self.conv3x3 = torch.nn.Conv2d(inp_planes,
                                       out_planes,
                                       kernel_size=3)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', inp_planes,
                                      out_planes, depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', inp_planes, out_planes,
                                      -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', inp_planes, out_planes,
                                      -1)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', inp_planes,
                                      out_planes, -1)
        self.conv1x1 = nn.Conv2d(inp_planes,out_planes,kernel_size=1,padding=0)

    def rep_params(self):
        ''' rep params '''
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        K5, B5 = self.conv1x1.weight, self.conv1x1.bias
        K5_pad =F.pad(K5, [1,1,1,1])
        RK, RB = (K0 + K1 + K2 + K3 + K4 + K5_pad), (B0 + B1 + B2 + B3 + B4 + B5)
        for i in range(self.n_feat):
            RK[i,i,1,1] += 1.0
        return RK,RB

    def forward(self, x):
        ''' forward '''
        x1 = x[:, :, 1:-1, 1:-1]
        x2 = self.conv1x1(x1)
        y = self.conv3x3(x) + self.conv1x1_3x3(x) + self.conv1x1_sbx(
            x) + self.conv1x1_sby(x) + self.conv1x1_lpl(x) + x1 + x2
        return y



class Scale_Atttention(nn.Module):
    def __init__(self,c_in,c_out,s,scale,bias=True):
        super(Scale_Atttention, self).__init__()
        self.tensor = nn.Parameter(
            scale * torch.ones((1, c_out, 1, 1)),
            requires_grad=False
        )
        self.scale_attn = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias),
            nn.SiLU(),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=1, padding=0, stride=1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.scale_attn(self.tensor)
        return attn*x


class Conv_SAE(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, s=1, scale=2, bias=True):
        super(Conv_SAE, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.gain = gain = gain1
        self.c_out = c_out
        self.scale_attn = Scale_Atttention(c_in, c_out, s, scale)
        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.mid_planes = int(c_in * gain)
        self.expand= torch.nn.Conv2d(c_in,
                                self.mid_planes,
                                kernel_size=1,
                                padding=1,
                                stride=1)
        self.conv = TDE_block(self.mid_planes,self.mid_planes,2)
        self.reduce = torch.nn.Conv2d(self.mid_planes,
                                c_out,
                                kernel_size=1,
                                padding=0)

    def forward(self, x):
        y0 = self.expand(x)
        x_1 = self.conv(y0)
        x_1 = self.reduce(x_1)
        x = self.scale_attn(self.sk(x))
        out = x_1 + x

        return out


@ARCH_REGISTRY.register()
class EdgeSR(nn.Module):

    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat, num_residual_feat):
        super().__init__()
        assert (num_feat > num_residual_feat >= num_in_ch) and (num_out_ch == num_in_ch)

        num_feat = num_feat - num_residual_feat

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)
        self.conv_residual_first = nn.Conv2d(num_in_ch, num_residual_feat, kernel_size=3, padding=1)

        backbone_convs = []
        residual_convs = []
        add_residual_convs = []
        add_expand_convs = []
        add_convs = []
        add_test = []
        for _ in range(num_block):
            backbone_convs.append(Conv_SAE(num_feat, num_feat, gain1=2, s=1, scale=upscale))
            residual_convs.append(nn.Conv2d(num_residual_feat, num_residual_feat, kernel_size=3, padding=1))
            add_residual_convs.append(nn.Conv2d(num_residual_feat, num_feat, kernel_size=3, padding=1))
            add_expand_convs.append(nn.Conv2d(num_feat,num_feat,kernel_size=3,padding=1))
            add_convs.append(nn.Conv2d(num_feat, num_feat, kernel_size=1, padding=0))
            add_test.append(nn.Conv2d(num_feat,num_feat,kernel_size=1,padding=0))
        self.backbone_convs = nn.ModuleList(backbone_convs)
        self.residual_convs = nn.ModuleList(residual_convs)
        self.add_residual_convs = nn.ModuleList(add_residual_convs)
        self.add_expand_convs = nn.ModuleList(add_expand_convs)
        self.add_convs = nn.ModuleList(add_convs)
        self.add_test = nn.ModuleList(add_test)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2), 3, padding=1)
        self.conv_residual_last = nn.Conv2d(num_residual_feat, num_out_ch * (upscale**2), kernel_size=3, padding=1)

        self.upsampler = nn.PixelShuffle(upscale)

        self.init_weights(num_in_ch, upscale, num_residual_feat)

    def init_weights(self, colors, upscale, num_residual_feat):
        ''' init weights (K_r --> I, K_{r2b} --> O, K_{b2r} --> O, and repeat --> repeat(I,n)) '''
        self.conv_residual_first.weight.data.fill_(0)
        for i in range(colors):
            self.conv_residual_first.weight.data[i, i, 1, 1] = 1
        self.conv_residual_first.bias.data.fill_(0)
        for residual_conv in self.residual_convs:
            residual_conv.weight.data.fill_(0)
            for i in range(num_residual_feat):
                residual_conv.weight.data[i, i, 1, 1] = 1
            residual_conv.bias.data.fill_(0)
        for add_residual_conv in self.add_residual_convs:
            add_residual_conv.weight.data.fill_(0)
            add_residual_conv.bias.data.fill_(0)
        for add_expand_conv in self.add_expand_convs:
            add_expand_conv.weight.data.fill_(0)
            add_expand_conv.bias.data.fill_(0)
        for add_conv in self.add_convs:
            add_conv.weight.data.fill_(0)
            add_conv.bias.data.fill_(0)
        for add_test in self.add_test:
            add_test.weight.data.fill_(0)
            add_test.bias.data.fill_(0)
        self.conv_residual_last.weight.data.fill_(0)
        for i in range(colors):
            for j in range(upscale**2):
                self.conv_residual_last.weight.data[i * (upscale**2) + j, i, 1, 1] = 1
        self.conv_residual_last.bias.data.fill_(0)

    def forward(self, input):
        ''' forward '''
        x = F.relu(self.conv_first(input))
        r = F.relu(self.conv_residual_first(input))

        for (backbone_conv, residual_conv, add_residual_conv,add_expand_convs
             ,add_convs,add_test) in zip(self.backbone_convs, self.residual_convs,
                                         self.add_residual_convs,self.add_expand_convs,
                                         self.add_convs,self.add_test):

            x, r = F.relu(backbone_conv(x) + add_convs(add_residual_conv(r)) + add_convs(add_expand_convs(x))), F.relu(residual_conv(r))

        x = self.upsampler(self.conv_last(x))
        r = self.upsampler(self.conv_residual_last(r))
        return x + r,r

@ARCH_REGISTRY.register()
class EdgeSRForInference(nn.Module):
    ''' ETDS for inference arch '''
    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat, num_residual_feat):
        super().__init__()
        assert (num_residual_feat >= num_in_ch) and (num_out_ch == num_in_ch)
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)

        backbone_convs = []
        for _ in range(num_block):
            backbone_convs.append(nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1))
        self.backbone_convs = nn.ModuleList(backbone_convs)

        if upscale == 3:
            # Reason for +1: extra channels are added to ensure that the number of channels is a multiple of 4.
            self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2) + 1, kernel_size=3, padding=1)
            self.conv_clip = nn.Conv2d(num_out_ch * (upscale**2) + 1, num_out_ch * (upscale**2) + 1, 1)
        else:
            self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2), kernel_size=3, padding=1)
            self.conv_clip = nn.Conv2d(num_out_ch * (upscale**2), num_out_ch * (upscale**2), 1)

        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, input):
        ''' forward '''
        x = F.relu(self.conv_first(input))

        for backbone_conv in self.backbone_convs:
            x = F.relu(backbone_conv(x))

        x = F.relu(self.conv_last(x))
        if self.upscale == 3:
            # Reason: extra channels are added to ensure that the number of channels is a multiple of 4.
            x = F.relu(self.conv_clip(x))[:, :-1, :, :]
        else:
            x = F.relu(self.conv_clip(x))

        x = self.upsampler(x)
        return x