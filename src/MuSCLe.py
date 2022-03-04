import torch
import copy
from torch import nn
import torch.nn.functional as F
from EfficientNet_PyTorch.efficientnet_pytorch.model import EfficientNet


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class H_Swish(nn.Module):
    def forward(self, x):
        return x*F.relu6(x, inplace=True)/6


class _BIFPN_Layer(nn.Module):
    def __init__(self, swish, bifpn_channels=256, last_pooling=True):
        super(_BIFPN_Layer, self).__init__()
        self.swish = swish
        self.convp67 = nn.Sequential(nn.Conv2d(2*bifpn_channels, bifpn_channels, 1), self.swish)
        self.convp56 = nn.Sequential(nn.Conv2d(2*bifpn_channels, bifpn_channels, 1), self.swish)
        self.convp45 = nn.Sequential(nn.Conv2d(2*bifpn_channels, bifpn_channels, 1), self.swish)
        self.convp34 = nn.Sequential(nn.Conv2d(2*bifpn_channels, bifpn_channels, 1), self.swish)
        
        self.out4 = nn.Sequential(nn.Conv2d(bifpn_channels, bifpn_channels, 1), nn.BatchNorm2d(bifpn_channels), self.swish)
        self.out5 = nn.Sequential(nn.Conv2d(bifpn_channels, bifpn_channels, 1), nn.BatchNorm2d(bifpn_channels), self.swish)
        self.out6 = nn.Sequential(nn.Conv2d(bifpn_channels, bifpn_channels, 1), nn.BatchNorm2d(bifpn_channels), self.swish)
        self.out7 = nn.Sequential(nn.Conv2d(bifpn_channels, bifpn_channels, 1), nn.BatchNorm2d(bifpn_channels), self.swish)

        self.last_pool = last_pooling

    def forward(self, p3, p4, p5, p6, p7):
        p6_mid = self.convp67(torch.cat([p6, p7], dim=1))
        p5_mid = self.convp56(torch.cat([p5, F.interpolate(p6_mid, size=p5.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        p4_mid = self.convp45(torch.cat([p4, p5], dim=1))
        p3_out = self.convp34(torch.cat([p3, F.interpolate(p4_mid, size=p3.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        p4_out = self.out4(p4 + p4_mid + F.interpolate(F.avg_pool2d(p3_out, kernel_size=3, stride=2, padding=1), size=p4.shape[2:], mode='bilinear', align_corners=True))
        p5_out = self.out5(p5 + p5_mid + p4_out)
        if self.last_pool:
            p6_out = self.out6(p6 + p6_mid + F.interpolate(F.avg_pool2d(p5_out, kernel_size=3, stride=2, padding=1), size=p6.shape[2:], mode='bilinear', align_corners=True))
        else:
            p6_out = self.out6(p6 + p6_mid + p5_out) ###if no pool on p5
        p7_out = self.out7(p7 + p6_out)
        return [p3_out, p4_out, p5_out, p6_out, p7_out]

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                        stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.activation(x)

        return x

class BIFPN(nn.Module):
    def __init__(self, swish, pretrained='efficientnet-b1', layers=1, bifpn_channels=256, last_pooling=True):
        super(BIFPN, self).__init__()
        self.swish = swish
        if pretrained == 'efficientnet-b1':
            inp_p2, inp_p3, inp_p4, inp_p5, inp_p6, inp_p7 = 24, 40, 80, 112, 192, 320
        elif pretrained == 'efficientnet-b3':
            inp_p2, inp_p3, inp_p4, inp_p5, inp_p6, inp_p7 = 32, 48, 96, 136, 232, 384
        elif pretrained == 'efficientnet-b5':
            inp_p2, inp_p3, inp_p4, inp_p5, inp_p6, inp_p7 = 40, 64, 128, 176, 304, 512
        elif pretrained == 'efficientnet-b7':
            inp_p2, inp_p3, inp_p4, inp_p5, inp_p6, inp_p7 = 48, 80, 160, 224, 384, 640

        
        self.inp3 = nn.Sequential(nn.Conv2d(inp_p3, bifpn_channels, 1), nn.BatchNorm2d(bifpn_channels), self.swish)
        self.inp4 = nn.Sequential(nn.Conv2d(inp_p4, bifpn_channels, 1), nn.BatchNorm2d(bifpn_channels), self.swish)
        self.inp5 = nn.Sequential(nn.Conv2d(inp_p5, bifpn_channels, 1), nn.BatchNorm2d(bifpn_channels), self.swish)
        self.inp6 = nn.Sequential(nn.Conv2d(inp_p6, bifpn_channels, 1), nn.BatchNorm2d(bifpn_channels), self.swish)
        self.inp7 = nn.Sequential(nn.Conv2d(inp_p7, bifpn_channels, 1), nn.BatchNorm2d(bifpn_channels), self.swish)

        self.BIFPN_Layers = self.get_clones(module=_BIFPN_Layer(swish=self.swish, bifpn_channels=bifpn_channels, last_pooling=last_pooling), N=layers)

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, p3, p4, p5, p6, p7):
        p3 = self.inp3(p3) 
        p4 = self.inp4(p4) 
        p5 = self.inp5(p5) 
        p6 = self.inp6(p6) 
        p7 = self.inp7(p7) 
        for layer in self.BIFPN_Layers:
            p3, p4, p5, p6, p7 = layer(p3, p4, p5, p6, p7)
        return [p3, p4, p5, p6, p7]


class View(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0),-1)
        return x

class MuSCLe(nn.Module):
    def __init__(self, num_classes, pretrained='efficientnet-b1', layers=1, MemoryEfficient=True, bifpn_channels=256, last_pooling=True, mode='enc'):
        super(MuSCLe, self).__init__()
        '''
            from p1 to p7, pretrained efficientnet strides are 2, 4, 8, 16, 16, 16, 16 (in CAM encoder training)
            from p1 to p7, pretrained efficientnet strides are 2, 4, 8, 16, 16, 32, 32 (in encoder decoder retraining)
        '''
        self.classes = num_classes
        self.swish = MemoryEfficientSwish() if MemoryEfficient else H_Swish()
        self.backbone = EfficientNet.from_pretrained(pretrained, num_classes=num_classes, last_pooling=last_pooling)
        
        if pretrained == 'efficientnet-b1':
            p1_ch, p2_ch, p3_ch, p4_ch, p5_ch, p6_ch, p7_ch = 16, 24, 40, 80, 112, 192, 320
            self.p1_seq , self.p2_seq, self.p3_seq, self.p4_seq, self.p5_seq, self.p6_seq, self.p7_seq = 1, 4, 7, 11, 15, 20, 22
        elif pretrained == 'efficientnet-b3':
            p1_ch, p2_ch, p3_ch, p4_ch, p5_ch, p6_ch, p7_ch = 24, 32, 48, 96, 136, 232, 384
            self.p1_seq , self.p2_seq, self.p3_seq, self.p4_seq, self.p5_seq, self.p6_seq, self.p7_seq = 1, 4, 7, 12, 17, 23, 25
        elif pretrained == 'efficientnet-b5':
            p1_ch, p2_ch, p3_ch, p4_ch, p5_ch, p6_ch, p7_ch = 24, 40, 64, 128, 176, 304, 512
            self.p1_seq , self.p2_seq, self.p3_seq, self.p4_seq, self.p5_seq, self.p6_seq, self.p7_seq = 2, 7, 12, 19, 26, 35, 38
        elif pretrained == 'efficientnet-b7':
            p1_ch, p2_ch, p3_ch, p4_ch, p5_ch, p6_ch, p7_ch = 32, 48, 80, 160, 224, 384, 640
            self.p1_seq , self.p2_seq, self.p3_seq, self.p4_seq, self.p5_seq, self.p6_seq, self.p7_seq = 3, 10, 17, 27, 37, 50, 54

        if mode == 'enc':
            self.fuse = nn.Conv2d(p1_ch+p3_ch+p5_ch, 128, 1, bias=True)
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(p7_ch, num_classes, bias=False)
        else:
            self.layers = layers
            self.BIFPN = BIFPN(swish=self.swish, pretrained=pretrained, layers=self.layers, bifpn_channels=bifpn_channels, last_pooling=last_pooling)
            # self.fuse = nn.Conv2d(bifpn_channels, 128, 1, bias=True)
        # self.se = SELayer(bifpn_channels)
        self.fuse_dec = nn.Conv2d(bifpn_channels, num_classes, 1)
        


    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm2d) and m.bias is not None:
            torch.nn.init.normal_(m.weight, 0, 1)
            torch.nn.init.zeros_(m.bias)

    
    def cam_maxnorm(self, cams):
        cams = torch.relu(cams)
        n,c,h,w = cams.shape
        cam_min = torch.min(cams.view(n,c,-1), dim=-1)[0].view(n,c,1,1)
        cam_max = torch.max(cams.view(n,c,-1), dim=-1)[0].view(n,c,1,1)
        norm_cam = (cams - cam_min - 1e-6)/ (cam_max - cam_min + 1e-6)
        foreground = norm_cam[:,1:,:,:]
        background = (1-torch.max(foreground, dim=1)[0]).unsqueeze(1)
        norm_cam = torch.relu(torch.cat([background, foreground], dim=1))
        return norm_cam

    def PCM(self, cam, f):
        n,c,h,w = f.size()
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w)
        f = self.fuse(f)
        f = f.view(n,-1,h*w)
        f = f/(torch.norm(f,dim=1,keepdim=True)+1e-5)

        aff = F.relu(torch.matmul(f.transpose(1,2), f),inplace=True)
        aff = aff/(torch.sum(aff,dim=1,keepdim=True)+1e-5)
        cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)
        return cam_rv



    def forward(self, x, cam='fm'):
        N, C, H, W = x.shape

        if cam == 'logits':
            FPs = self.backbone(x)
            p1, p2, p3, p4, p5, p6, p7 = FPs[self.p1_seq], FPs[self.p2_seq], FPs[self.p3_seq], FPs[self.p4_seq], FPs[self.p5_seq], FPs[self.p6_seq], FPs[self.p7_seq]
            emb = self.pool(p7).squeeze(2).squeeze(2) #N,C
            self.logits = self.fc(emb)
            return emb, self.logits

        elif cam == 'cam':
            FPs = self.backbone(x)
            p1, p2, p3, p4, p5, p6, p7 = FPs[self.p1_seq], FPs[self.p2_seq], FPs[self.p3_seq], FPs[self.p4_seq], FPs[self.p5_seq], FPs[self.p6_seq], FPs[self.p7_seq]
            emb = self.pool(p7).squeeze(2).squeeze(2) #N,C
            self.logits = self.fc(emb)
            cams = []
            for bs in range(p7.shape[0]):
                cam = p7[bs].unsqueeze(0) * self.fc.weight.data.unsqueeze(2).unsqueeze(2) #num_cls,ch,h,w
                cams.append(cam.sum(1).unsqueeze(0)) #1,num_cls,h,w
            cams = torch.cat(cams, dim=0) #N,num_cls,h,w
            cams = torch.relu(cams)
            with torch.no_grad():
                f1 = torch.relu(F.interpolate(p1, size=p7.shape[2:], mode='bilinear', align_corners=True))
                f2 = torch.relu(F.interpolate(p3, size=p7.shape[2:], mode='bilinear', align_corners=True))
                f3 = torch.relu(p5)
                fs = torch.cat([f1, f2, f3], dim=1)

            SGC = self.PCM(cams, fs.detach())

            cams = F.interpolate(cams, size=(H,W), mode='bilinear', align_corners=True)
            SGC = F.interpolate(SGC, size=(H,W), mode='bilinear', align_corners=True)
            return cams, SGC, emb, self.logits

        elif cam == 'pix':
            FPs = self.backbone(x)
            p1, p2, p3, p4, p5, p6, p7 = FPs[self.p1_seq], FPs[self.p2_seq], FPs[self.p3_seq], FPs[self.p4_seq], FPs[self.p5_seq], FPs[self.p6_seq], FPs[self.p7_seq]
            cams = []
            for bs in range(p7.shape[0]):
                cam = p7[bs].unsqueeze(0) * self.fc.weight.data.unsqueeze(2).unsqueeze(2) #p7_ch,num_cls,h,w
                cams.append(cam.sum(1).unsqueeze(0)) #1,num_cls,h,w
            cams = torch.cat(cams, dim=0) #N,num_cls,h,w
            cams = torch.relu(cams)
            with torch.no_grad():
                f1 = torch.relu(F.interpolate(p1, size=p7.shape[2:], mode='bilinear', align_corners=True))
                f2 = torch.relu(F.interpolate(p3, size=p7.shape[2:], mode='bilinear', align_corners=True))
                f3 = torch.relu(p5)
                fs = torch.cat([f1, f2, f3], dim=1)
 
            SGC = self.PCM(cams, fs.detach())

            cams = F.interpolate(cams, size=(H,W), mode='bilinear', align_corners=True)
            SGC = F.interpolate(SGC, size=(H,W), mode='bilinear', align_corners=True)
            return cams, SGC

        elif cam == 'seg':
            FPs = self.backbone(x)
            p1, p2, p3, p4, p5, p6, p7 = FPs[self.p1_seq], FPs[self.p2_seq], FPs[self.p3_seq], FPs[self.p4_seq], FPs[self.p5_seq], FPs[self.p6_seq], FPs[self.p7_seq]
            # p1 = F.interpolate(p1, size=p3.shape[2:], mode='bilinear', align_corners=True)
            # p2 = F.interpolate(p2, size=p3.shape[2:], mode='bilinear', align_corners=True)
            # x = F.interpolate(x, size=p3.shape[2:], mode='bilinear', align_corners=True)
            p3_dec, _, _, _, _ = self.BIFPN(p3, p4, p5, p6, p7)
            # f = p3_dec #torch.cat([p1, p2, x], dim=1)
            # p3_dec = self.PCM(p3_dec, f.detach())
            dense_ft = F.interpolate(p3_dec, size=(H,W), mode='bilinear', align_corners=True)
            seg_map = self.fuse_dec(dense_ft)
            return seg_map, dense_ft
            
        
        elif cam == 'vis':
            with torch.no_grad():
                FPs = self.backbone(x)
                p1, p2, p3, p4, p5, p6, p7 = FPs[self.p1_seq], FPs[self.p2_seq], FPs[self.p3_seq], FPs[self.p4_seq], FPs[self.p5_seq], FPs[self.p6_seq], FPs[self.p7_seq]

                p3_dec, _, _, _, _ = self.BIFPN(p3, p4, p5, p6, p7)
                dense_ft = F.interpolate(p3_dec, size=(H,W), mode='bilinear', align_corners=True)
                seg_map = self.fuse_dec(dense_ft)
            return seg_map, p7



    def get_parameter_groups(self):
        groups = ([], [])
        for n, p in self.named_parameters():
            if p.requires_grad:
                if 'BIFPN' or\
                    'deconv' or 'fuse_dec' in n:
                    groups[1].append(p)
                else:
                    groups[0].append(p)
        return groups



if __name__ == "__main__":
    model = MuSCLe(num_classes=21,pretrained='efficientnet-b5').cuda()
    inp = torch.rand(4, 3, 32, 32).cuda()
    label = torch.ones(4, 21).cuda()
    cam, SGC, emb, logits = model(inp, label)
    print(cam.shape, SGC.shape, emb.shape, logits.shape)
