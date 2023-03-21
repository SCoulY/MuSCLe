import random
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from data import *
from EffSeg import *
from loss_multilabel import *


class BGFilter(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        kernel = torch.ones(1,1,kernel_size,kernel_size)
        self.filter.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, seg_map):
        x = self.filter(seg_map)
        x = x/self.filter.weight.shape[2]/self.filter.weight.shape[3]
        return x

class Sobel(nn.Module):
    '''
    pre-defined 3x3 and 5x5 sobel filter    
    '''
    def __init__(self, kernel_size=3):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=kernel_size, 
                                stride=1, padding=kernel_size//2, bias=False)
        if kernel_size == 3:
            Gx = torch.tensor([[1.0, 1e-6, -1.0], [2.0, 1e-6, -2.0], [1.0, 1e-6, -1.0]])
            Gy = torch.tensor([[1.0, 2.0, 1.0], [1e-6, 1e-6, 1e-6], [-1.0, -2.0, -1.0]])
        elif kernel_size == 5:
            Gx = torch.tensor([[2.0, 1.0, 1e-6, -1.0, -2], [3.0, 2.0, 1e-6, -2.0, -3.0], [4.0, 3.0, 0.0, -3.0, -4.0],
                              [3.0, 2.0, 1e-6, -2.0, -3.0], [2.0, 1.0, 1e-6, -1.0, -2]])
            Gy = torch.tensor([[2.0, 3.0, 4.0, 3.0, 2.0], [1.0, 2.0, 3.0, 2.0, 1.0], [1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
                               [-1.0, -2.0, -3.0, -2.0, -1.0], [-2.0, -3.0, -4.0, -3.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, x, orient=False):
        x = self.filter(x)
        if orient:
            return x
        elif not orient:
            x = x * x
            x = torch.sum(x, dim=1, keepdim=True)
            x = torch.sqrt(x+1e-8)
            return x

class OrientQuantize(object):
    """
    input x of shape (N,2,H,W) 
    the first channel represents x gradient
    the second channel represents y gradient
    """
    def __init__(self):
        super().__init__()
        self.div = 3.1416/8
    
    def __call__(self, x):
        if len(x.shape)==4:#n,2,h,w
            mag = torch.sqrt(torch.sum(x**2, dim=1)+1e-8)
            orient = torch.atan2(x[:,1], x[:,0]) # n,h,w in (-pi, pi)
        elif len(x.shape)==5:#n,c-1,2,h,w
            mag = torch.sqrt(torch.sum(x**2, dim=2)+1e-8)
            orient = torch.atan2(x[:,:,1], x[:,:,0]) # n,c-1,h,w in (-pi, pi)
        mask1 = (3*self.div > orient) & (orient >= self.div)
        mask2 = (5*self.div > orient) & (orient >= 3*self.div)
        mask3 = (7*self.div > orient) & (orient >= 5*self.div)
        mask4 = ((8*self.div > orient) & (orient >= 7*self.div)) |\
                ((-7*self.div > orient) & (orient >= -8*self.div))
        mask5 = (-5*self.div > orient) & (orient >= -7*self.div)
        mask6 = (-3*self.div > orient) & (orient >= -5*self.div)
        mask7 = (-1*self.div > orient) & (orient >= -3*self.div)
        mask8 = (self.div > orient) & (orient >= -1*self.div)
        orient[mask1] = 0
        orient[mask2] = 1
        orient[mask3] = 2
        orient[mask4] = 3
        orient[mask5] = 4
        orient[mask6] = 5
        orient[mask7] = 6
        orient[mask8] = 7
        return mag, orient

class UnitVec(object):
    """
    input orientation map of shape (N,H,W) 
    output field map of shape (N,9,H,W) or (N,25,H,W)
    """
    def __init__(self):
        super().__init__()
        unit = 1/math.sqrt(2)
        self.filter_1 = torch.tensor([[unit],[unit]]).flatten()
        self.filter_2 = torch.tensor([[1e-6],[unit]]).flatten()
        self.filter_3 = torch.tensor([[-unit],[unit]]).flatten()
        self.filter_4 = torch.tensor([[-unit],[1e-6]]).flatten()
        self.filter_5 = torch.tensor([[-unit],[-unit]]).flatten()
        self.filter_6 = torch.tensor([[1e-6],[-unit]]).flatten()
        self.filter_7 = torch.tensor([[unit],[-unit]]).flatten()
        self.filter_8 = torch.tensor([[unit],[1e-6]]).flatten()

        filters = [self.filter_1, self.filter_2, self.filter_3, self.filter_4,
                   self.filter_5, self.filter_6, self.filter_7, self.filter_8]
        self.embed = nn.Embedding(8, 2)
        self.embed.weight.requires_grad = False

        if torch.cuda.is_available():
            self.embed= self.embed.cuda()
        for i in range(8):
            self.embed.weight[i] = filters[i]

    def __call__(self, orient):
        n,h,w = orient.shape
        unitvet = self.embed(orient.long())
        return unitvet #n,2,h,w

        

class FieldGenerator(object):
    """
    input orientation map of shape (N,H,W) 
    output field map of shape (N,9,H,W) or (N,25,H,W)
    """
    def __init__(self):
        super().__init__()

        self.k_1 = torch.tensor([[1e-6, 1, 1, 1, 1], [1e-6, 1e-6, 1, 1, 1], [1e-6, 1e-6, 1e-6, 1, 1],
                              [1e-6, 1e-6, 1e-6, 1e-6, 1], [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]]).flatten()
        self.k_2 = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
                              [1e-6, 1e-6, 1e-6, 1e-6, 1e-6], [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]]).flatten()
        self.k_3 = torch.tensor([[1, 1, 1, 1, 1e-6], [1, 1, 1, 1e-6, 1e-6], [1, 1, 1e-6, 1e-6, 1e-6],
                              [1, 1e-6, 1e-6, 1e-6, 1e-6], [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]]).flatten()
        self.k_4 = torch.tensor([[1, 1, 1e-6, 1e-6, 1e-6], [1, 1, 1e-6, 1e-6, 1e-6], [1, 1, 1e-6, 1e-6, 1e-6],
                              [1, 1, 1e-6, 1e-6, 1e-6], [1, 1, 1e-6, 1e-6, 1e-6]]).flatten()
        self.k_5 = torch.tensor([[1, 1e-6, 1e-6, 1e-6, 1e-6], [1, 1, 1e-6, 1e-6, 1e-6], [1, 1, 1, 1e-6, 1e-6],
                              [1, 1, 1, 1, 1e-6], [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]]).flatten()
        self.k_6 = torch.tensor([[1e-6, 1e-6, 1e-6, 1e-6, 1e-6], [1e-6, 1e-6, 1e-6, 1e-6, 1e-6], [1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
                              [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]).flatten()
        self.k_7 = torch.tensor([[1e-6, 1e-6, 1e-6, 1e-6, 1e-6], [1e-6, 1e-6, 1e-6, 1e-6, 1], [1e-6, 1e-6, 1e-6, 1, 1],
                              [1e-6, 1e-6, 1, 1, 1], [1e-6, 1, 1, 1, 1]]).flatten()
        self.k_8 = torch.tensor([[1e-6, 1e-6, 1e-6, 1, 1], [1e-6, 1e-6, 1e-6, 1, 1], [1e-6, 1e-6, 1e-6, 1, 1],
                              [1e-6, 1e-6, 1e-6, 1, 1], [1e-6, 1e-6, 1e-6, 1, 1]]).flatten()
        outs = [self.k_1, self.k_2, self.k_3, self.k_4,
                   self.k_5, self.k_6, self.k_7, self.k_8]
        
        ins = [self.k_5, self.k_6, self.k_7, self.k_8, self.k_1, self.k_2, self.k_3, self.k_4]
        self.embed_out = nn.Embedding(8, 25)
        self.embed_out.weight.requires_grad = False
        self.embed_in = nn.Embedding(8, 25)
        self.embed_in.weight.requires_grad = False

        if torch.cuda.is_available():
            self.embed_out = self.embed_out.cuda()
            self.embed_in = self.embed_in.cuda()
        for i in range(8):
            self.embed_out.weight[i] = outs[i]
            self.embed_in.weight[i] = ins[i]



    def __call__(self, orient):
        n,h,w = orient.shape
        outs = self.embed_out(orient.long()) #n,25,h,w
        out_idx = outs > 1e-5 #n,10,h,w
        ins = self.embed_in(orient.long()) #n,25,h,w
        in_idx = ins > 1e-5 #n,10,h,w
        return  outs, out_idx, ins, in_idx

class FieldLoss(nn.Module):
    """
    input edge_fg of shape (n,2,h,w)
    input edge_ori of shape (n,2,h,w)
    """
    def __init__(self,num_classes=21, gaussian_size=7, k=100,
                guassian_sigma=None, sobel_size=5, beta=1e3):
        super().__init__()
        # self.field = FieldGenerator()
        # self.unitvec = UnitVec()
        self.quantize = OrientQuantize()
        self.unfold = torch.nn.Unfold(kernel_size=(5,5), dilation=1, 
                                        padding=2, stride=1)
        # self.edge = Edge_detector(gaussian_size=gaussian_size, guassian_sigma=guassian_sigma, sobel_size=sobel_size)
        self.fg_edge = Mix_fg(sobel_size=sobel_size, num_classes=num_classes, beta=beta)
        self.ind_tensor = None
        self.num_fg_cls = num_classes-1
        self.k = k
        # self.valid_cls = [0,3,9,16,18]
    
    def in_out_div(self, orient, pos_bc, step_outs=[5], step_ins=[5]):
        """
        input orient is of shape (h,w)
        step_outs, step_ins controls the steps walking along gradient orientation
        
        return outs_bc, ins_bc contain a list of tensors store 
        the samples outside and inside boundary when walking in 
        different steps
        """
        h,w = self.ind_tensor.shape
        outs_bc = []
        ins_bc = []
        for step_out in step_outs:
            outs_bc.append(self.ind_tensor+(-step_out)**(1+(orient<4))*((orient%4==0)*w)+\
                                (-1)**(1+orient)*((orient==2)|(orient==6))) #h,w
        for step_in in step_ins:
            ins_bc.append(self.ind_tensor+(-step_in)**(orient<4)*((orient%4==0)*w)+\
                                (-1)**(orient)*((orient==2)|(orient==6))) #h,w
        
        out_bc = [x[pos_bc].unsqueeze(0) for x in outs_bc] #1,k
        out_bc = torch.cat(out_bc, dim=1).squeeze(0) #K
        in_bc = [x[pos_bc].unsqueeze(0) for x in ins_bc] #1,k
        in_bc = torch.cat(in_bc, dim=1).squeeze(0) #K

        ###remove marginal samples
        elim_out = (out_bc%(w-1)!=0) & (out_bc%(w-1)!=1) & (out_bc>0) & (out_bc<w*h-1)
        elim_in = (in_bc%(w-1)!=0) & (in_bc%(w-1)!=1) & (in_bc>0) & (in_bc<w*h-1)
        out_bc = out_bc[elim_out].long().flatten()
        in_bc = in_bc[elim_in].long().flatten()
        
        del outs_bc
        del ins_bc
        return out_bc, in_bc

    def loss_constructor(self, FP, FN, 
                         TP, TN, sim, dim=1):
        
        loss_beacon = 0

        if FP.sum()>0 :
            loss_FP = -sim.mean(dim)[FP].mean()
            loss_beacon += loss_FP
            del FP
            del loss_FP
        
        if FN.sum()>0 :
            loss_FN = sim.mean(dim)[FN].mean()
            loss_beacon += loss_FN
            del FN
            del loss_FN
            
        if TP.sum()>0:
            loss_TP = sim.mean(dim)[TP].mean()
            loss_beacon += loss_TP
            del TP
            del loss_TP

        if TN.sum()>0:
            loss_TN = -sim.mean(dim)[TN].mean()
            loss_beacon += loss_TN
            del TN
            del loss_TN
        
        return loss_beacon


    def bifilter(self, orient, dense_ft, mask, pos_idx, label_fg, step):
        '''
        orient is of shape (n,c-1,h,w) in range [0,7]
        dense_ft is of shape (n,ch,h,w)
        '''
        n,ch,h,w = dense_ft.shape
        if not torch.is_tensor(self.ind_tensor):
            self.ind_tensor = torch.tensor([range(h*w)]).view(h,w).to(dense_ft.device)
        loss = 0

        dense_ft = torch.softmax(dense_ft, dim=1)
        mask = torch.softmax(mask, dim=1)
        
        for b in range(n):
            pos_b = pos_idx[b]
            dense_b = dense_ft[b].view(ch,-1) #ch,hw
            mask_b = mask[b].view(self.num_fg_cls+1,-1) #ch,hw
            loss_b = 0

            for c in range(orient.shape[1]):
                if label_fg[b,c]==0:
                    continue
                pos_bc = pos_b[c] #h,w

                with torch.no_grad():
                    orient_bc = orient[b,c]+1 #h,w

                    out_bc, in_bc = self.in_out_div(orient_bc, pos_bc, step_outs=[step], step_ins=[step])

                    ins_mask = mask_b[:,in_bc] #ch,K
                    outs_mask = mask_b[:,out_bc] #ch,K

                ins = dense_b[:,in_bc] #ch,K
                outs = dense_b[:,out_bc] #ch,K
                

                if ins.shape[1]>self.k and outs.shape[1]>self.k:
                    rand_out = random.sample(range(outs.shape[1]), self.k)
                    rand_in = random.sample(range(ins.shape[1]), self.k)
                    ins = ins[:,rand_in] #k,1
                    outs = outs[:,rand_out]#k,1
                    
                    ins_mask = ins_mask[:,rand_in] #k,1
                    outs_mask = outs_mask[:,rand_out] #k,1

                elif ins.shape[1]>=10 and outs.shape[1]>=10:
                    continue
                
                else:
                    del out_bc
                    del in_bc
                    del ins
                    del outs
                    del ins_mask
                    del outs_mask
                    continue

                sim_out = outs.permute(1,0) @ ins.detach() #k,k
                sim_mask = outs_mask.permute(1,0) @ ins_mask.detach() #k,k

                sign_mask_out = (sim_mask.mean(1) > sim_mask.mean()).detach() #k
                sign_dense_out = (sim_out.mean(1) > sim_out.mean()).detach() #k
                FP = torch.bitwise_and(sign_mask_out, torch.bitwise_not(sign_dense_out))
                FN = torch.bitwise_and(torch.bitwise_not(sign_mask_out), sign_dense_out)
                TP = torch.bitwise_and(torch.bitwise_not(sign_mask_out), torch.bitwise_not(sign_dense_out))
                TN = torch.bitwise_and(sign_mask_out, sign_dense_out)

                loss_b += self.loss_constructor(FP, FN, TP,
                                               TN, sim_out, dim=1)


                sign_mask_in = (sim_mask.mean(0) > sim_mask.mean()).detach() #k
                sign_dense_in = (sim_out.mean(0) > sim_out.mean()).detach() #k
                    
                FP = torch.bitwise_and(sign_mask_in, torch.bitwise_not(sign_dense_in))
                FN = torch.bitwise_and(torch.bitwise_not(sign_mask_in), sign_dense_in)
                TP = torch.bitwise_and(torch.bitwise_not(sign_mask_in), torch.bitwise_not(sign_dense_in))
                TN = torch.bitwise_and(sign_mask_in, sign_dense_in)

                loss_b += self.loss_constructor(FP, FN, TP,
                                               TN, sim_out, dim=0)

                del out_bc
                del in_bc
                del ins
                del outs
                del ins_mask
                del outs_mask
                del sim_out
                del sim_mask

            loss += loss_b
            
        del dense_ft
        del dense_b
        del mask_b
        
    
        if torch.is_tensor(loss):
            return loss.mean()/n 
        return loss 

 
    def forward(self, seg_map, dense_ft, mask, label_with_bg, step=7):
        n,c,h,w = dense_ft.shape
        with torch.no_grad():
            edges_fg = self.fg_edge(seg_map, label_with_bg, reduction=False) #n,c-1,h,w
            mag_fg, orient_fg = self.quantize(edges_fg)

            max_fg =  torch.max(mag_fg.view(n,self.num_fg_cls,-1), dim=-1)[0].unsqueeze(2).unsqueeze(2)
            pos_idx = (mag_fg >= 0.8*max_fg) & (max_fg > 1)
            pos_idx = pos_idx * label_with_bg[:,1:].unsqueeze(2).unsqueeze(2).bool()
            pos_count = torch.sum(pos_idx)
            mag_fg = mag_fg.sum(1)

 
        loss = False
        if pos_count>=10:
            loss = self.bifilter(orient_fg, dense_ft, mask, pos_idx, label_fg=label_with_bg[:,1:], step=step)

        del edges_fg
        del orient_fg
        del pos_idx
        return loss, mag_fg

class ArgMax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, 1)
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        ctx.save_for_backward(idx)
        return idx.float()

    @staticmethod
    def backward(ctx, grad_output):
        idx, = ctx.saved_tensors
        grad_input = torch.zeros(ctx._input_shape, device=ctx._input_device, dtype=ctx._input_dtype)
        idx = idx.unsqueeze(1)
        grad_input.scatter_(1, idx, grad_output.sum())
        return grad_input

class Mix_fg(nn.Module):
    def __init__(self, sobel_size=3, num_classes=21, beta=1e3):
        super(Mix_fg, self).__init__()
        self.sobel = Sobel(kernel_size=sobel_size).cuda() if torch.cuda.is_available() else Sobel(kernel_size=sobel_size)
        self.num_classes = num_classes
        # self.argmax = ArgMax()
        self.beta = beta
    
    def to_one_hot(self, target, n_classes):
        n, h, w = target.size()
        one_hot = torch.zeros(n, n_classes, h, w, requires_grad=True)
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
            one_hot_clone = one_hot.clone()
        
        one_hot_clone = one_hot_clone.scatter(1, target.unsqueeze(1).long(), 1)
        return one_hot_clone

    def forward(self, seg_map, label_with_bg, reduction=True):
        n, c, h, w = seg_map.shape
        seg_map = torch.softmax(seg_map*self.beta, dim=1)
        seg_map = seg_map[:,1:] #exclude bg
        edges = []

        for i in range(c-1):
            seg_edge_i = self.sobel(seg_map[:,i].unsqueeze(1), orient=True) #n,2,h,w
            edges.append(seg_edge_i)

            
        edges = torch.stack(edges, dim=1) #n,c-1,2,h,w
        
        
        edges = edges * label_with_bg[:,1:].unsqueeze(2).unsqueeze(2).unsqueeze(2)
        if reduction:
            edges = edges.sum(1)
        return edges


class Edge_detector(nn.Module):
    def __init__(self, gaussian_size=7, guassian_sigma=None, sobel_size=3):
        super(Edge_detector, self).__init__()
        self.sobel = Sobel(kernel_size=sobel_size).cuda() if torch.cuda.is_available() else Sobel(kernel_size=sobel_size)
        self.gaussian_size = gaussian_size
        self.gaussian_sigma = guassian_sigma

    def denorm(self, x):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        x[:,0,:,:] = (x[:,0,:,:]*std[0] + mean[0])*255
        x[:,1,:,:] = (x[:,1,:,:]*std[1] + mean[1])*255
        x[:,2,:,:] = (x[:,2,:,:]*std[2] + mean[2])*255
        x[x > 255] = 255
        x[x < 0] = 0
        return x

    def forward(self, x):
        n,c,h,w = x.shape
        img_denorm = self.denorm(x.detach().clone())
        img_blur = transforms.functional.gaussian_blur(img_denorm, kernel_size=self.gaussian_size, sigma=self.gaussian_sigma)
        img_blur = transforms.functional.rgb_to_grayscale(img_blur)/255
        edge_blur = self.sobel(img_blur, orient=False) #n,h,w
        del x
        return edge_blur

