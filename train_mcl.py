import numpy as np
import torch
import random
import cv2
import time
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from data import *
import argparse

from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from src.loss_multilabel import *
from src.MuSCLe import *
from src import imutils, pyutils, torchutils

import pandas as pd


def cam_maxnorm(cams):
    cams = torch.relu(cams)
    n,c,h,w = cams.shape
    cam_min = torch.min(cams.view(n,c,-1), dim=-1)[0].view(n,c,1,1)
    cam_max = torch.max(cams.view(n,c,-1), dim=-1)[0].view(n,c,1,1)
    norm_cam = (cams - cam_min - 1e-6)/ (cam_max - cam_min + 1e-6)
    norm_cam = torch.relu(norm_cam)
    return norm_cam

def cam_softmaxnorm(cams):
    # cams = torch.relu(cams)
    n,c,h,w = cams.shape
    foreground = torch.softmax(cams[:,1:,:,:], dim=1)
    background = (1-torch.max(foreground, dim=1)[0]).unsqueeze(1)
    norm_cam = torch.cat([background, foreground], dim=1)
    return norm_cam


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = heatmap * 0.5 + np.float32(img) * 0.5
    cam = np.uint8(cam)
    return cam #H, W, C

def get_sample_weight(dataset):
    t1 = time.time()
    sample_weight = [0]*len(dataset)
    class_count = [590, 504, 705, 468, 714, 393, 1150, 1005, 1228, 267,
               613, 1188, 445, 492, 4155, 522, 300, 649, 503, 567]
    sum_instance = len(dataset)  #adjustable
    for i, (name, img, label) in enumerate(dataset):
        multihots = torch.where(label==1)[0]
        instance_count = 0
        for hot in multihots:
            instance_count += class_count[hot.item()]
        sample_weight[i] = sum_instance/instance_count
    print(f'calculate sample weight takes:{time.time()-t1}seconds')
    return sample_weight



# class_count = [590, 504, 705, 468, 714, 393, 1150, 1005, 1228, 267,
#                613, 1188, 445, 492, 4155, 522, 300, 649, 503, 567]
# class_count = torch.from_numpy(np.asarray(class_count)).cuda()
# class_weight = torch.sum(class_count)/class_count
# class_weight = torch.softmax(class_weight, dim=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=1e-6, type=float)
    parser.add_argument("--train_list", default="data/train_aug.txt", type=str)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--session_name", default="runs/EffSeg_mcl", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--voc12_root", default='data/VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='logs/tblog_mcl', type=str)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    print(vars(args))
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    model = MuSCLe(num_classes=args.num_classes, pretrained='efficientnet-b3', layers=3, MemoryEfficient=True, last_pooling=False)

    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.affine = False
            
    os.makedirs(args.tblog_dir, exist_ok=True)
    os.makedirs(args.session_name, exist_ok=True)
    tblogger = SummaryWriter(args.tblog_dir)	


    train_dataset = train_dataset = VOC12ClsPix(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(448, 768),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        np.asarray,
                        imutils.color_norm,
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy,
                        transforms.RandomErasing(p=0.5, scale=(0.02,0.2))
                    ]), view_size=(224,224))
    
    eval_dataset = VOC12ClsDatasetMSF("data/train.txt", voc12_root=args.voc12_root,
                                                  scales=[1],
                                                  inter_transform=transforms.Compose(
                                                       [np.asarray,
                                                        imutils.color_norm,
                                                        imutils.HWC_to_CHW]))

    eval_data_loader = DataLoader(eval_dataset, shuffle=False, num_workers=args.num_workers)

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn, shuffle=True)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr ,weight_decay=args.wt_dec)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=0, cooldown=0, factor=0.5, min_lr=1e-5)


    if args.weights:
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)

    model = model.cuda()
    # torch.autograd.set_detect_anomaly(True)
    criterion1 = FocalLoss()
    criterion2 = Log_Sum_Exp_Pairwise_Loss
    criterion3 = nn.MultiLabelSoftMarginLoss()
    criterion4 = EMD() 
    criterion5 = image_level_contrast
    timer = pyutils.Timer("Session started: ")

    valid_cam = 0
    for ep in range(args.max_epoches):
        for iter, pack in enumerate(train_data_loader):
            optimizer.zero_grad()
           
            model.train()
            name, img, label, view1, view2, coord1, coord2, ori_coord = pack

            _, C, H, W = img.shape

            if torch.cuda.is_available():
                label = label.cuda()
                img = img.cuda().float()
                view1 = view1.cuda().float()
                view2 = view2.cuda().float()

            with torch.no_grad():
                label_with_bg = label.clone()
                bg_score = torch.ones((label.shape[0], 1), device=label_with_bg.device)
                label_with_bg = torch.cat((bg_score, label_with_bg), dim=1)

 
            raw_cams, raw_sgcs, emb, logits = model(img, cam='cam')

            cams = cam_softmaxnorm(raw_cams).detach()
            sgcs = cam_softmaxnorm(raw_sgcs)

            valid_channel = int(label.sum().cpu().data)
            
            loss_focal = criterion1(torch.sigmoid(logits[:, 1:]), label) 
            loss_softmargin = criterion3(logits[:, 1:], label) 
            loss_pair = criterion2(torch.sigmoid(logits[:, 1:]), label).mean() 
            loss_cls = loss_pair + loss_softmargin + loss_focal

            cams = cams*label_with_bg.unsqueeze(2).unsqueeze(3)
            sgcs = sgcs*label_with_bg.unsqueeze(2).unsqueeze(3)
            n,c,h,w = cams.shape
            loss_er = torch.topk(torch.flatten(torch.abs(cams.detach()-sgcs), start_dim=1), k=int(0.2*valid_channel*h*w), dim=-1)[0].mean()
            loss = loss_cls + loss_er

            loss_imc = 0
            if ep >= 4:
                loss_imc = criterion5(emb, label)
                if torch.is_tensor(loss_imc):
                    loss = loss_cls + loss_imc + loss_er

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_pixpro = 0
            if ep >= 8:
                model.eval()
                _, sgcs_vw1 = model(view1, cam='pix')
                
                with torch.no_grad():
                    cams_vw2, _, = model(view2, cam='pix')

                loss_pixpro = PixPro(cam_maxnorm(sgcs_vw1)*label_with_bg.unsqueeze(2).unsqueeze(3), cam_maxnorm(cams_vw2)*label_with_bg.unsqueeze(2).unsqueeze(3), coord1, coord2)

            loss = loss_pixpro

            loss_emd = 0
            if ep >= 12:
                vw1 = cam_softmaxnorm(sgcs_vw1)
                vw2 = cam_softmaxnorm(cams_vw2)

                vw1 = F.normalize(vw1, dim=1)
                vw2 = F.normalize(vw2, dim=1)
                crops_vw1, crops_vw2, batch_indices = torchutils.get_dynamic_crops(vw1, coord1, vw2.detach(), coord2)
                loss_emd = criterion4(crops_vw1, crops_vw2, mode='dynamic') 
                del crops_vw1
                del crops_vw2
                loss += loss_emd

            if torch.is_tensor(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            del cams
            del sgcs

            if iter % 25 == 0:
                cams = cam_maxnorm(raw_cams)[0] #C,H,W
                sgcs = cam_maxnorm(raw_sgcs.detach())[0]
                norm_sgc = sgcs.cpu().data.numpy().transpose(1,2,0) # H,W,C
                norm_cam = cams.cpu().data.numpy().transpose(1,2,0) # H,W,
                timer.update_progress(iter / max_step + 1)


                print('Iter:%5d/%5d' % (iter + max_step//args.max_epoches*ep, max_step),
                    'loss_focal:%.4f' % (loss_focal),
                    'loss_softmargin:%.4f' % (loss_softmargin),
                    'loss_pair:%.4f' % (loss_pair),
                    'loss_er:%.4f' % (loss_er),
                    'loss_imc:%.4f' % (loss_imc),
                    'loss_pixc:%.4f' % (loss_pixpro),
                    'loss_emd:%.4f' % (loss_emd),
                    
                    'imps:%.1f' % ((iter+1) * args.batch_size  / timer.get_stage_elapsed()),
                    'Fin:%s' % (timer.str_estimated_complete()),
                    'lr: %.7f' % (optimizer.param_groups[0]['lr'])
                    )

                # Visualization for training process
                img_8 = img[0].cpu().numpy().transpose((1,2,0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
                img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
                img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)

                for i in range(1, norm_cam.shape[2]):
                    if label[0,i-1]>0: #if its interested classes
                        cam = norm_cam[:,:,i]
                        sgc = norm_sgc[:,:,i]
                        vis_cam = show_cam_on_image(img_8, cam)
                        vis_sgc = show_cam_on_image(img_8, sgc)
                        tblogger.add_image('cam_on_img', vis_cam.transpose(2,0,1), valid_cam)
                        tblogger.add_image('sgc_on_img', vis_sgc.transpose(2,0,1), valid_cam)

                        valid_cam += 1

        else:
            print('')
 
        # scheduler.step()
        torch.save(model.state_dict(), os.path.join(args.session_name,'_{}'.format(str(ep)) + '.pth'))
        
        ###rapid eval for lr scheduler
        model.eval()
        stamp = time.time()
 
        os.makedirs('./training_eval', exist_ok=True)
        for iter, (img_name, img_list, label) in enumerate(eval_data_loader):
            img_name = img_name[0]; label = label[0].unsqueeze(0)
            img = img_list[0]
            label_with_bg = label.clone()
            bg_score = torch.ones((label.shape[0], 1), device=label_with_bg.device)
            label_with_bg = torch.cat((bg_score, label_with_bg), dim=1)
            with torch.no_grad():
                _, pred, _, _ = model(img.cuda().float(), cam='cam')
                pred = cam_maxnorm(pred)
                pred *= label_with_bg.unsqueeze(2).unsqueeze(2).to(pred.device)
                pred = pred.cpu().squeeze().numpy().astype(np.half) #c,h,w
                sgc_dict = {}
                for i in range(20):
                    sgc_dict[i] = pred[i+1]
                np.save(os.path.join('./training_eval', img_name + '.npy'), sgc_dict)
        
        df = pd.read_csv('data/train.txt', names=['filename'])
        name_list = df['filename'].values
        from src.evaluation import do_python_eval
        mious = []
        for t in range(20,52, 2):
            t /= 100.0
            loglist = do_python_eval('training_eval', 'data/VOC2012/SegmentationClass', name_list, 21, 'npy', t, printlog=False)
            mious.append(loglist['mIoU'])
        max_miou = max(mious)
        max_t = mious.index(max_miou)*0.02 + 0.2
        print(f'\n Epoch:{ep} max miou:{max_miou} max t:{max_t}',
               f'Time elapse:{time.time()-stamp}s')
        scheduler.step(max_miou)
        timer.reset_stage()
    tblogger.close()

