from sched import scheduler
import numpy as np
import torch
import random
import cv2
import time
import os
from PIL import Image

from torch.utils.data import DataLoader

from data import *
import argparse

from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn

from torch.optim import lr_scheduler
from src.loss_multilabel import *
from src.MuSCLe import *
from src import imutils, pyutils


def cam_maxnorm(cams):
    cams = torch.relu(cams)
    n,c,h,w = cams.shape
    cam_min = torch.min(cams.view(n,c,-1), dim=-1)[0].view(n,c,1,1)
    cam_max = torch.max(cams.view(n,c,-1), dim=-1)[0].view(n,c,1,1)
    norm_cam = (cams - cam_min - 1e-6)/ (cam_max - cam_min + 1e-6)
    norm_cam = torch.relu(norm_cam)
    return norm_cam

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = heatmap * 0.5 + np.float32(img) * 0.5
    cam = np.uint8(cam)
    return cam #H, W, C

def get_sample_weight(dataset):

    if os.path.exists('sample_weight.npy'):
        with open('sample_weight.npy','rb') as f:
            sample_weight = np.load(f, allow_pickle=True)
            print(f'sample weight loaded!')
            return sample_weight
    
    else:
        t1 = time.time()
        sample_weight = [0]*len(dataset)
        class_count = [590, 504, 705, 468, 714, 393, 1150, 1005, 1228, 267,
                613, 1188, 445, 492, 4155, 522, 300, 649, 503, 567]
        sum_instance = len(dataset)  #adjustable
        for i, (_, label) in enumerate(dataset):
            multihots = torch.where(label==1)[0]
            instance_count = 0
            for hot in multihots:
                instance_count += class_count[hot.item()]
            sample_weight[i] = instance_count/sum_instance
        print(f'calculate sample weight takes:{time.time()-t1}seconds')
        np.save('sample_weight.npy', 1/np.array(sample_weight))
        return sample_weight


# class_count = [590, 504, 705, 468, 714, 393, 1150, 1005, 1228, 267,
#                613, 1188, 445, 492, 4155, 522, 300, 649, 503, 567]
# class_count = [sum(class_count)] + class_count
# class_count = np.array(class_count)
# class_count = torch.from_numpy(class_count).cuda()
# class_weight = torch.sum(class_count)/class_count
# class_weight = class_weight/class_weight.max()

# class_iou = np.array([82.69, 63.82, 38.65, 45.52, 39.09, 51.75, 74.02, 65.10, 52.07, 35.9, 73.12, 
#              52.62, 58.22, 71.72, 73.72, 56.11, 43.29, 78.29, 53.16, 62.58, 36.32])
# class_weight = class_iou/class_iou.max()
# class_weight = torch.from_numpy(class_weight).cuda().float()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=1e-5, type=float)
    parser.add_argument("--train_list", default="data/VOC2012/train_aug.txt", type=str)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--session_name", default="runs/muscle", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--voc12_root", default='data/VOC2012', type=str)
    parser.add_argument("--mask_root", help='PATH_TO_PSEUDO_LABEL_DIR', type=str)
    parser.add_argument("--k", default=128, type=int)
    parser.add_argument("--step", default=7, type=int)
    parser.add_argument("--lamb", default=5e-2, type=float)
    parser.add_argument("--tblog_dir", default='logs/tblog_muscle', type=str)
    parser.add_argument("--cls_dir", default=None, type=str)
    parser.add_argument("--crf", default=0, type=int)
    parser.add_argument("--seed", default=221, type=int)
    parser.add_argument("--pretrained", default='b7', type=str)
    parser.add_argument("--bifpn", default=3, type=int)
    args = parser.parse_args()

    print(vars(args))
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    model = MuSCLe(num_classes=args.num_classes, pretrained='efficientnet-'+args.pretrained, layers=args.bifpn, MemoryEfficient=True, mode='dec', last_pooling=True)

    # print(model._modules.items())

    # model.apply(model.weights_init)
            
    os.makedirs(args.tblog_dir, exist_ok=True)
    os.makedirs(args.session_name, exist_ok=True)
    tblogger = SummaryWriter(args.tblog_dir)	


    train_dataset = VOC12SegDataset(args.train_list, 
                                    args.voc12_root, 
                                    args.mask_root, 
                                    min_scale=0.5, 
                                    max_scale=1.75, 
                                    crop_size=args.crop_size, 
                                    mask_type='soft')
    

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   num_workers=args.num_workers, pin_memory=True, 
                                   drop_last=True, shuffle=True, prefetch_factor=4)
    
    eval_dataset = VOC12ClsDatasetMSF("data/val.txt", voc12_root=args.voc12_root,
                                                  scales=[1],
                                                  inter_transform=transforms.Compose(
                                                       [np.asarray,
                                                        imutils.color_norm,
                                                        imutils.HWC_to_CHW]))

    eval_data_loader = DataLoader(eval_dataset, shuffle=False, num_workers=args.num_workers)

    


    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_dec)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=0, cooldown=0, factor=0.5, min_lr=5e-6)


    if args.weights:
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)


    model = model.cuda()
    torch.autograd.set_detect_anomaly(True)
    
    criterion1 = nn.CrossEntropyLoss() 
    
    criterion2 = FieldLoss(sobel_size=5, beta=1e2, k=args.k)
    timer = pyutils.Timer("Session started: ")

    valid_cam = 0

    for ep in range(args.max_epoches):
        model.train()
        print('lr: %.6f' % (optimizer.param_groups[0]['lr']))
        for iter, pack in enumerate(train_data_loader):
            optimizer.zero_grad()

            name, img, label, mask = pack
            _, C, H, W = img.shape

            if torch.cuda.is_available():
                label = label.cuda()
                img = img.cuda().float()
                mask = mask.cuda().squeeze(1) #N,H,W

            with torch.no_grad():
                label_with_bg = label.clone()
                bg_score = torch.ones((label.shape[0], 1), device=label_with_bg.device)
                label_with_bg = torch.cat((bg_score, label_with_bg), dim=1)


            seg_map, dense_ft = model(img, cam='seg')
            hard_mask = torch.argmax(mask, dim=1)
            
            l1 = criterion1(seg_map, hard_mask)

            loss = l1
            l2 = 0
            if args.lamb > 0:
                l2, edge_fg = criterion2(seg_map, dense_ft, mask, label_with_bg, args.step)
                if torch.is_tensor(l2):
                    loss = l1 + args.lamb*l2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 9, norm_type=2)
            optimizer.step()
            # scheduler.step()

            del dense_ft
            del seg_map


            if iter % 25 == 0:
                timer.update_progress(iter / max_step + 1)

                print('Iter:%5d/%5d' % (iter + max_step//args.max_epoches*ep, max_step),
                    'loss_seg:%.4f' % (l1),
                    'loss_beacon:%.4f' % (l2),
                    'imps:%.1f' % ((iter+1) * args.batch_size  / timer.get_stage_elapsed()),
                    'Fin:%s' % (timer.str_estimated_complete()),
                    )


        torch.save(model.state_dict(), os.path.join(args.session_name, '_{}'.format(str(ep)) + '.pth'))
        # scheduler.step()

        ###rapid eval during training
        model.eval()
        stamp = time.time()
        res = [[]]*21
        TP = [0]*21
        P = [0]*21
        T = [0]*21
        IoU = []
        for iter, (img_name, img_list, label) in enumerate(eval_data_loader):
            img_name = img_name[0]; label = label[0].unsqueeze(0)
            # img = img_list[0]
            img_path = get_img_path(img_name, args.voc12_root)
            orig_img = np.asarray(Image.open(img_path))
            gt_file = os.path.join('data/VOC2012/SegmentationClass','%s.png'%img_name)
            gt = np.array(Image.open(gt_file))
            H, W = gt.shape
            label_with_bg = label.clone()
            bg_score = torch.ones((label.shape[0], 1), device=label_with_bg.device)
            label_with_bg = torch.cat((bg_score, label_with_bg), dim=1)
            seg_list = []

            for i, img in enumerate(img_list[:1]):
                with torch.no_grad():
                    seg, _ = model(img.cuda().float(), cam='seg')

                    seg = torch.softmax(seg, dim=1)
                    raw_seg = seg.squeeze(0).cpu().data.numpy() # C,H,W
                    raw_seg = raw_seg.transpose(1,2,0) #H,W,C
                    raw_seg = cv2.resize(raw_seg.astype(np.float32), (W, H))


                if i % 2 == 1:
                    raw_seg = np.flip(raw_seg, axis=1)

                raw_seg = raw_seg.transpose(2,0,1) #C,H,W
                seg_list.append(raw_seg)
                
            pred = np.mean(seg_list, axis=0)

            if args.cls_dir:
                cls_label = np.load(os.path.join(args.cls_dir, img_name+'.npy'), allow_pickle=True).squeeze()
                pred[1:] = pred[1:] * cls_label[1:, np.newaxis, np.newaxis]

            if args.crf:  
                pred = imutils.crf_inference(orig_img, pred, t=1)

            pred = np.argmax(pred, axis=0)
            cal = gt<255
            mask = (pred==gt) * cal

            for i in range(21):
                P[i] += np.sum((pred==i)*cal)
                T[i] += np.sum((gt==i)*cal)
                TP[i] += np.sum((gt==i)*mask)

        for i in range(21):
            IoU.append(TP[i]/(T[i]+P[i]-TP[i]+1e-10))
        miou = np.mean(np.array(IoU))
        print(f'\n Epoch:{ep} val miou:{miou}',
               f'Time elapse:{time.time()-stamp}s')
        scheduler.step(miou)
        timer.reset_stage()

    tblogger.close()

        
