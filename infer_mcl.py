import numpy as np
import torch
import cv2
import os
from data import *
from torch.utils.data import DataLoader
import torchvision
import imutils
import argparse
from src.MuSCLe import *
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

label_dict = {0:'Background', 1:'Aeroplane', 2:'Bicycle', 3:'Bird', 4:'Boat', 5:'Bottle', 6:'Bus',
              7:'Car', 8:'Cat', 9:'Chair', 10:'Cow', 11:'Diningtable',
              12:'Dog', 13:'Horse', 14:'Motorbike', 15:'Person', 16:'Pottedplant',
              17:'Sheep', 18:'Sofa', 19:'Train', 20:'TVmonitor'}

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = heatmap * 0.3 + np.float32(img) * 0.5
    cam = np.uint8(cam)
    return cam

def cam_maxnorm(cams):
    cams = torch.relu(cams)
    n,c,h,w = cams.shape
    cam_min = torch.min(cams.view(n,c,-1), dim=-1)[0].view(n,c,1,1)
    cam_max = torch.max(cams.view(n,c,-1), dim=-1)[0].view(n,c,1,1)
    norm_cam = (cams - cam_min - 1e-6)/ (cam_max - cam_min + 1e-6)
    norm_cam = torch.relu(norm_cam)
    return norm_cam

def cam_softmaxnorm(cams):
    cams = torch.relu(cams)
    n,c,h,w = cams.shape
    foreground = torch.softmax(cams[:,1:,:,:], dim=1)
    background = (1-torch.max(foreground, dim=1)[0]).unsqueeze(1)
    norm_cam = torch.cat([background, foreground], dim=1)
    return norm_cam

def accuracy(output, target, topk=(1,5), num_classes=20):
    maxk = max(topk)
    batch_size = target.size(0) #N, 20

    _, pred = output.float().topk(maxk, 1, True, True)
    # pred = F.one_hot(pred, num_classes) #N, 5, 20
    res = [0,0]
    pred_topk = pred[:,:maxk]
    for bs in range(batch_size):
        correct = [0]*maxk
        for k in range(maxk):
            correct[k] = 1 if target[bs, pred_topk[bs, k]]==1 else 0
        res[0] += correct[0]
        res[1] += max(correct)
    res[0]/=batch_size
    res[1]/=batch_size
    return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='/home/lunet/coky/CAM/EffSeg_pix_1000.pth', type=str)
    parser.add_argument("--infer_list", default="/home/lunet/coky/CAM/voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--tblog", default=None, type=str)
    parser.add_argument("--voc12_root", default='/home/lunet/coky/CAM/VOC2012', type=str)
    parser.add_argument("--out_npy", default=None, type=str)
    parser.add_argument("--out_png", default=None, type=str)



    args = parser.parse_args()
    model = MuSCLe(num_classes=args.num_classes, pretrained='efficientnet-b3', layers=3, MemoryEfficient=True, last_pooling=False)
    if '.ckpt' in args.weights:
        model.load_state_dict(torch.load(args.weights)['state_dict'], strict=False)
    else:
        model.load_state_dict(torch.load(args.weights), strict=False)

    model.eval()
    model.cuda()

    if args.tblog is not None:
        os.makedirs(args.tblog, exist_ok=True)
    writer = SummaryWriter(args.tblog)
    cmap = imutils.color_map()[:, np.newaxis, :]
    infer_dataset = VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                  scales=[0.5,1,1.5,2],
                                                  inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        imutils.color_norm,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers)

    model = model.cuda()
    model.eval()
    global_step = 0
    top1 = 0
    top5 = 0

    # os.makedirs(args.out_npy, exist_ok=True)
    if args.out_npy is not None:
        os.makedirs(args.out_npy+'_sgc', exist_ok=True)

    for iter, (img_name, img_list, label) in tqdm(enumerate(infer_data_loader)):
        img_name = img_name[0]; label = label[0].unsqueeze(0)

        # if label[0, 19] == 0:
        #     continue

        img_path = get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        H, W, _ = orig_img.shape
        raw_cam_list, SGC_list, score_list = [], [], []
        # seg_list = []

        label_with_bg = label.clone()
        bg_score = torch.ones((label.shape[0], 1), device=label_with_bg.device)
        label_with_bg = torch.cat((bg_score, label_with_bg), dim=1)

        for i, img in enumerate(img_list):
            with torch.no_grad():
                raw_cam, SGC, emb, score = model(img.cuda().float(), cam='cam')

                # raw_cam = cam_maxnorm(raw_cam)
                raw_cam = raw_cam.squeeze(0).cpu().data.numpy() # C,H,W
                raw_cam = raw_cam.transpose(1,2,0) #H,W,C
                raw_cam = cv2.resize(raw_cam.astype(np.float32), (W, H))

                # SGC = cam_maxnorm(SGC)
                SGC = SGC.squeeze(0).cpu().data.numpy() # C,H,W
                SGC = SGC.transpose(1,2,0) #H,W,C
                SGC = cv2.resize(SGC.astype(np.float32), (W, H))

            if i % 2 == 1:
                raw_cam = np.flip(raw_cam, axis=1)
                SGC = np.flip(SGC, axis=1)

            raw_cam = raw_cam.transpose(2,0,1) #C,H,W
            SGC = SGC.transpose(2,0,1) #C,H,W
 
            raw_cam_list.append(raw_cam[1:]) 
            SGC_list.append(SGC[1:])
            score_list.append(score[:,1:])

        score = torch.mean(torch.cat(score_list, dim=0), dim=0)
        score = torch.sigmoid(score)
 
        norm_cam = np.sum(raw_cam_list, axis=0)
        norm_cam[norm_cam < 0] = 0
        cam_max = np.max(norm_cam, (1,2), keepdims=True)
        cam_min = np.min(norm_cam, (1,2), keepdims=True)
        norm_cam[norm_cam < cam_min+1e-6] = 0
        norm_cam = (norm_cam-cam_min-1e-6) / (cam_max - cam_min + 1e-6)


        norm_SGC = np.sum(SGC_list, axis=0)
        norm_SGC[norm_SGC < 0] = 0
        SGC_max = np.max(norm_SGC, (1,2), keepdims=True)
        SGC_min = np.min(norm_SGC, (1,2), keepdims=True)
        norm_SGC[norm_SGC < SGC_min+1e-6] = 0
        norm_SGC = (norm_SGC-SGC_min-1e-6) / (SGC_max - SGC_min + 1e-6)

        cam_dict = {}
        sgc_dict = {}
        score_dict = {}
        for i in range(20):
            # print({label_dict[i+1]:score[i].item()})
            score_dict.update({label_dict[i]:score[i].item()})
            if label[:,i] > 1e-5:
                cam_dict[i] = norm_cam[i]
                sgc_dict[i] = norm_SGC[i]

        # prec1, prec5 = accuracy(score.view(1,-1).cpu(), label.view(1, -1), topk=(1, 5))
        # top1 += prec1
        # top5 += prec5

        if args.out_npy is not None:
            # np.save(os.path.join(args.out_npy, img_name + '.npy'), cam_dict)
            np.save(os.path.join(args.out_npy+'_sgc', img_name + '.npy'), sgc_dict)


        if args.tblog is not None:
            for c in range(20):
                if label[:,c] > 1e-5:
                    vis_cam = norm_cam[c]
                    vis_cam = show_cam_on_image(orig_img, vis_cam)
                    vis_cam = torch.from_numpy(vis_cam.transpose(2,0,1))

                    vis_sgc = norm_SGC[c]
                    vis_sgc = show_cam_on_image(orig_img, vis_sgc)
                    vis_sgc = torch.from_numpy(vis_sgc.transpose(2,0,1))

                    writer.add_image('cam_on_img', vis_cam, global_step)
                    writer.add_image('sgc_on_img', vis_sgc, global_step)

                    writer.add_scalars('confidence_score', score_dict, global_step)
                    global_step += 1
        



        print(img_name, iter)
    writer.close()
    # print(f'top1:{top1/len(infer_data_loader)} top5:{top5/len(infer_data_loader)}')
