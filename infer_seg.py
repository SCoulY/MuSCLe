import numpy as np
import torch
import cv2
import os
from torch.utils.data import DataLoader
import torchvision
import src.imutils as imutils
import argparse
from src import *
from PIL import Image
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

def softmax_np(x, axis=0):
    '''x is of shape (c,h,w)'''
    return np.exp(x)/np.sum(np.exp(x), axis=axis)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", help='PATH_TO_MuSCLe_WEIGHTS', type=str)
    parser.add_argument("--infer_list", default="data/val.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--tblog", default="logs/infer", type=str)
    parser.add_argument("--voc12_root", default='data/VOC2012', type=str)
    parser.add_argument("--cls_dir", default=None, type=str)
    parser.add_argument("--out_seg", default=None, type=str)
    parser.add_argument("--out_seg_pred", default=None, type=str)
    parser.add_argument("--crf", default=1, type=int)
    parser.add_argument("--bifpn", default=3, type=int)
    parser.add_argument("--pretrained", default='b7', type=str)


    args = parser.parse_args()
    model = MuSCLe(num_classes=args.num_classes, pretrained='efficientnet-'+args.pretrained, layers=args.bifpn, MemoryEfficient=True, last_pooling=True, mode='dec')
   
    if '.ckpt' in args.weights:
        model.load_state_dict(torch.load(args.weights)['state_dict'], strict=False)
    else:
        model.load_state_dict(torch.load(args.weights), strict=False)

    writer = SummaryWriter(args.tblog)
        
    infer_dataset = data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                  scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
                                                  inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        imutils.color_norm,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers)

    model = model.cuda()
    model.eval()
    global_step = 0



    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0].unsqueeze(0)

        img_path = data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        H, W, _ = orig_img.shape
        seg_list = []

        label_with_bg = label.clone()
        bg_score = torch.ones((label.shape[0], 1), device=label_with_bg.device)
        label_with_bg = torch.cat((bg_score, label_with_bg), dim=1)

        for i, img in enumerate(img_list):
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
 
        norm_seg = np.mean(seg_list, axis=0)
        # norm_seg = np_softmax(norm_seg, axis=0)


        # seg_dict = {}

        if args.cls_dir:
            cls_label = np.load(os.path.join(args.cls_dir, img_name+'.npy'), allow_pickle=True).squeeze()
            norm_seg[1:] = norm_seg[1:] * cls_label[1:, np.newaxis, np.newaxis]
        # norm_seg[1:] = norm_seg[1:] * label.squeeze(0).cpu().numpy()[:, np.newaxis, np.newaxis]

        if args.crf:  
            norm_seg = imutils.crf_inference(orig_img, norm_seg, t=4)

        if args.out_seg is not None:
            os.makedirs(args.out_seg, exist_ok=True)
            cv2.imwrite(os.path.join(args.out_seg, img_name + '.png'), np.argmax(norm_seg, axis=0))

        print(img_name, iter)
    writer.close()

