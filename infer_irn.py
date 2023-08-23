from PIL import Image
import torch
import torchvision

from src import *

import argparse
import importlib
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Random Walk Params
    parser.add_argument("--beta", default=8, type=int)
    parser.add_argument("--exp_times", default=6,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.", type=int)
    parser.add_argument("--sem_seg_bg_thres", default=0.35, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="src.backbones.resnet50_irn", type=str)
    parser.add_argument("--irn_weights_name", help="PATH_TO_IRN_WEIGHTS", type=str)
    parser.add_argument("--cam_dir", required=True, type=str)
    parser.add_argument("--sem_seg_out_dir", default="./irn_rw", type=str)

    parser.add_argument("--voc12_root", default='data/VOC2012', type=str)
    parser.add_argument("--infer_list", default="data/train.txt", type=str)
    parser.add_argument("--soft_output", default=0, type=int, help="output soft pseudo labels for BEACON training")
    args = parser.parse_args()

    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()
    model = model.cuda()

    infer_dataset = data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                        scales=[1.0],
                                        inter_transform=torchvision.transforms.Compose(
                                            [np.asarray,
                                            imutils.color_norm,
                                            imutils.HWC_to_CHW]))


    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=4, pin_memory=True)

    cmap = imutils.color_map()[:, np.newaxis, :]
    
    if args.soft_output:
        os.makedirs(args.sem_seg_out_dir, exist_ok=True)    
    os.makedirs(args.sem_seg_out_dir+'_png', exist_ok=True)
    
    with torch.no_grad():
        for (name, img_list, label) in tqdm(infer_data_loader):
            img_name = name[0]
            img = torch.cat(img_list, dim=0) # 2,c,h,w contains a image and its flipped version
            orig_img_size = img.shape
            edge, dp = model(img.cuda(non_blocking=True).float())

            cam = np.load(os.path.join(args.cam_dir, img_name + '.npy'), allow_pickle=True).item()

            cam_arr = np.zeros((20, orig_img_size[2], orig_img_size[3]), np.float32)

            for k, v in cam.items():
                cam_arr[k] = v

            cams = torch.from_numpy(cam_arr)
            downscale_cams = F.interpolate(cams.unsqueeze(0), size=edge.shape[1:], mode='bilinear', align_corners=False)
            rw = indexing.propagate_to_edge(downscale_cams.cuda(), edge, beta=args.beta, exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[2], :orig_img_size[3]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            

            if args.soft_output:
                rw_pred = torch.squeeze(rw_up_bg).permute(1,2,0).cpu().numpy()
                res = rw_pred.astype(np.half)
                np.save(os.path.join(args.sem_seg_out_dir, img_name + '.npy'), res)
            
            else:
                rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()
                res = rw_pred.astype(np.uint8)
                res = Image.fromarray(res)
                res.putpalette(cmap)
                res.save(os.path.join(args.sem_seg_out_dir+'_png', img_name + '.png'))