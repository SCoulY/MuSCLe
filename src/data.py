import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Subset
import imageio
from .imutils import *

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

def split_dataset(dataset, n_splits):

    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):
    cls_labels_dict = np.load('data/cls_labels.npy', allow_pickle=True).item()
    labels = [cls_labels_dict[x] for x in img_name_list]
    return labels

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()

    img_name_list = [img_gt_name.split(' ')[0].split('/')[-1].split('.')[0] for img_gt_name in img_gt_name_list]

    return img_name_list

class VOC12SegDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, mask_root,
                min_scale=0.5, max_scale=1.5, crop_size=448, mask_type='soft',
                inference=False):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.mask_root = mask_root
        self.mask_type = mask_type
        self.inference = inference
        if not self.inference:
            self.colorjitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            self.resizelong = RandomResizeLongWithMask(min_scale, max_scale, mask_type=mask_type)
            self.crop =  RandomCropWithMask(crop_size)
            self.flip =  RandomHorizontalFlipWithMask()
            # self.rot90 =  Rot90WithMask()
            # self.cutout =  Cutout(mask_size=66, p=0.5)

        

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        label = torch.from_numpy(self.label_list[idx])
        name = self.img_name_list[idx]
        # return name, label

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        if self.mask_type == 'hard':
            mask = PIL.Image.open(os.path.join(self.mask_root, name + '.png'))
        elif self.mask_type == 'soft':
            mask = np.load(os.path.join(self.mask_root, name + '.npy'), allow_pickle=True).astype(np.float)
        
        if not self.inference:
            img = self.colorjitter(img)
            img, mask = self.resizelong(img, mask)
        
            img =  color_norm(np.asarray(img))
            if self.mask_type == 'hard':
                mask = np.expand_dims(np.asarray(mask), axis=2) #h,w,1
            img, mask = self.crop(img, mask)
            img, mask = self.flip(img, mask)
            # img, mask = self.cutout(img, mask)
            # img, mask = self.rot90(img, mask)

        else:
            img =  color_norm(np.asarray(img))
            if self.mask_type == 'hard':
                mask = np.expand_dims(np.asarray(mask), axis=2) #h,w,1

        img = torch.from_numpy( HWC_to_CHW(img))
        mask = torch.from_numpy( HWC_to_CHW(mask))
        return name, img, label, mask


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return name, img

class VOC12ImageDatasetMS(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        return name, ms_img_list

class VOC12ImageDatasetMSF(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list

class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        #self.label_list = load_image_label_list_from_xml(self.img_name_list, self.voc12_root)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])
        return name, img, label

class VOC12ImageViews(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None, output_size=(224,224)):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.output_size = output_size
        self.view_transform = transforms.Compose([
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        np.asarray,
                         color_norm,
                         HWC_to_CHW,
                        torch.from_numpy
                    ])

    def __len__(self):
        return len(self.img_name_list)

    def get_inter(self, coord1, coord2):
        h11, w11, h12, w12 = coord1[0], coord1[1], coord1[0]+coord1[2], coord1[1]+coord1[3]
        h21, w21, h22, w22 = coord2[0], coord2[1], coord2[0]+coord2[2], coord2[1]+coord2[3]
        y_top = max(h11, h21)
        x_left = max(w11, w21)
        y_bot = min(h12, h22)
        x_right = min(w12, w22)

        if y_bot - y_top <= 0 or x_right - x_left <= 0:
            return False, False, False

        h_inter = y_bot - y_top  
        w_inter = x_right - x_left
        
        if (y_top, x_right) == (h11, w12):
            rel_h1 = 0
            rel_w1 = w21 - w11
            rel_h2 = h11 - h21
            rel_w2 = 0
        elif (y_bot, x_right) == (h12, w12):
            rel_h1 = h21 - h11
            rel_w1 = w21 - w11
            rel_h2 = 0
            rel_w2 = 0
        elif (y_top, x_left) == (h11, w11):
            rel_h1 = 0
            rel_w1 = 0
            rel_h2 = h11 - h21
            rel_w2 = w11 - w21
        elif (y_bot, x_left) == (h12, w11):
            rel_h1 = h21 - h11
            rel_w1 = 0
            rel_h2 = 0
            rel_w2 = w11 - w21
        else:
            print('coordinates error!')

        return (rel_h1, rel_w1, h_inter, w_inter), (rel_h2, rel_w2, h_inter, w_inter), (x_left, y_top, h_inter, w_inter)
    

    def get_views(self, img):
        w, h = img.size
        if w < 448 or h < 448:
            img = F.resize(img, size=(448,448))
        w, h = img.size
        th, tw = self.output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        i_1 = torch.randint(0, h - th + 1, size=(1, )).item()
        j_1 = torch.randint(0, w - tw + 1, size=(1, )).item()

        i_2 = torch.randint(0, h - th + 1, size=(1, )).item()
        j_2 = torch.randint(0, w - tw + 1, size=(1, )).item()

        coord1, coord2 = (i_1, j_1, th, tw), (i_2, j_2, th, tw)
        rel_coord1, rel_coord2, ori_coord = self.get_inter(coord1, coord2)
        while rel_coord1 == False:
            i_1 = torch.randint(0, h - th + 1, size=(1, )).item()
            j_1 = torch.randint(0, w - tw + 1, size=(1, )).item()

            i_2 = torch.randint(0, h - th + 1, size=(1, )).item()
            j_2 = torch.randint(0, w - tw + 1, size=(1, )).item()
            coord1, coord2 = (i_1, j_1, th, tw), (i_2, j_2, th, tw)
            rel_coord1, rel_coord2, ori_coord = self.get_inter(coord1, coord2)

        view_1 = F.crop(img, i_1, j_1, th, tw)
        view_2 = F.crop(img, i_2, j_2, th, tw)

        return view_1, view_2, rel_coord1, rel_coord2, ori_coord


    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        if torch.rand(1) < 0.5:
                img = F.hflip(img)
        view1, view2, coord1, coord2, ori_coord = self.get_views(img)
        if self.transform:
            img = self.transform(img)
            view1 = self.view_transform(view1)
            view2 = self.view_transform(view2) #torch.rot90(self.view_transform(view2), k=1, dims=(1,2))
        return name, img, view1, view2, coord1, coord2, ori_coord

class VOC12ClsPix(VOC12ImageViews):
    def __init__(self, img_name_list_path, voc12_root, transform=None, view_size=(224,224)):
        super().__init__(img_name_list_path, voc12_root, transform, output_size=view_size)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        #self.label_list = load_image_label_list_from_xml(self.img_name_list, self.voc12_root)

    def __getitem__(self, idx):
        name, img, view1, view2, coord1, coord2, ori_coord = super().__getitem__(idx)
        label = torch.from_numpy(self.label_list[idx])
        coord1 = torch.from_numpy(np.array(coord1)).long()
        coord2 = torch.from_numpy(np.array(coord2)).long()
        ori_coord = torch.from_numpy(np.array(ori_coord)).long()
        return name, img, label, view1, view2, coord1, coord2, ori_coord



class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

class VOC12ClsDatasetMS(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        return name, ms_img_list, label

class ExtractAffinityLabelInRadius():

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius-1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)

class VOC12AffDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_la_dir, label_ha_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_la_dir = label_la_dir
        self.label_ha_dir = label_ha_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_la_path = os.path.join(self.label_la_dir, name + '.npy')

        label_ha_path = os.path.join(self.label_ha_dir, name + '.npy')

        label_la = np.load(label_la_path, allow_pickle=True).item()
        label_ha = np.load(label_ha_path, allow_pickle=True).item()

        label = np.array(list(label_la.values()) + list(label_ha.values()))
        label = np.transpose(label, (1, 2, 0))

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        no_score_region = np.max(label, -1) < 1e-5
        label_la, label_ha = np.array_split(label, 2, axis=-1)
        label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
        label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
        label = label_la.copy()
        label[label_la == 0] = 255
        label[label_ha == 0] = 0
        label[no_score_region] = 255 # mostly outer of cropped region
        label = self.extract_aff_lab_func(label)

        return img, label

class VOC12AffGtDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_dir = label_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_path = os.path.join(self.label_dir, name + '.png')

        label = scipy.misc.imread(label_path)

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        label = self.extract_aff_lab_func(label)

        return img, label

class SBD(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = img_name_list_path
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(os.path.join(self.voc12_root, name+'.jpg')).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return name, img

class SBDMSF(SBD):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)

class VOC12SegmentationDataset(Dataset):

    def __init__(self, img_name_list_path, label_dir, crop_size, voc12_root,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_method = 'random'):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.label_dir = label_dir

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.voc12_root))
        label = imageio.imread(os.path.join(self.label_dir, name_str + '.png'))

        img = np.asarray(img)

        if self.rescale:
            img, label = random_scale((img, label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, label = random_lr_flip((img, label))

        if self.crop_method == "random":
            img, label = random_crop((img, label), self.crop_size, (0, 255))
        else:
            img = top_left_crop(img, self.crop_size, 0)
            label = top_left_crop(label, self.crop_size, 255)

        img =  HWC_to_CHW(img)

        return {'name': name, 'img': img, 'label': label}

class VOC12AffinityDataset(VOC12SegmentationDataset):
    def __init__(self, img_name_list_path, label_dir, crop_size, voc12_root,
                 indices_from, indices_to,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False, crop_method=None):
        super().__init__(img_name_list_path, label_dir, crop_size, voc12_root, rescale, img_normal, hor_flip, crop_method=crop_method)

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        reduced_label =  pil_rescale(out['label'], 0.25, 0)

        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = self.extract_aff_lab_func(reduced_label)

        return out

class VOC12ImageDatasetIRN(Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))

        if self.resize_long:
            img = random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = random_crop(img, self.crop_size, 0)
            else:
                img = top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = HWC_to_CHW(img)

        return {'name': name_str, 'img': img}