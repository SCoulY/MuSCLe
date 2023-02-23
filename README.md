# MuSCLe
 The implementation of: A Multi-Strategy Contrastive Learning Framework for Weakly Supervised Semantic Segmentation  (MuSCLe).
 
## Preparation

### Data
- Download PASCAL VOC 2012 devkit (follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit). Put the data under ./data/VOC2012 folder.

### Model
- Download model weights from [google drive](https://drive.google.com/drive/folders/1K3mMECLdWdu8YVrMq8YblppRdLtCcAaW?usp=sharing), including pretrained MCL, MuSCLe and IRN models.

### Packages
- install conda from [conda.io](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- build cond env from file
```
conda create --name muscle --file requirements.txt 
conda activate muscle
```

## MCL training
```
python train_mcl.py --voc12_root data --train_list data/train_aug.txt --weights PATH_TO_TRAINED_MODEL --tblog_dir logs/tblog_mcl
```

## CAM generation
```
python infer_mcl.py --voc12_root PATH_TO_VOC12 --infer_list PATH_TO_INFER_LIST --weights PATH_TO_TRAINED_MODEL --out_npy OUTPUT_DIR
```

## CAM refinement & Pseudo label generation
### turn on flag ```--soft_output 1 ``` to store soft pseudo labels for BEACON training
```
python infer_irn.py --cam_dir CAM_DIR --sem_seg_out_dir OUTPUT_PSEUDO_LABEL_DIR -- soft_output 0 --irn_weights_name PATH_TO_PRETRAINED_IRN_MODEL
```

## CAM quality evaluation
### Raw CAM evaluation
```
cd src
python evaluation.py --comment COMMENTS --type npy --list data/train.txt --predict_dir CAM_DIR --curve True
cd ..
```

### Refined CAM evaluation
```
cd src
python evaluation.py --comment COMMENTS --type png --list data/train.txt --predict_dir REFINED_CAM_DIR 
cd ..
```

### MuSCLe training
```
python train_muscle.py --voc12_root data --train_list data/train_aug.txt --weights PATH_TO_TRAINED_MODEL --tblog_dir logs/tblog_muscle --mask_root OUTPUT_PSEUDO_LABEL_DIR --session_name runs/muscle
``` 

### Semantic segmentation inference
- sepcify model to use in argument --pretrained (default b7)
```
python infer_seg.py --weights PATH_TO_TRAINED_MuSCLe_MODEL --out_seg OUTPUT_SEGMENTATION_MAP_DIR --infer_list PATH_TO_INFER_LIST --crf 1 --pretrained b7
```

## Acknowledges
We thank the author of [IRN](https://github.com/jiwoon-ahn/irn) and [SEAM](https://github.com/YudeWang/SEAM) for their great work. This repository heavily relys on their processing pipeline and evaluation codes.

## Citation
Please cite our work if it's helpful to your research.
```
@article{yuan2023multi,
  title={A Multi-Strategy Contrastive Learning Framework for Weakly Supervised Semantic Segmentation},
  author={Yuan, Kunhao and Schaefer, Gerald and Lai, Yu-Kun and Wang, Yifan and Liu, Xiyao and Guan, Lin and Fang, Hui},
  journal={Pattern Recognition},
  pages={109298},
  year={2023},
  publisher={Elsevier}
}
```