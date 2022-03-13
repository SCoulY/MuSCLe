# MuSCLe
 The implementation of: A Multi-Strategy Contrastive Learning Framework for Weakly Supervised Semantic Segmentation  (MuSCLe). Current repository only contains
inference and evaluation codes, training procedure will be made public after acceptance.

 
##Preparation

Data: Download PASCAL VOC 2012 devkit (follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit).

Model: Download model weights from [google drive](https://drive.google.com/drive/folders/1K3mMECLdWdu8YVrMq8YblppRdLtCcAaW?usp=sharing), including pretrained MCL, MuSCLe and IRN models.


##CAM generation
```
python infer_mcl.py --voc12_root PATH_TO_VOC12 --infer_list PATH_TO_INFER_LIST --weights PATH_TO_TRAINED_MODEL --out_npy OUTPUT_DIR
```

##CAM refinement
```
python infer_irn.py --cam_dir CAM_DIR --sem_seg_out_dir OUTPUT_PSEUDO_LABEL_DIR --irn_weights_name PATH_TO_PRETRAINED_IRN_MODEL
```

##CAM quality evaluation
###Raw CAM evaluation
```
cd src
python evaluation.py --comment COMMENTS --type npy --list PATH_TO_INFER_LIST --predict_dir CAM_DIR --curve True
```

###Refined CAM evaluation
```
cd src
python evaluation.py --comment COMMENTS --type png --list PATH_TO_INFER_LIST --predict_dir REFINED_CAM_DIR 
```

##Semantic segmentation inference
- sepcify model to use in argument --pretrained (default b7)
```
python infer_seg.py --weights PATH_TO_TRAINED_MuSCLe_MODEL --out_seg OUTPUT_SEGMENTATION_MAP_DIR --infer_list PATH_TO_INFER_LIST --crf 1 --pretrained b7
```

##Acknowledges
We thank the author of [IRN](https://github.com/jiwoon-ahn/irn) and [SEAM](https://github.com/YudeWang/SEAM) for their great work. This repository heavily relys on their processing pipeline and evaluation codes.
