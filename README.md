# ScribbleSimulation
This repository is created for reimplementation of the ScribbleCOCO and ScribbleCityscapes. The codes are available at [BaiduNetdisk](https://pan.baidu.com/s/1iE5thVH_z_Xm7q_h7Bar6Q). The extraction code is in the paper.

## Terms of use
- ScribbleCOCO: We follow the same license as [COCO dataset for images](https://cocodataset.org/#termsofuse). 
- ScribbleCityscapes: We follow the same license as [The Cityscapes Dataset](https://github.com/mcordts/cityscapesScripts).
- The datasets cannot be used for commercial purposes. The datasets are created for research purposes.
## ScribbleCOCO
The complete dataset is avaliable at [BaiduNetdisk](https://pan.baidu.com/s/1bTRDR9BqDyaLcfynN2bpvg?pwd=t817).
The ScribbleCOCO need at leat 24G space. The dataset is recommended to untar with the following structure:
```bash
ScribbleCOCO/
├── ImageSets/SegmentationAug
    ├──train.txt
    └──val.txt
├── coco2014_train_scribble_r1(The scribble masks, png files, about 923M.)
├── coco2014_train_scribble_r2
├── coco2014_train_scribble_r3
├── JPEGImages (The jpeg image files, about 19G)
└── SegmentationClassAug  (The ground truth masks, png files, about 1.1G)
(The followings are Optional)
├── pseudolabels (4.8G in total.)
    ├── toco (Pseudo label masks frome the original ToCo, png files, about 429M)
    ├── toco_r1 (Pseudo label masks from our scribble-promoted ToCo, png files, about 441M)
    ├── toco_r2
    ├── toco_r3
    ├── cutmix (Pseudo labels masks from the Cutmix, png files, about 418M)
    ├── has (Pseudo labels masks from the Has, png files, about 415m)
    └── recam (Pseudo labels masks from the ReCam, png files, about 360m)
├── scribble_dsmp (Distance maps of the scribbles. About 176G, recommend to generate from the code.)
└── pseudolabel_dsmp (Distance maps of the pseudo labels. About 160G, recommend to generate from the code.)
```
![ScribbleCOCO_vis](imgs/ScribbleCOCO_vis.png).

## ScribbleCityscapes
The complete dataset is available at [BaiduNetdisk](https://pan.baidu.com/s/1JDQkz211eXu_tzqlNw4stQ?pwd=hu5p).
```bash
ScribbleCityscapes/
├── ImageSets/SegmentationAug
    ├──train.txt
    └──val.txt
├── cityscapes_scribble_r1(The scribble masks, png files, about 74M.)
├── cityscapes_scribble_r2
├── cityscapes_scribble_r3
├── JPEGImages (The jpeg image files, about 1.6G)
└── SegmentationClassAug  (The ground truth masks, png files, about 171M)
(The followings are Optional)
├── pseudolabels (76M in total.)
    ├── ToCoR1 (Pseudo label masks from our scribble-promoted ToCo, png files, about 26M)
    ├── ToCoR2
    └── ToCoR3
├── scribble_dsmp (Distance maps of the scribbles. About 13G, recommend to generate from the code.)
└── pseudolabel_dsmp (Distance maps of the pseudo labels. About 16G, recommend to generate from the code.)
```
![ScribbleCityscapes_vis](imgs/ScribbleCityscapes_vis.png)



## ScribbleSup and ScribbleACDC
These two datasets are public datasets. 
**ScribbleSup**: The original scribble annotations were recorded as a serises of points, where you can find them in ../scribble_annotation/pascal_2012/*.xml. I recollected the ScribbleSup data in 2023 for scribble-supervised semantic segmentation. I convert them into the png files following this [code](https://github.com/meng-tang/rloss/blob/master/data/pascal_scribble/convertscribbles.m) by matlab. The complete dataset is availiable at [google drive](https://drive.google.com/file/d/1P_N_2RiJ0kYsz2A8-B5v3ltAxiXAmDGV/view?usp=sharing).

**ScribbleACDC**: The offical [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) dataset is a 3D medical image segmentation for Automated Cardiac Diagnosis Challenge. Valvano et. al provided the human annotated [scribble annotations](https://vios-s.github.io/multiscale-adversarial-attention-gates/data) in 2021. I here recollected this data following PASCAL VOC format. The complete dataset is availiable at [BaiduNetdisk](https://pan.baidu.com/s/1LGdEIFyjjmPcsX8sIDDt8Q?pwd=4wtu). The code is 4wtu.



