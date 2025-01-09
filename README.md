# ScribbleSimulation
This repository is created for reimplementation of the ScribbleCOCO and ScribbleCityscapes. 

## ScribbleCOCO
The complete dataset is avaliable at [BaiduNetdisk]().
![ScribbleCOCO_vis](imgs/ScribbleCOCO_vis.png)

## ScribbleCityscapes
The complete dataset is available at [BaiduNetdisk]().
![ScribbleCityscapes_vis](imgs/ScribbleCityscapes.png)

## ScribbleSup and ScribbleACDC
These two datasets are public datasets. 
**ScribbleSup**: The original scribble annotations were recorded as a serises of points, where you can find them in ../scribble_annotation/pascal_2012/*.xml. I recollected the ScribbleSup data in 2023 for scribble-supervised semantic segmentation. I convert them into the png files following this [code](https://github.com/meng-tang/rloss/blob/master/data/pascal_scribble/convertscribbles.m) by matlab. The complete dataset is availiable at [google drive](https://drive.google.com/file/d/1P_N_2RiJ0kYsz2A8-B5v3ltAxiXAmDGV/view?usp=sharing).

**ScribbleACDC**: The offical [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) dataset is a 3D medical image segmentation for Automated Cardiac Diagnosis Challenge. Valvano et. al provided the human annotated [scribble annotations](https://vios-s.github.io/multiscale-adversarial-attention-gates/data) in 2021. I here recollected this data following PASCAL VOC format. The complete dataset is availiable at [BaiduNetdisk]()



