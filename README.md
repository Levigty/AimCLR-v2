# 3s-AimCLR++

This is an official PyTorch implementation of **"Improving Self-Supervised Action Recognition from Extremely Augmented Skeleton Sequences"**. 

It is the extension of our conference paper "Contrastive Learning from Extremely Augmented Skeleton Sequences for Self-supervised Action Recognition" in AAAI2022 (Oral). Code: [AimCLR](https://github.com/Levigty/AimCLR).

![](./fig/pipe.png)

**The current repo extends AimCLR in several ways:** 

- We simplify the structure of AimCLR and explore the combinations of extreme augmentation to propose AimCLR++ for a single stream. 
- We propose an effificient strategy to aggregate and interact the multi-stream information. 
- We add a lot of experiments to prove the advance of our method, such as exploring the robustness of the method to different extreme augmentations, the number of most similar neighbors, the performance of transfer learning, etc. 
- We extend the quantitative and qualitative experiments of 3s-AimCLR++ to show its superiority.

## Requirements
  ![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)    ![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.4-blue.svg)

## Data Preparation
- Download the raw data of [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) and [PKU-MMD](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html).
- For NTU RGB+D dataset, preprocess data with `tools/ntu_gendata.py`. For PKU-MMD dataset, preprocess data with `tools/pku_part1_gendata.py`.
- Then downsample the data to 50 frames with `feeder/preprocess_ntu.py` and `feeder/preprocess_pku.py`.
- If you don't want to process the original data, download the file folder in Google Drive [action_dataset](https://drive.google.com/drive/folders/1VnD3CLcD7bT5fMGI3tDGPlcWZmBbXS0m?usp=sharing) or BaiduYun link [action_dataset](https://pan.baidu.com/s/1NRK1ksRHgng_NkOO1ZYTcQ), code: 0211. NTU-120 is also provided: [NTU-120-frame50](https://drive.google.com/drive/folders/1dn8VMcT9BYi0KHBkVVPFpiGlaTn2GnaX?usp=sharing).

## Installation
  ```bash
# Install torchlight
$ cd torchlight
$ python setup.py install
$ cd ..
  
# Install other python libraries
$ pip install -r requirements.txt
  ```

## Unsupervised Pre-Training

Example for unsupervised pre-training of **3s-AimCLR++**. You can change some settings of `.yaml` files in `config/three-stream/pretext` folder.
```bash
# train on NTU RGB+D xview for three-stream
$ python main.py pretrain_aimclr_v2_3views --config config/three-stream/pretext/pretext_aimclr_v2_3views_ntu60_xview.yaml

# train on NTU RGB+D xsub for three-stream
$ python main.py pretrain_aimclr_v2_3views --config config/three-stream/pretext/pretext_aimclr_v2_3views_ntu60_xsub.yaml
```

## Linear Evaluation

Example for linear evaluation of **3s-AimCLR++**. You can change `.yaml` files in `config/three-stream/linear` folder.
```bash
# Linear_eval on NTU RGB+D xview for three-stream
$ python main.py linear_evaluation --config config/three-stream/linear/linear_eval_aimclr_v2_3views_ntu60_xview.yaml

# Linear_eval on NTU RGB+D xsub for three-stream
$ python main.py linear_evaluation --config config/three-stream/linear/linear_eval_aimclr_v2_3views_ntu60_xsub.yaml
```

## Linear Evaluation Results

|          Model          | NTU 60 xsub (%) | NTU 60 xview (%) | PKU Part I (%) | PKU Part II (%) |
| :---------------------: | :-------------: | :--------------: | :------------: | :-------------: |
|        3s-AimCLR        |      79.18      |      84.02       |     87.79      |      38.52      |
| 3s-AimCLR++ (This repo) |    **80.9**     |     **85.4**     |    **90.4**    |    **41.2**     |


## Citation
Please cite our paper if you find this repository useful in your resesarch:

```
@inproceedings{guo2022aimclr,
  Title= {Contrastive Learning from Extremely Augmented Skeleton Sequences for Self-supervised Action Recognition},
  Author= {Tianyu, Guo and Hong, Liu and Zhan, Chen and Mengyuan, Liu and Tao, Wang  and Runwei, Ding},
  Booktitle= {AAAI},
  Year= {2022}
}
```

## Licence

This project is licensed under the terms of the MIT license.
