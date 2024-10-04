# NeXt-TDNN for Speaker Verification [ICASSP 2024]

This repository is the official implementation of "NeXt-TDNN: Modernizing Multi-Scale Temporal Convolution Backbone for Speaker Verification" accepted in ICASSP 2024 [Paper Link (Arxiv)](https://arxiv.org/abs/2312.08603) / [Paper Link (IEEE)](https://ieeexplore.ieee.org/abstract/document/10447037)

<p align="center"><img src="NeXt_TDNN_structure.png" width="550" /></p>

## News
🔥 December, 2023: We have uploaded the pre-trained models of our NeXt-TDNN in the `experiments` folder!

🔥 February 2024, the NeXt-TDNN model was updated with cyclic learning rate scheduling. This update improved the EER from 0.79/1.04/1.82% to 0.72/0.94/1.68% in VoxCeleb1-O/E/H. Changes were made to the LR scheduling, gradient clipping value, and batch size. Please check `configs/NeXt_TDNN_C256_B3_K65_7_cyclical_lr_step.py` for details.


## 0. Getting Start

### Prerequisites
This code requires the following:
* lightning == 2.1.2

### Installation

* CUDA, PyToch installation
```
# CUDA
conda install -c "nvidia/label/cuda-11.8.0" cuda

# PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
### Data preparation
- [VoxCeleb Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html#about)
  - To download VoxCeleb dataset fot train/test, execute the command described in the Data preparation section of the [voxceleb_trainer repository](https://github.com/clovaai/voxceleb_trainer)
  - Download [VoxCeleb1-O](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt), [VoxCeleb1-E](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt), and [VoxCeleb1-H](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt)  for test and locate it `data` directory
 
## 1. Model Training
To train ASV model, run main script in train mode. You can select the desired training configuration through config argument.

- to train NeXt-TDNN(C=256, B=3)
```bash
python main.py --mode train --config configs/NeXt_TDNN_C256_B3_K65_7
```



## 2. Model Test
To test on VoxCeleb1, run the script below. As in training, select the desired test configuration.

<p align="center"><img src="table_result.png"/></p>

```bash
# VoxCeleb1-O
python main.py --mode test --config configs/NeXt_TDNN_C256_B3_K65_7

# ⚡ VoxCeleb1-O, VoxCeleb1-E, VoxCeleb1-H
python main.py --mode test_all --config configs/NeXt_TDNN_C256_B3_K65_7
```


## 3. Reference
- https://github.com/facebookresearch/ConvNeXt-V2
- https://github.com/clovaai/voxceleb_trainer
- https://github.com/mechanicalsea/sugar
- https://github.com/TaoRuijie/ECAPA-TDNN
- https://github.com/speechbrain/speechbrain
- https://github.com/zyzisyz/mfa_conformer


## 4. Citation

If you find our work useful, please refer to 
```
@INPROCEEDINGS{10447037,
  author={Heo, Hyun-Jun and Shin, Ui-Hyeop and Lee, Ran and Cheon, YoungJu and Park, Hyung-Min},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={NeXt-TDNN: Modernizing Multi-Scale Temporal Convolution Backbone for Speaker Verification}, 
  year={2024},
  volume={},
  number={},
  pages={11186-11190},
  keywords={Convolution;Speech recognition;Transformers;Acoustics;Task analysis;Speech processing;speaker recognition;speaker verification;TDNN;ConvNeXt;multi-scale},
  doi={10.1109/ICASSP48485.2024.10447037}}
```
