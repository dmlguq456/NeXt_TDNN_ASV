# NeXt-TDNN for Speaker Verification


## 0. Getting Start

### Prerequisites
This code requires the following:
* pytorch-lightning == 2.1.2

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
 
```bash
python vad_generator.py --mode test --test_path /home/hhj/speaker_verification/voxceleb1 --test_list ./data/veri_test2.txt --save_path ./data/veri_test
``` 

 


The directory structure of speaker verification looks like this:

```
├── aggregation           <- raw waveform → audio features
├── backend               <- compute cosine similarity, euclidean distance
├── configs               <- configuration files for train
├── data                  <- Project data
├── experiments           <- experiment backup.
├── loss                  <- speaker embedding → logit → Loss
├── models                <- audio features → Frame-level feature
├── optimizer             <- optimizer
├── preprocessing         <- raw waveform → audio features 
├── scheduler             <- learning rate scheduler
├── SpeakerNet.py                    <- preprocessing + models + aggregation + loss
├── engine.py                        <- Define train/test loop
├── eval_metric.py                   <- compute EER
├── main.py                          <- run train/test
└── util.py                           <- This includes utility functions such as measuring RTF and MACs
```




## 1. Model Training
To train ASV model, run main script in train mode. You can select the desired training configuration through config argument.

- to train Next-TDNN-Light (C=256, B=3)
```bash
python main.py --mode train --config configs/NeXt_TDNN_light_C256_B3_K65
```
- to train NeXt-TDNN(C=256, B=3)
```bash
python main.py --mode train --config configs/NeXt_TDNN_C256_B3_K65
```



## 2. Model Test
To test on VoxCeleb1, run the script below. As in training, select the desired test configuration.


```bash
# VoxCeleb1-O
python main.py --mode test --config configs/NeXt_TDNN_light_C256_B3_K65

# ⚡ VoxCeleb1-O, VoxCeleb1-E, VoxCeleb1-H
python main.py --mode test_all --config configs/NeXt_TDNN_light_C256_B3_K65
```



