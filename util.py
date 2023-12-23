import os
from colorama import Fore, Style
from copy import deepcopy

import glob
import torch
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torchinfo import summary
from ptflops import get_model_complexity_info
from thop import profile

def get_min_eer_ckpt(config):
    """Get the minimum EER checkpoint from the config file."""

    lightning_logs_dir = os.path.join(config.TRAINER_CONFIG.get('default_root_dir'), 'lightning_logs')
    version_list = os.listdir(lightning_logs_dir) # e.g.) ['version_0', 'version_1']

    min_eer_per_version = []
    ckpt_files = []
    for version in version_list:
        checkpoint_dir = os.path.join(lightning_logs_dir, version, 'checkpoints')
        ckpt_list = os.listdir(checkpoint_dir) # e.g.) ['ECAPA_epoch=0-min_eer_seg=6.33.ckpt', ..., 'ECAPA_epoch=1-min_eer_seg=5.11.ckpt']
        ckpt_files.extend(ckpt_list)

        values = [float(item.split('min_eer_seg=')[-1].split('.ckpt')[0]) for item in ckpt_list]
        min_eer_per_version.append(min(values))

    # find version which contains the minimum EER
    min_eer_version = version_list[min_eer_per_version.index(min(min_eer_per_version))] # e.g.) version_1

    # find minimum EER checkpoint file name from overall checkpoints
    min_eer_ckpt = min(ckpt_files, key=lambda item: float(item.split('min_eer_seg=')[-1].split('.ckpt')[0])) # e.g.) 'ECAPA_epoch=120-min_eer_seg=1.98.ckpt'

    min_eer_ckpt_path = os.path.join(lightning_logs_dir, min_eer_version, 'checkpoints', min_eer_ckpt)

    return min_eer_ckpt_path



def logging_terminal(config):
    """
    prints a config file defined in the configs folder in terminal
    """
    for attr in dir(config):
        if not attr.startswith('__'):
            attr_value = getattr(config, attr)
            print(f"{Fore.BLUE + attr} : {attr_value}")
    return

def find_min_eer_values(logdir, tags):
    """
    Args:
        logdir: training log directory
        tags: list of tags to find min values
    return:
        df: dataframe of min values
    """
    
    log_list = glob.glob(f"{logdir}/*/*/events*")

    if len(log_list) == 0:
        return print("No log file in the directory")
    
    tags = tags # ['min_eer', 'min_eer_seg']

    dfs = []
    for log in log_list:    
        event_acc = EventAccumulator(log)
        event_acc.Reload()

        data = []

        for tag in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(tag)
            for event in events:
                step = event.step
                value = event.value
                data.append([step, tag, value])

        df_version = pd.DataFrame(data, columns=["Step", "Tag", "Value"])        
        dfs.append(df_version[df_version['Tag'].isin(tags)].groupby('Tag').min())

    df = pd.concat(dfs, axis=0)

    print(df)

    return df


def get_model_param_mmac(model, input_size):
    """
    Args:
        model: model
        input_size: (time)
    Returns:
        macs: MMac
        params: M
    """
    model_test = deepcopy(model)

    # ptflpos
    MACs_ptflops, params_ptflops = get_model_complexity_info(model_test, (input_size,))
    MACs_ptflops, params_ptflops = MACs_ptflops.replace(" MMac", ""), params_ptflops.replace(" M", "")

    # thop
    input = torch.randn(1, input_size)
    MACs_thop, params_thop = profile(model_test, inputs=(input, ), verbose=False)
    MACs_thop, params_thop = MACs_thop/1e6, params_thop/1e6

    # torchinfo
    model_profile = summary(model_test, input_size=(1, input_size))
    MACs_torchinfo, params_torchinfo = model_profile.total_mult_adds/1e6, model_profile.total_params/1e6

    # summary(model_test, input_size=(1, input_size))

    # pring detail
    print(f"ptflops: MMac: {MACs_ptflops}, Params: {params_ptflops}")
    print(f"thop: MMac: {MACs_thop}, Params: {params_thop}")
    print(f"torchinfo: MMac: {MACs_torchinfo}, Params: {params_torchinfo}")

    del model_test

    return MACs_ptflops, params_ptflops, MACs_thop, params_thop, MACs_torchinfo, params_torchinfo

    

def calculate_real_time_factor(model, input_size, device, num_repeats=5000, warm_up=100, config=None, mel='wo_mel'):
    """
    Calculates the real time factor of a model given the input size and the device
     - Reference: https://devopedia.org/speech-recognition
    Args:
        model: the model to calculate the real time factor for
        input_size(int): the input size of the model(seconds)
        device: the device to calculate the real time factor on
        num_repeats: the number of repeats to calculate the real time factor
        warm_up: the number of warm up iterations to calculate the real time factor
        mel (str): 'wi_mel' or 'wo_mel' to switch between the two modes
    Returns:
        time_result_mean: the mean time of the model inference. (unit: seconds)
        rtf_mean: the real time factor of the model inference. (unit: seconds)
    """
    model = model.to(device)
    model.eval()

    if mel == 'wi_mel':
        input = torch.randn(1, input_size * 16000).to(device).contiguous()
    elif mel == 'wo_mel':
        if (config.MODEL == 'ResNetSE34L') or (config.MODEL == 'idx29_2_DenseNet'):
            input = torch.randn(1, 1, 40, 202).to(device).contiguous()
        else:
            input = torch.randn(1, 80, 202).to(device).contiguous()
    else:
        raise ValueError("Invalid value for mel. Expected 'wi_mel' or 'wo_mel'.")

    print(f"calculating real time factor on device {device} with input size {input_size} & input shape {input.shape}")

    # warm up
    with torch.inference_mode():
        for _ in range(warm_up):
            model(input)

    starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)

    time_results = []
    rtf_results = []

    # real time factor
    with torch.inference_mode():
        total_time = 0
        for _ in range(num_repeats):
            starter.record()
            _ = model(input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
        time_per_single_data = total_time / (num_repeats)
        real_time_factor = time_per_single_data / input_size
        time_results.append(time_per_single_data)
        rtf_results.append(real_time_factor)

    time_results_mean = np.mean(time_results)
    rtf_results_mean = np.mean(rtf_results)

    return time_results_mean, rtf_results_mean


def tb_to_csv(log_dir_list, output_csv_file):
    """
    Args:
        log_dir_list: List of tensorboard log directories
                     (e.g. ["events.out.tfevents.1692696158.xavier.3269943.0", ...])
        output_csv_file: output csv file path (e.g. tb.csv)
    
    """
    
    # list of dataframes
    dfs = []
    epoch = 0

    for log_dir in log_dir_list:
        # Set Event accumulator
        event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
        event_acc.Reload()

        # scalars
        scalars_cos = event_acc.Scalars('eer_org_full_cos')
        scalars_eu = event_acc.Scalars('eer_org_full_eu')
        scalars_seg = event_acc.Scalars('min_eer_seg')

        # convert to pandas dataframe
        max_length = max(len(scalars_cos), len(scalars_eu), len(scalars_seg))

        df = pd.DataFrame({
            "Epoch": list(range(epoch, max_length + epoch)),
            "Min_of_Full_cos_or_eu": [min(scalars_cos[i].value, scalars_eu[i].value) if i < max_length else None for i in range(max_length)],
            "2_seg_min": [scalars_seg[i].value if i < len(scalars_seg) else None for i in range(max_length)]
        })

        dfs.append(df)

        epoch += max_length

    # concat dataframes
    final_df = pd.concat(dfs, ignore_index=True)

    # save to csv
    final_df.to_csv(output_csv_file, index=False)

    return final_df