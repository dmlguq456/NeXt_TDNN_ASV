
import torch
import torch.nn.functional as F
from typing import List


def split_segment(audio, num_seg:int=10, eval_frames:int=300, **kwargs)-> torch.FloatTensor:
    """
    split audio to num_seg segments
    before : [BATCH, audio_length] -> after : [BATCH, num_seg, EVAL_MAXFRAMES]

    Ref paper: In defence of metric learning for speaker recognition[https://arxiv.org/pdf/2003.11982.pdf]
    Ref code1: https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py
    Ref code2: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    Args:
        x (torch tensor): audio signal [BATCH, audio_length], where batch == 1
        num_seg (int): number of segment
        eval_frame (int): number of frame for evaluation
    Return:
        x (torch tensor): audio signal [BATCH, num_seg, EVAL_MAXFRAMES]
    """
    BATCH, LENGTH = audio.shape
    EVAL_MAXFRAMES = eval_frames * 160 + 240

    DEVICE = audio.device

    assert BATCH == 1, "batch size should be 1"

    # zero-padding
    if LENGTH <= EVAL_MAXFRAMES:
        shortage    = EVAL_MAXFRAMES - LENGTH
        # audio       = F.pad(audio[0], (0, shortage), mode= '') # audio[0]: select batch, this is not necessary
        audio       = F.pad(audio, (0, shortage), 'constant', 0) # same as above, see documents
        LENGTH   = audio.shape[1]
        assert LENGTH == EVAL_MAXFRAMES, "audiosize should be equal to MAXFRAMES"

    startframe = torch.linspace(0,LENGTH-EVAL_MAXFRAMES, steps=num_seg)

    audio_seg_list = []
    for asf in startframe:
        seg = audio[:, int(asf):int(asf)+EVAL_MAXFRAMES] # (BATCH, EVAL_MAXFRAMES)
        audio_seg_list.append(seg)

    audio_seg = torch.stack(audio_seg_list,axis=0) # [numseg, BATCH, EVAL_MAXFRAMES]
    audio_seg = audio_seg.permute(1,0,2) # [BATCH, numseg, EVAL_MAXFRAMES]

    if DEVICE.type == 'cuda':
        audio_seg = torch.cuda.FloatTensor(audio_seg)
    else:
        audio_seg = torch.FloatTensor(audio_seg)

    return audio_seg.to(DEVICE) # [BATCH, numseg, EVAL_MAXFRAMES]

def slice_short_utterance(audio, sampling_rate:int=16000, short_length:int=2, **kwargs)-> torch.FloatTensor:
    """
    slice short utterance to 'sampling_rate * short_length' based on middle point of audio
    before : [BATCH, audio_length] -> after : [BATCH, sliced_length]
    Args:
        audio (torch tensor): audio signal [BATCH, audio_length], where batch == 1
        sampling_rate (int): sampling rate
        short_length (int): audio length (unit: second)
    Return:
        audio (torch tensor): audio signal [BATCH, sliced_length], where batch == 1 & sliced_length == sampling_rate * short_length
    """
    BATCH, LENGTH = audio.shape
    SLICED_LENGTH = sampling_rate * short_length

    assert BATCH == 1, "batch size should be 1"

    center_index = LENGTH//2

    # calculate the number of samples for desired length(sliced_length) / 2
    half_sample = SLICED_LENGTH // 2

    sliced_data  = audio[:, max(0, center_index - half_sample) : min(LENGTH, center_index + half_sample)]

    return sliced_data  # [BATCH, sliced_length]
    
    
