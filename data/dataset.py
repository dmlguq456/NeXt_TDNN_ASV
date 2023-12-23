#! /usr/bin/python
# -*- encoding: utf-8 -*-
"""
reference
 - https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py
 - 
"""
# Reference
import random
import os
import itertools

import torch
import numpy as np
import glob
import librosa
import soundfile

import pandas as pd
from scipy import signal


# for classification learning
class train_dataset_classification(torch.utils.data.Dataset):
    """
    Ref1: https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py
    Ref2: https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/dataLoader.py
    Ref3: https://github.com/zyzisyz/mfa_conformer/blob/1b9c229948f8dbdbe9370937813ec75d4b06b097/module/augment.py#L10
    """
    def __init__(self, train_list, train_path, max_frames, augment, musan_path, rir_path, **kwargs):

        self.train_list = train_list
        self.train_path = train_path        
        self.max_frames = max_frames
        self.max_length = max_frames * 160 + 240
        self.musan_path = musan_path
        self.rir_path   = rir_path
        self.augment    = augment

        self.df = self.make_labels(self.train_list)
        self.file_name_list = list(self.df['filename'])
        self.label_list = list(self.df['label'])

        # print dataset info
        print(f"Number of speakers for training: {len(self.df['id'].unique())}") # 5994
        print(f"Number of utterances for training: {len(self.df)}") # 1092009


        # for data augmentation
        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {
            'noise':[],
            'speech':[],
            'music':[]
        }

        augment_files   = glob.glob(os.path.join(self.musan_path,'*/*/*/*.wav'))

        for file in augment_files:
            noise_type = file.split('/')[-4] # noise_type :  'noise' or 'speech' or 'music'
            assert noise_type in self.noisetypes, "noise type is not in ['noise','speech','music']"

            self.noiselist[noise_type].append(file)

        self.rir_files  = glob.glob(os.path.join(self.rir_path,'*/*/*.wav'))

    def make_labels(self, train_list):
        """
        Make labels for training data
        Args:
            train_list: list of training data(e.g. train_list.txt)

        Returns:
            df: dataframe of training data
                - id: speaker id (e.g. id00012, ... , id09272)
                - filename: path to wav file
                - label: speaker label (e.g. 0, 1, 2, ... , 5994)
        """
        # Read training files
        df = pd.read_csv(train_list, sep = " ", header = None, names = ['id', 'filename'])  
    
        # Exctract non overlap id list and copy it to 'label columns'
        id_list = list(df['id'])
        id_unique_list = list(df['id'].unique())

        map_id_to_label = { id : label for label, id in enumerate(id_unique_list) } # key: id, value: label
        label = [map_id_to_label[id] for id in id_list]
        df['label'] = label

        return df

    def __getitem__(self, idx):
        """
        Read wav file, zero-padding and data augmentation

        Ref1: https://github.com/zyzisyz/mfa_conformer/blob/master/module/dataset.py
        Ref2: https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py
        
        Returns:
            audio: audio data (shape: (self.max_length,))
        """
        # =====================================
        # 1. load audio & padding
        # =====================================
        filename = self.file_name_list[idx]
        file_path = os.path.join(self.train_path, filename)

        audio, _ = soundfile.read(file_path) # (129921,) (audiosize, )
        audio_length = audio.shape[0]

        # zero-padding
        if audio_length <= self.max_length:
            shortage    = self.max_length - audio_length
            audio       = np.pad(audio, (0, shortage), 'wrap')
            audio = audio.astype(float) # audio.shape : (self.max_length, )
        else:
            start = int(random.random()*(audio_length - self.max_length))
            assert start >= 0, "start index is negative" # for debugging, remove this line when you are sure
            audio = audio[start:start+self.max_length].astype(float) # audio.shape : (self.max_length, )


        # =====================================
        # 2. data augment
        # =====================================
        if self.augment:
            augtype = random.randint(0,5)
            if augtype == 1: # Reverberation
                audio   = self.add_rev(audio)
            elif augtype == 2: # Music
                audio   = self.add_noise('music',audio)
            elif augtype == 3: # Babble
                audio   = self.add_noise('speech',audio)
            elif augtype == 4: # Noise
                audio   = self.add_noise('noise',audio)
            elif augtype == 5: # Television noise
                audio = self.add_noise('speech', audio)
                audio = self.add_noise('music', audio)


        return torch.FloatTensor(audio), self.label_list[idx] # audio[0].shape : (32240,)

    def __len__(self):
        return len(self.df)
    
    def add_noise(self, noisecat, audio):
        """
        Ref1: https://github.com/zyzisyz/mfa_conformer/blob/1b9c229948f8dbdbe9370937813ec75d4b06b097/module/augment.py#L10
        Ref2: https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/dataLoader.py
        Args:
            noisecat: noise category (e.g. 'noise', 'speech', 'music')
            audio (np.array): audio data (shape: (self.max_length,))
        Returns:
            audio (np.array): audio data with noise (shape: (self.max_length,))
        """

        # compute DB
        val = max(0.0, np.mean(np.power(audio, 2)))
        clean_db = 10*np.log10(val+1e-4)

        # select noise
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        assert audio.ndim == 1, "audio dim is not 1" # for debugging, remove this line when you are sure
        audio_length = len(audio)

        noises = []

        for noise in noiselist:

            # read noise file
            noiseaudio, _  = soundfile.read(noise) # (noise_length, )
            noiseaudio = noiseaudio.astype(float)
            noise_length = len(noiseaudio)

            # zero-padding for noise to match audio length
            if noise_length <= audio_length:
                shortage = audio_length - noise_length
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
                noiseaudio = noiseaudio.astype(float)
            else:
                start_frame = int(random.random()*(noise_length - audio_length))
                noiseaudio = noiseaudio[start_frame:start_frame + audio_length]

            # set snr
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            val = max(0.0, np.mean(np.power(noiseaudio, 2)))
            noise_db = 10*np.log10(val+1e-4)

            noiseaudio = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio
            noises.append(noiseaudio)

        noise_stack = np.stack(noises,axis=0) # (numnoise, audio_length)
        noise_sum = np.sum(noise_stack,axis=0) # (audio_length,)

        assert noise_sum.shape[0] == audio.shape[0] == self.max_length, "noise_sum.shape != audio.shape" # for debugging, remove this line when you are sure

        return noise_sum + audio

    def add_rev(self, audio):
        """
        Ref1: https://github.com/zyzisyz/mfa_conformer/blob/1b9c229948f8dbdbe9370937813ec75d4b06b097/module/augment.py#L109
        Ref2: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html
        Args:
            audio (np.array): audio data (shape: (self.max_length,))
        Returns:
            audio (np.array): audio data with reverberation (shape: (self.max_length,))
        """

        rir_file    = random.choice(self.rir_files)
        
        rir, _     = soundfile.read(rir_file) # shape: (rir length,), (e.g. rir length == 8,000)
        rir         = rir / np.sqrt(np.sum(rir**2)) # (1, rir length)

        waveform = signal.convolve(audio, rir, mode='full')

        return waveform[:self.max_length] # (self.max_length)
    


    

# =====================================
# for test
# =====================================
class test_dataset(torch.utils.data.Dataset):
    def __init__(self, test_list, test_path, **kwargs):
        self.test_path  = test_path

        # make test list
        self.test_list = self.make_test_list(test_list)

        print(f"Number of utterances for test: {len(self.test_list)}") # 4715

    def make_test_list(self, test_list):
        """
        make test list
        it contains all test files
        in VoxCeleb1-o it contains 4715 files(utterances)
        """        
        with open(test_list) as f:
            lines = f.readlines()

        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))

        return sorted(setfiles)

    def __getitem__(self, index):
        """        
        return: audio segment(e.g. [audio length, ])
        """        
        file_path = os.path.join(self.test_path,self.test_list[index])
        audio, _ = soundfile.read(file_path)

        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)

